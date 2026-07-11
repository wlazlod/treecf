"""Public API: Explainer and result types (spec §10)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from treecf._errors import TreecfError
from treecf.aim.build import build_problem, swap_target
from treecf.aim.cells import cell_index, feature_cells
from treecf.aim.model import AimProblem, BuildInfeasible
from treecf.backends.base import Backend, BackendSolution, DiversityCut
from treecf.constraints.compile import compile_constraints
from treecf.constraints.objects import Constraint
from treecf.ir.evaluate import apply_link, raw_score
from treecf.ir.model import EnsembleIR, Link
from treecf.ir.parsers import parse_model
from treecf.objective import fit_normalizers
from treecf.plausibility import Plausibility
from treecf.targets import Target

FloatArray = npt.NDArray[np.float64]

_MAX_FIXED_POINT_RETRIES = 2


@dataclass(frozen=True)
class Grid:
    """Value policy: snap to ``anchor + k * step`` (spec §5.6)."""

    step: float
    anchor: float = 0.0


ValuePolicy = str | Grid | Callable[[float], float]


@dataclass(frozen=True)
class Counterfactual:
    x_cf: FloatArray
    changes: dict[str, tuple[float, float]]
    distance: float
    n_changed: int
    score_raw: float
    score_prob: float | None
    proof: str  # "optimal" | "feasible" | "heuristic"
    gap: float | None = None
    solver_stats: dict[str, object] = field(default_factory=dict)
    snapped: dict[str, bool] = field(default_factory=dict)  # value_policy outcome (§5.6)


@dataclass(frozen=True)
class Infeasible:
    reason: str
    relaxation_hint: str | None = None


class Explainer:
    """Counterfactual explainer for a tree-ensemble model (spec §10).

    ``model`` may be a native model object, a dump file path/dict, or an
    ``EnsembleIR``. ``background`` fits the distance normalizers (§4.1);
    alternatively pass ``normalizers`` explicitly (array or name->sigma dict).
    """

    def __init__(
        self,
        model: object,
        background: FloatArray | None = None,
        constraints: Sequence[Constraint] = (),
        weights: dict[str, float] | None = None,
        normalizers: FloatArray | dict[str, float] | None = None,
        value_policy: dict[str, ValuePolicy] | None = None,
        plausibility: Plausibility | None = None,
    ) -> None:
        self.ir = model if isinstance(model, EnsembleIR) else parse_model(model)
        names = self.ir.feature_names
        self.compiled = compile_constraints(constraints, names)
        self.plausibility = plausibility
        if plausibility is not None:
            if plausibility.if_ir.n_features != self.ir.n_features:
                raise TreecfError("plausibility forest must share the model's feature space")
            if self.compiled.allow_missing:
                raise TreecfError(
                    "plausibility with AllowMissing is not supported in v0.1 "
                    "(isolation forests define no NaN routing)"
                )
        self.background = (
            None if background is None else np.asarray(background, dtype=np.float64)
        )
        self.sigma = _resolve_sigma(names, background, normalizers)
        self.weights = np.array([(weights or {}).get(name, 1.0) for name in names])
        self.value_policy = value_policy or {}
        for name, policy in self.value_policy.items():
            if name not in names:
                raise TreecfError(f"value_policy references unknown feature {name!r}")
            if isinstance(policy, str) and policy not in ("raw", "integer"):
                raise TreecfError(f"unknown value policy {policy!r} for {name!r}")

    def explain(
        self,
        x: FloatArray,
        target: Target,
        backend: str = "cpsat",
        time_budget_s: float = 10.0,
        sparsity_weight: float = 0.0,
        num_workers: int = 0,
        seed: int | None = None,
        n_counterfactuals: int = 1,
        diversity_mode: str = "distinct_changes",
    ) -> Counterfactual | Infeasible | list[Counterfactual] | dict[str, object]:
        x = np.asarray(x, dtype=np.float64)
        if self.plausibility is not None and np.isnan(x).any():
            raise TreecfError(
                "plausibility with missing factual values is not supported in v0.1"
            )
        if target.bands_spec is not None:
            return self._explain_bands(
                x, target, backend, time_budget_s, sparsity_weight, num_workers, seed
            )
        interval = target.raw_interval(self.ir.link)
        if backend == "genetic":
            if n_counterfactuals > 1:
                raise TreecfError("n_counterfactuals > 1 requires backend='cpsat' in v0.1 (§8.3)")
            return self._explain_genetic(x, interval, time_budget_s, sparsity_weight, seed)
        solver = _select_backend(backend)
        if n_counterfactuals > 1:
            return self._explain_diverse(
                x, interval, solver, time_budget_s, sparsity_weight, num_workers,
                n_counterfactuals, diversity_mode,
            )

        scale_k = 10**6
        last_reason = "unknown"
        for _attempt in range(1 + _MAX_FIXED_POINT_RETRIES):
            problem = build_problem(
                self.ir, x, interval, self.compiled, self.sigma, self.weights,
                lam=sparsity_weight, scale_k=scale_k,
                plausibility=self._plausibility_bound(),
            )
            if isinstance(problem, BuildInfeasible):
                if problem.resolution_related:
                    scale_k *= 10
                    last_reason = problem.reason
                    continue
                return Infeasible(reason=problem.reason)

            solution = solver.solve(problem, time_budget_s=time_budget_s, num_workers=num_workers)
            if solution.status == "unknown":
                return Infeasible(
                    reason="time budget exhausted before any feasible solution was found "
                    "(not proven infeasible; raise time_budget_s)"
                )
            if solution.status == "infeasible":
                return Infeasible(
                    reason="no counterfactual satisfies target and constraints",
                    relaxation_hint=self._relaxation_hint(x, interval, time_budget_s),
                )

            x_cf = _extract_x_cf(problem, solution, x)
            verification = self._verify(x, x_cf, interval)
            if verification is None:
                x_cf, snapped = self._apply_value_policies(x, x_cf, interval)
                return self._result(
                    x, x_cf, solution.status, solution.gap, solution.stats, snapped
                )
            # Fixed-point artifact (should be prevented by §5.4 widening): retry finer.
            scale_k *= 10
            last_reason = verification

        return Infeasible(
            reason=f"fixed-point verification failed after retries: {last_reason}"
        )

    def _explain_bands(
        self,
        x: FloatArray,
        target: Target,
        backend: str,
        time_budget_s: float,
        sparsity_weight: float,
        num_workers: int,
        seed: int | None,
    ) -> dict[str, object]:
        """Rating ladder: one AIM compilation, one solve per band (§6, D9)."""
        intervals = target.band_intervals(self.ir.link)
        results: dict[str, object] = {}

        def full_solve(interval: tuple[float, float]) -> Counterfactual | Infeasible:
            single = Target.raw(range=interval)
            out = self.explain(
                x, single, backend=backend, time_budget_s=time_budget_s,
                sparsity_weight=sparsity_weight, num_workers=num_workers, seed=seed,
            )
            assert isinstance(out, Counterfactual | Infeasible)
            return out

        if backend == "genetic":
            return {name: full_solve(iv) for name, iv in intervals.items()}

        solver = _select_backend(backend)
        first = next(iter(intervals.values()))
        problem = build_problem(
            self.ir, x, first, self.compiled, self.sigma, self.weights,
            lam=sparsity_weight, plausibility=self._plausibility_bound(),
        )
        if isinstance(problem, BuildInfeasible):
            return {name: full_solve(iv) for name, iv in intervals.items()}

        for name, interval in intervals.items():
            band_problem = swap_target(problem, interval)
            if band_problem.score_lo > band_problem.score_hi:
                results[name] = Infeasible(reason="empty target interval")
                continue
            solution = solver.solve(
                band_problem, time_budget_s=time_budget_s, num_workers=num_workers
            )
            if solution.status not in ("optimal", "feasible"):
                reason = (
                    f"band {name!r} is unreachable under the constraints"
                    if solution.status == "infeasible"
                    else f"band {name!r}: time budget exhausted before a solution was found"
                )
                results[name] = Infeasible(reason=reason)
                continue
            x_cf = _extract_x_cf(band_problem, solution, x)
            if self._verify(x, x_cf, interval) is not None:
                results[name] = full_solve(interval)  # rare fixed-point retry path
                continue
            x_cf, snapped = self._apply_value_policies(x, x_cf, interval)
            results[name] = self._result(
                x, x_cf, solution.status, solution.gap, solution.stats, snapped
            )
        return results

    def _explain_diverse(
        self,
        x: FloatArray,
        interval: tuple[float, float],
        solver: Backend,
        time_budget_s: float,
        sparsity_weight: float,
        num_workers: int,
        n_counterfactuals: int,
        diversity_mode: str,
    ) -> list[Counterfactual] | Infeasible:
        """Iterative no-good cuts (§8.3); NaN flips count as changes (OQ3)."""
        if diversity_mode not in ("distinct_changes", "distinct_solution"):
            raise TreecfError("diversity_mode must be 'distinct_changes' or 'distinct_solution'")
        problem = build_problem(
            self.ir, x, interval, self.compiled, self.sigma, self.weights,
            lam=sparsity_weight, plausibility=self._plausibility_bound(),
        )
        if isinstance(problem, BuildInfeasible):
            return Infeasible(reason=problem.reason)

        results: list[Counterfactual] = []
        cuts: list[DiversityCut] = []
        from treecf.backends.cpsat import CpsatBackend

        assert isinstance(solver, CpsatBackend), "diversity requires the CP-SAT backend"
        for _ in range(n_counterfactuals):
            solution = solver.solve(
                problem, time_budget_s=time_budget_s, num_workers=num_workers, cuts=cuts
            )
            if solution.status not in ("optimal", "feasible"):
                break
            x_cf = _extract_x_cf(problem, solution, x)
            if self._verify(x, x_cf, interval) is not None:
                break
            x_cf, snapped = self._apply_value_policies(x, x_cf, interval)
            results.append(
                self._result(x, x_cf, solution.status, solution.gap, solution.stats, snapped)
            )
            all_positions = range(len(problem.features))
            cuts.append(
                DiversityCut(
                    mode=diversity_mode,
                    changed=solution.changed_positions,
                    unchanged=tuple(
                        p for p in all_positions if p not in solution.changed_positions
                    ),
                    chosen=solution.chosen_cells,
                )
            )
        if not results:
            return Infeasible(reason="no counterfactual satisfies target and constraints")
        return results

    def _relaxation_hint(
        self, x: FloatArray, interval: tuple[float, float], time_budget_s: float
    ) -> str | None:
        """Greedy relaxation ladder (§8.1): smallest constraint group whose removal helps."""
        from treecf.constraints.objects import (
            AllowMissing,
            Equals,
            Freeze,
            Monotone,
            Range,
        )

        constraints = self.compiled.constraints
        if not constraints:
            return None
        freedom: list[Constraint] = [c for c in constraints if isinstance(c, AllowMissing)]
        bounds: list[Constraint] = [
            c for c in constraints if isinstance(c, Freeze | Monotone | Range | Equals)
        ]
        budget = min(2.0, time_budget_s)

        def feasible_with(subset: list[Constraint]) -> bool:
            compiled = compile_constraints(subset, self.ir.feature_names)
            plaus = self._plausibility_bound()
            problem = build_problem(
                self.ir, x, interval, compiled, self.sigma, self.weights,
                lam=0.0, plausibility=plaus,
            )
            if isinstance(problem, BuildInfeasible):
                return False
            from treecf.backends.cpsat import CpsatBackend

            solution = CpsatBackend().solve(problem, time_budget_s=budget)
            return solution.status != "infeasible"

        if not feasible_with(freedom):
            return "the target interval is unreachable for this model even without constraints"
        if not feasible_with(freedom + bounds):
            return (
                "per-feature constraints (Freeze/Range/Monotone/Equals) make the target "
                "unreachable; relaxing them restores feasibility"
            )
        return (
            "relational constraints (Linear/Implies/OneHot) make the target unreachable; "
            "relaxing them restores feasibility"
        )

    def _explain_genetic(
        self,
        x: FloatArray,
        interval: tuple[float, float],
        time_budget_s: float,
        sparsity_weight: float,
        seed: int | None,
    ) -> Counterfactual | Infeasible:
        from treecf.backends.genetic import solve_genetic

        result = solve_genetic(
            self.ir,
            x,
            interval,
            self.compiled,
            self.sigma,
            self.weights,
            lam=sparsity_weight,
            background=self.background,
            plausibility=self._plausibility_bound(),
            seed=seed,
            time_budget_s=time_budget_s,
        )
        if result.x_cf is None:
            return Infeasible(reason="heuristic search exhausted (genetic backend, §8.2)")
        x_cf = result.x_cf
        verification = self._verify(x, x_cf, interval)
        if verification is not None:  # defensive: the GA only returns checked individuals
            return Infeasible(reason=f"heuristic solution failed verification: {verification}")
        x_cf, snapped = self._apply_value_policies(x, x_cf, interval)
        return self._result(x, x_cf, "heuristic", None, result.stats, snapped)

    def _verify(
        self, x: FloatArray, x_cf: FloatArray, interval: tuple[float, float]
    ) -> str | None:
        """Float-space re-check of target and constraints (§8.1 step 5). None = OK."""
        score = raw_score(self.ir, x_cf)
        if not (interval[0] <= score <= interval[1]):
            return f"score {score} outside target {interval}"
        lo, hi, _frozen = self.compiled.instance_bounds(x)  # bounds anchor at the factual x
        lo = np.where(np.isnan(lo), -math.inf, lo)
        hi = np.where(np.isnan(hi), math.inf, hi)
        for j, value in enumerate(x_cf):
            if math.isnan(value):
                if not math.isnan(x[j]) and j not in self.compiled.allow_missing:
                    return f"feature {self.ir.feature_names[j]!r} became NaN without AllowMissing"
                continue
            if not (lo[j] <= value <= hi[j]):
                return f"feature {self.ir.feature_names[j]!r} violates its bounds"

        slack = 1e-9  # integer-encoded constraints are exact; rounded coefs leave float dust
        for lin in self.compiled.linears:
            values = [x_cf[j] for j in lin.indices]
            if any(math.isnan(v) for v in values):
                if lin.missing_policy == "satisfied":
                    continue
                return "Linear constraint references a missing value"
            total = sum(c * v for c, v in zip(lin.coefs, values, strict=True))
            ok = (
                total <= lin.rhs + slack
                if lin.op == "<="
                else total >= lin.rhs - slack
                if lin.op == ">="
                else abs(total - lin.rhs) <= slack
            )
            if not ok:
                return f"Linear constraint violated: {lin.coefficients} {lin.op} {lin.rhs}"
        for imp in self.compiled.implications:
            if x_cf[imp.cond_index] == imp.cond_value and x_cf[imp.cons_index] != imp.cons_value:
                return "Implies constraint violated"
        for group in self.compiled.onehot_groups:
            if sum(x_cf[j] for j in group) != 1.0:
                return "OneHot constraint violated"
        if self.plausibility is not None:
            score_anomaly = self.plausibility.anomaly_score(x_cf)
            if score_anomaly > self.plausibility.max_anomaly_score + 1e-12:
                return f"anomaly score {score_anomaly:.4f} exceeds plausibility bound"
        return None

    def _plausibility_bound(self) -> tuple[EnsembleIR, float] | None:
        if self.plausibility is None:
            return None
        return self.plausibility.if_ir, self.plausibility.min_total_path

    def _apply_value_policies(
        self, x: FloatArray, x_cf: FloatArray, interval: tuple[float, float]
    ) -> tuple[FloatArray, dict[str, bool]]:
        """Snap changed values per policy inside their cells (§5.6); never break validity.

        The unsnapped ``x_cf`` is already verified, so reverting offending features
        one by one is guaranteed to terminate in a valid state.
        """
        applicable = [
            (j, name, self.value_policy[name])
            for j, name in enumerate(self.ir.feature_names)
            if name in self.value_policy
            and self.value_policy[name] != "raw"
            and not math.isnan(x_cf[j])
            and x_cf[j] != x[j]
        ]
        if not applicable:
            return x_cf, {}

        cells = feature_cells(self.ir)
        lo_b, hi_b, _ = self.compiled.instance_bounds(x)
        snapped: dict[str, bool] = {}
        candidate = x_cf.copy()
        for j, name, policy in applicable:
            cell = cells[j][cell_index(cells[j], x_cf[j])]
            value = _snap(x_cf[j], policy, cell.contains, float(lo_b[j]), float(hi_b[j]))
            if value is None:
                snapped[name] = False
            else:
                candidate[j] = value
                snapped[name] = True

        # Revert snapped features one at a time until the candidate verifies.
        order = [name for name in snapped if snapped[name]]
        while self._verify(x, candidate, interval) is not None and order:
            name = order.pop()
            j = self.ir.feature_names.index(name)
            candidate[j] = x_cf[j]
            snapped[name] = False
        if self._verify(x, candidate, interval) is not None:
            return x_cf, dict.fromkeys(snapped, False)
        return candidate, snapped

    def _result(
        self,
        x: FloatArray,
        x_cf: FloatArray,
        status: str,
        gap: float | None,
        stats: dict[str, object],
        snapped: dict[str, bool] | None = None,
    ) -> Counterfactual:
        changes: dict[str, tuple[float, float]] = {}
        distance = 0.0
        for j, name in enumerate(self.ir.feature_names):
            x_nan, cf_nan = math.isnan(x[j]), math.isnan(x_cf[j])
            if (x[j] == x_cf[j]) or (x_nan and cf_nan):
                continue
            changes[name] = (float(x[j]), float(x_cf[j]))
            if cf_nan:  # value -> NaN priced by delta_miss (§4.2)
                delta = self.compiled.allow_missing[j][0]
            elif x_nan:  # NaN -> value priced by delta_from_miss
                delta = self.compiled.allow_missing[j][1]
            else:
                delta = abs(x_cf[j] - x[j])
            distance += self.weights[j] * delta / self.sigma[j]
        score = raw_score(self.ir, x_cf)
        return Counterfactual(
            x_cf=x_cf,
            changes=changes,
            distance=float(distance),
            n_changed=len(changes),
            score_raw=score,
            score_prob=apply_link(Link.SIGMOID, score) if self.ir.link is Link.SIGMOID else None,
            proof=status,
            gap=gap,
            solver_stats=stats,
            snapped=snapped or {},
        )


def _extract_x_cf(problem: AimProblem, solution: BackendSolution, x: FloatArray) -> FloatArray:
    """Map solver values back to floats; the factual anchor means 'unchanged'."""
    assert solution.values_scaled is not None
    x_cf = x.copy()
    for block in problem.features:
        if solution.missing.get(block.index):
            x_cf[block.index] = math.nan
            continue
        v_int = solution.values_scaled[block.index]
        if block.x_cell is not None and v_int == block.x_scaled:
            x_cf[block.index] = x[block.index]
        else:
            x_cf[block.index] = v_int / problem.scale_k
    return x_cf


def _snap(
    value: float,
    policy: ValuePolicy,
    in_cell: Callable[[float], bool],
    lo: float,
    hi: float,
) -> float | None:
    """Nearest policy-conforming value inside the cell and bounds, or None."""
    if callable(policy):
        candidates = [float(policy(value))]
    elif policy == "integer":
        candidates = sorted({math.floor(value), math.ceil(value)}, key=lambda c: abs(c - value))
    else:
        assert isinstance(policy, Grid)
        base = policy.anchor + policy.step * round((value - policy.anchor) / policy.step)
        candidates = sorted(
            {base, base - policy.step, base + policy.step}, key=lambda c: abs(c - value)
        )
    for c in candidates:
        c = float(c)
        if in_cell(c) and lo <= c <= hi:
            return c
    return None


def _resolve_sigma(
    names: tuple[str, ...],
    background: FloatArray | None,
    normalizers: FloatArray | dict[str, float] | None,
) -> FloatArray:
    if normalizers is not None:
        if isinstance(normalizers, dict):
            missing = [n for n in names if n not in normalizers]
            if missing:
                raise TreecfError(f"normalizers missing features: {missing}")
            sigma = np.array([float(normalizers[n]) for n in names])
        else:
            sigma = np.asarray(normalizers, dtype=np.float64)
    elif background is not None:
        sigma = fit_normalizers(np.asarray(background, dtype=np.float64))
    else:
        raise TreecfError("provide either background (to fit normalizers) or normalizers")
    if len(sigma) != len(names) or np.any(sigma <= 0):
        raise TreecfError("normalizers must be positive, one per feature")
    return sigma


def _select_backend(name: str) -> Backend:
    if name == "cpsat":
        from treecf.backends.cpsat import CpsatBackend

        return CpsatBackend()
    if name == "highs":
        from treecf.backends.highs import HighsBackend

        return HighsBackend()
    raise TreecfError(f"unknown backend {name!r}; use 'cpsat' or 'genetic'")
