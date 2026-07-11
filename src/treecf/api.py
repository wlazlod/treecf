"""Public API: Explainer and result types (spec §10)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from treecf._errors import TreecfError
from treecf.aim.build import build_problem
from treecf.aim.model import BuildInfeasible
from treecf.backends.base import Backend
from treecf.constraints.compile import compile_constraints
from treecf.constraints.objects import Constraint
from treecf.ir.evaluate import apply_link, raw_score
from treecf.ir.model import EnsembleIR, Link
from treecf.ir.parsers import parse_model
from treecf.objective import fit_normalizers
from treecf.targets import Target

FloatArray = npt.NDArray[np.float64]

_MAX_FIXED_POINT_RETRIES = 2


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
    ) -> None:
        self.ir = model if isinstance(model, EnsembleIR) else parse_model(model)
        names = self.ir.feature_names
        self.compiled = compile_constraints(constraints, names)
        self.sigma = _resolve_sigma(names, background, normalizers)
        self.weights = np.array([(weights or {}).get(name, 1.0) for name in names])

    def explain(
        self,
        x: FloatArray,
        target: Target,
        backend: str = "cpsat",
        time_budget_s: float = 10.0,
        sparsity_weight: float = 0.0,
        num_workers: int = 0,
    ) -> Counterfactual | Infeasible:
        x = np.asarray(x, dtype=np.float64)
        interval = target.raw_interval(self.ir.link)
        solver = _select_backend(backend)

        scale_k = 10**6
        last_reason = "unknown"
        for _attempt in range(1 + _MAX_FIXED_POINT_RETRIES):
            problem = build_problem(
                self.ir, x, interval, self.compiled, self.sigma, self.weights,
                lam=sparsity_weight, scale_k=scale_k,
            )
            if isinstance(problem, BuildInfeasible):
                if problem.resolution_related:
                    scale_k *= 10
                    last_reason = problem.reason
                    continue
                return Infeasible(reason=problem.reason)

            solution = solver.solve(problem, time_budget_s=time_budget_s, num_workers=num_workers)
            if solution.status == "infeasible":
                return Infeasible(reason="no counterfactual satisfies target and constraints")

            assert solution.values_scaled is not None
            x_cf = x.copy()
            for block in problem.features:
                if solution.missing.get(block.index):
                    x_cf[block.index] = math.nan
                    continue
                v_int = solution.values_scaled[block.index]
                if block.x_cell is not None and v_int == block.x_scaled:
                    x_cf[block.index] = x[block.index]  # anchor value means "unchanged"
                else:
                    x_cf[block.index] = v_int / problem.scale_k
            verification = self._verify(x, x_cf, interval)
            if verification is None:
                return self._result(x, x_cf, solution.status, solution.gap, solution.stats)
            # Fixed-point artifact (should be prevented by §5.4 widening): retry finer.
            scale_k *= 10
            last_reason = verification

        return Infeasible(
            reason=f"fixed-point verification failed after retries: {last_reason}"
        )

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
        return None

    def _result(
        self,
        x: FloatArray,
        x_cf: FloatArray,
        status: str,
        gap: float | None,
        stats: dict[str, object],
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
        )


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
    if name == "genetic":
        raise NotImplementedError("the genetic backend arrives in M3")
    if name == "highs":
        from treecf.backends.highs import HighsBackend

        return HighsBackend()
    raise TreecfError(f"unknown backend {name!r}; use 'cpsat' or 'genetic'")
