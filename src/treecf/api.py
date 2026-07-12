"""Public API: Explainer and result types."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from treecf._errors import TreecfError
from treecf.aim.cells import cell_index, feature_cells
from treecf.constraints.compile import compile_constraints
from treecf.constraints.objects import Constraint
from treecf.ir.evaluate import apply_link, raw_score
from treecf.ir.model import EnsembleIR, Link
from treecf.ir.parsers import parse_model
from treecf.objective import fit_normalizers
from treecf.plausibility import Plausibility
from treecf.targets import Target

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Grid:
    """Value policy: snap to ``anchor + k * step``."""

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
    proof: str  # "heuristic" — the genetic engine never claims optimality
    solver_stats: dict[str, object] = field(default_factory=dict)
    snapped: dict[str, bool] = field(default_factory=dict)  # value_policy outcome


@dataclass(frozen=True)
class Infeasible:
    reason: str


class Explainer:
    """Counterfactual explainer for a tree-ensemble model.

    ``model`` may be a native model object, a dump file path/dict, or an
    ``EnsembleIR``. ``background`` fits the distance normalizers;
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
                    "plausibility with AllowMissing is not supported "
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
        backend: str = "genetic",
        time_budget_s: float = 10.0,
        sparsity_weight: float = 0.0,
        seed: int | None = None,
    ) -> Counterfactual | Infeasible | dict[str, object]:
        """Search for a counterfactual (or one per band for ``Target.bands``).

        ``backend="genetic"`` runs the bundled Rust engine (default);
        ``backend="python"`` runs the reference numpy implementation of the
        same algorithm. Every result is float-verified before being returned.
        """
        x = np.asarray(x, dtype=np.float64)
        if self.plausibility is not None and np.isnan(x).any():
            raise TreecfError("plausibility with missing factual values is not supported")
        if backend in ("genetic", "genetic-rust"):
            rust = True
        elif backend == "python":
            rust = False
        else:
            raise TreecfError(f"unknown backend {backend!r}; use 'genetic' or 'python'")

        if target.bands_spec is not None:
            results: dict[str, object] = {}
            for name, interval in target.band_intervals(self.ir.link).items():
                results[name] = self._explain_genetic(
                    x, interval, time_budget_s, sparsity_weight, seed, rust=rust
                )
            return results
        interval = target.raw_interval(self.ir.link)
        return self._explain_genetic(
            x, interval, time_budget_s, sparsity_weight, seed, rust=rust
        )

    def _explain_genetic(
        self,
        x: FloatArray,
        interval: tuple[float, float],
        time_budget_s: float,
        sparsity_weight: float,
        seed: int | None,
        rust: bool = True,
    ) -> Counterfactual | Infeasible:
        if rust:
            from treecf.backends.genetic_rust import solve_genetic_rust

            if not hasattr(self, "_rust_cache"):
                self._rust_cache: dict[str, object] = {}
            result = solve_genetic_rust(
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
                cache=self._rust_cache,
            )
        else:
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
            return Infeasible(reason="heuristic search exhausted (genetic backend)")
        x_cf = result.x_cf
        verification = self._verify(x, x_cf, interval)
        if verification is not None:  # defensive: the GA only returns checked individuals
            return Infeasible(reason=f"heuristic solution failed verification: {verification}")
        x_cf, snapped = self._apply_value_policies(x, x_cf, interval)
        return self._result(x, x_cf, "heuristic", result.stats, snapped)

    def _verify(
        self, x: FloatArray, x_cf: FloatArray, interval: tuple[float, float]
    ) -> str | None:
        """Float-space re-check of target and constraints. None = OK."""
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

        slack = 1e-9
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
        """Snap changed values per policy inside their cells; never break validity.

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
            if cf_nan:  # value -> NaN priced by delta_miss
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
            solver_stats=stats,
            snapped=snapped or {},
        )


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
