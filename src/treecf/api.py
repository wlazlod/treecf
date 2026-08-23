"""Public API: Explainer and result types."""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from treecf._errors import TreecfError, TreecfWarning
from treecf.aim.cells import cell_index, feature_cells
from treecf.constraints.compile import compile_constraints
from treecf.constraints.objects import Constraint
from treecf.ir.evaluate import TreeArrays, apply_link, prepare_tree_arrays, raw_score
from treecf.ir.model import EnsembleIR, Link
from treecf.ir.parsers import parse_model
from treecf.objective import fit_normalizers
from treecf.plausibility import Plausibility
from treecf.targets import Target

if TYPE_CHECKING:
    from treecf.backends.exact import ExactResult
    from treecf.backends.genetic import GeneticResult
    from treecf.regions import RecourseRegion

FloatArray = npt.NDArray[np.float64]

_ALL_LEVERS = "(all levers)"  # reserved coalition name for the unrestricted baseline


def _validate_coalitions(
    coalitions: Mapping[str, Sequence[str]],
    feature_names: Sequence[str],
    include_full: bool,
) -> dict[str, tuple[str, ...]]:
    """Normalize a coalition mapping; overlaps are allowed, unknown names are not."""
    if not coalitions:
        raise TreecfError("coalitions must contain at least one named group")
    known = set(feature_names)
    normalized: dict[str, tuple[str, ...]] = {}
    for name, members in coalitions.items():
        if include_full and name == _ALL_LEVERS:
            raise TreecfError(
                f"coalition name {_ALL_LEVERS!r} is reserved for the include_full baseline"
            )
        members = tuple(members)
        if not members:
            raise TreecfError(f"coalition {name!r} is empty")
        unknown = [f for f in members if f not in known]
        if unknown:
            raise TreecfError(f"coalition {name!r} references unknown features: {unknown}")
        normalized[name] = members
    return normalized


@dataclass(frozen=True)
class Grid:
    """Value policy: snap a changed feature to ``anchor + k * step`` for integer ``k``.

    Pass as a ``value_policy`` entry (``Explainer(..., value_policy={"tenor_months":
    Grid(step=6.0)})``) to keep a feature on a fixed lattice (loan tenors in
    6-month increments, prices in 5-cent ticks, and similar). The exact backend
    treats the grid as a hard constraint on its own candidates; the genetic
    backend snaps its winning row afterward, reverting the snap if it would break
    feasibility — see
    [Certification — value policies](concepts/certification.md#value-policies-under-certification).

    Attributes:
        step: Spacing between adjacent grid points. Must be positive.
        anchor: Offset of the grid from zero; grid points are
            ``anchor + k * step`` for integer ``k``. Defaults to ``0.0``.
    """

    step: float
    anchor: float = 0.0


ValuePolicy = str | Grid | Callable[[float], float]


@dataclass(frozen=True)
class Counterfactual:
    """One verified counterfactual: the changed row, its cost, and how strong
    a claim the search makes about it being the cheapest one.

    ``proof`` is always one of exactly three values:
    ``{"heuristic", "optimal", "optimal_within_gap"}``. The genetic and
    python backends always report ``"heuristic"`` — they never claim
    optimality. The exact backend reports ``"optimal"`` when it proved no
    cheaper row exists, ``"optimal_within_gap"`` when ``gap > 0`` and it only
    proved none exists more than that relative fraction cheaper, and — more
    rarely — ``"heuristic"`` for a row it is not claiming is cheapest: see
    ``Explainer.explain`` for when that happens. See
    [Certification](concepts/certification.md) for the full proof taxonomy.

    Attributes:
        x_cf: The full counterfactual feature vector, same order and length as
            the factual; unchanged features keep the factual's own value.
        changes: ``{feature: (factual_value, counterfactual_value)}`` for every
            feature that actually differs (including a transition to or from
            ``NaN``); unchanged features are omitted.
        distance: The weighted, normalized sum of per-feature changes
            (``sum(weight * |delta| / sigma)``, with ``AllowMissing``'s
            ``delta_miss``/``delta_from_miss`` pricing any NaN transition),
            excluding the sparsity term. When ``sparsity_weight > 0`` the
            search minimizes ``distance + sparsity_weight * n_changed``, so
            ``distance`` alone does not reproduce the search's own ranking.
        n_changed: ``len(changes)`` — the number of features actually changed.
        score_raw: The model's raw score at ``x_cf`` (pre-link, i.e. margin for
            a sigmoid-link model).
        score_prob: ``sigmoid(score_raw)`` for a sigmoid-link model, otherwise
            ``None``.
        proof: The optimality claim this result makes; see above.
        solver_stats: Backend-specific diagnostics. Populated by the exact
            backend (``nodes_expanded``, ``nodes_pruned_score``,
            ``nodes_pruned_cost``, ``lower_bound``, ``gap``, ``completed``,
            ``warm_start_used``); empty or backend-specific for genetic/python.
        snapped: ``{feature: bool}`` for every feature under a ``value_policy``
            that also changed — ``True`` when the genetic backend's post-hoc
            snap held, ``False`` when it was reverted (or never applied) to keep
            the result feasible. Empty when no changed feature carries a policy,
            or on the exact backend (policies are baked into its own search and
            never post-hoc snapped).
        region: The certified box around ``x_cf``, set only when the search ran
            with ``region=True`` (``Explainer.explain``/``explain_batch``/
            ``explain_coalitions``); ``None`` otherwise.
    """

    x_cf: FloatArray
    changes: dict[str, tuple[float, float]]
    distance: float
    n_changed: int
    score_raw: float
    score_prob: float | None
    proof: str  # "heuristic" | "optimal" | "optimal_within_gap"
    solver_stats: dict[str, object] = field(default_factory=dict)
    snapped: dict[str, bool] = field(default_factory=dict)  # value_policy outcome
    region: RecourseRegion | None = None  # set when `explain(..., region=True)`


@dataclass(frozen=True)
class Infeasible:
    """No counterfactual returned — the search made no claim, or proved none exists.

    ``proof`` is always one of exactly two values:
    ``{"search_exhausted", "certified"}``. ``"search_exhausted"`` (the
    default) means the search ran out of budget, hit a heuristic dead end, or
    gave up an optimality certificate along the way — nothing is proven about
    whether a counterfactual exists at all. ``"certified"`` is exact-backend
    only: every assignment the searched grid allows was tried and none was
    feasible, so ``reason`` names the node count behind that proof. See
    [Certification](concepts/certification.md) for the full proof taxonomy.

    Attributes:
        reason: Human-readable explanation of why no counterfactual was
            returned; names the node count behind a ``"certified"`` proof, or
            describes the exhaustion/repair cause for ``"search_exhausted"``.
        proof: The claim this non-result makes; see above.
        solver_stats: Backend-specific diagnostics, populated the same way as
            ``Counterfactual.solver_stats`` for the exact backend; empty for
            genetic/python.
    """

    reason: str
    proof: str = "search_exhausted"  # "search_exhausted" | "certified"
    solver_stats: dict[str, object] = field(default_factory=dict)


# documented defaults for the exact-only kwargs; ``None`` at the public call
# sites is the sentinel for "not explicitly passed"
_DEFAULT_WARM_START = True
_DEFAULT_NODE_BUDGET = 2_000_000
_DEFAULT_GAP = 0.0


def _resolve_exact_kwargs(
    backend: str,
    warm_start: bool | None,
    node_budget: int | None,
    gap: float | None,
) -> tuple[bool, int, float]:
    """Normalize the exact-only kwargs and reject them for other backends.

    ``None`` means "not explicitly passed" and normalizes to the documented
    default. An explicit non-default value together with a backend other
    than ``"exact"`` raises ``ValueError`` — a Python-level argument
    combination error, deliberately not the usual ``TreecfError``.
    """
    resolved_warm_start = _DEFAULT_WARM_START if warm_start is None else warm_start
    resolved_node_budget = _DEFAULT_NODE_BUDGET if node_budget is None else node_budget
    resolved_gap = _DEFAULT_GAP if gap is None else gap
    if backend != "exact" and (
        resolved_warm_start is not _DEFAULT_WARM_START
        or resolved_node_budget != _DEFAULT_NODE_BUDGET
        or resolved_gap != _DEFAULT_GAP
    ):
        raise ValueError(
            "warm_start, node_budget, and gap are only valid with backend='exact'"
        )
    return resolved_warm_start, resolved_node_budget, resolved_gap


@dataclass(frozen=True)
class _Degradation:
    """One exact-backend result with ``stats["completed"] is False``, collected
    instead of warned immediately when a caller aggregates over several solves
    (``Target.bands``, ``explain_coalitions``, ``explain_batch``)."""

    kind: str  # "exhausted" | "withdrawn"
    message: str  # standalone warning body; the seed clause is applied separately
    unseeded: bool  # seed clause applies (seed is None and warm_start_used)


_SEED_CLAUSE = (
    " Warm start was unseeded; rerunning may return a different heuristic result — "
    "pass seed= for reproducibility."
)


def _gap_parenthetical(distance: float | None, lower_bound: float) -> str:
    """`` (lower bound X, gap <= Y%)``, or ``""`` when there is no row, or the
    lower bound is absent, non-positive, or non-finite."""
    if distance is None or not math.isfinite(lower_bound) or lower_bound <= 0.0:
        return ""
    displayed_gap = (distance - lower_bound) / lower_bound
    return f" (lower bound {lower_bound:.4g}, gap ≤ {displayed_gap:.1%})"


def _exhausted_message(
    nodes_expanded: int, has_row: bool, distance: float | None, lower_bound: float
) -> str:
    if has_row:
        return (
            f"exact search exhausted its budget after {nodes_expanded:,} nodes; the "
            "result is the best found, not proven optimal"
            f"{_gap_parenthetical(distance, lower_bound)}; raise node_budget/time_budget_s "
            "or set gap= to accept a proven tolerance."
        )
    return (
        f"exact search exhausted its budget after {nodes_expanded:,} nodes without "
        "finding a feasible counterfactual; this is NOT a certified infeasibility — "
        "raise budgets or use backend='genetic'."
    )


def _withdrawn_message(has_row: bool, distance: float | None, lower_bound: float) -> str:
    if has_row:
        return (
            "exact search withdrew its optimality certificate after a conservative "
            "constraint repair without exhausting its budget; the result is verified "
            f"but not proven cheapest{_gap_parenthetical(distance, lower_bound)}."
        )
    return (
        "exact search withdrew its optimality certificate after a conservative "
        "constraint repair without exhausting its budget; no feasible counterfactual "
        "was found, and this is NOT a certified infeasibility."
    )


def _degradation_for(
    res: ExactResult, node_budget: int, time_budget_s: float, elapsed: float, seed: int | None
) -> _Degradation:
    """Classify one degraded exact-search result (``stats["completed"] is False``).

    Elapsed time is measured around the whole solver call, including the
    Python<->Rust marshaling overhead — that overhead only ever pushes
    ``elapsed`` up, so it only ever biases the classification toward
    "exhausted", never toward the unearned "conservative withdrawal" reading.
    """
    stats = res.stats
    nodes_expanded = cast(int, stats["nodes_expanded"])
    lower_bound = cast(float, stats["lower_bound"])
    exhausted = nodes_expanded >= node_budget or elapsed >= time_budget_s
    has_row = res.x_cf is not None
    unseeded = seed is None and bool(stats["warm_start_used"])
    if exhausted:
        kind = "exhausted"
        message = _exhausted_message(nodes_expanded, has_row, res.distance, lower_bound)
    else:
        kind = "withdrawn"
        message = _withdrawn_message(has_row, res.distance, lower_bound)
    return _Degradation(kind=kind, message=message, unseeded=unseeded)


def _degraded_summary(
    degraded: list[_Degradation], affected: int, total: int, unit: str
) -> str | None:
    """Message body for one aggregate degraded-result warning, or ``None`` when
    nothing degraded. Counts stay broken down by kind so the aggregate never
    claims exhaustion for a withdrawn result, or the reverse. The caller emits
    the actual ``warnings.warn`` itself, so its own stacklevel stays accurate."""
    if not degraded:
        return None
    counts: dict[str, int] = {}
    for d in degraded:
        counts[d.kind] = counts.get(d.kind, 0) + 1
    summary = "; ".join(f"{kind}: {n} solve{'s' if n != 1 else ''}" for kind, n in counts.items())
    unseeded = any(d.unseeded for d in degraded)
    return (
        f"exact search returned a degraded result in {affected}/{total} {unit} "
        f"({summary}); see each result's own proof/solver_stats for which case applies."
        + (_SEED_CLAUSE if unseeded else "")
    )


class Explainer:
    """Counterfactual explainer for a tree-ensemble model.

    Parses the model, compiles the constraints, and fits the distance
    normalizers once at construction, so repeated ``explain``/``explain_batch``/
    ``explain_coalitions`` calls reuse that work.

    Args:
        model: A native model object (XGBoost/LightGBM/CatBoost/sklearn
            ensemble), a JSON dump file path or dict, or an already-parsed
            ``EnsembleIR``. See [Models and the IR](concepts/models.md) for
            which native types are supported.
        background: Sample used to fit the per-feature distance normalizers
            (``sigma``, one per feature). Required unless ``normalizers`` is
            given instead; ignored when it is.
        constraints: Constraint objects (``Freeze``, ``Range``, ``Monotone``,
            ``Linear``, ``Implies``, ``OneHot``, ``AllowMissing``, or a string
            parsed by ``constraint()``) compiled and validated immediately.
            Defaults to no constraints. See [Constraints](concepts/constraints.md).
        weights: Per-feature multiplier on distance cost, ``{feature: weight}``;
            a feature not listed defaults to ``1.0``. Use to make some levers
            relatively cheaper or more expensive than the normalized default.
        normalizers: Per-feature distance scale ``sigma``, either an array
            aligned to the model's feature order or a ``{feature: sigma}``
            dict. Pass this instead of ``background`` to reuse known scales
            (e.g. across several explainers on the same features).
        value_policy: Per-feature snapping rule, ``{feature: policy}``, where
            a policy is ``"raw"`` (no snapping; the default for a feature not
            listed), ``"integer"`` (round to the nearest feasible integer), a
            ``Grid(step, anchor=0.0)`` (snap to a fixed lattice), or a callable
            ``float -> float``. The exact backend treats a policy as a hard
            constraint on its candidates; the genetic backend snaps its
            winning row afterward and reverts the snap if it would break
            feasibility (``Counterfactual.snapped`` records which happened) —
            see [Certification](concepts/certification.md#value-policies-under-certification).
        plausibility: Optional hard isolation-forest bound keeping every
            returned counterfactual inside the data manifold (see
            ``Plausibility.isolation_forest``). Cannot be combined with
            ``AllowMissing``, and ``explain``/``explain_batch``/
            ``explain_coalitions`` reject a factual containing NaN once it is
            set (isolation forests define no NaN routing) — see
            [Plausibility](concepts/plausibility.md).

    Raises:
        TreecfError: If neither ``background`` nor ``normalizers`` is given, if
            ``normalizers`` omits a feature or resolves to a non-positive
            scale, if ``value_policy`` names an unknown feature or an
            unrecognized string policy, or if ``plausibility`` is given
            together with ``AllowMissing`` or a mismatched feature space.
        ConstraintValidationError: If ``constraints`` contains a malformed or
            self-contradictory constraint.
    """

    _rust_cache: dict[str, object]  # marshaled Rust objects, filled on first solve
    _prepared_trees: tuple[TreeArrays, ...]  # vectorized-verify arrays, created on first batch

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
        self._rust_cache = {}
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
        warm_start: bool | None = None,
        node_budget: int | None = None,
        gap: float | None = None,
        region: bool = False,
    ) -> Counterfactual | Infeasible | dict[str, object]:
        """Search for a counterfactual (or one per band for ``Target.bands``).

        ``x`` is the factual instance (one row, aligned to the model's feature
        order); ``target`` bounds the model output the counterfactual must
        reach. ``time_budget_s`` caps wall time per solve (per band, when
        ``target`` is a ``Target.bands`` ladder); ``math.inf`` is accepted
        and removes the time cut, letting an exact search run until it
        proves its answer (Ctrl-C still aborts promptly). ``sparsity_weight`` makes
        the search minimize ``distance + sparsity_weight * n_changed``
        instead of plain ``distance``, trading a cheaper plan that touches
        more features against a sparser one that costs more per feature; the
        returned ``Counterfactual.distance`` itself always excludes the
        sparsity term. ``0.0`` (the default) does not penalize sparsity at
        all. ``seed`` fixes the genetic search's (and, on the
        exact backend, the warm start's) randomness for reproducibility;
        ``None`` draws a fresh one each call.

        ``backend="genetic"`` runs the bundled Rust engine (default);
        ``backend="python"`` runs the reference numpy implementation of the
        same algorithm; ``backend="exact"`` runs a branch-and-bound search
        over the same candidate grid that proves optimality when it finds a
        counterfactual and proves infeasibility when it does not, at the cost
        of a potentially longer solve. Every result is float-verified before
        being returned.

        ``warm_start`` (default ``True``), ``node_budget`` (default
        ``2_000_000``), and ``gap`` (default ``0.0``) configure the exact
        backend only; passing a non-default value together with another
        backend raises ``ValueError`` — deliberately not the usual
        ``TreecfError``, since this rejects a Python-level argument
        combination rather than a modeling error. ``warm_start=True`` runs a
        short genetic pass first (about a quarter of ``time_budget_s``,
        capped at 2 seconds) and, if it lands a verified counterfactual, feeds
        it to the exact search as a starting incumbent; the exact search
        still gets the full ``time_budget_s`` afterwards, so warm start is
        additive rather than deducted from the budget. ``gap`` lets the exact
        search settle for a counterfactual within that relative fraction of
        the true optimum, reported through ``proof="optimal_within_gap"``.

        An exact search can return a feasible row with ``proof="heuristic"``
        without exhausting ``node_budget`` or ``time_budget_s``: conservative
        repair of some constraint shapes can withdraw the optimality
        certificate honestly rather than claim a cheapest row it did not
        prove — the row itself is still real and verified, only the
        "cheapest possible" claim is dropped. Whenever the exact search
        returns any result with ``solver_stats["completed"] is False`` — for
        that reason or because the budget genuinely ran out — a
        ``TreecfWarning`` names which of the two happened, never the
        other one. ``Target.bands``, ``explain_coalitions``, and
        ``explain_batch`` collapse this into one aggregate warning per call
        instead of one per solve.

        If the factual itself violates a constraint, a ``TreecfWarning``
        is emitted: the returned plan will include changes made solely to
        satisfy the constraint set.

        ``region=True`` widens every successful ``Counterfactual`` into a
        certified ``RecourseRegion`` (``cf.region``) —
        works with any backend, genetic included. Costs one oracle call per
        attempted per-feature, per-direction expansion; see
        ``Explainer.recourse_region``.

        Returns:
            A single ``Counterfactual`` or ``Infeasible`` when ``target`` is a
            plain interval (``Target.raw``/``probability``/``calibrated``); a
            ``{band_name: Counterfactual | Infeasible}`` dict, one entry per
            band in solved order, when ``target`` is a ``Target.bands``
            ladder. ``Infeasible`` means the search found no verified
            counterfactual — see ``Infeasible.proof`` for whether that is a
            certified impossibility or just an unsuccessful search.

        Raises:
            ValueError: If ``warm_start``, ``node_budget``, or ``gap`` is
                given a non-default value together with a ``backend`` other
                than ``"exact"``.
            TreecfError: If ``backend`` is not one of ``"genetic"``,
                ``"python"``, or ``"exact"``, or if ``plausibility`` is
                configured on this explainer and ``x`` contains NaN.
            ConstraintValidationError: If ``backend="exact"`` and this
                explainer's constraints include an unsupported multi-feature
                ``Linear`` shape, or a callable ``value_policy``; the message
                names ``backend="genetic"`` as the fallback.
        """
        return self._explain(
            x, target, backend, time_budget_s, sparsity_weight, seed, warn_factual=True,
            warm_start=warm_start, node_budget=node_budget, gap=gap, region=region,
        )

    def _explain(
        self,
        x: FloatArray,
        target: Target,
        backend: str,
        time_budget_s: float,
        sparsity_weight: float,
        seed: int | None,
        *,
        warn_factual: bool,
        warm_start: bool | None = None,
        node_budget: int | None = None,
        gap: float | None = None,
        region: bool = False,
        degraded: list[_Degradation] | None = None,
        incumbent: tuple[float, FloatArray] | None = None,
    ) -> Counterfactual | Infeasible | dict[str, object]:
        """``explain`` body; ``explain_batch`` calls it with ``warn_factual=False``
        after emitting its own aggregate warning. ``degraded`` collects exact-backend
        degradations for an external caller's own aggregate instead of warning
        immediately; ignored (a fresh local collector is used instead) when
        ``target.bands_spec`` is set, since batch/coalitions never pass bands through.
        ``incumbent`` forwards to ``_explain_exact`` for the single-interval case only
        (``target.bands_spec`` rows never receive one — batch, the only caller that
        passes one, already rejects bands)."""
        x = np.asarray(x, dtype=np.float64)
        if warn_factual:
            violations = self.compiled.factual_violations(x)
            if violations:
                warnings.warn(
                    f"factual violates {len(violations)} constraint(s): "
                    + "; ".join(violations)
                    + ". The returned plan will include changes made solely to satisfy them.",
                    TreecfWarning,
                    stacklevel=3,  # _explain <- explain <- user code
                )
        if self.plausibility is not None and np.isnan(x).any():
            raise TreecfError("plausibility with missing factual values is not supported")
        if backend not in ("genetic", "genetic-rust", "python", "exact"):
            raise TreecfError(f"unknown backend {backend!r}; use 'genetic', 'python', or 'exact'")
        resolved_warm_start, resolved_node_budget, resolved_gap = _resolve_exact_kwargs(
            backend, warm_start, node_budget, gap
        )
        rust = backend in ("genetic", "genetic-rust")

        if target.bands_spec is not None:
            results: dict[str, object] = {}
            band_degraded: list[_Degradation] = []
            intervals = target.band_intervals(self.ir.link)
            for name, interval in intervals.items():
                outcome = (
                    self._explain_exact(
                        x, interval, time_budget_s, resolved_warm_start,
                        resolved_node_budget, resolved_gap, sparsity_weight, seed,
                        degraded=band_degraded,
                    )
                    if backend == "exact"
                    else self._explain_genetic(
                        x, interval, time_budget_s, sparsity_weight, seed, rust=rust
                    )
                )
                if region and isinstance(outcome, Counterfactual):
                    outcome = replace(outcome, region=self._region_for(x, outcome.x_cf, interval))
                results[name] = outcome
            message = _degraded_summary(band_degraded, len(band_degraded), len(intervals), "bands")
            if message is not None:
                warnings.warn(
                    message, TreecfWarning, stacklevel=3  # _explain <- explain <- user code
                )
            return results
        interval = target.raw_interval(self.ir.link)
        result = (
            self._explain_exact(
                x, interval, time_budget_s, resolved_warm_start,
                resolved_node_budget, resolved_gap, sparsity_weight, seed,
                incumbent=incumbent, degraded=degraded,
            )
            if backend == "exact"
            else self._explain_genetic(
                x, interval, time_budget_s, sparsity_weight, seed, rust=rust
            )
        )
        if region and isinstance(result, Counterfactual):
            result = replace(result, region=self._region_for(x, result.x_cf, interval))
        return result

    def explain_batch(
        self,
        X: FloatArray,
        target: Target,
        n_per_example: int = 1,
        diversity: str = "seeds",
        ids: Sequence[object] | None = None,
        backend: str = "genetic",
        time_budget_s: float = 10.0,
        sparsity_weight: float = 0.0,
        seed: int = 0,
        coalitions: Mapping[str, Sequence[str]] | None = None,
        include_full: bool = False,
        warm_start: bool | None = None,
        node_budget: int | None = None,
        gap: float | None = None,
        region: bool = False,
        allow_exact_batch: bool = False,
    ) -> Any:
        """Mass-produce counterfactuals for a dataset; see ``treecf.batch``.

        ``X`` is the factual dataset (one row per instance, aligned to the
        model's feature order); ``ids`` labels each row (defaults to its
        integer index) and must have one entry per row of ``X``.
        ``n_per_example`` alternatives per row via ``diversity="seeds"`` (distinct
        change-sets from different seeds, best-effort) or ``"lever-blocking"``
        (freeze each plan's biggest lever; also records essential levers).
        ``diversity="coalitions"`` instead produces one plan per named feature
        group in ``coalitions`` per row (``n_per_example`` unused; see
        ``explain_coalitions`` for ``coalitions``/``include_full`` semantics,
        which are only valid in this mode). The returned ``BatchResult`` supports
        save/load/for_id/to_frame. ``time_budget_s``, ``sparsity_weight``, and
        ``seed`` carry the same meaning as in ``Explainer.explain``, applied
        per solve (``seed`` is combined with each row's index to derive a
        distinct per-row seed).

        Solves run in parallel inside the Rust engine; ``time_budget_s`` is
        per solve, so a solve that hits its wall-clock budget while sharing
        cores may stop earlier than it would sequentially (results are
        otherwise identical to solving row by row).

        ``backend="exact"`` has no vectorized population to parallelize, so
        this loops the single-instance exact solve per row (and per plan, for
        lever-blocking) sequentially — expect roughly linear-in-rows wall
        time rather than the Rust engine's parallel wave scheduling, and each
        row still gets the full, undiminished ``time_budget_s``. Because that
        wall time is easy to underestimate, ``backend="exact"`` here is
        opt-in: without ``allow_exact_batch=True`` this raises ``ValueError``
        naming an estimate (rows × plans × ``time_budget_s``, hours-formatted)
        instead of silently running -- a floor, not a ceiling, since
        ``diversity="seeds"`` can retry each plan up to 3x on a seed
        collision; passing it through with any other backend also raises
        ``ValueError``. Opting in additionally
        replaces ``warm_start``'s N sequential per-row (or, in seeds mode,
        per-attempt) genetic warm passes with a single vectorized one across
        every row — see ``treecf.batch.explain_batch`` for exactly which
        modes it covers and which keep 0.2.0's per-solve behavior.
        ``node_budget``/``gap`` thread through to every solve unchanged; see
        ``Explainer.explain``. A ``KeyboardInterrupt`` during any batch solve
        discards whatever the batch has not yet finished — there is no
        partial ``BatchResult``. ``region=True`` attaches a certified
        ``RecourseRegion`` (``BatchRecord.region``) to every feasible record,
        at the same one-oracle-call-per-expansion cost.

        Returns:
            A ``BatchResult`` holding one ``BatchRecord`` per (row, plan) pair
            — infeasible rows/plans get a record with ``feasible=False`` and
            no ``x_cf`` rather than being omitted.

        Raises:
            TreecfError: If ``target`` is a ``Target.bands`` ladder, if
                ``diversity`` is not ``"seeds"``, ``"lever-blocking"``, or
                ``"coalitions"``, if ``coalitions``/``include_full`` is given
                outside ``diversity="coalitions"`` (or omitted inside it), or
                if ``ids`` does not have one entry per row of ``X``.
            ValueError: If ``warm_start``, ``node_budget``, or ``gap`` is
                given a non-default value together with a ``backend`` other
                than ``"exact"``; if ``backend="exact"`` is requested without
                ``allow_exact_batch=True`` (message names the wall time
                estimate, a floor rather than a ceiling); or if
                ``allow_exact_batch=True`` is passed with a ``backend`` other
                than ``"exact"``.
        """
        from treecf.batch import explain_batch

        return explain_batch(
            self, X, target, n_per_example=n_per_example, diversity=diversity,
            ids=ids, backend=backend, time_budget_s=time_budget_s,
            sparsity_weight=sparsity_weight, seed=seed,
            coalitions=coalitions, include_full=include_full,
            warm_start=warm_start, node_budget=node_budget, gap=gap, region=region,
            allow_exact_batch=allow_exact_batch,
        )

    def explain_coalitions(
        self,
        x: FloatArray,
        target: Target,
        coalitions: Mapping[str, Sequence[str]],
        include_full: bool = False,
        backend: str = "genetic",
        time_budget_s: float = 10.0,
        sparsity_weight: float = 0.0,
        seed: int | None = None,
        warm_start: bool | None = None,
        node_budget: int | None = None,
        gap: float | None = None,
        region: bool = False,
    ) -> dict[str, Counterfactual | Infeasible]:
        """One counterfactual per named feature coalition (opt-in mode).

        ``x``/``target``/``backend``/``time_budget_s``/``sparsity_weight``/
        ``seed`` carry the same meaning as in ``Explainer.explain``, applied
        once per coalition. ``coalitions`` maps a group name to the features
        it may change; each coalition is solved with every feature *outside*
        it frozen, so a plan only ever asks for changes within one group —
        grouped recourse instead of one plan that mixes unrelated levers.
        Coalitions may overlap; features in no coalition are never modified;
        an ``Infeasible`` for a coalition means that group alone cannot reach
        the target. ``include_full=True`` prepends an unrestricted baseline
        under the reserved key ``"(all levers)"``. One solve per coalition
        (milliseconds each); this mode is optional and never the default.
        ``warm_start``/``node_budget``/``gap``/``region`` thread through to every
        coalition's solve; see ``Explainer.explain``. A degraded exact result
        (``solver_stats["completed"] is False``) in any coalition's solve is
        collapsed into one aggregate ``TreecfWarning`` for the whole call,
        rather than one per coalition.

        Returns:
            ``{coalition_name: Counterfactual | Infeasible}``, one entry per
            key of ``coalitions`` plus ``"(all levers)"`` when
            ``include_full=True``, in that insertion order.

        Raises:
            TreecfError: If ``target`` is a ``Target.bands`` ladder, if
                ``coalitions`` is empty, names a coalition with no members, or
                references an unknown feature, or if ``include_full=True`` and
                a coalition is named ``"(all levers)"`` (the reserved key).
            ValueError: If ``warm_start``, ``node_budget``, or ``gap`` is
                given a non-default value together with a ``backend`` other
                than ``"exact"``.
        """
        if target.bands_spec is not None:
            raise TreecfError(
                "Target.bands is not supported in explain_coalitions; loop bands explicitly"
            )
        normalized = _validate_coalitions(coalitions, self.ir.feature_names, include_full)
        results: dict[str, Counterfactual | Infeasible] = {}
        degraded: list[_Degradation] = []
        if include_full:
            results[_ALL_LEVERS] = self._explain_one(
                x, target, backend, time_budget_s, sparsity_weight, seed,
                warm_start=warm_start, node_budget=node_budget, gap=gap, region=region,
                degraded=degraded,
            )
        for name, clone in self._coalition_explainers(normalized).items():
            results[name] = clone._explain_one(
                x, target, backend, time_budget_s, sparsity_weight, seed,
                warm_start=warm_start, node_budget=node_budget, gap=gap, region=region,
                degraded=degraded,
            )
        message = _degraded_summary(degraded, len(degraded), len(results), "coalitions")
        if message is not None:
            warnings.warn(message, TreecfWarning, stacklevel=2)  # explain_coalitions <- user code
        return results

    def _explain_one(
        self,
        x: FloatArray,
        target: Target,
        backend: str,
        time_budget_s: float,
        sparsity_weight: float,
        seed: int | None,
        warn_factual: bool = True,
        warm_start: bool | None = None,
        node_budget: int | None = None,
        gap: float | None = None,
        region: bool = False,
        degraded: list[_Degradation] | None = None,
        incumbent: tuple[float, FloatArray] | None = None,
    ) -> Counterfactual | Infeasible:
        """`explain` for a single-interval target, with the bands arm ruled out."""
        result = self._explain(
            x, target, backend, time_budget_s, sparsity_weight, seed,
            warn_factual=warn_factual,
            warm_start=warm_start, node_budget=node_budget, gap=gap, region=region,
            degraded=degraded, incumbent=incumbent,
        )
        assert not isinstance(result, dict)  # bands are rejected by the callers
        return result

    def _coalition_explainers(
        self, coalitions: dict[str, tuple[str, ...]]
    ) -> dict[str, Explainer]:
        """One freeze-complement clone per coalition (Rust ensemble shared)."""
        names = self.ir.feature_names
        return {
            name: self._with_extra_freezes([f for f in names if f not in set(members)])
            for name, members in coalitions.items()
        }

    def _with_extra_freezes(self, features: Sequence[str]) -> Explainer:
        """Clone with additional Freeze constraints (lever-blocking, coalitions).

        ``AllowMissing`` on a newly frozen feature is dropped: a frozen value
        cannot transition to NaN, and keeping both would (correctly) fail
        constraint validation.
        """
        from treecf.constraints.objects import AllowMissing, Freeze

        frozen = set(features)
        kept = [
            c
            for c in self.compiled.constraints
            if not (isinstance(c, AllowMissing) and c.feature in frozen)
        ]
        clone = Explainer(
            self.ir,
            background=self.background,
            constraints=kept + [Freeze(f) for f in features],
            weights=dict(zip(self.ir.feature_names, self.weights.tolist(), strict=True)),
            normalizers=self.sigma,
            value_policy=self.value_policy,
            plausibility=self.plausibility,
        )
        # Same frozen IR -> the marshaled Rust ensembles are reusable; only the
        # constraints differ, so that cache entry is deliberately left out.
        clone._rust_cache = {
            key: self._rust_cache[key]
            for key in ("ensemble", "if_ensemble", "missing_defined", "if_missing_defined")
            if key in self._rust_cache
        }
        return clone

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
            return Infeasible(
                reason="heuristic search exhausted (genetic backend)",
                proof="search_exhausted",
            )
        return self._finalize_candidate(x, result.x_cf, interval, result.stats)

    def _finalize_candidate(
        self,
        x: FloatArray,
        x_cf: FloatArray,
        interval: tuple[float, float],
        stats: dict[str, object],
        score: float | None = None,
    ) -> Counterfactual | Infeasible:
        """Verify, snap, and package one solver candidate.

        ``score`` is an optional precomputed ``raw_score(self.ir, x_cf)``
        (e.g. from a vectorized batch evaluation); it is recomputed whenever
        value policies modify the candidate.
        """
        if score is None:
            score = raw_score(self.ir, x_cf)
        verification = self._verify(x, x_cf, interval, score=score)
        if verification is not None:  # defensive: the GA only returns checked individuals
            return Infeasible(
                reason=f"heuristic solution failed verification: {verification}",
                proof="search_exhausted",
            )
        x_cf = self._prune_changes(x, x_cf, interval)
        final_cf, snapped = self._apply_value_policies(x, x_cf, interval)
        score = raw_score(self.ir, final_cf)
        return self._result(x, final_cf, "heuristic", stats, snapped, score=score)

    def _explain_exact(
        self,
        x: FloatArray,
        interval: tuple[float, float],
        time_budget_s: float,
        warm_start: bool,
        node_budget: int,
        gap: float,
        sparsity_weight: float,
        seed: int | None,
        incumbent: tuple[float, FloatArray] | None = None,
        degraded: list[_Degradation] | None = None,
    ) -> Counterfactual | Infeasible:
        """Exact-backend counterfactual for one target interval.

        ``warm_start`` runs a short genetic pass first, exactly as
        ``_explain_genetic`` does (Rust engine, same seed), with
        ``time_budget_s`` cut to ``min(time_budget_s * 0.25, 2.0)``. A
        verified counterfactual from that pass is re-costed on the exact
        backend's own objective and handed to ``solve_exact`` as an
        incumbent — the exact search still runs with the full, undiminished
        ``time_budget_s`` afterwards.

        ``incumbent``, when given, is used as that starting incumbent
        directly and this method's own internal warm pass never runs (the
        pass only runs when ``incumbent is None and warm_start``) — used by
        ``explain_batch``'s opt-in exact path, which computes one incumbent
        per row in a single vectorized genetic pass instead of every row (or,
        in seeds mode, every attempt) running its own.

        The search itself dispatches rust-first: when the `_treecf_core`
        extension is importable, ``exact_rust.solve_exact_rust`` runs instead
        of the pure-Python ``solve_exact``. The rust engine is a bit-parity
        mirror, not a heuristic stand-in — every fixture in
        ``tests/fixtures/exact/`` proves the two produce a RESULT-IDENTICAL
        answer (same ``x_cf`` or both ``None``, same ``distance``, ``proof``,
        and all seven ``stats`` keys), so the fallback only ever changes
        which engine ran, never what it found.

        A ``ConstraintValidationError`` from the exact backend's constraint
        validation (an unsupported multi-feature ``Linear`` shape, or a
        callable ``value_policy``) propagates unchanged; it already names
        ``backend="genetic"`` as the fallback.

        When the returned ``stats["completed"] is False``, ``degraded`` (when
        given) collects a ``_Degradation`` instead of warning here
        directly, for an external caller's own aggregate warning; ``None``
        (the default) warns immediately.
        """
        from treecf.backends._exact_domains import _cost_of_row
        from treecf.backends.exact_rust import _rust_available, solve_exact_rust

        if incumbent is None and warm_start:
            warm_budget = min(time_budget_s * 0.25, 2.0)
            warm = self._explain_genetic(
                x, interval, warm_budget, sparsity_weight, seed, rust=True
            )
            if isinstance(warm, Counterfactual) and self._verify(x, warm.x_cf, interval) is None:
                cost = _cost_of_row(
                    x, warm.x_cf, self.sigma, self.weights, sparsity_weight,
                    self.compiled.allow_missing,
                )
                incumbent = (cost, warm.x_cf)

        # Wraps only the solver call: any Python<->Rust marshaling overhead this
        # measurement picks up can only push `elapsed` up, which only ever
        # biases the exhaustion classifier below toward "exhausted" -- the
        # honest direction, never toward the stronger "conservative
        # withdrawal" reading it has not earned.
        start = time.monotonic()
        if _rust_available():
            res = solve_exact_rust(
                self.ir,
                x,
                interval,
                self.compiled,
                self.sigma,
                self.weights,
                sparsity_weight,
                value_policies=self.value_policy,
                plausibility=self._plausibility_bound(),
                node_budget=node_budget,
                gap=gap,
                time_budget_s=time_budget_s,
                incumbent=incumbent,
                cache=self._rust_cache,
            )
        else:
            from treecf.backends.exact import solve_exact

            res = solve_exact(
                self.ir,
                x,
                interval,
                self.compiled,
                self.sigma,
                self.weights,
                sparsity_weight,
                value_policies=self.value_policy,
                plausibility=self._plausibility_bound(),
                node_budget=node_budget,
                gap=gap,
                time_budget_s=time_budget_s,
                incumbent=incumbent,
            )
        elapsed = time.monotonic() - start

        if res.stats["completed"] is False:
            degradation = _degradation_for(res, node_budget, time_budget_s, elapsed, seed)
            if degraded is None:
                warnings.warn(
                    degradation.message + (_SEED_CLAUSE if degradation.unseeded else ""),
                    TreecfWarning,
                    stacklevel=4,  # _explain_exact <- _explain <- explain/_explain_one <- caller
                )
            else:
                degraded.append(degradation)

        if res.x_cf is None:
            # Certification is read from stats["completed"], never from
            # res.proof: proof carries no meaning on an infeasible result
            # (see solve_exact's docstring).
            if res.stats["completed"] is True:
                return Infeasible(
                    reason=(
                        "no counterfactual exists in the target interval under the "
                        f"given constraints (certified; {res.stats['nodes_expanded']} nodes)"
                    ),
                    proof="certified",
                    solver_stats=res.stats,
                )
            # completed=False does not always mean the budget ran out: a
            # conservative order-pair repair can withdraw the certificate
            # without spending the whole budget, so this reason names both
            # possibilities rather than claiming the budget was exhausted.
            return Infeasible(
                reason=(
                    "exact search ended without an infeasibility certificate "
                    "(budget exhausted or conservative pruning)"
                ),
                proof="search_exhausted",
                solver_stats=res.stats,
            )
        return self._finalize_exact(x, res, interval)

    def _finalize_exact(
        self,
        x: FloatArray,
        res: ExactResult,
        interval: tuple[float, float],
    ) -> Counterfactual | Infeasible:
        """Verify and package an exact-backend result.

        No ``_prune_changes``/``_apply_value_policies`` here: the exact
        search already reasons over the refined constraint geometry
        (``_constraint_cells``) and bakes value policies into its own
        domains, so post-hoc pruning or snapping would second-guess a
        solution the search already committed to.
        """
        assert res.x_cf is not None
        verification = self._verify(x, res.x_cf, interval)
        if verification is not None:  # defensive: the search only returns checked rows
            return Infeasible(
                reason=f"exact solution failed verification: {verification}",
                proof="search_exhausted",
                solver_stats=res.stats,
            )
        return self._result(x, res.x_cf, res.proof, res.stats, res.snapped)

    def _prune_changes(
        self, x: FloatArray, x_cf: FloatArray, interval: tuple[float, float]
    ) -> FloatArray:
        """Greedily revert changes that verification proves unnecessary.

        The search's revert-to-factual mutation is stochastic, so a stalled
        run can leave residual micro-changes that cross no decision threshold
        — they cost distance without moving the score. Reverting candidates
        one at a time (cheapest change first) can only lower the objective,
        and every kept revert is re-verified in float space, so the returned
        plan keeps all its guarantees.
        """
        allow = self.compiled.allow_missing

        def effort(j: int) -> float:
            source, dest = x[j], x_cf[j]
            if math.isnan(dest):
                delta = allow[j][0]
            elif math.isnan(source):
                delta = allow[j][1]
            else:
                delta = abs(dest - source)
            return float(self.weights[j] * delta / self.sigma[j])

        changed = [
            j
            for j in range(len(x))
            if (x[j] != x_cf[j]) and not (math.isnan(x[j]) and math.isnan(x_cf[j]))
        ]
        if len(changed) < 2:  # a single change is necessary by feasibility
            return x_cf
        candidate = x_cf.copy()
        for j in sorted(changed, key=effort):
            trial = candidate.copy()
            trial[j] = x[j]
            if self._verify(x, trial, interval) is None:
                candidate = trial
        return candidate

    def _prepared_tree_arrays(self) -> tuple[TreeArrays, ...]:
        if not hasattr(self, "_prepared_trees"):
            self._prepared_trees = prepare_tree_arrays(self.ir)
        return self._prepared_trees

    def _solve_batch(
        self,
        X: FloatArray,
        tasks: Sequence[tuple[int, int]],
        interval: tuple[float, float],
        time_budget_s: float,
        sparsity_weight: float,
    ) -> list[GeneticResult]:
        """Run independent seeded searches in one parallel Rust call."""
        from treecf.backends.genetic_rust import solve_genetic_batch_rust

        return solve_genetic_batch_rust(
            self.ir,
            X,
            tasks,
            interval,
            self.compiled,
            self.sigma,
            self.weights,
            lam=sparsity_weight,
            background=self.background,
            plausibility=self._plausibility_bound(),
            time_budget_s=time_budget_s,
            cache=self._rust_cache,
        )

    def _verify(
        self,
        x: FloatArray,
        x_cf: FloatArray,
        interval: tuple[float, float],
        score: float | None = None,
    ) -> str | None:
        """Float-space re-check of target and constraints. None = OK."""
        if score is None:
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
            # exact float equality is intentional: repair writes literal 0.0/1.0,
            # and a tolerance would mask genuinely broken candidates
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

    def recourse_region(
        self, x: FloatArray, x_cf: FloatArray, target: Target
    ) -> RecourseRegion:
        """Certify a per-feature box around an already-verified counterfactual.

        ``x`` is the original factual and ``x_cf`` the counterfactual to widen
        (typically ``Counterfactual.x_cf`` from a prior ``explain`` call);
        ``target`` is the single interval ``x_cf`` was solved against. ``x_cf``
        must independently pass the same float-space re-check
        ``explain`` runs on its own results; a row that fails it raises
        ``TreecfError`` naming the reason, since there is nothing sound to
        widen. Works for a counterfactual from any backend. Costs one oracle
        call — a full interval-tree walk of every ensemble tree — per
        attempted per-feature, per-direction expansion; see
        ``RecourseRegion``. The returned region is certified
        but neither maximal nor monotone in ``target``: a strictly narrower
        target can still grow a strictly wider region on some feature. See
        [Certification](concepts/certification.md#regions-certified-not-maximal-not-monotone).

        Returns:
            The certified ``RecourseRegion`` around ``x_cf``.

        Raises:
            TreecfError: If ``target`` is a ``Target.bands`` ladder (pass the
                single band's own interval instead), or if ``x_cf`` fails the
                float-space re-check against ``x``/``target`` — the message
                names the specific check that failed.
        """
        x = np.asarray(x, dtype=np.float64)
        x_cf = np.asarray(x_cf, dtype=np.float64)
        if target.bands_spec is not None:
            raise TreecfError(
                "Target.bands is not supported in recourse_region; pass the single "
                "band's own interval via Target.raw/probability/calibrated"
            )
        interval = target.raw_interval(self.ir.link)
        try:
            verification = self._verify(x, x_cf, interval)
        except ValueError as exc:
            # _verify's own raw_score re-check raises ValueError when x_cf's
            # path hits a split with no missing routing defined (an unrouted
            # NaN) -- surfaced here as the TreecfError this method's docstring
            # promises, since _verify itself stays a float-space re-check that
            # never wraps its own scoring call.
            raise TreecfError(
                f"cannot certify a region for an unverified counterfactual: {exc}"
            ) from exc
        if verification is not None:
            raise TreecfError(
                f"cannot certify a region for an unverified counterfactual: {verification}"
            )
        return self._region_for(x, x_cf, interval)

    def certificate(
        self,
        x: FloatArray,
        result: Counterfactual | Infeasible,
        target: Target,
        *,
        band: str | None = None,
        seed: int | None = None,
        node_budget: int | None = None,
        gap: float | None = None,
        time_budget_s: float | None = None,
        warm_start: bool | None = None,
    ) -> dict[str, object]:
        """Issue an audit certificate for a stored result (post-hoc, like
        ``recourse_region``).

        A certificate is a reproducibility record plus a fresh verification —
        it binds the result to a model fingerprint, a constraint fingerprint,
        and the solve parameters, and re-verifies the returned plan at issue
        time; it does not cryptographically prove that a search ran or that a
        ``proof="optimal"`` claim is true — re-running with the recorded seed
        and budgets on a fingerprint-matching model is how a validator checks
        that. See
        [Certification — audit certificates](concepts/certification.md#audit-certificates)
        for the schema.

        The certificate is a plain ``dict`` (``"schema_version": 1``) that
        serializes with ``json.dumps(cert, allow_nan=False, sort_keys=True)``;
        non-finite floats are encoded as the strings ``"NaN"``/``"Infinity"``/
        ``"-Infinity"``. Accepts a ``Counterfactual`` or an ``Infeasible`` —
        the certified "no" is exactly the case a validator cares most about.
        The verification block is computed fresh here, never copied from the
        solve: the plan's score, target membership, and constraint check are
        recomputed (plus the plausibility bound when configured, and a sampled
        set of region points when the result carries a region). A certificate
        whose fresh verification fails is still returned, with the failing
        booleans recorded — but a ``TreecfWarning`` names the failed check.

        ``seed``/``node_budget``/``gap``/``time_budget_s``/``warm_start`` are
        recorded under ``solve.declared`` when given: the result object does
        not carry them, so they are caller-supplied, and the block's name
        makes that provenance explicit.

        Args:
            x: The factual instance the result was solved from.
            result: The ``Counterfactual`` or ``Infeasible`` to certify.
            target: The target the result was solved against.
            band: For a ``Target.bands`` result, the band this result belongs
                to; required then, invalid otherwise.
            seed: The seed the solve ran with, if the caller wants it recorded.
            node_budget: The node budget the solve ran with, likewise.
            gap: The relative gap the solve ran with, likewise.
            time_budget_s: The time budget the solve ran with, likewise.
            warm_start: The warm-start setting the solve ran with, likewise.

        Returns:
            The certificate as a strict-JSON-serializable ``dict``.

        Raises:
            TreecfError: If ``target`` is a ``Target.bands`` ladder and
                ``band`` is missing or unknown, or if ``band`` is given for a
                plain-interval target.
        """
        from treecf.audit import build_certificate

        return build_certificate(
            self, x, result, target, band=band, seed=seed, node_budget=node_budget,
            gap=gap, time_budget_s=time_budget_s, warm_start=warm_start,
        )

    def check_certificate(
        self, cert: dict[str, object], *, calibrator: object | None = None
    ) -> dict[str, object]:
        """Validate a stored certificate against *this* explainer.

        Recomputes both fingerprints (model and constraints) against this
        explainer and re-runs the certificate's verification block from its
        stored factual/plan, so a tampered ``x_cf``, a swapped model, or a
        changed constraint set each flips the corresponding boolean. This
        method reports — it never raises on a mismatch.

        Without ``calibrator=``, a calibrated-target certificate is still
        fully verifiable in *plan geometry*: the resolved ``raw_interval``
        is stored, so this proves the plan reaches the stored interval — it
        does not prove which calibrator produced that interval. Passing
        ``calibrator=`` adds exactly that: the duck-typed ``fingerprint()``
        is compared with the stored one, and the certificate's calibrated
        ``lo``/``hi`` are re-inverted through the supplied calibrator and
        compared with the stored interval (rtol 1e-9, infinities by
        identity). Neither mode requires treecf to import a calibration
        library.

        Args:
            cert: A certificate produced by ``Explainer.certificate`` (a
                ``json.loads`` round trip of one works identically).
            calibrator: Optional duck-typed calibrator (the object handed to
                ``Target.calibrated``) to additionally verify calibrator
                provenance against a calibrated-target certificate.

        Returns:
            ``{"model_match": bool, "constraints_match": bool,
            "verification_ok": bool, "mismatches": [...]}`` with one
            human-readable string per mismatch, plus ``"calibrator_match":
            bool`` when ``calibrator=`` was given.
        """
        from treecf.audit import check_certificate

        return check_certificate(self, cert, calibrator=calibrator)

    def _region_for(
        self, x: FloatArray, x_cf: FloatArray, interval: tuple[float, float]
    ) -> RecourseRegion:
        """Build the region for an already-verified ``x_cf`` (no re-verification)."""
        from treecf.regions import _recourse_region

        if_ir, min_total_path = (None, 0.0)
        plaus = self._plausibility_bound()
        if plaus is not None:
            if_ir, min_total_path = plaus
        return _recourse_region(
            self.ir, x, x_cf, interval, self.compiled, if_ir, min_total_path,
            cache=self._rust_cache,
        )

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

        # Genetic-path-only: this is post-hoc snapping onto the unrefined
        # routing grid. The exact backend never calls this function — its
        # geometry lives in `_constraint_cells` (refined for constraints) and
        # its value policies are already baked into `_build_domains`'
        # candidate states, so routing a winning exact row back through here
        # would snap it against the wrong grid.
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
        score: float | None = None,
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
        if score is None:
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
