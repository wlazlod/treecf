"""Mass production of counterfactuals over a dataset (batch API).

``Explainer.explain_batch`` runs the (Rust) genetic search once per row and
alternative, producing a ``BatchResult`` that can be saved to portable JSON,
reloaded, queried per id, or turned into a pandas frame — so a day's worth of
counterfactuals is computed once and then simply looked up.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from treecf._errors import TreecfError, TreecfWarning
from treecf._json import decode_floats, encode_floats
from treecf.ir.evaluate import raw_score_batch_prepared

if TYPE_CHECKING:
    from treecf.api import Counterfactual, Explainer, Infeasible, _Degradation
    from treecf.backends.genetic import GeneticResult
    from treecf.regions import RecourseRegion
    from treecf.targets import Target

FloatArray = npt.NDArray[np.float64]

_SEED_ATTEMPT_FACTOR = 3  # try up to 3k seeds per row when hunting k distinct plans


@dataclass(frozen=True)
class BatchRecord:
    """One counterfactual (or the infeasibility marker) for one dataset row.

    Fields mirror ``Counterfactual`` (``x_cf``, ``changes``, ``distance``,
    ``n_changed``, ``score_raw``, ``score_prob``, ``proof``, ``solver_stats``,
    ``region``), plus batch bookkeeping: ``id`` and ``k`` place the record in
    the dataset, and ``feasible`` distinguishes a real plan from the
    infeasibility marker.

    Attributes:
        id: The row identifier this record belongs to (an element of
            ``explain_batch``'s ``ids``, or the row's integer index when
            ``ids`` was not given).
        k: Rank of this plan among the row's feasible alternatives,
            ``0``-based, ascending by distance (``0`` is always the
            cheapest). For a wholly infeasible row (``diversity="seeds"``/
            ``"lever-blocking"``), the single infeasibility marker gets
            ``k=0``; for ``diversity="coalitions"``, an infeasible
            coalition's marker instead continues the same row's ascending
            sequence after its feasible plans, so each coalition still gets
            a distinct ``k``.
        feasible: ``False`` marks the infeasibility marker for a row (or
            coalition) that produced no plan; ``x_cf``/``changes``/
            ``distance``/``n_changed``/``score_raw``/``score_prob`` are then
            ``None``/``{}`` rather than real values.
        x_cf: The full counterfactual feature vector, or ``None`` when
            ``feasible`` is ``False``.
        changes: ``{feature: (factual_value, counterfactual_value)}`` for
            every feature that differs; ``{}`` when ``feasible`` is ``False``.
        distance: The weighted, normalized sum of per-feature changes,
            excluding the sparsity term (see ``Counterfactual.distance``), or
            ``None`` when ``feasible`` is ``False``.
        n_changed: ``len(changes)``, or ``None`` when ``feasible`` is
            ``False``.
        score_raw: The model's raw score at ``x_cf``, or ``None`` when
            ``feasible`` is ``False``.
        score_prob: ``sigmoid(score_raw)`` for a sigmoid-link model, ``None``
            for an identity-link model or when ``feasible`` is ``False``.
        seed: The seed that produced this plan, set only for
            ``diversity="seeds"``; ``None`` otherwise.
        blocked_lever: The feature frozen to produce this plan, set only for
            ``diversity="lever-blocking"`` alternatives (not the primary
            plan, ``k=0``); ``None`` otherwise.
        coalition: The coalition name this plan belongs to, set only for
            ``diversity="coalitions"`` (including the reserved
            ``"(all levers)"`` baseline when ``include_full=True``); ``None``
            otherwise.
        region: The certified box around ``x_cf``, set only when
            ``explain_batch`` ran with ``region=True`` and ``feasible`` is
            ``True``; ``None`` otherwise.
        proof: The claim this record makes, mirroring the single-instance
            result that produced it: ``Counterfactual.proof`` (``"heuristic"``
            | ``"optimal"`` | ``"optimal_within_gap"``) for a feasible
            record, ``Infeasible.proof`` (``"search_exhausted"`` |
            ``"certified"``) for an infeasibility marker.
        solver_stats: Exact-backend diagnostics for the solve behind this
            record, same keys as ``Counterfactual.solver_stats``; empty for
            genetic/python solves (those engines report no per-row stats).
        calibrator_fingerprint: The duck-typed ``fingerprint()`` of the
            calibrated target's calibrator, when it exposes one; ``None``
            for raw/probability targets or fingerprint-less calibrators.
            Repeated on every record so each file line is self-contained.
        score_calibrated: The calibrator's probability at ``x_cf`` for a
            calibrated target whose calibrator exposes ``predict_proba``;
            presentational only — the engine optimized and verified on the
            resolved raw interval. ``None`` otherwise.
    """

    id: object
    k: int
    feasible: bool
    x_cf: FloatArray | None
    changes: dict[str, tuple[float, float]]
    distance: float | None
    n_changed: int | None
    score_raw: float | None
    score_prob: float | None
    seed: int | None = None  # diversity="seeds": the seed that produced this plan
    blocked_lever: str | None = None  # diversity="lever-blocking": the frozen lever
    coalition: str | None = None  # diversity="coalitions": the group this plan may touch
    region: RecourseRegion | None = None  # set by explain_batch(..., region=True)
    proof: str = "heuristic"  # mirrors Counterfactual.proof / Infeasible.proof
    solver_stats: dict[str, object] = field(default_factory=dict)  # exact-backend only
    # Calibrated-target provenance and read-out (0.2.4). The fingerprint is one
    # value repeated per record on purpose: every JSON line stays self-contained
    # for a validator who receives only a slice of the file.
    calibrator_fingerprint: str | None = None
    score_calibrated: float | None = None  # presentational; the engine used raw_interval


@dataclass(frozen=True)
class BatchResult:
    """Counterfactuals for a whole dataset, addressable by row id.

    Returned by ``Explainer.explain_batch``; supports ``len()``, iteration
    over its ``records``, id lookup (``for_id``), a JSON round trip
    (``save``/``load``), and a pandas view (``to_frame``).

    Attributes:
        feature_names: The model's feature names, in the order ``x_cf``
            arrays are indexed by.
        diversity: The ``diversity`` mode ``explain_batch`` ran with
            (``"seeds"``, ``"lever-blocking"``, or ``"coalitions"``).
        records: Every ``BatchRecord``, feasible and infeasible, across every
            row and alternative/coalition; order matches the originating
            ``explain_batch`` call.
        essential_levers: ``{row_id: [feature, ...]}`` — for
            ``diversity="lever-blocking"`` rows only, the features whose
            freezing made every alternative infeasible (so the primary plan
            has no substitute for that lever). Empty for other diversity
            modes.
    """

    feature_names: tuple[str, ...]
    diversity: str
    records: tuple[BatchRecord, ...]
    essential_levers: dict[object, list[str]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[BatchRecord]:
        return iter(self.records)

    def for_id(self, row_id: object) -> list[BatchRecord]:
        """Every record (all alternatives/coalitions) for one dataset row.

        Args:
            row_id: A value from ``explain_batch``'s ``ids`` (or the row's
                integer index when ``ids`` was not given).

        Returns:
            The matching records, in their original order; ``[]`` if
            ``row_id`` is not present in this result.
        """
        return [r for r in self.records if r.id == row_id]

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write this result to a portable JSON file, reloadable with ``load``.

        Every field is encoded explicitly (NaN/Infinity-safe floats via
        ``encode_floats``), including ``region`` when set, so a round trip
        through ``save``/``load`` is lossless.

        Args:
            path: Destination file path; overwritten if it already exists.
        """
        data = {
            "feature_names": list(self.feature_names),
            "diversity": self.diversity,
            "essential_levers": {str(k): v for k, v in self.essential_levers.items()},
            "essential_lever_ids": [encode_floats(k) for k in self.essential_levers],
            "records": [
                {
                    "id": record.id,
                    "k": record.k,
                    "feasible": record.feasible,
                    "x_cf": None if record.x_cf is None else encode_floats(record.x_cf),
                    "changes": {
                        name: encode_floats(list(pair))
                        for name, pair in record.changes.items()
                    },
                    "distance": record.distance,
                    "n_changed": record.n_changed,
                    "score_raw": record.score_raw,
                    "score_prob": record.score_prob,
                    "seed": record.seed,
                    "blocked_lever": record.blocked_lever,
                    "coalition": record.coalition,
                    "proof": record.proof,
                    "calibrator_fingerprint": record.calibrator_fingerprint,
                    "score_calibrated": record.score_calibrated,
                    "solver_stats": {
                        key: encode_floats(value)
                        for key, value in record.solver_stats.items()
                    },
                    "region": (
                        None
                        if record.region is None
                        else {
                            "lo": encode_floats(record.region.lo),
                            "hi": encode_floats(record.region.hi),
                            "feature_intervals": {
                                name: encode_floats(list(pair))
                                for name, pair in record.region.feature_intervals.items()
                            },
                            "certified": record.region.certified,
                        }
                    ),
                }
                for record in self.records
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> BatchResult:
        """Read a ``BatchResult`` previously written by ``save``.

        A file saved without ``region=True``, or by a version of treecf
        before regions existed, loads with every record's ``region`` set to
        ``None``; a file saved before coalition support loads with every
        record's ``coalition`` set to ``None``; a file saved before per-record
        proofs existed loads with ``proof`` defaulted by feasibility
        (``"heuristic"``/``"search_exhausted"``) and empty ``solver_stats``.

        Args:
            path: Path to a file written by ``save``.

        Returns:
            The reconstructed ``BatchResult``.
        """
        from treecf.regions import RecourseRegion

        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        records = []
        for raw in data["records"]:
            raw_region = raw.get("region")  # absent key (pre-region files) -> None
            region = (
                None
                if raw_region is None
                else RecourseRegion(
                    lo=np.asarray(decode_floats(raw_region["lo"]), dtype=np.float64),
                    hi=np.asarray(decode_floats(raw_region["hi"]), dtype=np.float64),
                    feature_intervals={
                        name: tuple(decode_floats(pair))
                        for name, pair in raw_region["feature_intervals"].items()
                    },
                    certified=bool(raw_region["certified"]),
                )
            )
            records.append(
                BatchRecord(
                    id=raw["id"],
                    k=int(raw["k"]),
                    feasible=bool(raw["feasible"]),
                    x_cf=(
                        None
                        if raw["x_cf"] is None
                        else np.asarray(decode_floats(raw["x_cf"]), dtype=np.float64)
                    ),
                    changes={
                        name: tuple(decode_floats(pair))
                        for name, pair in raw["changes"].items()
                    },
                    distance=raw["distance"],
                    n_changed=raw["n_changed"],
                    score_raw=raw["score_raw"],
                    score_prob=raw["score_prob"],
                    seed=raw["seed"],
                    blocked_lever=raw["blocked_lever"],
                    coalition=raw.get("coalition"),  # absent in pre-coalition files
                    region=region,
                    # pre-0.2.2 files carry neither field; default by feasibility
                    proof=raw.get(
                        "proof", "heuristic" if raw["feasible"] else "search_exhausted"
                    ),
                    # absent in files written before 0.2.4
                    calibrator_fingerprint=raw.get("calibrator_fingerprint"),
                    score_calibrated=raw.get("score_calibrated"),
                    solver_stats={
                        key: decode_floats(value)
                        for key, value in raw.get("solver_stats", {}).items()
                    },
                )
            )
        essential_ids = [decode_floats(k) for k in data.get("essential_lever_ids", [])]
        essential_values = list(data.get("essential_levers", {}).values())
        return cls(
            feature_names=tuple(data["feature_names"]),
            diversity=data["diversity"],
            records=tuple(records),
            essential_levers=dict(zip(essential_ids, essential_values, strict=True)),
        )

    def to_frame(self) -> Any:
        """One row per (id, k), wide ``cf_<feature>`` columns (pandas, lazy import).

        Every ``BatchRecord`` field except ``x_cf``/``changes``/``region``/
        ``solver_stats`` becomes its own column (``solver_stats`` stays
        record-only — read it off the ``BatchRecord`` directly); ``x_cf`` is
        spread into one ``cf_<feature>`` column per model feature (``NaN`` for
        an infeasible record, or an unchanged feature's factual-equal value);
        ``changes`` is summarized as a ``changed_features`` column (sorted
        feature names).

        Returns:
            A pandas ``DataFrame`` with one row per record.

        Raises:
            TreecfError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - exercised without pandas
            raise TreecfError("to_frame() requires pandas: pip install pandas") from exc
        rows = []
        for record in self.records:
            row: dict[str, object] = {
                "id": record.id,
                "k": record.k,
                "feasible": record.feasible,
                "distance": record.distance,
                "n_changed": record.n_changed,
                "score_raw": record.score_raw,
                "score_prob": record.score_prob,
                "seed": record.seed,
                "blocked_lever": record.blocked_lever,
                "coalition": record.coalition,
                "proof": record.proof,
                "changed_features": sorted(record.changes),
            }
            for j, name in enumerate(self.feature_names):
                row[f"cf_{name}"] = (
                    float(record.x_cf[j]) if record.x_cf is not None else math.nan
                )
            rows.append(row)
        return pd.DataFrame(rows)


def explain_batch(
    explainer: Explainer,
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
) -> BatchResult:
    """See ``Explainer.explain_batch``.

    With the Rust backend, solves run in parallel inside the extension.
    ``time_budget_s`` stays per solve, but concurrent solves share cores: a
    solve that hits its wall-clock budget under contention may stop at a
    different generation than it would sequentially. Results are otherwise
    identical to solving row by row (stall/max-generation stops are
    deterministic).

    ``diversity="coalitions"`` produces one record per named coalition per
    row (``n_per_example`` is not used); ``coalitions``/``include_full`` are
    only valid in that mode.

    ``backend="exact"`` has no vectorized population, so this loops the
    single-instance exact solve per row (and per plan, for lever-blocking)
    sequentially instead of running the Rust engine's parallel waves -- each
    row still gets the full, undiminished ``time_budget_s``. Because that
    wall time is easy to underestimate, it requires ``allow_exact_batch=True``
    to opt in explicitly: without it this raises ``ValueError`` naming an
    estimate instead (``rows`` × ``plans`` × ``time_budget_s``, hours-formatted,
    where ``plans`` is ``n_per_example`` for ``"seeds"``/``"lever-blocking"`` or
    the coalition count for ``"coalitions"``) -- a floor, not a ceiling, since
    ``diversity="seeds"`` can retry each plan up to
    ``_SEED_ATTEMPT_FACTOR``x (3x) on a seed collision, pushing actual wall
    time higher; passing ``allow_exact_batch=True`` with any other backend
    also raises ``ValueError``. ``node_budget``/``gap`` thread through to
    every one of those solves unchanged; see ``Explainer.explain``.

    Opting in also replaces ``warm_start``'s (default ``True``) N sequential
    per-row genetic warm passes with a single vectorized one across every row
    (one ``Explainer._solve_batch`` call, ``min(time_budget_s * 0.25, 2.0)``)
    -- for ``diversity="seeds"`` every attempt of a row shares that one
    incumbent instead of each attempt warm-starting its own (so, unlike
    0.2.0, a batch run with ``n_per_example > 1`` is not required to explore
    as many distinct warm starts per row as an equivalent sequence of
    ``explain`` calls would; with ``n_per_example=1`` the result matches a
    sequential ``explain(..., backend="exact")`` call exactly). A row whose
    warm draw is infeasible gets no incumbent and runs unwarmed -- there is
    no per-row genetic fallback. For ``diversity="lever-blocking"`` only the
    primary solve (the unrestricted plan) uses the shared incumbent; the
    per-lever frozen clones keep 0.2.0's own per-solve ``warm_start`` (their
    constraint set differs by one ``Freeze``, so the primary's incumbent does
    not necessarily still verify for them). ``diversity="coalitions"``
    likewise keeps 0.2.0's per-coalition-solver behavior throughout -- each
    coalition's constraint set differs the same way. A ``KeyboardInterrupt``
    during any of this discards whatever the batch has not yet finished --
    there is no partial ``BatchResult``.

    ``region=True`` attaches a certified ``RecourseRegion`` to every feasible
    record's ``region`` field; ``BatchResult.save``/``load`` persist it
    (``lo``/``hi``/``feature_intervals``/``certified``, all explicit -- a
    file saved without ``region=True``, or by an older version, loads with
    every record's ``region`` set to ``None``). Unlike the wave-parallel GA
    search above, region growth runs one record at a time (each is one
    ``Explainer._region_for`` call, rust-first when the extension is
    importable, exactly as a single ``explain(..., region=True)`` call would
    run it) -- there is no batched/parallel region path.
    """
    from treecf.api import _degraded_summary, _resolve_exact_kwargs

    if target.bands_spec is not None:
        raise TreecfError("Target.bands is not supported in explain_batch; loop bands explicitly")
    if diversity not in ("seeds", "lever-blocking", "coalitions"):
        raise TreecfError("diversity must be 'seeds', 'lever-blocking', or 'coalitions'")
    if diversity == "coalitions" and coalitions is None:
        raise TreecfError("diversity='coalitions' requires a coalitions mapping")
    if diversity != "coalitions" and (coalitions is not None or include_full):
        raise TreecfError("coalitions/include_full are only valid with diversity='coalitions'")
    # Validated here too (not only inside `_explain`) because the rust
    # wave-parallel paths below (`_rows_by_seed_waves`, `_lever_primaries`)
    # never call `_explain` and would otherwise silently ignore the kwargs.
    resolved_warm_start, _, _ = _resolve_exact_kwargs(backend, warm_start, node_budget, gap)
    X = np.asarray(X, dtype=np.float64)
    if backend == "exact" and not allow_exact_batch:
        if diversity == "coalitions":
            assert coalitions is not None  # validated above
            plans = len(coalitions) + (1 if include_full else 0)
        else:
            plans = n_per_example
        hours = len(X) * plans * time_budget_s / 3600.0
        raise ValueError(
            "backend='exact' inside explain_batch loops the single-instance exact "
            "solve sequentially, one solve per (row, plan) pair -- no vectorized "
            f"population to parallelize -- estimated {len(X)} rows x {plans} plans x "
            f"{time_budget_s:.4g}s time_budget_s each ~= {hours:.1f} hours at least "
            "(diversity='seeds' can retry each plan up to 3x on a seed collision, "
            "pushing this higher); pass allow_exact_batch=True to opt in explicitly. "
            "Opting in also switches warm_start (default True) from one genetic pass "
            "per row (or, in seeds mode, per attempt) to a single vectorized pass "
            "across every row."
        )
    if allow_exact_batch and backend != "exact":
        raise ValueError("allow_exact_batch is only valid with backend='exact'")
    row_ids: Sequence[object] = range(len(X)) if ids is None else list(ids)
    if len(row_ids) != len(X):
        raise TreecfError("ids must have one entry per row of X")

    # one aggregate factual-violation warning for the whole batch; the per-row
    # solve paths all run with warn_factual=False so nothing warns twice
    counts: dict[str, int] = {}
    affected = 0
    for row in X:
        items = explainer.compiled._factual_violation_items(row)
        if items:
            affected += 1
            for label, _ in items:
                counts[label] = counts.get(label, 0) + 1
    if affected:
        summary = "; ".join(f"{label}: {n} rows" for label, n in counts.items())
        warnings.warn(
            f"factual constraint violations in {affected}/{len(X)} rows ({summary}); "
            "affected plans include changes made solely to satisfy them.",
            TreecfWarning,
            stacklevel=2,
        )

    records: list[BatchRecord] = []
    essential: dict[object, list[str]] = {}
    # Per-row degraded-exact-result buckets, whatever the diversity mode; a
    # row counts as affected once, however many of its solves degraded --
    # the same "affected/total rows" shape `explain_batch`'s own
    # factual-violation aggregate above uses.
    row_degraded: list[list[_Degradation]] = [[] for _ in row_ids]
    if diversity == "coalitions":
        assert coalitions is not None  # narrowed by the validation above
        records = _rows_by_coalitions(
            explainer, X, target, row_ids, coalitions, include_full,
            backend, time_budget_s, sparsity_weight, seed=seed,
            warm_start=warm_start, node_budget=node_budget, gap=gap, region=region,
            row_degraded=row_degraded,
        )
    elif diversity == "seeds" and backend in ("genetic", "genetic-rust"):
        # Same attempts, dedup, and stopping rule as `_row_by_seeds`, but each
        # wave solves every unfinished row's next attempts in one parallel
        # Rust call — the output is identical to the sequential loop.
        records = _rows_by_seed_waves(
            explainer, X, target, row_ids, n_per_example,
            time_budget_s, sparsity_weight, seed=seed, region=region,
        )
    else:
        primaries: list[Counterfactual | Infeasible] | None = None
        # One row -> one incumbent (or None), the vectorized warm pass's
        # output; None throughout means either warm_start=False or (below)
        # this diversity mode not sharing an incumbent at all -- either way
        # every row's own explain call runs unwarmed rather than falling back
        # to a per-row genetic pass.
        row_incumbents: list[tuple[float, FloatArray] | None] | None = None
        if diversity == "lever-blocking" and backend in ("genetic", "genetic-rust"):
            # All rows' primary solves share the constraints, so they run as
            # one parallel Rust call; the per-lever loop stays sequential.
            primaries = _lever_primaries(
                explainer, X, target, time_budget_s, sparsity_weight, seed=seed
            )
        elif backend == "exact":
            # One vectorized warm pass replaces the per-row (seeds: per-attempt)
            # internal warm starts `_row_by_seeds`/`_row_by_lever_blocking` would
            # otherwise each run on their own; see `allow_exact_batch`'s docstring.
            interval = target.raw_interval(explainer.ir.link)
            row_seeds = (
                [seed * 1_000_003 + i * 1_009 for i in range(len(X))]
                if diversity == "seeds"
                else [seed] * len(X)
            )
            row_incumbents = (
                _batch_warm_incumbents(
                    explainer, X, row_seeds, interval, time_budget_s, sparsity_weight
                )
                if resolved_warm_start
                else [None] * len(X)
            )
            if diversity == "lever-blocking":
                primaries = _exact_lever_primaries(
                    explainer, X, target, time_budget_s, sparsity_weight,
                    node_budget, gap, seed, row_incumbents, row_degraded,
                )
        for i, row_id in enumerate(row_ids):
            if diversity == "seeds":
                row_records = _row_by_seeds(
                    explainer, X[i], target, row_id, n_per_example,
                    backend, time_budget_s, sparsity_weight,
                    master_seed=seed * 1_000_003 + i * 1_009,
                    warm_start=False if backend == "exact" else warm_start,
                    node_budget=node_budget, gap=gap, region=region,
                    degraded=row_degraded[i],
                    incumbent=None if row_incumbents is None else row_incumbents[i],
                )
            else:
                row_records, row_essential = _row_by_lever_blocking(
                    explainer, X[i], target, row_id, n_per_example,
                    backend, time_budget_s, sparsity_weight, seed=seed,
                    primary=None if primaries is None else primaries[i],
                    warm_start=warm_start, node_budget=node_budget, gap=gap, region=region,
                    degraded=row_degraded[i],
                )
                essential[row_id] = row_essential
            records.extend(row_records)

    degraded_all = [d for bucket in row_degraded for d in bucket]
    affected_rows = sum(1 for bucket in row_degraded if bucket)
    message = _degraded_summary(degraded_all, affected_rows, len(row_ids), "rows")
    if message is not None:
        warnings.warn(message, TreecfWarning, stacklevel=2)

    if target.space == "calibrated":
        from dataclasses import replace as _replace

        from treecf.api import _calibrated_readout
        from treecf.audit import _duck_fingerprint

        calibrator_fp = _duck_fingerprint(target.calibrator)
        if calibrator_fp is not None:
            records = [_replace(r, calibrator_fingerprint=calibrator_fp) for r in records]
        # The Rust wave paths assemble Counterfactuals without going through
        # Explainer._explain, so the read-out is filled here for every path.
        records = [
            _replace(r, score_calibrated=_calibrated_readout(target, r.score_raw))
            if r.feasible and r.score_calibrated is None and r.score_raw is not None
            else r
            for r in records
        ]

    return BatchResult(
        feature_names=explainer.ir.feature_names,
        diversity=diversity,
        records=tuple(records),
        essential_levers=essential,
    )


def _exact_stats(stats: dict[str, object]) -> dict[str, object]:
    """Stats worth mirroring onto a record: the exact backend's per-solve
    diagnostics (recognized by their ``completed`` key). Genetic/python engine
    stats are not mirrored — those engines report no per-row diagnostics."""
    return stats if "completed" in stats else {}


def _record_from(
    row_id: object,
    k: int,
    cf: Counterfactual,
    seed: int | None = None,
    blocked_lever: str | None = None,
    coalition: str | None = None,
    region: RecourseRegion | None = None,
) -> BatchRecord:
    return BatchRecord(
        id=row_id,
        k=k,
        feasible=True,
        x_cf=cf.x_cf,
        changes=cf.changes,
        distance=cf.distance,
        n_changed=cf.n_changed,
        score_raw=cf.score_raw,
        score_prob=cf.score_prob,
        score_calibrated=cf.score_calibrated,
        seed=seed,
        blocked_lever=blocked_lever,
        coalition=coalition,
        region=region,
        proof=cf.proof,
        solver_stats=_exact_stats(cf.solver_stats),
    )


def _infeasible_record(
    row_id: object,
    k: int = 0,
    coalition: str | None = None,
    infeasible: Infeasible | None = None,
) -> BatchRecord:
    return BatchRecord(
        id=row_id, k=k, feasible=False, x_cf=None, changes={},
        distance=None, n_changed=None, score_raw=None, score_prob=None,
        coalition=coalition,
        proof="search_exhausted" if infeasible is None else infeasible.proof,
        solver_stats={} if infeasible is None else _exact_stats(infeasible.solver_stats),
    )


def _rows_by_seed_waves(
    explainer: Explainer,
    X: FloatArray,
    target: Target,
    row_ids: Sequence[object],
    n_per_example: int,
    time_budget_s: float,
    sparsity_weight: float,
    seed: int,
    region: bool = False,
) -> list[BatchRecord]:
    """Wave-parallel `_row_by_seeds` over all rows (Rust backend only).

    Wave ``w`` solves attempts ``w*n .. w*n + n - 1`` of every row that still
    needs plans; results are then consumed per row in attempt order with the
    sequential dedup/stop logic, so extra attempts computed past a row's
    stopping point are simply discarded.
    """
    from treecf.api import Counterfactual

    if explainer.plausibility is not None and np.isnan(X).any():
        raise TreecfError("plausibility with missing factual values is not supported")
    interval = target.raw_interval(explainer.ir.link)
    n = n_per_example
    found: list[dict[frozenset[str], tuple[Counterfactual, int]]] = [
        {} for _ in row_ids
    ]
    active = list(range(len(row_ids)))
    for wave in range(_SEED_ATTEMPT_FACTOR):
        if not active:
            break
        tasks = [
            (i, seed * 1_000_003 + i * 1_009 + wave * n + a)
            for i in active
            for a in range(n)
        ]
        results = explainer._solve_batch(X, tasks, interval, time_budget_s, sparsity_weight)
        scores = _wave_scores(explainer, results)
        for t, ((i, attempt_seed), result) in enumerate(zip(tasks, results, strict=True)):
            if len(found[i]) == n or result.x_cf is None:
                continue
            outcome = explainer._finalize_candidate(
                X[i], result.x_cf, interval, result.stats, score=scores.get(t)
            )
            if isinstance(outcome, Counterfactual):
                key = frozenset(outcome.changes)
                if key not in found[i]:
                    found[i][key] = (outcome, attempt_seed)
        active = [i for i in active if len(found[i]) < n]

    records: list[BatchRecord] = []
    for i, row_id in enumerate(row_ids):
        if not found[i]:
            records.append(_infeasible_record(row_id))
            continue
        ranked = sorted(found[i].values(), key=lambda pair: pair[0].distance)[:n]
        records.extend(
            _record_from(
                row_id, k, cf, seed=cf_seed,
                region=explainer._region_for(X[i], cf.x_cf, interval) if region else None,
            )
            for k, (cf, cf_seed) in enumerate(ranked)
        )
    return records


def _wave_scores(explainer: Explainer, results: Sequence[GeneticResult]) -> dict[int, float]:
    """One vectorized IR pass over a wave's candidates. NaN candidates keep the
    scalar path so models without missing routing fail exactly as in a single
    explain."""
    candidates = {
        t: result.x_cf for t, result in enumerate(results) if result.x_cf is not None
    }
    scorable = [t for t, cf in candidates.items() if not np.isnan(cf).any()]
    if not scorable:
        return {}
    stacked = np.stack([candidates[t] for t in scorable])
    wave_scores = raw_score_batch_prepared(
        explainer._prepared_tree_arrays(), explainer.ir.base_score, stacked
    )
    return dict(zip(scorable, (float(s) for s in wave_scores), strict=True))


def _lever_primaries(
    explainer: Explainer,
    X: FloatArray,
    target: Target,
    time_budget_s: float,
    sparsity_weight: float,
    seed: int,
) -> list[Counterfactual | Infeasible]:
    """All rows' primary lever-blocking solves in one parallel Rust call.

    Each outcome is bitwise-identical to ``explainer.explain(X[i], ..., seed=seed)``.
    """
    from treecf.api import Infeasible

    if explainer.plausibility is not None and np.isnan(X).any():
        raise TreecfError("plausibility with missing factual values is not supported")
    interval = target.raw_interval(explainer.ir.link)
    tasks = [(i, seed) for i in range(len(X))]
    results = explainer._solve_batch(X, tasks, interval, time_budget_s, sparsity_weight)
    scores = _wave_scores(explainer, results)
    outcomes: list[Counterfactual | Infeasible] = []
    for t, result in enumerate(results):
        if result.x_cf is None:
            outcomes.append(
                Infeasible(
                    reason="heuristic search exhausted (genetic backend)",
                    proof="search_exhausted",
                )
            )
        else:
            outcomes.append(
                explainer._finalize_candidate(
                    X[t], result.x_cf, interval, result.stats, score=scores.get(t)
                )
            )
    return outcomes


def _batch_warm_incumbents(
    explainer: Explainer,
    X: FloatArray,
    row_seeds: Sequence[int],
    interval: tuple[float, float],
    time_budget_s: float,
    sparsity_weight: float,
) -> list[tuple[float, FloatArray] | None]:
    """One vectorized genetic warm pass for every row of an exact batch: a
    single ``Explainer._solve_batch`` call at ``min(time_budget_s * 0.25,
    2.0)``, replacing the N sequential per-row (or, in seeds mode,
    per-attempt) warm passes ``warm_start=True`` would otherwise run one at a
    time through ``_explain_exact``'s own internal pass. ``row_seeds[i]`` is
    the seed row ``i``'s own internal warm pass would have used, so with a
    single plan per row this reproduces a sequential
    ``explain(..., backend="exact")`` call's incumbent exactly.

    A row whose warm draw is infeasible, or fails float-space verification,
    contributes ``None`` -- there is no per-row genetic fallback here; the
    exact search for that row runs unwarmed, exactly as ``warm_start=False``
    would.
    """
    from treecf.api import Counterfactual
    from treecf.backends._exact_domains import _cost_of_row

    if explainer.plausibility is not None and np.isnan(X).any():
        raise TreecfError("plausibility with missing factual values is not supported")
    warm_budget = min(time_budget_s * 0.25, 2.0)
    tasks = [(i, row_seeds[i]) for i in range(len(X))]
    results = explainer._solve_batch(X, tasks, interval, warm_budget, sparsity_weight)
    incumbents: list[tuple[float, FloatArray] | None] = []
    for i, result in enumerate(results):
        if result.x_cf is None:
            incumbents.append(None)
            continue
        candidate = explainer._finalize_candidate(X[i], result.x_cf, interval, result.stats)
        if isinstance(candidate, Counterfactual) and (
            explainer._verify(X[i], candidate.x_cf, interval) is None
        ):
            cost = _cost_of_row(
                X[i], candidate.x_cf, explainer.sigma, explainer.weights, sparsity_weight,
                explainer.compiled.allow_missing,
            )
            incumbents.append((cost, candidate.x_cf))
        else:
            incumbents.append(None)
    return incumbents


def _exact_lever_primaries(
    explainer: Explainer,
    X: FloatArray,
    target: Target,
    time_budget_s: float,
    sparsity_weight: float,
    node_budget: int | None,
    gap: float | None,
    seed: int,
    row_incumbents: Sequence[tuple[float, FloatArray] | None],
    row_degraded: list[list[_Degradation]],
) -> list[Counterfactual | Infeasible]:
    """Every row's lever-blocking primary exact solve, sequentially -- the
    exact backend has no vectorized population to parallelize -- each
    pre-seeded with its row's ``_batch_warm_incumbents`` incumbent instead of
    running its own internal warm pass (``warm_start=False`` throughout, so a
    ``None`` incumbent runs that row's primary unwarmed rather than falling
    back to a per-row genetic pass). The per-lever frozen clones computed
    afterwards by ``_row_by_lever_blocking`` are not covered here -- their
    constraint set differs by one ``Freeze``, so this incumbent does not
    necessarily still verify for them; they keep their own per-solve
    ``warm_start``.
    """
    return [
        explainer._explain_one(
            X[i], target, "exact", time_budget_s, sparsity_weight, seed,
            warn_factual=False, warm_start=False, node_budget=node_budget, gap=gap,
            degraded=row_degraded[i], incumbent=row_incumbents[i],
        )
        for i in range(len(X))
    ]


def _rows_by_coalitions(
    explainer: Explainer,
    X: FloatArray,
    target: Target,
    row_ids: Sequence[object],
    coalitions: Mapping[str, Sequence[str]],
    include_full: bool,
    backend: str,
    time_budget_s: float,
    sparsity_weight: float,
    seed: int,
    warm_start: bool | None = None,
    node_budget: int | None = None,
    gap: float | None = None,
    region: bool = False,
    row_degraded: list[list[_Degradation]] | None = None,
) -> list[BatchRecord]:
    """One record per named coalition per row (plus the optional baseline).

    With the Rust backend each coalition's rows solve in one parallel wave on
    a freeze-complement clone; ``backend="python"`` loops the same clones row
    by row. Per row, feasible plans are ranked by distance (k = 0, 1, ...),
    then infeasible coalitions follow in coalition order — an infeasible
    record with ``coalition`` set means that group alone cannot reach the
    target. ``row_degraded`` (one bucket per row, indexed like ``row_ids``)
    collects exact-backend degradations across every coalition's solve for
    ``explain_batch``'s own aggregate warning.
    """
    from treecf.api import _ALL_LEVERS, Counterfactual, Infeasible, _validate_coalitions

    normalized = _validate_coalitions(coalitions, explainer.ir.feature_names, include_full)
    solvers: dict[str, Explainer] = {}
    if include_full:
        solvers[_ALL_LEVERS] = explainer
    solvers.update(explainer._coalition_explainers(normalized))

    if explainer.plausibility is not None and np.isnan(X).any():
        raise TreecfError("plausibility with missing factual values is not supported")
    interval = target.raw_interval(explainer.ir.link)

    outcomes: dict[str, list[Counterfactual | Infeasible]] = {}
    rust = backend in ("genetic", "genetic-rust")
    for name, solver in solvers.items():
        if rust:
            tasks = [(i, seed) for i in range(len(X))]
            results = solver._solve_batch(X, tasks, interval, time_budget_s, sparsity_weight)
            scores = _wave_scores(solver, results)
            outcomes[name] = [
                Infeasible(
                    reason="heuristic search exhausted (genetic backend)",
                    proof="search_exhausted",
                )
                if result.x_cf is None
                else solver._finalize_candidate(
                    X[i], result.x_cf, interval, result.stats, score=scores.get(i)
                )
                for i, result in enumerate(results)
            ]
        else:
            outcomes[name] = [
                solver._explain_one(
                    X[i], target, backend, time_budget_s, sparsity_weight, seed,
                    warn_factual=False,
                    warm_start=warm_start, node_budget=node_budget, gap=gap,
                    degraded=None if row_degraded is None else row_degraded[i],
                )
                for i in range(len(X))
            ]

    records: list[BatchRecord] = []
    for i, row_id in enumerate(row_ids):
        feasible = [
            (name, outcome)
            for name, outcome in ((n, outcomes[n][i]) for n in solvers)
            if isinstance(outcome, Counterfactual)
        ]
        feasible.sort(key=lambda pair: pair[1].distance)
        k = 0
        for name, cf in feasible:
            reg = solvers[name]._region_for(X[i], cf.x_cf, interval) if region else None
            records.append(_record_from(row_id, k, cf, coalition=name, region=reg))
            k += 1
        for name in solvers:
            outcome = outcomes[name][i]
            if isinstance(outcome, Infeasible):
                records.append(_infeasible_record(row_id, k=k, coalition=name, infeasible=outcome))
                k += 1
    return records


def _row_by_seeds(
    explainer: Explainer,
    x: FloatArray,
    target: Target,
    row_id: object,
    n_per_example: int,
    backend: str,
    time_budget_s: float,
    sparsity_weight: float,
    master_seed: int,
    warm_start: bool | None = None,
    node_budget: int | None = None,
    gap: float | None = None,
    region: bool = False,
    degraded: list[_Degradation] | None = None,
    incumbent: tuple[float, FloatArray] | None = None,
) -> list[BatchRecord]:
    """``incumbent``, when given, is shared by every attempt below instead of
    each attempt warm-starting its own (the caller is expected to pass
    ``warm_start=False`` alongside it, so a ``None`` incumbent -- an
    infeasible warm draw -- also runs unwarmed rather than falling back to a
    per-attempt genetic pass; see ``explain_batch``'s ``allow_exact_batch``)."""
    from treecf.api import Counterfactual, Infeasible

    found: dict[frozenset[str], tuple[Counterfactual, int]] = {}
    last_infeasible: Infeasible | None = None
    for attempt in range(_SEED_ATTEMPT_FACTOR * n_per_example):
        attempt_seed = master_seed + attempt
        result = explainer._explain(
            x, target, backend, time_budget_s, sparsity_weight, attempt_seed,
            warn_factual=False,  # explain_batch already warned in aggregate
            warm_start=warm_start, node_budget=node_budget, gap=gap,
            degraded=degraded, incumbent=incumbent,
        )
        if isinstance(result, Counterfactual):
            key = frozenset(result.changes)
            if key not in found:
                found[key] = (result, attempt_seed)
                if len(found) == n_per_example:
                    break
        elif isinstance(result, Infeasible):  # bands are rejected by explain_batch
            last_infeasible = result
    if not found:
        return [_infeasible_record(row_id, infeasible=last_infeasible)]
    ranked = sorted(found.values(), key=lambda pair: pair[0].distance)[:n_per_example]
    interval = target.raw_interval(explainer.ir.link) if region else None
    return [
        _record_from(
            row_id, k, cf, seed=cf_seed,
            region=explainer._region_for(x, cf.x_cf, interval) if interval is not None else None,
        )
        for k, (cf, cf_seed) in enumerate(ranked)
    ]


def _row_by_lever_blocking(
    explainer: Explainer,
    x: FloatArray,
    target: Target,
    row_id: object,
    n_per_example: int,
    backend: str,
    time_budget_s: float,
    sparsity_weight: float,
    seed: int,
    primary: Counterfactual | Infeasible | None = None,
    warm_start: bool | None = None,
    node_budget: int | None = None,
    gap: float | None = None,
    region: bool = False,
    degraded: list[_Degradation] | None = None,
) -> tuple[list[BatchRecord], list[str]]:
    from treecf.api import Counterfactual

    if primary is None:
        explained = explainer._explain(
            x, target, backend, time_budget_s, sparsity_weight, seed,
            warn_factual=False,  # explain_batch already warned in aggregate
            warm_start=warm_start, node_budget=node_budget, gap=gap,
            degraded=degraded,
        )
        assert not isinstance(explained, dict)  # bands are rejected by explain_batch
        primary = explained
    if not isinstance(primary, Counterfactual):
        return [_infeasible_record(row_id, infeasible=primary)], []

    interval = target.raw_interval(explainer.ir.link) if region else None
    primary_region = (
        explainer._region_for(x, primary.x_cf, interval) if interval is not None else None
    )
    records = [_record_from(row_id, 0, primary, region=primary_region)]
    seen = {frozenset(primary.changes)}
    essential: list[str] = []
    names = explainer.ir.feature_names
    index = {name: j for j, name in enumerate(names)}
    levers = sorted(
        primary.changes,
        key=lambda f: abs(primary.changes[f][1] - primary.changes[f][0])
        / explainer.sigma[index[f]],
        reverse=True,
    )
    for lever in levers:
        if len(records) >= n_per_example:
            break
        clone = explainer._with_extra_freezes([lever])
        alternative = clone._explain(
            x, target, backend, time_budget_s, sparsity_weight, seed,
            warn_factual=False,  # explain_batch already warned in aggregate
            warm_start=warm_start, node_budget=node_budget, gap=gap,
            degraded=degraded,
        )
        if isinstance(alternative, Counterfactual):
            key = frozenset(alternative.changes)
            if key not in seen:
                seen.add(key)
                alt_region = (
                    clone._region_for(x, alternative.x_cf, interval)
                    if interval is not None
                    else None
                )
                records.append(
                    _record_from(
                        row_id, len(records), alternative, blocked_lever=lever, region=alt_region
                    )
                )
        else:
            essential.append(lever)
    return records, essential
