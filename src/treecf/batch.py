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
    from treecf.api import Counterfactual, Explainer, Infeasible
    from treecf.backends.genetic import GeneticResult
    from treecf.regions import RecourseRegion
    from treecf.targets import Target

FloatArray = npt.NDArray[np.float64]

_SEED_ATTEMPT_FACTOR = 3  # try up to 3k seeds per row when hunting k distinct plans


@dataclass(frozen=True)
class BatchRecord:
    """One counterfactual (or the infeasibility marker) for one dataset row."""

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


@dataclass(frozen=True)
class BatchResult:
    """Counterfactuals for a whole dataset, addressable by row id."""

    feature_names: tuple[str, ...]
    diversity: str
    records: tuple[BatchRecord, ...]
    essential_levers: dict[object, list[str]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[BatchRecord]:
        return iter(self.records)

    def for_id(self, row_id: object) -> list[BatchRecord]:
        return [r for r in self.records if r.id == row_id]

    def save(self, path: str | os.PathLike[str]) -> None:
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
        """One row per (id, k), wide ``cf_<feature>`` columns (pandas, lazy import)."""
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
    sequentially instead of running the Rust engine's parallel waves.
    ``warm_start``/``node_budget``/``gap`` configure that solve; they are
    only valid together with ``backend="exact"`` (see ``Explainer.explain``).
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
    from treecf.api import _resolve_exact_kwargs

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
    _resolve_exact_kwargs(backend, warm_start, node_budget, gap)
    X = np.asarray(X, dtype=np.float64)
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
    if diversity == "coalitions":
        assert coalitions is not None  # narrowed by the validation above
        records = _rows_by_coalitions(
            explainer, X, target, row_ids, coalitions, include_full,
            backend, time_budget_s, sparsity_weight, seed=seed,
            warm_start=warm_start, node_budget=node_budget, gap=gap, region=region,
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
        if diversity == "lever-blocking" and backend in ("genetic", "genetic-rust"):
            # All rows' primary solves share the constraints, so they run as
            # one parallel Rust call; the per-lever loop stays sequential.
            primaries = _lever_primaries(
                explainer, X, target, time_budget_s, sparsity_weight, seed=seed
            )
        for i, row_id in enumerate(row_ids):
            if diversity == "seeds":
                row_records = _row_by_seeds(
                    explainer, X[i], target, row_id, n_per_example,
                    backend, time_budget_s, sparsity_weight,
                    master_seed=seed * 1_000_003 + i * 1_009,
                    warm_start=warm_start, node_budget=node_budget, gap=gap, region=region,
                )
            else:
                row_records, row_essential = _row_by_lever_blocking(
                    explainer, X[i], target, row_id, n_per_example,
                    backend, time_budget_s, sparsity_weight, seed=seed,
                    primary=None if primaries is None else primaries[i],
                    warm_start=warm_start, node_budget=node_budget, gap=gap, region=region,
                )
                essential[row_id] = row_essential
            records.extend(row_records)

    return BatchResult(
        feature_names=explainer.ir.feature_names,
        diversity=diversity,
        records=tuple(records),
        essential_levers=essential,
    )


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
        seed=seed,
        blocked_lever=blocked_lever,
        coalition=coalition,
        region=region,
    )


def _infeasible_record(row_id: object, k: int = 0, coalition: str | None = None) -> BatchRecord:
    return BatchRecord(
        id=row_id, k=k, feasible=False, x_cf=None, changes={},
        distance=None, n_changed=None, score_raw=None, score_prob=None,
        coalition=coalition,
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
) -> list[BatchRecord]:
    """One record per named coalition per row (plus the optional baseline).

    With the Rust backend each coalition's rows solve in one parallel wave on
    a freeze-complement clone; ``backend="python"`` loops the same clones row
    by row. Per row, feasible plans are ranked by distance (k = 0, 1, ...),
    then infeasible coalitions follow in coalition order — an infeasible
    record with ``coalition`` set means that group alone cannot reach the
    target.
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
            if not isinstance(outcomes[name][i], Counterfactual):
                records.append(_infeasible_record(row_id, k=k, coalition=name))
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
) -> list[BatchRecord]:
    from treecf.api import Counterfactual

    found: dict[frozenset[str], tuple[Counterfactual, int]] = {}
    for attempt in range(_SEED_ATTEMPT_FACTOR * n_per_example):
        attempt_seed = master_seed + attempt
        result = explainer._explain(
            x, target, backend, time_budget_s, sparsity_weight, attempt_seed,
            warn_factual=False,  # explain_batch already warned in aggregate
            warm_start=warm_start, node_budget=node_budget, gap=gap,
        )
        if isinstance(result, Counterfactual):
            key = frozenset(result.changes)
            if key not in found:
                found[key] = (result, attempt_seed)
                if len(found) == n_per_example:
                    break
    if not found:
        return [_infeasible_record(row_id)]
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
) -> tuple[list[BatchRecord], list[str]]:
    from treecf.api import Counterfactual

    if primary is None:
        explained = explainer._explain(
            x, target, backend, time_budget_s, sparsity_weight, seed,
            warn_factual=False,  # explain_batch already warned in aggregate
            warm_start=warm_start, node_budget=node_budget, gap=gap,
        )
        assert not isinstance(explained, dict)  # bands are rejected by explain_batch
        primary = explained
    if not isinstance(primary, Counterfactual):
        return [_infeasible_record(row_id)], []

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
