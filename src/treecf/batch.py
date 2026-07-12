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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from treecf._errors import TreecfError
from treecf._json import decode_floats, encode_floats
from treecf.ir.evaluate import raw_score_batch_prepared

if TYPE_CHECKING:
    from treecf.api import Counterfactual, Explainer
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
                }
                for record in self.records
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> BatchResult:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        records = []
        for raw in data["records"]:
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
) -> BatchResult:
    if target.bands_spec is not None:
        raise TreecfError("Target.bands is not supported in explain_batch; loop bands explicitly")
    if diversity not in ("seeds", "lever-blocking"):
        raise TreecfError("diversity must be 'seeds' or 'lever-blocking'")
    X = np.asarray(X, dtype=np.float64)
    row_ids: Sequence[object] = range(len(X)) if ids is None else list(ids)
    if len(row_ids) != len(X):
        raise TreecfError("ids must have one entry per row of X")

    records: list[BatchRecord] = []
    essential: dict[object, list[str]] = {}
    if diversity == "seeds" and backend in ("genetic", "genetic-rust"):
        # Same attempts, dedup, and stopping rule as `_row_by_seeds`, but each
        # wave solves every unfinished row's next attempts in one parallel
        # Rust call — the output is identical to the sequential loop.
        records = _rows_by_seed_waves(
            explainer, X, target, row_ids, n_per_example,
            time_budget_s, sparsity_weight, seed=seed,
        )
    else:
        for i, row_id in enumerate(row_ids):
            if diversity == "seeds":
                row_records = _row_by_seeds(
                    explainer, X[i], target, row_id, n_per_example,
                    backend, time_budget_s, sparsity_weight,
                    master_seed=seed * 1_000_003 + i * 1_009,
                )
            else:
                row_records, row_essential = _row_by_lever_blocking(
                    explainer, X[i], target, row_id, n_per_example,
                    backend, time_budget_s, sparsity_weight, seed=seed,
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
    )


def _infeasible_record(row_id: object) -> BatchRecord:
    return BatchRecord(
        id=row_id, k=0, feasible=False, x_cf=None, changes={},
        distance=None, n_changed=None, score_raw=None, score_prob=None,
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
        # One vectorized IR pass scores the wave's candidates (§8.1 stays in
        # float space through the IR). NaN candidates keep the scalar path so
        # models without missing routing fail exactly as in a single explain.
        candidates = {
            t: result.x_cf for t, result in enumerate(results) if result.x_cf is not None
        }
        scorable = [t for t, cf in candidates.items() if not np.isnan(cf).any()]
        scores: dict[int, float] = {}
        if scorable:
            stacked = np.stack([candidates[t] for t in scorable])
            wave_scores = raw_score_batch_prepared(
                explainer._prepared_tree_arrays(), explainer.ir.base_score, stacked
            )
            scores = dict(zip(scorable, (float(s) for s in wave_scores), strict=True))
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
            _record_from(row_id, k, cf, seed=cf_seed) for k, (cf, cf_seed) in enumerate(ranked)
        )
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
) -> list[BatchRecord]:
    from treecf.api import Counterfactual

    found: dict[frozenset[str], tuple[Counterfactual, int]] = {}
    for attempt in range(_SEED_ATTEMPT_FACTOR * n_per_example):
        attempt_seed = master_seed + attempt
        result = explainer.explain(
            x, target, backend=backend, time_budget_s=time_budget_s,
            sparsity_weight=sparsity_weight, seed=attempt_seed,
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
    return [
        _record_from(row_id, k, cf, seed=cf_seed)
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
) -> tuple[list[BatchRecord], list[str]]:
    from treecf.api import Counterfactual

    primary = explainer.explain(
        x, target, backend=backend, time_budget_s=time_budget_s,
        sparsity_weight=sparsity_weight, seed=seed,
    )
    if not isinstance(primary, Counterfactual):
        return [_infeasible_record(row_id)], []

    records = [_record_from(row_id, 0, primary)]
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
        alternative = explainer._with_extra_freezes([lever]).explain(
            x, target, backend=backend, time_budget_s=time_budget_s,
            sparsity_weight=sparsity_weight, seed=seed,
        )
        if isinstance(alternative, Counterfactual):
            key = frozenset(alternative.changes)
            if key not in seen:
                seen.add(key)
                records.append(
                    _record_from(row_id, len(records), alternative, blocked_lever=lever)
                )
        else:
            essential.append(lever)
    return records, essential
