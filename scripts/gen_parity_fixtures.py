"""Generate cross-language parity fixtures.

Run with: uv run python scripts/gen_parity_fixtures.py
Regenerating overwrites tests/fixtures/parity/*.json — do this ONLY when the
Python GA's behavior changes deliberately; the fixtures freeze it otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tests.conftest import make_random_ir
from tests.parity import harness
from treecf.constraints import compile_constraints
from treecf.constraints.flatten import flatten_constraints
from treecf.ir.evaluate import raw_score
from treecf.ir.flatten import flatten_ir
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

GOLDEN_SEEDS = list(range(5))
DIST_SEEDS = list(range(1000, 1200))  # 200 seeds
GA = {
    "population": 60,
    "max_generations": 120,
    "stall_generations": 25,
    "time_budget_s": 1e9,  # stall/max-gen stopping only: machine-independent
}


def _leaf(i: int, v: float) -> Node:
    return Node(i, None, None, None, None, None, None, v)


def _stump(feature: int, threshold: float, left: float, right: float) -> Tree:
    return Tree(
        nodes=(
            Node(0, feature, threshold, SplitOp.LT, True, 1, 2, None),
            _leaf(1, left),
            _leaf(2, right),
        )
    )


def flags_ir() -> EnsembleIR:
    return EnsembleIR(
        trees=(
            _stump(0, 0.5, 0.0, 1.0),
            _stump(1, 0.5, 0.0, 0.5),
            _stump(2, 0.5, 0.0, 0.25),
        ),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=3,
        feature_names=("f1", "f2", "f3"),
        meta={},
    )


def target_from_percentile(ir: EnsembleIR, rng: np.random.Generator, pct: float) -> float:
    scores = [raw_score(ir, rng.normal(scale=3.0, size=ir.n_features)) for _ in range(80)]
    return float(np.percentile(scores, pct))


def scenario(
    name: str,
    ir: EnsembleIR,
    x: np.ndarray,
    descriptors: list[dict[str, Any]],
    interval: tuple[float, float],
    sigma: np.ndarray | None = None,
    background: np.ndarray | None = None,
    if_ir: EnsembleIR | None = None,
    min_total_path: float | None = None,
    lam: float = 0.05,
) -> dict[str, Any]:
    p = ir.n_features
    sigma = np.ones(p) if sigma is None else sigma
    constraints = harness.build_constraints(descriptors)
    compiled = compile_constraints(constraints, ir.feature_names)
    data: dict[str, Any] = {
        "name": name,
        "ensemble": _encode_ensemble(flatten_ir(ir)),
        "if_ensemble": _encode_ensemble(flatten_ir(if_ir)) if if_ir else None,
        "min_total_path": min_total_path,
        "x": harness.encode_floats(x.astype(np.float64)),
        "sigma": harness.encode_floats(sigma.astype(np.float64)),
        "weights": harness.encode_floats(np.ones(p)),
        "lam": lam,
        "interval": harness.encode_floats(list(interval)),
        "constraints": descriptors,
        "constraints_flat": {
            k: (v if isinstance(v, int) else harness.encode_floats(np.asarray(v)))
            for k, v in flatten_constraints(compiled).items()
        },
        "background": harness.encode_floats(background) if background is not None else None,
        "ga": GA,
    }
    # records via the harness's own loader path (round-trip: what tests will see)
    tmp = harness.FIXTURES_DIR / f"_tmp_{name}.json"
    data["golden"] = []
    data["dist"] = {"feasible": [], "j": [], "n_changed": [], "generations": []}
    data["dist_seeds"] = DIST_SEEDS
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    loaded = harness.load_scenario(tmp)
    tmp.unlink()

    golden = [harness.run_python(loaded, seed) for seed in GOLDEN_SEEDS]
    dist: dict[str, list[Any]] = {"feasible": [], "j": [], "n_changed": [], "generations": []}
    for seed in DIST_SEEDS:
        record = harness.run_python(loaded, seed)
        dist["feasible"].append(record["feasible"])
        dist["j"].append(record["j"])
        dist["n_changed"].append(record["n_changed"])
        dist["generations"].append(record["generations"])
    data["golden"] = golden
    data["dist"] = dist
    feas_rate = float(np.mean(dist["feasible"]))
    print(f"  {name}: feasibility {feas_rate:.2%}, "
          f"median gens {int(np.median(dist['generations']))}")
    return data


def _encode_ensemble(flat: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in flat.items():
        out[k] = harness.encode_floats(v) if isinstance(v, np.ndarray) else v
    return out


def main() -> None:
    rng = np.random.default_rng(2026)
    out_dir = harness.FIXTURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios: list[dict[str, Any]] = []

    ir1 = make_random_ir(rng, n_features=3, n_trees=4, depth=3)
    bg1 = rng.normal(scale=2.0, size=(50, 3))
    scenarios.append(scenario(
        "01-basic-lt-le", ir1, rng.normal(scale=2.0, size=3), [],
        (target_from_percentile(ir1, rng, 65), float("inf")), background=bg1,
    ))

    ir2 = make_random_ir(rng, n_features=4, n_trees=5, depth=3)
    x2 = rng.normal(scale=2.0, size=4)
    scenarios.append(scenario(
        "02-bounds-mix", ir2, x2,
        [
            {"type": "Freeze", "feature": "x0"},
            {"type": "Monotone", "feature": "x1", "direction": "increase"},
            {"type": "Range", "feature": "x2", "lo": float(x2[2] - 4), "hi": float(x2[2] + 4)},
        ],
        (target_from_percentile(ir2, rng, 60), float("inf")),
    ))

    ir3 = make_random_ir(rng, n_features=3, n_trees=4, depth=3)
    x3 = rng.normal(scale=2.0, size=3)
    x3[1] = np.nan
    scenarios.append(scenario(
        "03-nan-fixed-factual", ir3, x3, [],
        (target_from_percentile(ir3, rng, 55), float("inf")),
    ))

    ir4 = make_random_ir(rng, n_features=3, n_trees=4, depth=3)
    scenarios.append(scenario(
        "04-allow-missing-to-nan", ir4, rng.normal(scale=2.0, size=3),
        [{"type": "AllowMissing", "feature": "x0", "delta_miss": 0.2}],
        (target_from_percentile(ir4, rng, 60), float("inf")),
    ))

    ir5 = make_random_ir(rng, n_features=3, n_trees=4, depth=3)
    x5 = rng.normal(scale=2.0, size=3)
    x5[0] = np.nan
    scenarios.append(scenario(
        "05-allow-missing-from-nan", ir5, x5,
        [{"type": "AllowMissing", "feature": "x0", "delta_miss": 0.3,
          "delta_from_miss": 1.5}],
        (target_from_percentile(ir5, rng, 60), float("inf")),
    ))

    ir6 = make_random_ir(rng, n_features=4, n_trees=5, depth=3)
    scenarios.append(scenario(
        "06-linear-policies", ir6, rng.normal(scale=2.0, size=4),
        [
            {"type": "Linear", "coefficients": {"x0": 1.0, "x1": -1.0}, "op": "<=",
             "rhs": 0.0},
            {"type": "Linear", "coefficients": {"x2": 1.0, "x3": 1.0}, "op": "<=",
             "rhs": 6.0, "missing_policy": "forbid_missing"},
            {"type": "AllowMissing", "feature": "x2", "delta_miss": 0.4},
        ],
        (target_from_percentile(ir6, rng, 60), float("inf")),
    ))

    ir7 = flags_ir()
    scenarios.append(scenario(
        "07-flags-onehot-implies", ir7, np.array([0.0, 0.0, 1.0]),
        [
            {"type": "OneHot", "features": ["f1", "f2", "f3"]},
        ],
        (0.9, float("inf")),
    ))

    ir8 = make_random_ir(rng, n_features=4, n_trees=6, depth=3)
    bg8 = rng.normal(scale=2.0, size=(400, 4))
    from sklearn.ensemble import IsolationForest

    from treecf.plausibility import Plausibility
    iso = IsolationForest(n_estimators=25, random_state=0).fit(bg8)
    plaus = Plausibility.isolation_forest(iso, max_anomaly_score=0.65)
    scenarios.append(scenario(
        "08-plausibility", ir8, bg8[0], [],
        (target_from_percentile(ir8, rng, 55), float("inf")),
        background=bg8[:50], if_ir=plaus.if_ir, min_total_path=plaus.min_total_path,
    ))

    ir9 = make_random_ir(rng, n_features=3, n_trees=4, depth=3)
    scenarios.append(scenario(
        "09-infeasible", ir9, rng.normal(scale=2.0, size=3),
        [{"type": "Freeze", "feature": n} for n in ir9.feature_names],
        (1e6, float("inf")),
    ))

    import xgboost as xgb

    from treecf.ir.parsers import parse_model
    Xr = rng.normal(size=(3000, 6)) * np.array([0.1, 1, 10, 100, 1, 5])
    yr = (Xr @ rng.normal(size=6) + rng.logistic(scale=2.0, size=3000) > 0).astype(float)
    clf = xgb.XGBClassifier(n_estimators=12, max_depth=3, random_state=0)
    clf.fit(Xr, yr)
    ir10 = parse_model(clf)
    proba_scores = [raw_score(ir10, Xr[i]) for i in range(300)]
    scenarios.append(scenario(
        "10-real-xgboost", ir10, Xr[int(np.argmax(proba_scores))],
        [{"type": "Monotone", "feature": ir10.feature_names[0], "direction": "decrease"}],
        (float("-inf"), float(np.percentile(proba_scores, 40))),
        sigma=np.maximum(np.std(Xr, axis=0), 1e-6), background=Xr[:50],
    ))

    for data in scenarios:
        path = out_dir / f"{data['name']}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"))
        print(f"wrote {path} ({path.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
