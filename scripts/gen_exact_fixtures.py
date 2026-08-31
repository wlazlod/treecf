"""Generate golden exact-backend fixtures.

Run with: uv run python scripts/gen_exact_fixtures.py

Regenerating overwrites tests/fixtures/exact/*.json — do this ONLY when the
Python exact backend's behavior changes deliberately; the fixtures freeze it
otherwise (the Rust port has to reproduce them bit-for-bit). Every
input here is either a fixed-seed draw or a hand-built ensemble, so two runs
of this script always produce byte-identical files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tests.conftest import make_random_ir, make_random_mixed_ir
from tests.exactness import fixture_utils
from tests.parity.harness import build_constraints
from treecf.api import Grid
from treecf.backends.exact import solve_exact
from treecf.backends.genetic import solve_genetic
from treecf.constraints.compile import compile_constraints
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree


def _leaf(i: int, v: float) -> Node:
    return Node(i, None, None, None, None, None, None, v)


def _stump(
    feature: int, threshold: float, left_v: float, right_v: float, missing_left: bool = True
) -> Tree:
    return Tree(
        nodes=(
            Node(0, feature, threshold, SplitOp.LT, missing_left, 1, 2, None),
            _leaf(1, left_v),
            _leaf(2, right_v),
        )
    )


def _write(payload: dict[str, Any]) -> None:
    result = fixture_utils.solve_payload(payload)
    payload["golden"] = fixture_utils.golden_block(result)
    out = fixture_utils.FIXTURES_DIR / f"{payload['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, separators=(",", ":"))
    print(
        f"  {payload['name']}: x_cf={result.x_cf}, distance={result.distance}, "
        f"proof={result.proof!r}, completed={result.stats['completed']}, "
        f"nodes_expanded={result.stats['nodes_expanded']}"
    )


def _scenario_01_basic() -> dict[str, Any]:
    """Unconstrained search over a plain LT/LE-mixed random ensemble."""
    rng = np.random.default_rng(2026)
    ir = make_random_ir(rng, n_features=3, n_trees=4, depth=3)
    x = rng.normal(scale=2.0, size=3)
    lo = raw_score(ir, x) + 0.15  # just above the factual: forces real search, stays reachable
    return fixture_utils.build_fixture_payload(
        "01-basic-lt-le", ir, x, (lo, float("inf")), [], lam=0.05,
    )


def _scenario_02_nan_both_directions() -> dict[str, Any]:
    """One feature moves value->NaN (delta_miss), another NaN->value (delta_from_miss)."""
    # x0: NaN routes right (like a value >= 0.0); factual -5.0 routes left (low leaf).
    tree0 = _stump(0, 0.0, 0.0, 2.0, missing_left=False)
    # x1: NaN routes left (low leaf); a value >= 5.0 routes right (high leaf).
    tree1 = _stump(1, 5.0, 0.0, 2.0, missing_left=True)
    ir = EnsembleIR(
        trees=(tree0, tree1), base_score=0.0, link=Link.IDENTITY,
        n_features=2, feature_names=("x0", "x1"), meta={},
    )
    x = np.array([-5.0, np.nan])
    constraints = [
        {"type": "AllowMissing", "feature": "x0", "delta_miss": 0.1},
        {"type": "AllowMissing", "feature": "x1", "delta_miss": 0.5, "delta_from_miss": 0.2},
    ]
    return fixture_utils.build_fixture_payload(
        "02-nan-both-directions", ir, x, (3.5, float("inf")), constraints, lam=0.05,
    )


def _order_pair_ir() -> EnsembleIR:
    tree0 = Tree(
        nodes=(
            Node(0, 0, 1.0, SplitOp.LT, True, 1, 2, None),
            _leaf(1, 0.0),
            Node(2, 0, 3.0, SplitOp.LT, True, 3, 4, None),
            _leaf(3, 1.0),
            _leaf(4, 5.0),
        )
    )
    tree1 = _stump(1, 1.0, 0.0, 0.1)
    return EnsembleIR(
        trees=(tree0, tree1), base_score=0.0, link=Link.IDENTITY,
        n_features=2, feature_names=("x0", "x1"), meta={},
    )


def _scenario_03_order_pair_boundary() -> dict[str, Any]:
    """The cheapest per-cell candidates for x0 and x1 land the wrong way round
    (x0=3.0 > x1=0.0/1.0), so the winning row only exists after a boundary
    repair moves both onto their cells' shared overlap (x0 == x1 == 3.0) —
    off either feature's own per-cell nearest-to-factual point."""
    ir = _order_pair_ir()
    x = np.array([0.0, 0.0])
    constraints = [{"type": "Linear", "coefficients": {"x0": 1.0, "x1": -1.0}, "op": "<=",
                     "rhs": 0.0}]
    return fixture_utils.build_fixture_payload(
        "03-order-pair-boundary", ir, x, (5.0, float("inf")), constraints,
    )


def _scenario_04_onehot_implies() -> dict[str, Any]:
    """Reaching the target forces the one-hot group to flip onto f3, which in
    turn forces g to the implication's exact demanded value (not merely into
    g's routing cell)."""
    tree_f1 = _stump(0, 0.5, 0.0, 0.1)
    tree_f2 = _stump(1, 0.5, 0.0, 0.2)
    tree_f3 = _stump(2, 0.5, 0.0, 5.0)
    tree_g = _stump(3, 0.5, 0.0, 2.0)
    ir = EnsembleIR(
        trees=(tree_f1, tree_f2, tree_f3, tree_g), base_score=0.0, link=Link.IDENTITY,
        n_features=4, feature_names=("f1", "f2", "f3", "g"), meta={},
    )
    x = np.array([1.0, 0.0, 0.0, 0.0])
    constraints = [
        {"type": "OneHot", "features": ["f1", "f2", "f3"]},
        {"type": "Implies", "cond_feature": "f3", "cond_value": 1.0,
         "cons_feature": "g", "cons_value": 1.0},
    ]
    return fixture_utils.build_fixture_payload(
        "04-onehot-implies", ir, x, (3.0, float("inf")), constraints, lam=0.05,
    )


def _scenario_05_pinned_features() -> dict[str, Any]:
    """Freeze pins one feature to its factual value; Equals pins another to a
    binary value off the factual — both hold throughout the search."""
    rng = np.random.default_rng(555)
    ir = make_random_ir(rng, n_features=4, n_trees=5, depth=3)
    x = rng.normal(scale=2.0, size=4)
    x[1] = 0.0  # binary-valued factual so Equals can pin it off-factual
    scores = [raw_score(ir, rng.normal(scale=3.0, size=4)) for _ in range(80)]
    lo = float(np.percentile(scores, 55))
    constraints = [
        {"type": "Freeze", "feature": "x0"},
        {"type": "Equals", "feature": "x1", "value": 1.0},
    ]
    return fixture_utils.build_fixture_payload(
        "05-pinned-features", ir, x, (lo, float("inf")), constraints, lam=0.05,
    )


def _plausibility_ir() -> tuple[EnsembleIR, EnsembleIR, float]:
    tree0 = Tree(
        nodes=(
            Node(0, 0, 1.0, SplitOp.LT, True, 1, 2, None),
            _leaf(1, 0.0),
            Node(2, 0, 3.0, SplitOp.LT, True, 3, 4, None),
            _leaf(3, 1.0),
            _leaf(4, 5.0),
        )
    )
    tree1 = _stump(1, 1.0, 0.0, 0.5)
    tree2 = _stump(2, 5.0, 0.0, 5.0)
    ir = EnsembleIR(
        trees=(tree0, tree1, tree2), base_score=0.0, link=Link.IDENTITY,
        n_features=3, feature_names=("x0", "x1", "x2"), meta={},
    )
    # Total path length: short (implausible) for x0 >= 3.0, long (plausible) below it;
    # indifferent to x1/x2.
    if_tree = _stump(0, 3.0, 10.0, 0.0)
    if_ir = EnsembleIR(
        trees=(if_tree,), base_score=0.0, link=Link.IDENTITY,
        n_features=3, feature_names=("x0", "x1", "x2"), meta={},
    )
    return ir, if_ir, 5.0


def _scenario_06_plausibility_pruning() -> dict[str, Any]:
    """Without the plausibility bound the cheapest route moves x0 to 3.0
    (distance 3.0). The isolation-forest surrogate makes that route
    implausible (its path length collapses right where x0 crosses 3.0), so
    the search is pruned onto the pricier, still-plausible x2 route
    (distance 5.0) — the plausibility bound changes the answer, not just the
    proof."""
    ir, if_ir, min_total_path = _plausibility_ir()
    x = np.array([0.0, 0.0, 0.0])
    return fixture_utils.build_fixture_payload(
        "06-plausibility-pruning", ir, x, (5.0, float("inf")), [],
        if_ir=if_ir, min_total_path=min_total_path,
    )


def _scenario_07_gap() -> dict[str, Any]:
    """gap=0.15 widens the pruning threshold enough that the search settles
    for a within-gap incumbent instead of exhausting the space for the exact
    optimum, so it reports ``optimal_within_gap``."""
    rng = np.random.default_rng(10)
    ir = make_random_ir(rng, n_features=4, n_trees=6, depth=3)
    x = rng.normal(scale=2.0, size=4)
    scores = [raw_score(ir, rng.normal(scale=3.0, size=4)) for _ in range(80)]
    lo = float(np.percentile(scores, 55))
    return fixture_utils.build_fixture_payload(
        "07-gap-within-tolerance", ir, x, (lo, float("inf")), [], lam=0.05, gap=0.15,
    )


def _warm_start_ir() -> EnsembleIR:
    tree0 = Tree(
        nodes=(
            Node(0, 0, 1.0, SplitOp.LT, True, 1, 2, None),
            _leaf(1, 0.0),
            Node(2, 0, 3.0, SplitOp.LT, True, 3, 4, None),
            _leaf(3, 1.0),
            _leaf(4, 5.0),
        )
    )
    tree1 = _stump(1, 1.0, 0.0, 0.5)
    tree2 = _stump(2, 5.0, 0.0, 5.0)
    return EnsembleIR(
        trees=(tree0, tree1, tree2), base_score=0.0, link=Link.IDENTITY,
        n_features=3, feature_names=("x0", "x1", "x2"), meta={},
    )


def _scenario_08_warm_start_on() -> dict[str, Any]:
    """A pinned incumbent plus ``node_budget=1``: the search cannot expand
    even one node on its own, so it hands the pinned row straight back
    (``proof="heuristic"``, ``completed=False``, ``warm_start_used=True``)."""
    ir = _warm_start_ir()
    x = np.array([0.0, 0.0, 0.0])
    incumbent_row = np.array([0.0, 0.0, 5.0])
    incumbent = (5.0, incumbent_row)
    return fixture_utils.build_fixture_payload(
        "08-warm-start-on", ir, x, (5.0, float("inf")), [],
        node_budget=1, incumbent=incumbent,
    )


def _scenario_09_warm_start_off() -> dict[str, Any]:
    """Same model, target and ``node_budget=1`` as scenario 8, minus the
    incumbent: with nothing to fall back on, one node buys no complete
    assignment at all (``x_cf=None``, ``completed=False``)."""
    ir = _warm_start_ir()
    x = np.array([0.0, 0.0, 0.0])
    return fixture_utils.build_fixture_payload(
        "09-warm-start-off", ir, x, (5.0, float("inf")), [], node_budget=1,
    )


def _scenario_10_certified_infeasible() -> dict[str, Any]:
    """The target sits far above the ensemble's reachable maximum (2.4): the
    search proves infeasibility after a handful of score-pruned nodes."""
    tree_a = _stump(0, 1.0, 0.0, 1.0)
    tree_b = _stump(1, 1.0, 0.0, 0.8)
    tree_c = _stump(2, 1.0, 0.0, 0.6)
    ir = EnsembleIR(
        trees=(tree_a, tree_b, tree_c), base_score=0.0, link=Link.IDENTITY,
        n_features=3, feature_names=("a", "b", "c"), meta={},
    )
    x = np.zeros(3)
    return fixture_utils.build_fixture_payload(
        "10-certified-infeasible", ir, x, (100.0, float("inf")), [],
    )


def _scenario_11_value_policies() -> dict[str, Any]:
    """Neither feature alone reaches the target, so both must move; each
    snaps onto its own policy grid (nearest integer, and a 50-wide grid)."""
    tree_amount = _stump(0, 2.5, 0.0, 2.0)
    tree_step = _stump(1, 1000.3, 0.0, 2.0)
    ir = EnsembleIR(
        trees=(tree_amount, tree_step), base_score=0.0, link=Link.IDENTITY,
        n_features=2, feature_names=("amount", "step"), meta={},
    )
    x = np.zeros(2)
    value_policies = {"amount": "integer", "step": Grid(step=50.0, anchor=0.0)}
    return fixture_utils.build_fixture_payload(
        "11-value-policies", ir, x, (3.5, float("inf")), [], value_policies=value_policies,
    )


def _scenario_12_categorical_blocks() -> dict[str, Any]:
    """Mixed numeric + categorical ensemble: block domains drive the search."""
    rng = np.random.default_rng(3026)
    ir = make_random_mixed_ir(rng, n_features=4, n_trees=5, depth=3, categorical={1: 5, 3: 8})
    x = np.array([0.5, 2.0, -0.3, 6.0])
    lo = raw_score(ir, x) + 0.15
    return fixture_utils.build_fixture_payload(
        "12-categorical-blocks", ir, x, (lo, float("inf")), [], lam=0.05,
    )


def _scenario_13_categorical_allowed() -> dict[str, Any]:
    """AllowedCategories narrows the block domains; NaN factual on a numeric feature."""
    rng = np.random.default_rng(3027)
    ir = make_random_mixed_ir(rng, n_features=4, n_trees=5, depth=3, categorical={1: 6})
    x = np.array([0.2, 4.0, np.nan, 1.1])
    lo = raw_score(ir, np.nan_to_num(x, nan=0.0)) + 0.1
    constraints = [
        {"type": "AllowedCategories", "feature": "x1", "allowed": [0, 2, 4]},
        {"type": "AllowMissing", "feature": "x2", "delta_miss": 0.3},
    ]
    return fixture_utils.build_fixture_payload(
        "13-categorical-allowed", ir, x, (lo, float("inf")), constraints, lam=0.05,
    )


def _scenario_14_categorical_frozen_nan() -> dict[str, Any]:
    """A frozen categorical, an AllowMissing categorical, and a plain one."""
    rng = np.random.default_rng(3028)
    ir = make_random_mixed_ir(
        rng, n_features=4, n_trees=4, depth=3, categorical={0: 4, 1: 5, 3: 3}
    )
    x = np.array([1.0, 3.0, 0.4, 2.0])
    lo = raw_score(ir, x) + 0.1
    constraints = [
        {"type": "Freeze", "feature": "x0"},
        {"type": "AllowMissing", "feature": "x1", "delta_miss": 0.4},
    ]
    return fixture_utils.build_fixture_payload(
        "14-categorical-frozen-nan", ir, x, (lo, float("inf")), constraints, lam=0.05,
    )


# One builder per fixture, in file order. Exposed as a module constant (not just
# a literal inside main()) so tests can call each builder twice and diff the
# payload dicts directly -- the double-generation determinism check without
# round-tripping through the filesystem.
SCENARIO_BUILDERS = (
    _scenario_01_basic,
    _scenario_02_nan_both_directions,
    _scenario_03_order_pair_boundary,
    _scenario_04_onehot_implies,
    _scenario_05_pinned_features,
    _scenario_06_plausibility_pruning,
    _scenario_07_gap,
    _scenario_08_warm_start_on,
    _scenario_09_warm_start_off,
    _scenario_10_certified_infeasible,
    _scenario_11_value_policies,
    _scenario_12_categorical_blocks,
    _scenario_13_categorical_allowed,
    _scenario_14_categorical_frozen_nan,
)


# --------------------------------------------------------------------------
# Region fixtures: a verified x_cf widened into a certified box.
# --------------------------------------------------------------------------


def _write_region(payload: dict[str, Any]) -> None:
    lo, hi = fixture_utils.solve_region_payload(payload)
    payload["golden"] = fixture_utils.region_golden_block(lo, hi)
    out = fixture_utils.REGION_FIXTURES_DIR / f"{payload['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, separators=(",", ":"))
    print(f"  {payload['name']}: lo={lo}, hi={hi}")


def _scenario_region_01_genetic_widened() -> dict[str, Any]:
    """A genetic-found x_cf (fixed seed, deterministic) widened into a region."""
    rng = np.random.default_rng(4242)
    ir = make_random_ir(rng, n_features=3, n_trees=4, depth=3)
    x = rng.normal(scale=2.0, size=3)
    interval = (raw_score(ir, x) + 0.2, float("inf"))
    compiled = compile_constraints([], ir.feature_names)
    result = solve_genetic(ir, x, interval, compiled, np.ones(3), np.ones(3), 0.0, seed=7)
    assert result.x_cf is not None
    return fixture_utils.build_region_fixture_payload(
        "region-01-genetic-widened", ir, x, result.x_cf, interval, [],
    )


def _scenario_region_02_exact_found() -> dict[str, Any]:
    """An exact-found x_cf on a small deterministic ensemble, widened into a region."""
    ir = _warm_start_ir()
    x = np.array([0.0, 0.0, 0.0])
    interval = (3.0, float("inf"))
    compiled = compile_constraints([], ir.feature_names)
    result = solve_exact(ir, x, interval, compiled, np.ones(3), np.ones(3), 0.0)
    assert result.x_cf is not None
    return fixture_utils.build_region_fixture_payload(
        "region-02-exact-found", ir, x, result.x_cf, interval, [],
    )


def _scenario_region_03_plausibility() -> dict[str, Any]:
    """A plausibility-constrained exact-found x_cf, widened under the
    isolation-forest bound -- exercises the `if_ir` bracket in the oracle."""
    ir, if_ir, min_total_path = _plausibility_ir()
    x = np.array([0.0, 0.0, 0.0])
    interval = (5.0, float("inf"))
    compiled = compile_constraints([], ir.feature_names)
    result = solve_exact(
        ir, x, interval, compiled, np.ones(3), np.ones(3), 0.0,
        plausibility=(if_ir, min_total_path),
    )
    assert result.x_cf is not None
    return fixture_utils.build_region_fixture_payload(
        "region-03-plausibility", ir, x, result.x_cf, interval, [],
        if_ir=if_ir, min_total_path=min_total_path,
    )


def _scenario_region_04_order_pair() -> dict[str, Any]:
    """An order-pair-constrained exact-found x_cf; the canonical `a - b <= 0`
    shape is never pinned by `_degenerate_features`, so growth must hold it
    at its worst corner rather than only at the factual point."""
    ir = _order_pair_ir()
    x = np.array([0.0, 0.0])
    constraints = [
        {"type": "Linear", "coefficients": {"x0": 1.0, "x1": -1.0}, "op": "<=", "rhs": 0.0}
    ]
    interval = (5.0, float("inf"))
    compiled = compile_constraints(build_constraints(constraints), ir.feature_names)
    result = solve_exact(ir, x, interval, compiled, np.ones(2), np.ones(2), 0.0)
    assert result.x_cf is not None
    return fixture_utils.build_region_fixture_payload(
        "region-04-order-pair", ir, x, result.x_cf, interval, constraints,
    )


REGION_SCENARIO_BUILDERS = (
    _scenario_region_01_genetic_widened,
    _scenario_region_02_exact_found,
    _scenario_region_03_plausibility,
    _scenario_region_04_order_pair,
)


def main() -> None:
    fixture_utils.FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for build in SCENARIO_BUILDERS:
        _write(build())
    fixture_utils.REGION_FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for build in REGION_SCENARIO_BUILDERS:
        _write_region(build())


if __name__ == "__main__":
    main()
