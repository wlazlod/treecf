"""The certification trace the exact backend records alongside its counters."""

from __future__ import annotations

import math

import numpy as np

from treecf.backends._exact_trace import _Trace
from treecf.backends.exact import solve_exact
from treecf.constraints import Freeze, compile_constraints

from ..conftest import make_random_ir

STATS_KEYS = {
    "nodes_expanded",
    "nodes_pruned_score",
    "nodes_pruned_cost",
    "lower_bound",
    "gap",
    "completed",
    "warm_start_used",
    "presolve_removed",
    "presolve_certified",
    "search",
    "coarse_accepts",
    "refinements",
    "trace",
}


class TestTraceRecorder:
    def test_samples_are_plain_tuples_in_order(self) -> None:
        trace = _Trace()
        trace.record(1, None, 0.5, is_incumbent=False)
        trace.record(2, 3.0, 0.5, is_incumbent=True)
        assert trace.samples() == [(1, None, 0.5), (2, 3.0, 0.5)]

    def test_same_node_count_replaces_the_last_sample(self) -> None:
        trace = _Trace()
        trace.record(4, None, 0.5, is_incumbent=False)
        trace.record(4, 2.0, 0.5, is_incumbent=True)
        assert trace.samples() == [(4, 2.0, 0.5)]

    def test_cap_thins_non_incumbent_samples_and_keeps_incumbents(self) -> None:
        trace = _Trace()
        for n in range(1, _Trace.CAP + 1):
            trace.record(n, None if n % 50 else float(n), 0.0, is_incumbent=n % 50 == 0)
        assert len(trace.samples()) == _Trace.CAP
        trace.record(_Trace.CAP + 1, None, 0.0, is_incumbent=False)
        samples = trace.samples()
        assert len(samples) < _Trace.CAP
        incumbent_nodes = [n for n, cost, _ in samples if cost is not None]
        assert incumbent_nodes == [50, 100, 150, 200, 250]
        assert samples[-1] == (_Trace.CAP + 1, None, 0.0)


def _problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    ir = make_random_ir(rng, n_features=4, n_trees=5, depth=3)
    x = rng.normal(size=4)
    compiled = compile_constraints([], ir.feature_names)
    sigma = np.ones(4)
    weights = np.ones(4)
    return ir, x, compiled, sigma, weights


def test_classic_search_reports_the_full_stats_key_set() -> None:
    ir, x, compiled, sigma, weights = _problem()
    res = solve_exact(ir, x, (0.2, 0.9), compiled, sigma, weights, 0.0)
    assert set(res.stats) == STATS_KEYS
    assert res.stats["search"] == "classic"
    assert res.stats["coarse_accepts"] == 0
    assert res.stats["refinements"] == 0


def test_trace_ends_at_the_reported_counters() -> None:
    ir, x, compiled, sigma, weights = _problem()
    res = solve_exact(ir, x, (0.2, 0.9), compiled, sigma, weights, 0.0)
    trace = res.stats["trace"]
    assert isinstance(trace, list) and trace
    nodes, incumbent, bound = trace[-1]
    assert nodes == res.stats["nodes_expanded"]
    assert bound == res.stats["lower_bound"]
    assert incumbent == res.distance
    incumbents = [c for _, c, _ in trace if c is not None]
    assert incumbents == sorted(incumbents, reverse=True)
    counts = [n for n, _, _ in trace]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)
    assert all(math.isfinite(b) or b == math.inf for _, _, b in trace)


def test_trace_bound_is_non_decreasing_without_gap_or_order_pairs() -> None:
    for seed in range(6):
        ir, x, compiled, sigma, weights = _problem(seed)
        res = solve_exact(ir, x, (0.2, 0.9), compiled, sigma, weights, 0.0)
        bounds = [b for _, _, b in res.stats["trace"]]
        assert bounds == sorted(bounds), (seed, bounds)


def test_budget_limited_trace_bound_stays_below_incumbent() -> None:
    ir, x, compiled, sigma, weights = _problem(3)
    res = solve_exact(ir, x, (0.2, 0.9), compiled, sigma, weights, 0.0, node_budget=3)
    assert res.stats["completed"] is False
    for _, incumbent, bound in res.stats["trace"]:
        if incumbent is not None:
            assert bound <= incumbent


def test_early_exits_carry_a_single_sample() -> None:
    ir, x, compiled, sigma, weights = _problem()
    score = float(np.sum([0.0]))  # the factual is inside a wide-open target
    res = solve_exact(ir, x, (-math.inf, math.inf), compiled, sigma, weights, 0.0)
    assert res.distance == 0.0 and score == 0.0
    assert res.stats["trace"] == [(0, 0.0, 0.0)]

    frozen = compile_constraints([Freeze(name) for name in ir.feature_names], ir.feature_names)
    res = solve_exact(ir, x, (1e9, 1e10), frozen, sigma, weights, 0.0)
    assert res.x_cf is None and res.stats["completed"] is True
    assert res.stats["trace"] == [(0, None, math.inf)]
