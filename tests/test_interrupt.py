"""Ctrl-C during a Rust-held search: `py.rs` polls `check_signals` from inside
the released GIL and raises `KeyboardInterrupt` promptly, with no result
returned.

Every fixture here is deliberately oversized: its uninterrupted baseline was
measured once during development (noted in each test's docstring) to run well
past the 1-second interrupt timer below, so a broken probe would fail the
`elapsed < 5.0` assertion loudly instead of the test passing by accident
because the search happened to finish on its own first.
"""

from __future__ import annotations

import _thread
import sys
import threading
import time

import numpy as np
import pytest

from treecf import Explainer, Target
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

pytestmark = [
    pytest.mark.rust,
    pytest.mark.skipif(
        sys.platform == "win32", reason="_thread.interrupt_main signal delivery is unix-only here"
    ),
]

_treecf_core = pytest.importorskip("treecf._treecf_core")


def _leaf(i: int, v: float) -> Node:
    return Node(i, None, None, None, None, None, None, v)


def _stump(feature: int, threshold: float, right_value: float) -> Tree:
    return Tree(
        nodes=(
            Node(0, feature, threshold, SplitOp.LT, True, 1, 2, None),
            _leaf(1, 0.0),
            _leaf(2, right_value),
        )
    )


def _wide_stump_ensemble(n_features: int) -> EnsembleIR:
    """``n_features`` independent 0/1 levers. With the target on the
    unreachable half-integer between two adjacent achievable integer scores,
    no branch prunes on score before a leaf, so the search is exhaustive."""
    trees = tuple(_stump(j, 1.0, 1.0) for j in range(n_features))
    return EnsembleIR(
        trees=trees,
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=n_features,
        feature_names=tuple(f"x{j}" for j in range(n_features)),
        meta={},
    )


def _dense_random_ensemble(
    rng: np.random.Generator, n_features: int, n_trees: int, depth: int, pool_size: int
) -> EnsembleIR:
    """Many features, deep trees, a large shared threshold pool: each feature
    ends up with many distinct joint-grid cells (so region growth needs many
    rounds to close), and every oracle/fitness call walks a large ensemble."""
    threshold_pool = np.round(rng.normal(scale=2.0, size=pool_size), 4)

    def build(nodes: list[Node | None], d: int) -> int:
        idx = len(nodes)
        nodes.append(None)
        if d == 0 or rng.random() < 0.1:
            nodes[idx] = Node(idx, None, None, None, None, None, None, float(rng.normal()))
            return idx
        feature = int(rng.integers(0, n_features))
        threshold = float(rng.choice(threshold_pool))
        op = SplitOp.LT if rng.random() < 0.5 else SplitOp.LE
        missing_left = bool(rng.random() < 0.5)
        left = build(nodes, d - 1)
        right = build(nodes, d - 1)
        nodes[idx] = Node(idx, feature, threshold, op, missing_left, left, right, None)
        return idx

    trees = []
    for _ in range(n_trees):
        nodes: list[Node | None] = []
        build(nodes, depth)
        trees.append(Tree(nodes=tuple(n for n in nodes if n is not None)))
    return EnsembleIR(
        trees=tuple(trees),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=n_features,
        feature_names=tuple(f"x{j}" for j in range(n_features)),
        meta={},
    )


def test_exact_search_raises_keyboard_interrupt_promptly() -> None:
    """28 independent levers, target on the unreachable half-integer 14.5:
    every one of the 2**28 leaf assignments is expanded and only pruned on
    score at the leaf, so ``warm_start=False`` + ``node_budget=1e9`` never
    lets the search finish early. Measured uninterrupted: ~23 s (>> 5 s)."""
    n = 28
    ir = _wide_stump_ensemble(n)
    exp = Explainer(ir, normalizers=np.ones(n))
    x0 = np.zeros(n)
    mid = n / 2 + 0.5
    target = Target.raw(range=(mid - 0.01, mid + 0.01))

    timer = threading.Timer(1.0, _thread.interrupt_main)
    timer.start()
    t0 = time.perf_counter()
    try:
        with pytest.raises(KeyboardInterrupt):
            exp.explain(
                x0, target, backend="exact", warm_start=False,
                node_budget=10**9, time_budget_s=30.0,
            )
        assert time.perf_counter() - t0 < 5.0
    finally:
        timer.cancel()


def test_region_growth_raises_keyboard_interrupt_promptly() -> None:
    """120 features, 1900 trees, depth 10, a 10 000-value shared threshold
    pool: the region-growth oracle walks every tree on every attempted
    expansion, and this fixture needs well over the 64-attempt poll interval
    to close even one round. Measured uninterrupted: ~14 s (>> 10 s, and far
    more than 64 attempts)."""
    n_features, n_trees, depth, pool = 120, 1900, 10, 10_000
    rng = np.random.default_rng(3)
    ir = _dense_random_ensemble(rng, n_features, n_trees, depth, pool)
    exp = Explainer(ir, normalizers=np.ones(n_features))
    x0 = np.zeros(n_features)
    s0 = raw_score(ir, x0)
    target = Target.raw(range=(s0 - 0.5, s0 + 0.5))  # x0 itself already verifies

    timer = threading.Timer(1.0, _thread.interrupt_main)
    timer.start()
    t0 = time.perf_counter()
    try:
        with pytest.raises(KeyboardInterrupt):
            exp.recourse_region(x0, x0, target)
        assert time.perf_counter() - t0 < 5.0
    finally:
        timer.cancel()


def test_batch_solve_raises_keyboard_interrupt_promptly() -> None:
    """700 tasks (three 256-task chunks) against an unreachable target, so
    every task runs the full generation budget instead of stalling out early.
    The batch probe is only checked between chunks, so interruption during
    chunk 0 is only caught at the start of chunk 1 -- chunk 0 alone measured
    ~1.8 s uninterrupted, comfortably outliving the 1 s timer below while
    staying well under the 5 s bound."""
    n_features, n_trees, depth, pool = 15, 150, 5, 1000
    rng = np.random.default_rng(3)
    ir = _dense_random_ensemble(rng, n_features, n_trees, depth, pool)
    exp = Explainer(ir, normalizers=np.ones(n_features))
    X = np.zeros((1, n_features))
    target = Target.raw(range=(1e6, 1e6 + 1.0))  # unreachable: no task ever converges early
    interval = target.raw_interval(ir.link)
    tasks = [(0, seed) for seed in range(700)]

    timer = threading.Timer(1.0, _thread.interrupt_main)
    timer.start()
    t0 = time.perf_counter()
    try:
        with pytest.raises(KeyboardInterrupt):
            exp._solve_batch(X, tasks, interval, time_budget_s=10.0, sparsity_weight=0.0)
        assert time.perf_counter() - t0 < 5.0
    finally:
        timer.cancel()
