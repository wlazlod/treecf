"""Public-API smoke for backend='exact', rust-first dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from treecf import Counterfactual, Explainer, Freeze, Infeasible, Target
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

pytestmark = pytest.mark.rust


def _stump() -> EnsembleIR:
    """Splits on ``a`` only; ``b`` never changes the score."""
    nodes = (
        Node(0, 0, 1.0, SplitOp.LT, True, 1, 2, None),
        Node(1, None, None, None, None, None, None, -1.0),
        Node(2, None, None, None, None, None, None, 1.0),
    )
    return EnsembleIR(
        trees=(Tree(nodes=nodes),),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=2,
        feature_names=("a", "b"),
        meta={},
    )


def test_rust_path_is_taken_even_if_python_solve_exact_would_raise(monkeypatch) -> None:
    """Canary: with the extension present, the python reference is never
    called at all, not tried-then-abandoned. If dispatch ever regressed to
    a rust-then-python fallback (or python-first), this raises."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("python solve_exact must not run when rust is available")

    monkeypatch.setattr("treecf.backends.exact.solve_exact", _boom)

    exp = Explainer(_stump(), normalizers=np.ones(2))
    res = exp.explain(
        np.array([0.0, 0.0]),
        target=Target.raw(op=">=", value=0.5),
        backend="exact",
        warm_start=False,
    )
    assert isinstance(res, Counterfactual)
    assert res.proof == "optimal"


def test_warm_start_feeds_the_exact_search() -> None:
    exp = Explainer(_stump(), normalizers=np.ones(2))
    res = exp.explain(
        np.array([0.0, 0.0]),
        target=Target.raw(op=">=", value=0.5),
        backend="exact",
        warm_start=True,
        seed=1,
    )
    assert isinstance(res, Counterfactual)
    assert res.solver_stats["warm_start_used"] is True
    assert res.proof in ("optimal", "optimal_within_gap")


def test_certified_infeasible_end_to_end() -> None:
    exp = Explainer(
        _stump(), normalizers=np.ones(2), constraints=[Freeze("a"), Freeze("b")]
    )
    res = exp.explain(
        np.array([0.0, 0.0]),
        target=Target.raw(op=">=", value=0.5),
        backend="exact",
        warm_start=False,
    )
    assert isinstance(res, Infeasible)
    assert res.proof == "certified"
    assert res.solver_stats["completed"] is True


def test_rust_cache_is_reused_across_calls() -> None:
    exp = Explainer(_stump(), normalizers=np.ones(2))
    x = np.array([0.0, 0.0])
    target = Target.raw(op=">=", value=0.5)

    exp.explain(x, target=target, backend="exact", warm_start=False)
    assert "ensemble" in exp._rust_cache and "constraints" in exp._rust_cache
    ensemble_before = exp._rust_cache["ensemble"]
    constraints_before = exp._rust_cache["constraints"]

    exp.explain(x, target=target, backend="exact", warm_start=False)
    assert exp._rust_cache["ensemble"] is ensemble_before
    assert exp._rust_cache["constraints"] is constraints_before


def test_repeat_calls_are_bitwise_deterministic() -> None:
    exp = Explainer(_stump(), normalizers=np.ones(2))
    x = np.array([0.0, 0.0])
    target = Target.raw(op=">=", value=0.5)

    results = [
        exp.explain(x, target=target, backend="exact", warm_start=False) for _ in range(5)
    ]
    assert all(isinstance(r, Counterfactual) for r in results)
    first = results[0]
    for other in results[1:]:
        np.testing.assert_array_equal(other.x_cf, first.x_cf)
        assert other.distance == first.distance
        assert other.proof == first.proof
        assert other.solver_stats == first.solver_stats


def test_explain_coalitions_reuses_the_parent_marshaled_ensemble(monkeypatch) -> None:
    """`_with_extra_freezes` shares the ensemble cache entry across coalition
    clones (only ``constraints`` differs per coalition); this proves the
    exact-rust path actually takes that shortcut rather than remarshaling."""
    import treecf.backends.exact_rust as exact_rust_mod

    real_build = exact_rust_mod.build_rust_ensemble
    calls: list[int] = []

    def _counting_build(ir: object) -> object:
        calls.append(1)
        return real_build(ir)

    monkeypatch.setattr(exact_rust_mod, "build_rust_ensemble", _counting_build)

    exp = Explainer(_stump(), normalizers=np.ones(2))
    x = np.array([0.0, 0.0])
    target = Target.raw(op=">=", value=0.5)

    exp.explain(x, target=target, backend="exact", warm_start=False)
    assert len(calls) == 1

    results = exp.explain_coalitions(
        x,
        target=target,
        coalitions={"a-only": ["a"], "b-only": ["b"]},
        backend="exact",
        warm_start=False,
    )
    assert len(calls) == 1  # both coalition clones reused the parent's marshaled ensemble
    assert isinstance(results["a-only"], Counterfactual)
    assert isinstance(results["b-only"], Infeasible)  # b alone never moves the score
