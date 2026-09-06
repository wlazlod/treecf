"""Recourse menus over lever sets and lever-diverse plans."""

from __future__ import annotations

import itertools
import json
import math
import warnings
from collections.abc import Mapping

import numpy as np
import pytest

from treecf import (
    Counterfactual,
    DiverseSet,
    Explainer,
    Freeze,
    Infeasible,
    RecourseMenu,
    Target,
    TreecfError,
    TreecfWarning,
)
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

from .conftest import make_random_ir


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


def _ir() -> EnsembleIR:
    """Three independent levers worth 1.0 / 0.8 / 0.6 on features a/b/c."""
    return EnsembleIR(
        trees=(_stump(0, 1.0, 1.0), _stump(1, 1.0, 0.8), _stump(2, 1.0, 0.6)),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=3,
        feature_names=("a", "b", "c"),
        meta={},
    )


@pytest.fixture()
def exp() -> Explainer:
    return Explainer(_ir(), normalizers=np.ones(3))


x0 = np.zeros(3)
ONE_LEVER = Target.raw(op=">=", value=0.5)  # any single lever suffices
TWO_LEVERS = Target.raw(op=">=", value=1.5)  # a+b (1.8) or a+c (1.6); b+c (1.4) falls short


def _all_sets(levers: list[str], max_levers: int) -> list[tuple[str, ...]]:
    return [
        combo
        for size in range(1, max_levers + 1)
        for combo in itertools.combinations(sorted(levers), size)
    ]


def _brute_force(
    exp: Explainer, x: np.ndarray, target: Target, max_levers: int, **kwargs: object
) -> dict[tuple[str, ...], Counterfactual | Infeasible]:
    """One coalition solve per lever set, independent of the menu code."""
    levers = exp.recourse_menu(x, target, max_levers=1, **kwargs).levers  # type: ignore[arg-type]
    out = {}
    for combo in _all_sets(list(levers), max_levers):
        plans = exp.explain_coalitions(
            x, target, coalitions={"s": list(combo)}, **kwargs  # type: ignore[arg-type]
        )
        out[combo] = plans["s"]
    return out


class TestEnumeration:
    def test_all_mode_matches_one_coalition_solve_per_set(self, exp: Explainer) -> None:
        menu = exp.recourse_menu(
            x0, TWO_LEVERS, max_levers=2, mode="all", backend="exact", seed=0,
        )
        expected = _brute_force(exp, x0, TWO_LEVERS, 2, backend="exact", seed=0)
        assert set(menu.levers) == {"a", "b", "c"}
        for combo, result in expected.items():
            key = "+".join(combo)
            entry = menu[menu.implied.get(key, key)]
            assert type(entry) is type(result)
            if isinstance(result, Counterfactual):
                assert isinstance(entry, Counterfactual)
                assert entry.distance == result.distance
                assert sorted(entry.changes) == sorted(result.changes)
            else:
                assert isinstance(entry, Infeasible)
                assert entry.proof == result.proof

    def test_random_model_all_mode_matches_brute_force(self) -> None:
        rng = np.random.default_rng(3)
        ir = make_random_ir(rng, n_features=5, n_trees=4, depth=3)
        exp = Explainer(ir, normalizers=np.ones(5))
        x = rng.normal(size=5)
        target = Target.raw(op=">=", value=raw_score(ir, x) + 0.3)
        menu = exp.recourse_menu(
            x, target, max_levers=2, mode="all", backend="exact", seed=0, warm_start=False,
        )
        expected = _brute_force(
            exp, x, target, 2, backend="exact", seed=0, warm_start=False,
        )
        assert len(expected) == len(menu.levers) + math.comb(len(menu.levers), 2)
        for combo, result in expected.items():
            key = "+".join(combo)
            if key in menu.unresolved:
                continue
            entry = menu[menu.implied.get(key, key)]
            if isinstance(result, Counterfactual):
                assert isinstance(entry, Counterfactual)
                assert entry.distance == result.distance
            else:
                assert isinstance(entry, Infeasible)

    def test_minimal_mode_lists_the_frontier(self, exp: Explainer) -> None:
        menu = exp.recourse_menu(x0, TWO_LEVERS, max_levers=3, backend="exact", seed=0)
        assert menu.mode == "minimal"
        assert set(menu.minimal) == {"a+b", "a+c"}
        # the frontier is an antichain, and the three-lever set was never solved
        for one, other in itertools.permutations(menu.minimal, 2):
            assert not set(one.split("+")) < set(other.split("+"))
        assert "a+b+c" not in menu
        assert "a+b+c" not in menu.implied
        # every feasible set of the brute force contains a listed set
        expected = _brute_force(exp, x0, TWO_LEVERS, 3, backend="exact", seed=0)
        for combo, result in expected.items():
            if isinstance(result, Counterfactual):
                assert any(set(m.split("+")) <= set(combo) for m in menu.minimal)

    def test_minimal_frontier_in_all_mode_too(self, exp: Explainer) -> None:
        menu = exp.recourse_menu(
            x0, TWO_LEVERS, max_levers=3, mode="all", backend="exact", seed=0,
        )
        assert set(menu.minimal) == {"a+b", "a+c"}
        assert "a+b+c" in menu.implied  # solved, changed a strict subset

    def test_keys_are_sorted_names_joined_by_plus(self, exp: Explainer) -> None:
        menu = exp.recourse_menu(x0, TWO_LEVERS, max_levers=2, backend="exact", seed=0)
        for key, entry in menu.items():
            if isinstance(entry, Counterfactual):
                assert key == "+".join(sorted(entry.changes))
            assert key == "+".join(sorted(key.split("+")))

    def test_ordering_minimal_then_feasible_then_infeasible(self, exp: Explainer) -> None:
        menu = exp.recourse_menu(
            x0, TWO_LEVERS, max_levers=2, mode="all", backend="exact", seed=0,
        )
        keys = list(menu)
        feasible = [k for k in keys if isinstance(menu[k], Counterfactual)]
        infeasible = [k for k in keys if isinstance(menu[k], Infeasible)]
        assert keys == feasible + infeasible
        costs = [menu[k].distance for k in feasible]  # type: ignore[union-attr]
        assert costs == sorted(costs)
        sizes = [len(k.split("+")) for k in infeasible]
        assert sizes == sorted(sizes)
        assert infeasible[-1] == "b+c"

    def test_same_seed_same_menu(self, exp: Explainer) -> None:
        first = exp.recourse_menu(x0, TWO_LEVERS, max_levers=2, mode="all", seed=4)
        second = exp.recourse_menu(x0, TWO_LEVERS, max_levers=2, mode="all", seed=4)
        assert list(first) == list(second)
        assert first.to_dict() == second.to_dict()

    def test_frozen_features_are_never_levers(self) -> None:
        exp = Explainer(_ir(), normalizers=np.ones(3), constraints=[Freeze("c")])
        menu = exp.recourse_menu(x0, ONE_LEVER, max_levers=3, mode="all", backend="exact")
        assert menu.levers == ("a", "b")
        assert all("c" not in key for key in menu)

    def test_factual_already_in_target_raises(self, exp: Explainer) -> None:
        with pytest.raises(TreecfError, match="already"):
            exp.recourse_menu(np.array([2.0, 0.0, 0.0]), ONE_LEVER, max_levers=1)


class TestCertification:
    def test_exact_menu_is_complete_and_certified(self, exp: Explainer) -> None:
        menu = exp.recourse_menu(
            x0, TWO_LEVERS, max_levers=2, mode="all", backend="exact", seed=0,
        )
        assert menu.complete is True
        assert menu.unresolved == ()
        for entry in menu.values():
            if isinstance(entry, Counterfactual):
                assert entry.proof in ("optimal", "optimal_within_gap")
            else:
                assert entry.proof == "certified"
        assert set(menu.certified_infeasible) == {"a", "b", "c", "b+c"}

    def test_genetic_menu_is_never_complete(self, exp: Explainer) -> None:
        menu = exp.recourse_menu(x0, TWO_LEVERS, max_levers=2, backend="genetic", seed=0)
        assert menu.complete is False
        assert any(isinstance(e, Counterfactual) for e in menu.values())

    def test_total_budget_leaves_sets_unresolved_with_one_warning(
        self, exp: Explainer
    ) -> None:
        with pytest.warns(TreecfWarning) as record:
            menu = exp.recourse_menu(
                x0, TWO_LEVERS, max_levers=2, mode="all", backend="exact", seed=0,
                total_budget_s=0.0,
            )
        assert len(menu) == 0
        assert set(menu.unresolved) == {"a", "b", "c", "a+b", "a+c", "b+c"}
        assert menu.complete is False
        assert len(record) == 1
        assert "6/6" in str(record[0].message)

    def test_no_warning_when_everything_is_certified(self, exp: Explainer) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exp.recourse_menu(x0, TWO_LEVERS, max_levers=2, backend="exact", seed=0)

    def test_genetic_menu_does_not_warn_about_missing_certificates(
        self, exp: Explainer
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exp.recourse_menu(x0, ONE_LEVER, max_levers=1, backend="genetic", seed=0)


class TestImpliedAndGuard:
    def test_a_coalition_that_changed_a_subset_files_under_the_subset(
        self, exp: Explainer
    ) -> None:
        menu = exp.recourse_menu(
            x0, ONE_LEVER, max_levers=3, mode="all", backend="exact", seed=0,
        )
        assert set(menu.minimal) == {"a", "b", "c"}
        assert "a+b+c" in menu.implied
        assert menu.implied["a+b+c"] in ("a", "b", "c")
        assert "a+b+c" not in menu

    def test_contradictory_verdicts_raise(
        self, exp: Explainer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def free_levers(explainer: Explainer) -> set[str]:
            frozen = {c.feature for c in explainer.compiled.constraints if isinstance(c, Freeze)}
            return set(explainer.ir.feature_names) - frozen

        def stub(self: Explainer, x: np.ndarray, target: Target, *args: object, **kw: object):
            free = free_levers(self)
            if free == {"a"}:
                x_cf = x.copy()
                x_cf[0] = 1.0
                return Counterfactual(
                    x_cf=x_cf, changes={"a": (0.0, 1.0)}, distance=1.0, n_changed=1,
                    score_raw=1.0, score_prob=None, proof="optimal",
                    solver_stats={"completed": True},
                )
            return Infeasible(reason="stub", proof="certified", solver_stats={"completed": True})

        monkeypatch.setattr(Explainer, "_explain_one", stub)
        with pytest.raises(TreecfError, match=r"'a\+b'.*'a'|'a'.*'a\+b'"):
            exp.recourse_menu(x0, ONE_LEVER, max_levers=2, mode="all", backend="exact")


class TestSharedMarshaling:
    def test_a_large_menu_marshals_the_ensemble_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from treecf.backends import exact_rust, genetic_rust
        from treecf.backends.exact_rust import _rust_available

        if not _rust_available():
            pytest.skip("Rust core not built")
        calls: list[int] = []
        original = genetic_rust.build_rust_ensemble

        def counting(ir: EnsembleIR) -> object:
            calls.append(1)
            return original(ir)

        monkeypatch.setattr(genetic_rust, "build_rust_ensemble", counting)
        monkeypatch.setattr(exact_rust, "build_rust_ensemble", counting)
        # nine independent levers: 9 + 36 + 84 = 129 lever sets up to size three
        ir = EnsembleIR(
            trees=tuple(_stump(j, 1.0, 1.0) for j in range(9)),
            base_score=0.0, link=Link.IDENTITY, n_features=9,
            feature_names=tuple(f"f{j}" for j in range(9)), meta={},
        )
        exp = Explainer(ir, normalizers=np.ones(9))
        menu = exp.recourse_menu(
            np.zeros(9), ONE_LEVER, max_levers=3, mode="all", backend="exact",
            seed=0, warm_start=False,
        )
        assert len(menu) + len(menu.implied) == 129
        assert len(calls) == 1


class TestSerialization:
    def test_to_dict_is_strict_json(self, exp: Explainer) -> None:
        menu = exp.recourse_menu(
            x0, TWO_LEVERS, max_levers=2, mode="all", backend="exact", seed=0,
        )
        payload = menu.to_dict()
        text = json.dumps(payload, allow_nan=False)
        back = json.loads(text)
        assert back["menu_schema_version"] == 1
        assert back["mode"] == "all" and back["max_levers"] == 2
        assert back["levers"] == ["a", "b", "c"]
        assert set(back["entries"]) == set(menu)
        assert back["entries"]["a+b"]["feasible"] is True
        assert back["entries"]["b+c"]["feasible"] is False
        assert back["entries"]["b+c"]["proof"] == "certified"
        assert back["minimal"] == list(menu.minimal)

    def test_to_frame_rows(self, exp: Explainer) -> None:
        menu = exp.recourse_menu(
            x0, TWO_LEVERS, max_levers=2, mode="all", backend="exact", seed=0,
        )
        rows = menu.to_frame()
        assert [row["key"] for row in rows] == list(menu)
        first = rows[0]
        assert set(first) >= {"key", "size", "feasible", "distance", "proof", "changed"}
        assert first["size"] == 2 and first["feasible"] is True

    def test_describe_phrases_certified_infeasibility(self, exp: Explainer) -> None:
        menu = exp.recourse_menu(
            x0, TWO_LEVERS, max_levers=2, mode="all", backend="exact", seed=0,
        )
        described = menu.describe()
        assert set(described) == set(menu)
        assert described["b+c"] == (
            "no acceptance is reachable by changing only b, c"
        )
        assert described["a+b"].startswith("change a, b")

    def test_menu_is_a_mapping(self, exp: Explainer) -> None:
        menu = exp.recourse_menu(x0, ONE_LEVER, max_levers=1, backend="exact", seed=0)
        assert isinstance(menu, RecourseMenu)
        assert isinstance(menu, Mapping)
        assert dict(menu) == menu.entries


class TestDiverse:
    def test_lever_diverse_plans_are_pairwise_distinct_and_cost_ordered(
        self, exp: Explainer
    ) -> None:
        diverse = exp.explain_diverse(x0, TWO_LEVERS, k=5, backend="exact", seed=0)
        assert isinstance(diverse, DiverseSet)
        assert diverse.criterion == "levers"
        keys = list(diverse)
        assert keys == ["a+b", "a+c"]
        changed = [frozenset(diverse[k].changes) for k in keys]
        assert len(set(changed)) == len(changed)
        costs = [diverse[k].distance for k in keys]
        assert costs == sorted(costs)
        assert diverse.complete is True  # the whole menu was certified and had only two
        assert diverse.menu is not None and diverse.menu.mode == "minimal"

    def test_k_reached_means_not_complete(self, exp: Explainer) -> None:
        diverse = exp.explain_diverse(x0, ONE_LEVER, k=2, backend="exact", seed=0)
        assert len(diverse) == 2
        assert diverse.complete is False

    def test_jaccard_matrix(self, exp: Explainer) -> None:
        diverse = exp.explain_diverse(x0, TWO_LEVERS, k=5, backend="exact", seed=0)
        j = diverse.jaccard
        assert j.shape == (2, 2)
        assert np.allclose(j, j.T)
        assert np.all(np.diag(j) == 0.0)
        assert j[0, 1] == pytest.approx(1.0 - 1.0 / 3.0)  # {a,b} vs {a,c}

    def test_coalition_ladder_uses_unions_only_when_needed(
        self, exp: Explainer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        solved: list[frozenset[str]] = []
        original = Explainer._explain_one

        def spy(self: Explainer, *args: object, **kw: object):
            frozen = {c.feature for c in self.compiled.constraints if isinstance(c, Freeze)}
            solved.append(frozenset(set(self.ir.feature_names) - frozen))
            return original(self, *args, **kw)

        monkeypatch.setattr(Explainer, "_explain_one", spy)
        coalitions = {"first": ["a"], "second": ["b"], "third": ["c"]}
        # every single coalition suffices: no union is ever solved
        diverse = exp.explain_diverse(
            x0, ONE_LEVER, k=3, diversity="coalitions", coalitions=coalitions,
            backend="exact", seed=0,
        )
        assert list(diverse) == ["first", "second", "third"]
        assert all(len(s) == 1 for s in solved)
        solved.clear()
        # two levers needed: singles fail, the pairs are tried, the triple is not
        diverse = exp.explain_diverse(
            x0, TWO_LEVERS, k=2, diversity="coalitions", coalitions=coalitions,
            backend="exact", seed=0,
        )
        assert list(diverse) == ["first+second", "first+third"]
        assert diverse.criterion == "coalitions"
        assert diverse.menu is None
        assert max(len(s) for s in solved) == 2

    def test_coalition_ladder_exhausted_is_complete(self, exp: Explainer) -> None:
        coalitions = {"first": ["a"], "second": ["b"], "third": ["c"]}
        diverse = exp.explain_diverse(
            x0, TWO_LEVERS, k=5, diversity="coalitions", coalitions=coalitions,
            backend="exact", seed=0,
        )
        assert list(diverse) == ["first+second", "first+third"]
        assert diverse.complete is True

    def test_unknown_criterion_and_missing_coalitions(self, exp: Explainer) -> None:
        with pytest.raises(ValueError, match=r"levers.*coalitions"):
            exp.explain_diverse(x0, ONE_LEVER, diversity="routing")
        with pytest.raises(TreecfError, match="coalitions"):
            exp.explain_diverse(x0, ONE_LEVER, diversity="coalitions")

    def test_to_frame(self, exp: Explainer) -> None:
        diverse = exp.explain_diverse(x0, TWO_LEVERS, k=5, backend="exact", seed=0)
        rows = diverse.to_frame()
        assert [row["key"] for row in rows] == list(diverse)
        assert rows[0]["distance"] == diverse["a+b"].distance
