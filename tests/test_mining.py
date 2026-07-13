"""Constraint mining: planted invariants, reduction, ranking."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from treecf.constraints.objects import Linear, OneHot
from treecf.mining import suggest_constraints, transitive_reduction


def _planted_data(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """DPD-window hierarchy, a one-hot trio, a missingness link, noise columns."""
    rng = np.random.default_rng(seed)
    dpd_7d = np.floor(rng.exponential(1.0, n))
    dpd_30d = dpd_7d + np.floor(rng.exponential(1.0, n))
    dpd_90d = dpd_30d + np.floor(rng.exponential(1.0, n))
    channel = rng.integers(0, 3, n)
    web = (channel == 0).astype(float)
    app = (channel == 1).astype(float)
    branch = (channel == 2).astype(float)
    noise_a = rng.normal(size=n)
    noise_b = rng.normal(size=n) * 100
    # missingness link: months_since_delinq missing => delinq_amount missing
    months = rng.exponential(10.0, n)
    amount = rng.exponential(500.0, n)
    no_history = rng.random(n) < 0.3
    months[no_history] = np.nan
    amount[no_history | (rng.random(n) < 0.1)] = np.nan
    X = np.column_stack(
        [dpd_7d, dpd_30d, dpd_90d, web, app, branch, noise_a, noise_b, months, amount]
    )
    names = [
        "max_dpd_7d", "max_dpd_30d", "max_dpd_90d",
        "channel_web", "channel_app", "channel_branch",
        "noise_a", "noise_b", "months_since_delinq", "delinq_amount",
    ]
    return X, names


@pytest.fixture(scope="module")
def suggestions() -> object:
    X, names = _planted_data()
    return suggest_constraints(X, feature_names=names)


class TestPlantedRecovery:
    def test_dpd_chain_recovered_and_reduced(self, suggestions: object) -> None:
        orders = [
            s for s in suggestions
            if s.kind == "order" and isinstance(s.constraint, Linear)
        ]
        as_pairs = {
            (min(s.constraint.coefficients, key=s.constraint.coefficients.get),
             max(s.constraint.coefficients, key=s.constraint.coefficients.get))
            for s in orders
        }
        # a <= b is stored as {a: 1, b: -1}: max-coef key is the smaller feature
        chain = {("max_dpd_7d", "max_dpd_30d"), ("max_dpd_30d", "max_dpd_90d")}
        found = {(b, a) for a, b in as_pairs}  # (smaller, larger)
        assert chain <= found
        # transitive edge suppressed
        assert ("max_dpd_7d", "max_dpd_90d") not in found
        assert all(s.support == 1.0 for s in orders)

    def test_onehot_group_recovered(self, suggestions: object) -> None:
        groups = [s for s in suggestions if s.kind == "onehot"]
        assert any(
            isinstance(s.constraint, OneHot)
            and set(s.constraint.features)
            == {"channel_web", "channel_app", "channel_branch"}
            for s in groups
        )

    def test_missingness_link_recovered(self, suggestions: object) -> None:
        links = [s for s in suggestions if s.kind == "missing_link"]
        assert any("months_since_delinq" in s.rationale for s in links)

    def test_integer_valuedness_detected(self, suggestions: object) -> None:
        integers = {s.rationale for s in suggestions if s.kind == "integer"}
        assert any("max_dpd_7d" in r for r in integers)

    def test_as_code_is_pasteable(self, suggestions: object) -> None:
        order = next(s for s in suggestions if s.kind == "order")
        code = order.as_code()
        assert "constraint(" in code and "support=" in code


class TestNearInvariants:
    def test_single_violation_becomes_finding_not_suggestion(self) -> None:
        X, names = _planted_data()
        X[7, 0] = X[7, 1] + 5.0  # one ETL-style violation of dpd_7d <= dpd_30d
        result = suggest_constraints(X, feature_names=names)
        pairs = [
            s for s in result
            if s.kind == "order" and "max_dpd_7d" in getattr(s.constraint, "coefficients", {})
            and "max_dpd_30d" in s.constraint.coefficients
        ]
        assert not pairs  # not suggested as a constraint...
        assert any(
            "max_dpd_7d" in f.description and f.n_violations == 1
            for f in result.findings
        )  # ...but surfaced as a data-quality finding with evidence


class TestNoFalsePositives:
    def test_shuffled_columns_produce_no_onehot(self) -> None:
        rng = np.random.default_rng(3)
        X = (rng.random((3000, 6)) < 0.4).astype(float)  # independent binaries
        result = suggest_constraints(X, feature_names=[f"b{i}" for i in range(6)])
        assert not [s for s in result if s.kind == "onehot"]


class TestRanking:
    def test_token_matched_pairs_rank_above_unrelated(self) -> None:
        X, names = _planted_data()
        result = suggest_constraints(X, feature_names=names)
        orders = [s for s in result if s.kind == "order"]
        token_scores = [
            ("dpd" in " ".join(s.constraint.coefficients), i) for i, s in enumerate(orders)
        ]
        first_dpd = min(i for is_dpd, i in token_scores if is_dpd)
        unrelated = [i for is_dpd, i in token_scores if not is_dpd]
        if unrelated:  # accidental orderings of unrelated features rank lower
            assert first_dpd < min(unrelated)


class TestTransitiveReduction:
    @settings(max_examples=50, deadline=None)
    @given(st.integers(min_value=0, max_value=10_000))
    def test_reduction_preserves_reachability_without_implied_edges(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 8))
        edges = {
            (a, b)
            for a in range(n)
            for b in range(a + 1, n)  # DAG by construction
            if rng.random() < 0.4
        }
        reduced = transitive_reduction(edges)

        def reachable(es: set[tuple[int, int]], a: int, b: int) -> bool:
            frontier, seen = {a}, set()
            while frontier:
                cur = frontier.pop()
                if cur == b:
                    return True
                seen.add(cur)
                frontier |= {y for (x, y) in es if x == cur and y not in seen}
            return False

        for a, b in edges:
            assert reachable(reduced, a, b)  # reachability preserved
        for a, b in reduced:
            assert not reachable(reduced - {(a, b)}, a, b)  # no redundant edge
