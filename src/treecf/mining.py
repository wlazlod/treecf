"""Data-driven constraint suggestion — numpy only.

Mines candidate invariants from a background sample and returns them for
explicit human review; nothing is ever auto-applied. Mined constraints are
sample invariants, not domain truths: ``min_support = 1.0`` on a finite sample
can be coincidence, and near-invariants (support in [report_threshold, 1)) are
returned as ``DataQualityFinding`` records — a handful of violations of an
otherwise universal rule usually signals an ETL defect, not a domain exception.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from treecf.constraints.objects import Constraint, Equals, Implies, Linear, OneHot

FloatArray = npt.NDArray[np.float64]

_EVIDENCE_ROWS = 5


@dataclass(frozen=True)
class SuggestedConstraint:
    constraint: Constraint | None  # None for advisory kinds (missing_link, integer)
    kind: str  # "order" | "equality" | "implication" | "onehot" | "missing_link" | "integer"
    support: float
    n_rows_checked: int
    n_violations: int
    evidence: list[dict[str, object]] = field(default_factory=list)
    rationale: str = ""

    def as_code(self) -> str:
        tail = f"  # support={self.support:.4f}, n={self.n_rows_checked}"
        if self.kind == "order" and isinstance(self.constraint, Linear):
            coeffs = self.constraint.coefficients
            smaller = max(coeffs, key=lambda k: coeffs[k])
            larger = min(coeffs, key=lambda k: coeffs[k])
            return f'constraint("{smaller} <= {larger}")' + tail
        if self.kind == "equality" and isinstance(self.constraint, Linear):
            a, b = list(self.constraint.coefficients)
            return f'# equality: {a} == {b} — likely a redundant feature' + tail
        if self.kind == "implication" and isinstance(self.constraint, Implies):
            c = self.constraint
            return (
                f'Implies(Equals("{c.condition.feature}", {c.condition.value:g}), '
                f'Equals("{c.consequence.feature}", {c.consequence.value:g}))' + tail
            )
        if self.kind == "onehot" and isinstance(self.constraint, OneHot):
            inner = ", ".join(f'"{f}"' for f in self.constraint.features)
            return f"OneHot(({inner}))" + tail
        return f"# {self.kind}: {self.rationale}" + tail


@dataclass(frozen=True)
class DataQualityFinding:
    kind: str  # "near_invariant"
    description: str
    support: float
    n_rows_checked: int
    n_violations: int
    evidence: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class SuggestionSet:
    suggestions: tuple[SuggestedConstraint, ...]
    findings: tuple[DataQualityFinding, ...]

    def __iter__(self) -> Iterator[SuggestedConstraint]:
        return iter(self.suggestions)

    def __len__(self) -> int:
        return len(self.suggestions)

    def __getitem__(self, item: int | slice) -> SuggestedConstraint | list[SuggestedConstraint]:
        if isinstance(item, slice):
            return list(self.suggestions[item])
        return self.suggestions[item]


def suggest_constraints(
    X: FloatArray,
    feature_names: Sequence[str] | None = None,
    min_support: float = 1.0,
    top_k: int = 50,
    report_threshold: float = 0.999,
    include_ranges: bool = False,
) -> SuggestionSet:
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(p)]
    present = ~np.isnan(X)

    binary = [
        j
        for j in range(p)
        if present[:, j].any() and np.isin(X[present[:, j], j], (0.0, 1.0)).all()
    ]
    binary_set = set(binary)

    suggestions: list[SuggestedConstraint] = []
    findings: list[DataQualityFinding] = []

    # --- pairwise order / equality (O(p^2 n), vectorized per anchor column) ---
    order_edges: dict[tuple[int, int], SuggestedConstraint] = {}
    equal_pairs: list[tuple[int, int]] = []
    for a in range(p):
        for b in range(a + 1, p):
            both = present[:, a] & present[:, b]
            n_both = int(both.sum())
            if n_both == 0:
                continue
            va, vb = X[both, a], X[both, b]
            viol_ab = int((va > vb).sum())  # violations of a <= b
            viol_ba = int((vb > va).sum())
            if viol_ab == 0 and viol_ba == 0:
                equal_pairs.append((a, b))
                suggestions.append(
                    SuggestedConstraint(
                        constraint=Linear({names[a]: 1.0, names[b]: -1.0}, op="==", rhs=0.0),
                        kind="equality",
                        support=1.0,
                        n_rows_checked=n_both,
                        n_violations=0,
                        rationale=f"{names[a]} == {names[b]} on every co-present row; "
                        "usually a redundant feature, not a constraint to impose",
                    )
                )
                continue
            for lo_idx, hi_idx, viol in ((a, b, viol_ab), (b, a, viol_ba)):
                support = 1.0 - viol / n_both
                if support >= min_support:
                    order_edges[(lo_idx, hi_idx)] = SuggestedConstraint(
                        constraint=Linear(
                            {names[lo_idx]: 1.0, names[hi_idx]: -1.0}, op="<=", rhs=0.0
                        ),
                        kind="order",
                        support=support,
                        n_rows_checked=n_both,
                        n_violations=viol,
                        evidence=_violation_evidence(X, present, lo_idx, hi_idx),
                        rationale=_order_rationale(names[lo_idx], names[hi_idx]),
                    )
                elif support >= report_threshold:
                    findings.append(
                        DataQualityFinding(
                            kind="near_invariant",
                            description=f"{names[lo_idx]} <= {names[hi_idx]} holds on "
                            f"{support:.4%} of rows — likely an ETL defect",
                            support=support,
                            n_rows_checked=n_both,
                            n_violations=viol,
                            evidence=_violation_evidence(X, present, lo_idx, hi_idx),
                        )
                    )

    # equality-class collapse, then transitive reduction of the <= graph
    representative = _union_find(p, equal_pairs)
    rep_edges = {
        (representative[a], representative[b])
        for (a, b) in order_edges
        if representative[a] != representative[b]
    }
    reduced = transitive_reduction(rep_edges)
    for (a, b), suggestion in order_edges.items():
        edge = (representative[a], representative[b])
        if edge in reduced and edge[0] != edge[1]:
            suggestions.append(suggestion)
            reduced.discard(edge)  # one edge per class pair

    # --- binary implications A=1 => B=1 ---
    for a in binary:
        a_is_one = present[:, a] & (X[:, a] == 1.0)
        if not a_is_one.any():
            continue
        for b in binary:
            if a == b:
                continue
            checked = a_is_one & present[:, b]
            if not checked.any():
                continue
            if (X[checked, b] == 1.0).all():
                suggestions.append(
                    SuggestedConstraint(
                        constraint=Implies(Equals(names[a], 1.0), Equals(names[b], 1.0)),
                        kind="implication",
                        support=1.0,
                        n_rows_checked=int(checked.sum()),
                        n_violations=0,
                        rationale=f"{names[a]}=1 always co-occurs with {names[b]}=1",
                    )
                )

    # --- one-hot groups: exclusivity components with row sum == 1 ---
    complete_binary = [j for j in binary if present[:, j].all()]
    for component in _exclusivity_components(X, complete_binary):
        if len(component) < 2:
            continue
        if np.all(X[:, component].sum(axis=1) == 1.0):
            suggestions.append(
                SuggestedConstraint(
                    constraint=OneHot(tuple(names[j] for j in component)),
                    kind="onehot",
                    support=1.0,
                    n_rows_checked=n,
                    n_violations=0,
                    rationale="binary columns with row sum identically 1",
                )
            )

    # --- missingness links miss(A) => miss(B) ---
    for a in range(p):
        miss_a = ~present[:, a]
        if not miss_a.any():
            continue
        for b in range(p):
            if a == b or present[:, b].all():
                continue
            if (~present[miss_a, b]).all():
                both_ways = bool((~present[~present[:, b], a]).all())
                suggestions.append(
                    SuggestedConstraint(
                        constraint=None,
                        kind="missing_link",
                        support=1.0,
                        n_rows_checked=int(miss_a.sum()),
                        n_violations=0,
                        rationale=(
                            f"miss({names[a]}) {'<=>' if both_ways else '=>'} miss({names[b]}); "
                            "consider joint AllowMissing / missing_policy"
                        ),
                    )
                )
            if bool((~present[~present[:, b], a]).all()):
                break  # symmetric link already reported from this anchor

    # --- integer-valuedness -> value_policy suggestion ---
    for j in range(p):
        col = X[present[:, j], j]
        if len(col) and j not in binary_set and np.all(col == np.round(col)):
            suggestions.append(
                SuggestedConstraint(
                    constraint=None,
                    kind="integer",
                    support=1.0,
                    n_rows_checked=len(col),
                    n_violations=0,
                    rationale=(
                        f"{names[j]} is integer-valued; "
                        f'value_policy={{"{names[j]}": "integer"}}'
                    ),
                )
            )

    if include_ranges:
        for j in range(p):
            col = X[present[:, j], j]
            if len(col):
                lo, hi = np.percentile(col, [1, 99])
                pad = 0.1 * (hi - lo)
                suggestions.append(
                    SuggestedConstraint(
                        constraint=None,
                        kind="range",
                        support=1.0,
                        n_rows_checked=len(col),
                        n_violations=0,
                        rationale=f"observed 1-99% range [{lo:.4g}, {hi:.4g}] padded by {pad:.4g}",
                    )
                )

    suggestions.sort(key=_rank_key, reverse=True)
    return SuggestionSet(suggestions=tuple(suggestions[:top_k]), findings=tuple(findings))


_NodeT = TypeVar("_NodeT", bound=Hashable)


def transitive_reduction(edges: set[tuple[_NodeT, _NodeT]]) -> set[tuple[_NodeT, _NodeT]]:
    """Minimal generating set of a DAG's reachability relation."""
    adjacency: dict[_NodeT, set[_NodeT]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
    empty: set[_NodeT] = set()

    def reachable_without(a: _NodeT, b: _NodeT) -> bool:
        frontier = list(adjacency.get(a, empty) - {b})
        seen: set[_NodeT] = set()
        while frontier:
            cur = frontier.pop()
            if cur == b:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            frontier.extend(adjacency.get(cur, empty))
        return False

    return {(a, b) for a, b in edges if not reachable_without(a, b)}


def _union_find(p: int, pairs: list[tuple[int, int]]) -> list[int]:
    parent = list(range(p))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    return [find(j) for j in range(p)]


def _exclusivity_components(X: FloatArray, cols: list[int]) -> list[list[int]]:
    """Connected components of the 'never both 1' graph over complete binary columns."""
    index = {j: i for i, j in enumerate(cols)}
    parent = list(range(len(cols)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in cols:
        for b in cols:
            if a < b and not np.any((X[:, a] == 1.0) & (X[:, b] == 1.0)):
                parent[find(index[a])] = find(index[b])
    components: dict[int, list[int]] = {}
    for j in cols:
        components.setdefault(find(index[j]), []).append(j)
    return list(components.values())


def _violation_evidence(
    X: FloatArray, present: npt.NDArray[np.bool_], lo_idx: int, hi_idx: int
) -> list[dict[str, object]]:
    both = present[:, lo_idx] & present[:, hi_idx]
    rows = np.flatnonzero(both & (X[:, lo_idx] > X[:, hi_idx]))[:_EVIDENCE_ROWS]
    return [
        {"row": int(r), "values": (float(X[r, lo_idx]), float(X[r, hi_idx]))} for r in rows
    ]


def _order_rationale(a: str, b: str) -> str:
    shared = sorted(set(a.split("_")) & set(b.split("_")))
    if shared:
        return f"shared name tokens: {shared}"
    return "no shared name tokens"


def _rank_key(s: SuggestedConstraint) -> tuple[float, int, int]:
    tokens = 1 if "shared name tokens: [" in s.rationale else 0
    kind_priority = {"order": 3, "onehot": 3, "implication": 2, "missing_link": 2}.get(s.kind, 1)
    return (s.support, tokens, kind_priority)
