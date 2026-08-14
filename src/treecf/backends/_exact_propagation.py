"""Domain settling for the exact backend: implications and one-hot groups.

Split out of ``treecf.backends.exact`` for size only: ``exact``, this file,
``_exact_domains`` and ``_exact_orderpairs`` are one implementation, and the
Rust mirror has to match all four bit-for-bit.
"""

from __future__ import annotations

from treecf.backends._exact_domains import _State
from treecf.constraints.compile import CompiledConstraints

# what one assignment changed in the propagation state: the features it settled
# (with their previous setting) and the one-hot counters it moved (with their
# previous readings). Both are stored as old values rather than deltas, for the
# same reason the cost is: putting a saved value back cannot drift.
_PropFrame = tuple[
    tuple[tuple[int, float | None], ...],
    tuple[tuple[int, int, int], ...],
]


class _Propagation:
    """What assigning one feature settles about the features that follow it.

    Two constraint kinds reach past the feature they name. An implication
    settles its consequence the moment its condition is met, and a one-hot
    group settles its last free member once every other member is a zero. A
    later state that disagrees with such a settlement cannot be completed into
    anything the arbiter would accept, so the search cuts it unexplored.

    ``apply`` reports what it changed so ``restore`` can put the previous
    state back on the way out of a branch, and reports separately whether the
    assignment contradicts something already settled — on a contradiction the
    changes made up to that point are still reported, since the caller
    restores the frame either way.

    One-hot bookkeeping only runs for groups whose members can hold nothing
    but 0 and 1. A member offering some other value — a factual that is not
    binary, or a missing state — could still make its group sum to one in ways
    these counters do not model, so such a group is left to the arbiter alone.

    ``assigned`` and ``values`` are the search's own arrays, shared by
    reference, so this reads the one assignment everything else reads.
    """

    def __init__(
        self,
        compiled: CompiledConstraints,
        domains: list[list[_State]],
        assigned: list[bool],
        values: list[float],
    ) -> None:
        self.implications = compiled.implications
        self.groups = compiled.onehot_groups
        self.assigned = assigned
        self.values = values
        self.group_of: dict[int, int] = {}
        for g_idx, group in enumerate(self.groups):
            if all(s.value in (0.0, 1.0) for f in group for s in domains[f]):
                for f in group:
                    self.group_of[f] = g_idx
        self.forced_value: list[float | None] = [None] * len(assigned)
        self.ones = [0] * len(self.groups)
        self.zeros = [0] * len(self.groups)

    def apply(self, j: int, v: float) -> tuple[_PropFrame, bool]:
        """Settle what follows from ``j`` taking ``v``; report any contradiction."""
        settled: list[tuple[int, float | None]] = []
        counters: list[tuple[int, int, int]] = []

        def force(f: int, value: float) -> bool:
            if self.assigned[f]:
                return self.values[f] == value
            current = self.forced_value[f]
            if current is not None:
                return current == value
            settled.append((f, None))
            self.forced_value[f] = value
            return True

        def done(conflict: bool) -> tuple[_PropFrame, bool]:
            return (tuple(settled), tuple(counters)), conflict

        if self.forced_value[j] is not None and v != self.forced_value[j]:
            return done(True)
        g_idx = self.group_of.get(j)
        if g_idx is not None:
            group = self.groups[g_idx]
            counters.append((g_idx, self.ones[g_idx], self.zeros[g_idx]))
            if v == 1.0:
                self.ones[g_idx] += 1
            else:
                self.zeros[g_idx] += 1
            # a second one, or nothing but zeros: the group can no longer sum
            # to one. The all-zeros reading is a backstop -- settling the last
            # free member normally catches that case one assignment earlier.
            if self.ones[g_idx] > 1 or self.zeros[g_idx] == len(group):
                return done(True)
            if self.ones[g_idx] == 0 and self.zeros[g_idx] == len(group) - 1:
                last = next(f for f in group if f != j and not self.assigned[f])
                if not force(last, 1.0):
                    return done(True)
        for imp in self.implications:
            triggered = imp.cond_index == j and v == imp.cond_value
            if triggered and not force(imp.cons_index, imp.cons_value):
                return done(True)
        return done(False)

    def restore(self, frame: _PropFrame) -> None:
        """Put back everything one ``apply`` settled."""
        settled, counters = frame
        for f, previous in reversed(settled):
            self.forced_value[f] = previous
        for g_idx, ones, zeros in reversed(counters):
            self.ones[g_idx] = ones
            self.zeros[g_idx] = zeros
