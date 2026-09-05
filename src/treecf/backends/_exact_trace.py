"""The certification trace: how the incumbent and the lower bound moved.

The exact search records ``(nodes_expanded, incumbent_cost, lower_bound)``
at every incumbent update and at every power-of-two node count, plus one
terminal sample. The list is bounded: past ``CAP`` entries, every second
sample that is not an incumbent update is dropped before the next one is
appended, so a long search keeps its shape at a fixed size. The Rust core
records the identical list, sample for sample.
"""

from __future__ import annotations

TraceSample = tuple[int, float | None, float]


class _Trace:
    CAP = 256

    def __init__(self) -> None:
        self._samples: list[TraceSample] = []
        self._incumbent_flags: list[bool] = []

    def record(
        self, nodes: int, incumbent: float | None, bound: float, *, is_incumbent: bool
    ) -> None:
        sample: TraceSample = (nodes, incumbent, bound)
        if self._samples and self._samples[-1][0] == nodes:
            self._samples[-1] = sample
            self._incumbent_flags[-1] = self._incumbent_flags[-1] or is_incumbent
            return
        if len(self._samples) >= self.CAP:
            kept: list[TraceSample] = []
            flags: list[bool] = []
            plain_seen = 0
            for existing, flag in zip(self._samples, self._incumbent_flags, strict=True):
                if flag:
                    kept.append(existing)
                    flags.append(True)
                    continue
                if plain_seen % 2 == 0:
                    kept.append(existing)
                    flags.append(False)
                plain_seen += 1
            self._samples = kept
            self._incumbent_flags = flags
        self._samples.append(sample)
        self._incumbent_flags.append(is_incumbent)

    def samples(self) -> list[TraceSample]:
        return list(self._samples)
