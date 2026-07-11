"""Solver backends. Selection is always explicit — no silent fallback (D1)."""

from treecf.backends.base import Backend, BackendSolution

__all__ = ["Backend", "BackendSolution"]
