"""Exception types. Parsers and backends raise — they never degrade silently (D1, §3.4)."""

from __future__ import annotations


class TreecfError(Exception):
    """Base class for all treecf errors."""


class UnsupportedModelError(TreecfError):
    """The model (or one of its nodes/objectives) cannot be represented in the IR."""


class MissingExtraError(TreecfError):
    """An optional dependency is required; message carries the pip install command."""
