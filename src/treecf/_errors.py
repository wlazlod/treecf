"""Exception types. Parsers and backends raise — they never degrade silently."""

from __future__ import annotations


class TreecfError(Exception):
    """Base class for all treecf errors."""


class UnsupportedModelError(TreecfError):
    """The model (or one of its nodes/objectives) cannot be represented in the IR."""


class MissingExtraError(TreecfError):
    """An optional dependency is required; message carries the pip install command."""


class ConstraintValidationError(TreecfError):
    """The constraint set is invalid for this model's feature space."""


class ConstraintParseError(TreecfError):
    """A string constraint failed to parse; the message carries a caret position."""


class TargetError(TreecfError):
    """The target specification is malformed or incompatible with the model link."""
