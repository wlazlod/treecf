"""treecf — constrained, threshold-aware counterfactual explanations for tree ensembles."""

from treecf._errors import (
    ConstraintValidationError,
    MissingExtraError,
    TargetError,
    TreecfError,
    UnsupportedModelError,
)
from treecf.api import Counterfactual, Explainer, Infeasible
from treecf.constraints import Freeze, Monotone, Range
from treecf.targets import Target

__version__ = "0.1.0.dev0"

__all__ = [
    "ConstraintValidationError",
    "Counterfactual",
    "Explainer",
    "Freeze",
    "Infeasible",
    "MissingExtraError",
    "Monotone",
    "Range",
    "Target",
    "TargetError",
    "TreecfError",
    "UnsupportedModelError",
    "__version__",
]
