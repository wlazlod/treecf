"""treecf — constrained, threshold-aware counterfactual explanations for tree ensembles."""

from treecf._errors import (
    ConstraintParseError,
    ConstraintValidationError,
    MissingExtraError,
    TargetError,
    TreecfError,
    UnsupportedModelError,
)
from treecf.api import Counterfactual, Explainer, Infeasible
from treecf.constraints import (
    AllowMissing,
    Equals,
    Freeze,
    Implies,
    Linear,
    Monotone,
    OneHot,
    Range,
    constraint,
)
from treecf.targets import Target

__version__ = "0.1.0.dev0"

__all__ = [
    "AllowMissing",
    "ConstraintParseError",
    "ConstraintValidationError",
    "Counterfactual",
    "Equals",
    "Explainer",
    "Freeze",
    "Implies",
    "Infeasible",
    "Linear",
    "MissingExtraError",
    "Monotone",
    "OneHot",
    "Range",
    "Target",
    "TargetError",
    "TreecfError",
    "UnsupportedModelError",
    "__version__",
    "constraint",
]
