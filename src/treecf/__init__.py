"""treecf — constrained, threshold-aware counterfactual explanations for tree ensembles."""

from treecf._errors import (
    ConstraintParseError,
    ConstraintValidationError,
    MissingExtraError,
    TargetError,
    TreecfError,
    UnsupportedModelError,
)
from treecf.api import Counterfactual, Explainer, Grid, Infeasible
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
from treecf.mining import DataQualityFinding, SuggestedConstraint, suggest_constraints
from treecf.plausibility import Plausibility
from treecf.targets import Target

__version__ = "0.1.0.dev0"

__all__ = [
    "AllowMissing",
    "ConstraintParseError",
    "ConstraintValidationError",
    "Counterfactual",
    "DataQualityFinding",
    "Equals",
    "Explainer",
    "Freeze",
    "Grid",
    "Implies",
    "Infeasible",
    "Linear",
    "MissingExtraError",
    "Monotone",
    "OneHot",
    "Plausibility",
    "Range",
    "SuggestedConstraint",
    "Target",
    "TargetError",
    "TreecfError",
    "UnsupportedModelError",
    "__version__",
    "constraint",
    "suggest_constraints",
]
