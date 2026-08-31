"""treecf — constrained, threshold-aware counterfactual explanations for tree ensembles."""

from treecf._errors import (
    ConstraintParseError,
    ConstraintValidationError,
    MissingExtraError,
    ParserError,
    TargetError,
    TreecfError,
    TreecfWarning,
    UnsupportedModelError,
)
from treecf.api import Counterfactual, Explainer, Grid, Infeasible
from treecf.audit import constraints_fingerprint, ir_fingerprint
from treecf.batch import BatchRecord, BatchResult
from treecf.constraints import (
    AllowedCategories,
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
from treecf.regions import RecourseRegion
from treecf.targets import Target

__version__ = "0.3.0"

__all__ = [
    "AllowMissing",
    "AllowedCategories",
    "BatchRecord",
    "BatchResult",
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
    "ParserError",
    "Plausibility",
    "Range",
    "RecourseRegion",
    "SuggestedConstraint",
    "Target",
    "TargetError",
    "TreecfError",
    "TreecfWarning",
    "UnsupportedModelError",
    "__version__",
    "constraint",
    "constraints_fingerprint",
    "ir_fingerprint",
    "suggest_constraints",
]
