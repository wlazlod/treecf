# Constrain the search

Constraints are what make a counterfactual a *plan* rather than an
adversarial example: they encode what can change, in which direction, and
what must remain consistent. They are declared once on the `Explainer` and
compiled for every backend identically — the semantics never depend on which
engine runs. This page is the working catalog;
[Constraints](../concepts/constraints.md) covers compilation and violation
handling.

## Declaring

```python
# docs: no-run — the catalog; feature names stand in for your own schema
from treecf import (
    AllowedCategories, Equals, Freeze, Implies, Monotone, OneHot, Range, constraint,
)

Freeze("age_of_bureau_file")                    # immutable
Monotone("age", "increase")                     # directional
Range("utilization", 0.0, 1.5)                  # hard bounds
constraint("max_dpd_30d <= max_dpd_12m")        # string sugar -> Linear
constraint("2*a - b <= c + 5")                  # any linear expression
Implies(Equals("has_mortgage", 0), Equals("mortgage_balance", 0))
OneHot(("channel_web", "channel_app", "channel_branch"))
AllowedCategories("occupation", ["clerk", "manager"])   # categorical whitelist
```

Pass them at construction:

```python
# X_bg: the docs background matrix; model construction as in the models guide
# docs: no-run — model.json stands in for your own dump file
from treecf import Explainer, Freeze, Monotone

exp = Explainer("model.json", background=X_bg,
                constraints=[Freeze("tenure_months"), Monotone("dpd_12m", "decrease")])
```

## Categorical features

A feature with native categorical splits is unordered: `Freeze` and
`AllowedCategories` are its two constraint forms, and `AllowedCategories`
accepts display names (when `categories=` named them) or raw codes. Order-
and arithmetic-shaped constraints (`Range`, `Monotone`, `Linear`, `Equals`,
`Implies`, `OneHot`) are rejected on categorical features at construction
with `ConstraintValidationError` — there is no order to be monotone in. See
[Categorical features](../concepts/categorical.md).

## Missing values

`AllowMissing("feature")` opts a feature into NaN as a counterfactual value,
with a transition cost; the interaction with linear constraints and the NaN
routing rules are in [Missing values](../concepts/missing-values.md).

## Mining candidates from data

`suggest_constraints` scans the background for near-invariant order pairs
and data-quality findings, and returns candidates you accept explicitly —
mined rules are suggestions, never silently applied:

```python
# exp, X_bg: the docs explainer and its background rows
import treecf

suggestions = treecf.suggest_constraints(X_bg, feature_names=exp.ir.feature_names)
accepted = [s.constraint for s in suggestions if s.kind == "order"]
```

## Next

With the levers declared, [run the search](explain.md); the target came from
[set the target](targets.md).
