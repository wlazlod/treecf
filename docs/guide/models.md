# Bring your model

The first step of every treecf workflow: hand `Explainer` a trained tree
ensemble. treecf parses it into its own intermediate representation once, at
construction, and never touches the native object again — so anything on this
page is settled before the first `explain` call.

## What you can pass

`Explainer(model, ...)` accepts a native model object, a dump `dict`, or a
path to a dump file:

| Library | Native objects | Dump input | Native categorical splits |
|---|---|---|---|
| XGBoost | `Booster`, sklearn wrappers | `save_model("*.json")` path or dict | `enable_categorical=True` models |
| LightGBM | `Booster`, sklearn wrappers | `dump_model()` dict or its JSON | `categorical_feature` models |
| CatBoost | classifier/regressor | `save_model(format="json")` | one-hot and single-feature-statistic splits; `categories=` required |
| scikit-learn | RandomForest, GradientBoosting, HistGradientBoosting | — | `HistGradientBoosting` with `categorical_features` |

JSON dumps parse without the training library installed — a scoring
environment that holds only the dump file and `treecf` (numpy-only) can
explain the model. The docs' own explainer is built exactly that way, from a
committed LightGBM dump:

```python
# docs: no-run — model.json stands in for your own dump file
from treecf import Explainer

exp = Explainer(
    "model.json",
    background=X_bg,
    categories={"occupation": OCCUPATIONS},
)
```

```python
# exp: the docs explainer, itself built from a committed LightGBM dump
sorted(exp.ir.categorical)   # feature indices with native categorical splits
```

Binary classifiers (`binary:logistic`, LightGBM `binary`, CatBoost
`Logloss`, sklearn classifiers) and regression objectives are supported;
multiclass, dart, and gblinear raise `UnsupportedModelError` — parsers never
degrade silently. The IR-level details, per-family raw-score semantics, and
the float32 pitfalls the parsers absorb are in
[Models and the tree IR](../concepts/models.md).

## Categorical features and `categories=`

A model trained with native categorical splits parses into set-membership
nodes, and treecf treats those features as unordered codes end to end — see
[Categorical features](../concepts/categorical.md) for the semantics. The
`categories=` argument maps a feature name to its display names, in code
order:

- **Optional** for LightGBM, XGBoost, and integer-coded
  HistGradientBoosting: the dump already fixes the code order; `categories=`
  adds names (for `res.changes`, plots, certificates) and may *extend* the
  cardinality beyond what training saw.
- **Required** for CatBoost models with native categorical features, and for
  HistGradientBoosting trained on string categories: those models identify
  categories by hash or by encoder state, so treecf needs the explicit list
  to know which real-world values the codes stand for. Omitting it raises
  `ParserError` with the exact argument to supply.

A CatBoost model that used categorical feature *combinations* cannot be
parsed exactly; the `ParserError` names the retraining recipe
(`max_ctr_complexity=1`).

## Background data

`background=` is a sample of real rows (a few hundred is plenty). It powers
the per-feature cost normalizers, [constraint mining](constraints.md), and
the genetic backend's initialization. Categorical columns hold the integer
codes as floats, like every other column — validation rejects non-integral
or out-of-range codes at construction with the feature name in the message.

## Verify the parse

Every parser is gated by a conformance suite comparing IR evaluation against
native predictions on ≥10k probes, including NaN patterns and
threshold-adjacent points, and every counterfactual any backend returns is
re-verified through the parsed model in float space before you see it. A
parse that cannot guarantee parity raises rather than approximating.

## Next

With the model parsed, state what the model's output should become:
[set the target](targets.md).
