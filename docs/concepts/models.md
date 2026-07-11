# Models and the tree IR

Every supported model is parsed into one intermediate representation
(`EnsembleIR`): trees of `(feature, threshold, op, missing_left)` nodes plus a
raw-space intercept and an output link. Backends only ever see the IR — no
backend touches a native model object.

## Supported inputs

| Library | Native objects | Dump input | Notes |
|---|---|---|---|
| XGBoost | `Booster`, sklearn wrappers | `save_model("*.json")` path or dict | `binary:logistic`, `reg:squarederror`; LT convention |
| LightGBM | `Booster`, sklearn wrappers | `dump_model()` dict or its JSON | `binary`, `regression`; LE convention; `zero_as_missing` and categorical splits raise |
| CatBoost | classifier/regressor | `save_model(format="json")` | oblivious trees expanded to binary trees |
| scikit-learn | RF, GB, HistGB | — | see raw-score semantics below |

Unsupported constructs (multiclass, dart/gblinear, native categorical splits)
raise `UnsupportedModelError` — parsers never degrade silently. Every parser is
gated by a conformance suite that compares IR evaluation against native
predictions on ≥10k probes including NaN patterns and threshold-adjacent points.

## Raw-score semantics per family

- **XGBoost / LightGBM / CatBoost / GradientBoosting / HistGradientBoosting**:
  raw score = margin (log-odds for binary classifiers, SIGMOID link).
- **RandomForestClassifier**: the raw score is the *averaged class-1
  probability* with an IDENTITY link. Use `Target.raw(range=(0.0, 0.3))` — a
  `Target.probability` on a forest raises, because there is no sigmoid to invert.

## Float32 pitfalls handled for you

GBDT libraries store thresholds and inputs as float32 and their JSON dumps use
shortest-round-trip decimals; treecf casts thresholds back through float32 where
the library compares in float32, and handles LightGBM's zeroing of values with
magnitude below 1e-35. Without this, counterfactual values equal to a threshold
would route differently in the deployed model.
