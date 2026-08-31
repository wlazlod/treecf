# Categorical features

A tree ensemble trained with native categorical support splits on *set
membership* — "occupation ∈ {clerk, manager} goes left" — rather than on a
threshold. treecf parses those splits exactly and treats the feature as what
it is: an unordered set of codes, with no arithmetic, no ordering, and no
"between two categories". This page is the semantics; the per-workflow
surface is spread across the [models](../guide/models.md),
[constraints](../guide/constraints.md), and [certify](../guide/certify.md)
guides.

## Representation and routing

A categorical feature holds integer *codes* stored as floats, like every
other column. A set-membership node routes a row left exactly when its code
is in the node's stored set; three edge cases are fixed by the training
libraries' own behavior, verified by conformance suites:

- **NaN** follows the node's missing direction, the same mechanism as
  numeric splits.
- A **non-integral value** is never a member of any set — it routes as
  out-of-set, matching how the libraries treat it.
- An **unseen code** (valid integer, never encountered in training) routes
  out-of-set — which is exactly what makes declaring extra categories via
  `categories=` sound: the model has one, fixed answer for all of them.

`res.changes` reports a categorical change as a code pair, and every
user-facing surface — plots, certificates, `RecourseRegion` — uses the
display names when `categories=` provided them.

## Category blocks

The model cannot distinguish two codes that appear on the same side of every
split that mentions the feature. treecf groups codes into these
routing-equivalence classes — *category blocks* — and searches over blocks,
not codes:

- The genetic and exact backends both try one representative per block
  (plus the factual's own code), so the search space scales with how finely
  the ensemble actually partitions the feature, not with its cardinality —
  200 merchant codes falling into 7 blocks cost the search 7 candidates.
- Two codes in one block are provably interchangeable *to the model*;
  constraints can still separate them (an `AllowedCategories` whitelist may
  admit only part of a block, and the search then uses the smallest allowed
  member as the block's representative).

Blocks are the categorical analog of the numeric [cells](cells.md) grid, and
the exact backend's optimality proof quantifies over them the same way: a
claim over every block is a claim over every code.

## Cost

Categorical distance is flat: any change of code costs the same one unit
(scaled by the feature's weight), because there is no meaningful magnitude
between codes. Changing occupation from `student` to `clerk` costs exactly
what changing it to `retired` costs; sparsity pressure, not distance, decides
whether the feature moves at all.

## Constraints

`Freeze` and `AllowedCategories` are the two constraint forms on a
categorical feature — the first pins it, the second whitelists codes (by
display name or raw code). Order- and arithmetic-shaped constraints
(`Range`, `Monotone`, `Linear`, `Equals`, `Implies`, `OneHot`) are rejected
at construction with `ConstraintValidationError`, because their semantics
presuppose an order the feature does not have. The `OneHot` constraint
remains the right tool for *manually* one-hot-encoded columns — native
categorical features never need it.

## Certified regions and certificates

A [certified region](certification.md#regions-certified-not-maximal-not-monotone)
over a categorical feature is a *set of codes*, every member of which keeps
every row in the region feasible: `RecourseRegion.feature_categories` maps
the feature to the certified codes, `category_names` to their display names.
Certificates store these sets (as the current schema version); a
version-1 certificate — regions over numeric features only — still verifies.

## Per-library parsing

| Library | Trained with | What treecf reads |
|---|---|---|
| LightGBM | `categorical_feature=[...]` | Set splits from the dump; category names recovered from pandas categoricals when present; NaN always routes right at a categorical split (LightGBM's own rule) |
| XGBoost | `enable_categorical=True` | Set splits from the JSON dump, including its right-child set encoding and per-node missing direction |
| sklearn `HistGradientBoosting` | `categorical_features=[...]` | Bitset splits, mapped back through the estimator's own encoder and column permutation; string categories need `categories=` to name the codes |
| CatBoost | `cat_features=[...]` | One-hot splits and single-feature target statistics, reproduced bit-exactly (including CatBoost's hashing); `categories=` is required, and feature *combinations* raise `ParserError` with the `max_ctr_complexity=1` retraining recipe |

In every case the conformance gate is the same as for numeric models: IR
evaluation must match native prediction on ≥10k probes, including NaN,
unseen-code, and non-integral inputs — a parse that cannot guarantee parity
raises instead of approximating.

## What stays unchanged

Numeric-only models are entirely untouched by all of this: their parsing,
routing, cost, fingerprints, and results are byte-identical to a
categorical-unaware build — pinned by an invariance test suite, not just
asserted.
