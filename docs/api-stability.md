# API stability

**Status:** beta on PyPI. Until 1.0, breaking changes bump the minor
version and are listed in the changelog with the reasoning. Serialized
artifacts have their own, stronger promise: **every 0.x release reads every
earlier batch JSON file and both certificate schema versions (1 and 2)**,
enforced by committed golden files in CI.

## Public surface

The public API is exactly the export lists below; anything prefixed with
`_` or not exported is internal and may change without notice.

- `treecf.__all__`: `Explainer`, the result types (`Counterfactual`,
  `Infeasible`, `BatchResult`, `BatchRecord`, `RecourseRegion`), `Target`,
  the constraint objects (`Freeze`, `Monotone`, `Range`, `Linear`, `Equals`,
  `Implies`, `OneHot`, `AllowMissing`, `AllowedCategories`, the
  `constraint()` mini-language), `Plausibility`, `Grid`, constraint mining
  (`suggest_constraints`, `SuggestedConstraint`, `DataQualityFinding`),
  the fingerprints (`ir_fingerprint`, `constraints_fingerprint`), and the
  error taxonomy (`TreecfError`, `UnsupportedModelError`, `ParserError`,
  `MissingExtraError`, `ConstraintValidationError`, `ConstraintParseError`,
  `TargetError`, `TreecfWarning`).
- `treecf.constraints.__all__`: the constraint objects plus
  `CompiledConstraints`, `compile_constraints`, and `constraint`.
- `treecf.audit.__all__`: `build_certificate`, `check_certificate`,
  `ir_fingerprint`, `constraints_fingerprint`, `portfolio_report`.
- `treecf.viz.__all__` (extra `treecf[viz]`): `plot_changes`,
  `plot_counterfactuals`, `plot_ladder`, `plot_alternatives`,
  `plot_tradeoff`, `plot_recourse_map`, `plot_waterfall`, `plot_effort`,
  `plot_region`, `plot_certification_trace`.
- `treecf.viz_batch.__all__` (extra `treecf[viz]`): `plot_batch_levers`,
  `plot_batch_matrix`, `plot_batch_summary`, `plot_batch_deltas`,
  `plot_recourse_burden`, `recourse_burden_table`.

## The artifact promise

Three artifact kinds leave the library, all plain JSON, none ever unpickled:

- **Batch files** (`BatchResult.save`/`load`): every 0.x release reads every
  file an earlier 0.x release wrote; fields added later default when absent.
- **Certificates**: schema version 2 is current (it adds the certified
  category sets of a recourse region); `check_certificate` verifies versions
  1 and 2, and a committed version-1 golden file keeps that promise honest.
- **Model dumps** are inputs, not outputs — the parsers read the training
  libraries' own JSON formats.

Keys are added, never repurposed. A reader of a certificate or a batch file
must tolerate keys it does not know: a release may add a key to any block
(a new solver counter, a new region flag) without bumping the schema
version, and `check_certificate` verifies such a file exactly as it would
without the addition. Only removing a key, or changing what an existing key
means, bumps `schema_version`.

## Added in 0.3.1

New public symbols in this release, as one running list:

- `explain(..., search="refine")` / `explain_batch` / `explain_coalitions`:
  the coarse-to-fine exact search; `"classic"` stays the default and is
  unchanged. `Explainer.certificate(..., search=)` records the choice under
  `declared`.
- `explain(..., region_mode="maximal", region_budget=)` and
  `Explainer.recourse_region(mode=, budget=, keep_witnesses=)`: budgeted
  proof that a region side cannot grow, with witness points on request.
- `RecourseRegion.maximal` / `.maximal_categories` / `.witnesses`: per-side
  proof flags, per-feature category flags, and the witnesses that closed
  each side (`None` unless kept); certificates store the flags under
  `plan.region_maximal` and `plan.region_maximal_categories` without a
  schema bump.
- `RecourseRegion.integer_features`: the features under an `"integer"`
  value policy when the region was built; `describe()` phrases those on the
  integers, and every phrase is now strict where a rounded endpoint would
  overstate the box.
- `Explainer.search_profile(x, target=None)`: per-feature domain sizes and
  the total search-space size before an exact solve; the budget-exhaustion
  warning quotes that size.
- `solver_stats["search"]`, `["coarse_accepts"]`, `["refinements"]`,
  `["trace"]` on every exact result: the search mode, its counters, and the
  sampled incumbent/bound trace; certificates carry them as JSON lists.
- `treecf.audit.portfolio_report`: a batch-level audit report as JSON, a
  self-contained HTML page, or markdown with figures beside it.
- `treecf.viz.plot_certification_trace`: the incumbent and the proven lower
  bound against nodes expanded, ending at the named outcome.

## Added in 0.3.0

New public symbols in this release, as one running list (extend this list
rather than starting a new one for later additions in the same release):

- `AllowedCategories` (`treecf`): restrict a categorical feature to a set of
  category codes or names; see *Constraints*.
- `ParserError` (`treecf`): a model dump was recognized but cannot be parsed
  as given — the message names the argument to supply or the retraining
  recipe.
- `RecourseRegion.feature_categories` / `.category_names` / `.cat_sets`:
  certified category sets on regions over categorical features; certificates
  store them as schema version 2.
- `Explainer(categories=...)`: display names (and declared cardinalities)
  for categorical features; required for CatBoost models with native
  categorical features.
- `plot_region` (`treecf.viz`): the certified recourse region, per feature,
  with what stopped each bound.
- `plot_recourse_burden`, `recourse_burden_table` (`treecf.viz_batch`):
  recourse cost and availability by segment.
- `solver_stats["presolve_removed"]` / `["presolve_certified"]`: how many
  candidate states the exact backend's pre-search reachability filter
  removed, and whether it certified infeasibility outright.
