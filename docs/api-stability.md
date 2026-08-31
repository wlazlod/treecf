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
  `ir_fingerprint`, `constraints_fingerprint`.
- `treecf.viz.__all__` (extra `treecf[viz]`): `plot_changes`,
  `plot_counterfactuals`, `plot_ladder`, `plot_alternatives`,
  `plot_tradeoff`, `plot_recourse_map`, `plot_waterfall`, `plot_effort`,
  `plot_region`.
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
