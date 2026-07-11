# FAQ

**Why do I get `MissingExtraError` when calling `explain`?**
Backend selection is explicit and treecf never falls back silently (a
heuristic answer silently substituted for a proven-optimal one would be a
correctness bug in a credit process). Install the extra the error names —
`pip install treecf[cpsat]` — or explicitly choose `backend="genetic"`.

**Why does `Target.probability` fail on my RandomForest?**
Forest classifiers average probabilities; there is no sigmoid link to invert.
Their raw score *is* the averaged probability — use
`Target.raw(range=(0.0, 0.3))`.

**Why is my counterfactual `Infeasible`?**
Read `reason` and `relaxation_hint`: the hint reports whether the target is
unreachable even unconstrained, or whether per-feature bounds or relational
constraints block it.

**Can I run treecf where xgboost/ortools cannot be installed?**
Yes. Parsers accept JSON dumps (`Booster.save_model("model.json")`,
`dump_model()`, CatBoost `format="json"`), and the genetic backend needs only
numpy: `pip install treecf` on the scoring host, ship the dump file.

**Are mined constraints safe to apply automatically?**
No, by design. They are sample invariants, not domain truths; the API returns
them for review (`as_code()`), and near-invariants are flagged as data-quality
findings instead of constraints.

**Do NaN flips count as "changes" for sparsity and diversity?**
Yes — flipping a value to NaN (or back) increments `n_changed`, pays the
configured delta, and counts in `distinct_changes` diversity cuts.
