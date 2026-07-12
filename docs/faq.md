# FAQ

**Why does `Target.probability` fail on my RandomForest?**
Forest classifiers average probabilities; there is no sigmoid link to invert.
Their raw score *is* the averaged probability — use
`Target.raw(range=(0.0, 0.3))`.

**Why is my counterfactual `Infeasible`?**
The search exhausted its budget without a candidate satisfying the target and
every constraint. Check for contradictory constraints (e.g. everything frozen),
an unreachable target interval, or raise `time_budget_s`.

**Can I run treecf where xgboost cannot be installed?**
Yes. Parsers accept JSON dumps (`Booster.save_model("model.json")`,
`dump_model()`, CatBoost `format="json"`), and the genetic backend has no
dependencies beyond the wheel itself: `pip install treecf` on the scoring host,
ship the dump file.

**What is the Rust core, and do I need a Rust toolchain?**
`backend="genetic"` runs a compiled Rust engine bundled inside the platform
wheel (44–58× faster than the equivalent numpy implementation — see the
[benchmarks](benchmarks-genetic-rust.md)). Installing from a wheel needs no
toolchain; only building from the sdist compiles Rust. The engine is held to
bitwise parity with Python on tree evaluation and constraint checking, and to
statistical parity on end-to-end GA outcomes; every result is float-verified
in Python before being returned.

**When would I use `backend="python"`?**
It is the original numpy implementation of the same genetic algorithm, kept as
a reference engine (and as the behavioral baseline the Rust core is tested
against). Use it to cross-check results or in environments where the compiled
extension cannot load; expect identical result quality, just slower.

**Are mined constraints safe to apply automatically?**
No, by design. They are sample invariants, not domain truths; the API returns
them for review (`as_code()`), and near-invariants are flagged as data-quality
findings instead of constraints.

**Do NaN flips count as "changes" for sparsity and diversity?**
Yes — flipping a value to NaN (or back) increments `n_changed`, pays the
configured delta, and counts in `distinct_changes` diversity cuts.
