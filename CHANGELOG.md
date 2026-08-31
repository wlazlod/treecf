# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- Release checklist:
  1. Run `uv run python scripts/bump_version.py X.Y.Z` (rewrites every version
     location, promotes [Unreleased], refreshes both lockfiles).
  2. Review the diff.
  3. Tag vX.Y.Z -> release.yml checks version consistency, builds, smoke-tests,
     and publishes.
-->

## [Unreleased]

### Added

- **Native categorical splits.** Models trained with native categorical support now parse
  exactly into set-membership IR nodes: LightGBM (`categorical_feature`), XGBoost
  (`enable_categorical`), scikit-learn `HistGradientBoosting` (`categorical_features`,
  including string categories via `categories=`), and CatBoost (`cat_features`; one-hot and
  single-feature-statistic splits, hashing reproduced bit-exactly). CatBoost models built with
  categorical feature *combinations* raise `ParserError` naming the `max_ctr_complexity=1`
  retraining recipe; the new `ParserError` type covers "recognized but unparseable as given".
- **`Explainer(categories=...)`.** Display names (and, where useful, declared cardinalities
  beyond training) for categorical features; required for CatBoost with native categorical
  features and for HistGradientBoosting trained on string categories, optional elsewhere.
- **Category blocks.** Every backend searches categorical features over routing-equivalence
  classes of codes, so search cost scales with how finely the ensemble partitions the feature,
  not its cardinality. Categorical distance is flat: any change of code costs one weighted
  unit.
- **`AllowedCategories` constraint.** Whitelist a categorical feature's codes by display name
  or raw code; order- and arithmetic-shaped constraints (`Range`, `Monotone`, `Linear`,
  `Equals`, `Implies`, `OneHot`) are rejected on categorical features at construction.
- **Categorical exact search and regions.** `backend="exact"` proves optimality and certified
  infeasibility over the block grid; `region=True` certifies *category sets* per categorical
  feature (`RecourseRegion.feature_categories` / `.category_names` / `.cat_sets`), stored in
  certificates as schema version 2. Schema version 1 certificates still verify, pinned by a
  committed golden file.
- **Presolve.** The exact backend filters each feature's candidate states by reachable score
  and plausibility brackets before branching; `solver_stats` gains `presolve_removed` and
  `presolve_certified`, and an emptied domain certifies infeasibility with zero nodes
  expanded. Results are bit-identical with presolve on; only node counts drop.
- **Visualization.** `plot_region` (the certified box, with per-bound cap markers and
  categorical tiles) and `plot_recourse_burden` / `recourse_burden_table` (feasible share and
  cost distribution by segment, kept side by side).
- **Docs.** Reader-oriented navigation (workflow guides, grouped concepts, split API pages,
  benchmarks and changelog pages); every fenced snippet executed in CI against a committed
  docs model; a structure test pins that no published URL disappears and every plot function
  ships a committed figure.
- `SECURITY.md`, `CONTRIBUTING.md`, `scripts/bump_version.py` (with a version-consistency
  test), and `#![forbid(unsafe_code)]` in the Rust core.

### Changed

- **Exact-search performance.** Presolve, a feature-to-trees index, and per-tree bracket
  caching in region growth. Measured before/after (same machine, same seeds, medians):

MEASURED_TABLE_PLACEHOLDER

### Invariants

- Numeric-model results are byte-identical to the previous release: fingerprints, solves,
  regions, and stored fixtures are unchanged, pinned by a dedicated invariance suite.
- No genetic or parity fixture was regenerated; exact fixtures were regenerated only under an
  equality guard asserting identical plans, distances, and proofs.
- The Python and Rust engines remain byte-identical on every solve, domain, and region,
  including the new categorical paths.

## [0.2.4] - 2026-08-23

### Added

- **Calibrator provenance in certificates.** For calibrated-space targets the certificate's
  `target.calibrator` block is now structured — `{embedded: false, fingerprint, type,
  buffer_logit}` — with the fingerprint duck-read from the calibrator's own `fingerprint()`
  (`null` when absent; probcal calibrators provide one). `check_certificate` accepts an
  optional `calibrator=` keyword: when given, the report gains `calibrator_match`, true only
  if the fingerprints agree and re-inverting the stored calibrated bounds through the passed
  calibrator reproduces the stored raw interval.
- **Calibrator provenance in batch records.** `BatchRecord.calibrator_fingerprint` repeats the
  target calibrator's fingerprint on every row, so each JSON line stays self-contained.
- **`score_calibrated` read-out.** `Counterfactual`, `BatchRecord`, and the certificate's
  `factual` block now carry the calibrator's probability at the result (and at the factual)
  for calibrated targets whose calibrator exposes `predict_proba`; `None` otherwise.
  Presentational only: the engine still optimizes and verifies against the raw interval.
- **Plateau-aware exactness tests.** Calibrated targets on and one float above step-calibrator
  plateau levels, cross-checked against brute-force enumeration in calibrated space and
  against real probcal isotonic/centered-isotonic fits.
- **probcal test matrix.** New optional `test` extra (and probcal in the `dev` extra):
  7 fitted probcal calibrators x target ops x buffer levels on sklearn and LightGBM models,
  every plan re-verified through the model and calibrator; dedicated CI job with pinned
  probcal + lightgbm. `src/` never imports probcal — the duck-typed protocol is unchanged.
- **Docs.** `concepts/calibration.md` gains provenance, read-out, and worked-example
  sections, and pins the guarantee that `explain_batch` calls `interval_inverse` exactly
  once per call (once per band for ladders), backed by counting tests.

### Compatibility

- Strictly additive. All new dataclass fields default to `None`; 0.2.x batch JSON and
  certificates load with the new fields defaulted. `check_certificate` without `calibrator=`
  produces byte-identical reports to 0.2.3. Calibrators missing optional duck members
  (`fingerprint`, `predict_proba`) degrade to `null`/`None`, never an error.

## [0.2.3] - 2026-08-23

### Fixed

- **sklearn `tree_`-based ensembles (RandomForest, GradientBoosting, IsolationForest) routed
  differently from sklearn itself at split boundaries**, because sklearn casts inputs to
  float32 before comparing against the float64 threshold while the IR evaluates in float64.
  A counterfactual whose coordinate landed exactly on a split threshold — the natural optimum
  of a smallest-change search, since `<=` cells are closed on the left — could flip through
  many trees at once: in the reproducing case (`GradientBoostingClassifier`,
  `subsample=0.8`), the exact backend stamped `proof="optimal"` on an `x_cf` whose true
  `decision_function` margin was 3.09 raw-score units away from the reported `score_raw`,
  silently violating the target. Thresholds are now re-expressed at parse time as the exact
  float64 boundary of the float32 cast (largest float64 `T` with `float32(T) <= t`,
  round-half-to-even handled), so float64 IR routing reproduces sklearn bit-for-bit for
  every input — search, certificates, and `score_raw` included. Verified by a 138k-probe
  property sweep, new unquantized conformance tests (exact-threshold and float64-neighbour
  probes; the old harness quantized all probes to the float32 grid, which is exactly why
  this never surfaced), and probcal's joint recourse scenarios. HistGradientBoosting
  predicts on the float64 grid and is unchanged; XGBoost also casts features to float32
  natively and should get the same treatment once a reproducing case is confirmed
  (follow-up).

### Internal

- Restructured a late-initialized binding in the exact search's proof/lower-bound
  epilogue (behavior-identical) — clippy 1.98's `needless_late_init` began rejecting
  the old form under `-D warnings` on the freshly installed stable toolchain in CI.

## [0.2.2] - 2026-08-19

### Added

- **Audit certificates**: `Explainer.certificate(x, result, target)` turns any stored
  `Counterfactual` or `Infeasible` (the certified "no" included) into a strict-JSON-serializable
  audit record — a reproducibility record plus a fresh verification. It binds the claim to a
  model fingerprint, a constraint fingerprint, and the solve parameters, and re-verifies the
  returned plan (score, target membership, constraint check, plausibility, sampled region
  points) at issue time; it does not cryptographically prove that a search ran or that a
  `proof="optimal"` claim is true — re-running with the recorded seed/budgets on a
  fingerprint-matching model is how a validator checks that. A certificate whose fresh
  verification fails is still issued with the failing checks recorded, plus a `TreecfWarning`
  naming them. `Explainer.check_certificate(cert)` is the validator's tool: it recomputes both
  fingerprints against the current explainer, re-runs the verification block, and reports
  (`model_match`/`constraints_match`/`verification_ok`/`mismatches`) without ever raising on a
  mismatch. The new `treecf.audit` module exposes the underlying `ir_fingerprint` and
  `constraints_fingerprint` (SHA-256 over canonical byte encodings — stable across Python
  versions, platforms, and dict ordering; a callable `value_policy` has no canonical encoding
  and marks the certificate `"reproducible": false` with a reason).
- `BatchRecord.proof` and `BatchRecord.solver_stats`: every batch record now carries the claim
  and (for exact solves) the diagnostics of the single-instance result that produced it —
  `Counterfactual.proof` values for feasible records, `Infeasible.proof`
  (`"search_exhausted"`/`"certified"`) for infeasibility markers. Genetic/python records carry
  empty stats (those engines report no per-row diagnostics). `BatchResult.to_frame` gains a
  `proof` column (`solver_stats` stays record-only); `save`/`load` round-trip both fields, and
  files from earlier versions load with feasibility-based defaults.

### Fixed

- The batch aggregate degraded-result warning pointed at "each result's own
  proof/solver_stats" while `BatchRecord` exposed neither field; the fields now exist, so the
  message is true as written (the wording itself is unchanged).

### Notes

- No solver behavior changes; no fixtures touched; no Rust source changes (only the mirrored
  version in `rust/Cargo.toml`/`Cargo.lock`).

## [0.2.1] - 2026-08-15

### Added

- **Interruptibility**: a `Ctrl-C` during an exact search, a certified-region growth, or a
  batch genetic solve now raises `KeyboardInterrupt` promptly instead of waiting for the
  whole search to finish. The Rust core polls for it from inside its released GIL (about
  every 2^18 nodes for the exact search); the pure-Python exact fallback already raised
  promptly, since Python delivers signals between bytecode instructions. Reliable only when
  the call happens on the main thread. Nothing is returned on interrupt -- whatever
  incumbent or partially grown region existed is discarded.
- The exact backend now always warns when it returns a degraded result
  (`solver_stats["completed"] is False`): a `TreecfWarning` names whether the search
  genuinely ran out of budget or instead withdrew its optimality certificate through a
  conservative constraint repair without touching the budget -- the two causes are never
  conflated, and the message includes a lower-bound/gap parenthetical when one is
  available, plus an unseeded-warm-start clause when the incumbent came from an unseeded
  warm pass. `Target.bands`, `explain_coalitions`, and `explain_batch` collapse every
  degraded solve in one call into a single aggregate warning instead of one per solve.
- `explain_batch(..., backend="exact")` is now opt-in behind `allow_exact_batch=True`:
  without it, raises `ValueError` naming a worst-case wall-time estimate (rows × plans ×
  `time_budget_s`) instead of running unbounded. Opting in also replaces `warm_start`'s
  per-row (or, in `diversity="seeds"` mode, per-attempt) genetic warm passes with a single
  vectorized warm pass shared across the whole batch -- in seeds mode this means every
  attempt of a row now shares one incumbent instead of each attempt warm-starting its own
  (with `n_per_example=1` the result still matches a sequential `explain(...,
  backend="exact")` call exactly).

### Changed

- README and the package's one-line description refreshed to cover the exact backend,
  certified infeasibility, and recourse regions (updates the PyPI project page on the next
  release).
- No result of any non-interrupted call changes in this release, and no fixtures were
  regenerated: the full pre-existing test suite passes unchanged.

### Fixed

- `Ctrl-C` latency during an exact search, region growth, or batch genetic solve used to
  equal however much of the search remained; it is now near-immediate (see
  Interruptibility above).

### Docs

- Every public API object rendered on the API reference now documents its parameters
  (with default semantics, not just default values), return shape, and deliberate raises
  to the same depth, with cross-references to the relevant concepts pages; several
  objects that previously had no docstring at all (`Target.raw`/`bands`,
  `BatchResult.for_id`/`save`/`load`, `suggest_constraints` and its result types,
  `Plausibility.isolation_forest`/`anomaly_score`) were entirely absent from the rendered
  docs and are now covered.
- README quick-look comment recalibrated to name the warned-degrade case alongside
  `proof="optimal"` and a certified "no"; new [Certification](https://wlazlod.github.io/treecf/concepts/certification/)
  sections cover interruption and the always-on degraded-result warning.

## [0.2.0] - 2026-08-14

### Added

- **`backend="exact"`**: a branch-and-bound search over the same routing-atomic cell grid the
  genetic backend shares, reporting `proof="optimal"` (no cheaper feasible row exists) or
  `proof="optimal_within_gap"` (within a `gap > 0` relative fraction of the optimum) instead of
  `"heuristic"`. Rust-first with a byte-identical pure-Python fallback when the extension is not
  importable. `warm_start` (default `True`) seeds the search with a short genetic pass;
  `node_budget` (default 2,000,000 assignments) and `gap` (default `0.0`) trade proof strength
  against wall time. Supports single-feature `Linear` constraints and the canonical two-feature
  order pair exactly; any other multi-feature `Linear` or a callable `value_policy` raises
  `ConstraintValidationError` naming `backend="genetic"` as the fallback.
- **Certified infeasibility**: `Infeasible.proof="certified"` from the exact backend means the
  whole reachable grid was tried and every row rejected — not merely that a budget ran out. New
  `Infeasible.proof` (`"search_exhausted"` | `"certified"`) and `Infeasible.solver_stats` fields;
  both default so existing code is unaffected.
- **Recourse regions**: `explain(..., region=True)` (also on `explain_coalitions`/
  `explain_batch`) widens a verified counterfactual into a `RecourseRegion` — a per-feature box
  where every point is certified feasible by interval arithmetic over the whole box, not
  sampled. Works with every
  backend via `Explainer.recourse_region`; `Counterfactual.region` carries it, and
  `BatchRecord.region` persists it through batch save/load.
- `plot_recourse_map`: one-axes map of a single applicant's recourse options — model
  output on x, recourse cost J on y — with the accept band, an arrow per plan, infeasible
  coalitions marked, and a `schematic=True` slide-style mode.
- Docs: new [Certification](https://wlazlod.github.io/treecf/concepts/certification/) concepts page covering the proof
  taxonomy, what a certificate does and does not cover, and the region layer's guarantees.

### Changed

- PyPI development-status classifier raised to `4 - Beta`.
- Rust core: rand upgraded 0.9 → 0.10 (with rand_distr 0.6 and rand_pcg 0.10).
  Seeded runs stay deterministic for a given treecf version, but the random
  stream may differ from builds against rand 0.9, so genetic-search results for
  the same seed can change across this upgrade.
- The genetic backend itself is unchanged by this release: the full pre-existing test suite
  passes without fixture regeneration.

### Fixed

- Derived per-feature bounds from single-feature linear constraints now include the
  linear check's tolerance, so they no longer exclude counterfactuals the
  constraint itself admits (previously possible with very small or very large
  coefficients).

## [0.1.1] - 2026-08-08

### Added

- `TreecfWarning`, emitted when a factual violates its constraints — once per
  `explain` call, and as a single per-constraint aggregate in `explain_batch`.
  The warning spells out that the returned plan includes changes made solely
  to satisfy the violated constraints.
- Derived per-feature bounds for single-feature `Linear` constraints
  (`constraint("income >= 100")` now clips candidates like the equivalent
  `Range`); vacuous zero-coefficient linears are dropped, unsatisfiable ones
  rejected at compile time.
- Declared Rust MSRV (rustc 1.86) in `rust/Cargo.toml` with an enforcing CI
  job; building from the sdist needs 1.86+, wheels need no toolchain.
- Wheel smoke tests in the release workflow: every runnable wheel target is
  installed into a fresh venv (musllinux inside an Alpine container) and runs
  one `explain` per backend before upload.
- `CITATION.cff` version is now checked against `treecf.__version__` in the
  test suite.

### Fixed

- Satisfiable `Linear` constraints whose feasible set lies far from the
  factual no longer come back `Infeasible`: single-feature linears lower into
  bounds, and multi-feature linears get halfspace-projection repair.
- `apply_link` no longer raises `OverflowError` for raw scores below ≈ −710;
  mid-range sigmoid outputs are bit-for-bit unchanged.
- `CITATION.cff` and `rust/Cargo.toml` version drift (both said 0.0.1 while
  the released package was 0.1.0).

### Changed

- Repair for non-canonical linear constraints now runs a 3-round cyclic
  halfspace projection; **seeded results from 0.1.0 that involve such
  constraints are not reproducible in 0.1.1**. The canonical order-pair
  repair (`a - b <= 0`) is unchanged, and the existing parity fixtures
  regenerated byte-identical; a new `11-linear-projection` fixture pins the
  projection behavior.

## [0.1.0] - 2026-07-23

### Added

- **Calibrated targets**: `Target.calibrated(calibrator, ...)` expresses the
  target on the post-hoc *calibrated* probability scale and lazily inverts it
  through the calibrator's duck-typed generalized inverse
  (`interval_inverse(lo, hi, *, space="logit", buffer_logit=...)` +
  `is_monotone_`) — no calibration-library dependency. `Target.bands` accepts
  `space="calibrated"` with `calibrator=`/`buffer_logit=` for masterscales
  defined on calibrated PD. `Target.probability` now documents that it targets
  the *uncalibrated* model probability.

### Fixed

- **`Target.band_intervals` field propagation**: per-band targets were rebuilt
  from `(space, lo, hi)` only, silently dropping any other field — now all
  fields propagate (surfaced by the calibrated-bands work).

- **Competitor benchmark**: `scripts/bench_vs_competitors.py` (PEP 723,
  self-contained via `uv run`) compares treecf with DiCE and NICE on two
  model scales; results published in *Backends and proofs* — 8–3400× faster
  than DiCE with far cheaper plans, cheapest plans overall, 157 rows/s batch
  production on the medium model; NICE's per-instance speed and treecf's own
  misses reported as-is.
- **Post-solve pruning**: every returned plan now drops changes that
  verification proves unnecessary (cheapest first, each revert re-verified in
  float space). The search's revert-to-factual mutation is stochastic, so a
  stalled run could ship a residual micro-change that crossed no decision
  threshold — pure distance cost with zero score effect.
- `CITATION.cff`.

### Changed

- Compiled extensions are no longer tracked in git (history rewritten to drop
  the committed `.so`; wheels come from CI, local builds via maturin).
- Publish steps skip files already on the index, making tag-triggered
  re-releases idempotent; retroactive `v0.0.1` tag and GitHub release created.
- PyPI keywords no longer mention the removed CP-SAT backend; README/docs
  state the published version (0.0.1) consistently.

## [0.0.1] - 2026-07-13

First published release (PyPI). Version deliberately resets BELOW 0.1.0 (which
was never published): the Rust-backed rebuild supersedes the prior pure-Python
implementation outright and restarts the version line.

### Changed

- **The genetic backend runs on a Rust core by default** (44-58x faster
  than the numpy implementation on realistic workloads; 24.6x single-threaded).
  `backend="genetic"` uses Rust; the pure-Python GA remains available as
  `backend="python"`.
- Packaging switched from hatchling (pure Python) to maturin (single mixed
  Rust/Python package). Installing from source now requires a Rust toolchain;
  platform wheels are built in CI. The numpy-only-core dependency policy ends;
  runtime Python dependencies are unchanged (numpy only).
- **Release**: platform wheels now include **linux-aarch64** (Graviton, Docker
  on Apple Silicon) alongside linux/musllinux x86_64, macOS arm64/x86_64, and
  Windows x64; the PyPI description no longer mentions the removed CP-SAT
  backend.
- **Docs**: the standalone Benchmarks page is gone; the headline numbers, the
  single-core explanation, and the batch-parallelism caveat now live in a
  "Performance" section of *Backends and proofs*. Full protocol and
  reproduction stay in `scripts/bench_genetic.py` / `scripts/bench_batch.py`.
- **`explain_batch` runs its solves in parallel inside the Rust core**: the
  seeds path solves one wave of independently seeded attempts per Rust call
  (rayon across tasks, GIL released) and lever-blocking batches all primary
  solves; per-wave verification scores come from one vectorized IR pass.
  Records are identical to the former sequential per-row loop (same seeds,
  dedup, and stopping rule), with one caveat: a solve that hits its
  per-task `time_budget_s` under core contention may stop at a different
  generation than it would sequentially. Also: routing-atomic cells are now
  cached on the Rust ensemble instead of rebuilt per solve, and
  lever-blocking clones reuse the parent's marshaled Rust ensembles.
  ~1.7x batch throughput on a 4-core machine (`scripts/bench_batch.py`);
  the gain grows with core count.

### Added

- **Coalitions mode (opt-in)**: `Explainer.explain_coalitions(x, target,
  coalitions={...}, include_full=False)` produces one counterfactual per named
  feature group, each solve allowed to change only that group (everything else
  frozen); `Infeasible` per group means that group alone cannot reach the
  target. `explain_batch(..., diversity="coalitions")` scales it to datasets
  (one record per group per row; new `coalition` field on `BatchRecord`,
  persisted and exposed in `to_frame()`). `plot_alternatives`/`plot_tradeoff`
  accept the outcome mapping directly, labeling plans by coalition name.
  Never the default mode. Documented in a new Concepts page, a "Grouped
  recourse" section of How it works, and a tutorial section.
- **Single-instance comparison plots** (`treecf.viz`): `plot_alternatives`
  (every alternative plan's changes on shared axes, one color per plan,
  σ-standardized with an explainer) and `plot_tradeoff` (cost vs achieved
  score per plan, with target lines). Both accept `Counterfactual` objects or
  feasible `BatchRecord` entries.
- **Docs**: pipeline and genetic-loop diagrams (Mermaid) in "How it works";
  reorganized Home and Getting started (single install section, alternatives
  walkthrough, "where next" links), pipeline-ordered Concepts nav, and the
  stale `proof` values from the removed CP-SAT era corrected.
- **Batch visualizations** (`treecf.viz_batch`, `[viz]` extra): `plot_batch_levers`
  (which levers plans use, by direction, with essential-lever annotations),
  `plot_batch_matrix` (plans × features heatmap, effort-shaded with an
  explainer), `plot_batch_summary` (cost / sparsity / feasibility panel), and
  `plot_batch_deltas` (per-lever delta distributions, σ-standardized with an
  explainer). Demonstrated in the credit-risk tutorial.
- **Docs**: long-form ["How treecf finds counterfactuals"](https://wlazlod.github.io/treecf/how-it-works/)
  article walking one applicant from objective to verified counterfactual;
  MathJax wired into the docs build for the objective and plausibility formulas.
- **Batch production**: `Explainer.explain_batch(X, target, n_per_example=k,
  diversity="seeds"|"lever-blocking", ids=...)` mass-produces counterfactuals
  for a dataset (~ms/row via the Rust engine); `BatchResult` persists to
  portable JSON (`save`/`load`), supports `for_id` lookup and a lazy-pandas
  `to_frame()`. Lever-blocking mode also records per-row *essential levers*.
- **New visualizations**: `plot_waterfall` (SHAP-style waterfall of exact
  score deltas per change, cutoff line, probability space for sigmoid models)
  and `plot_effort` (decomposition of the distance J across changes).
- `treecf._treecf_core` extension: tree-IR evaluation (bitwise-identical to
  the Python evaluator), constraint check/repair (bitwise-identical), and the
  genetic algorithm (statistically indistinguishable across 200 seeds x 10
  scenarios; every result float-verified in Python).
- Parity harness: flat-array cross-language contract
  (`treecf.ir.flatten`, `treecf.constraints.flatten`), golden per-seed
  fixtures and 200-seed distributional baselines under tests/fixtures/parity/.

### Removed

- **The exact CP-SAT backend, entirely**: `backend="cpsat"`, the `[cpsat]`/ortools
  extra, the AIM integer encoding, the HiGHS stub, optimality proofs
  (`proof="optimal"`), `n_counterfactuals`/diversity cuts, infeasibility
  `relaxation_hint`, and the bands single-compilation amortization. The
  genetic engines are the sole backends (`"genetic"` = Rust default,
  `"python"` = numpy reference); `Target.bands` still works (one search per
  band). Users needing provable optimality should pair the IR with a
  dedicated exact-optimization package. The brute-force oracle remains the
  test-suite's optimality bracket.

### Fixed

- Counterfactual values adjacent to open cell bounds now step one **float32**
  ulp inside (previously float64): a float64-ulp neighbour of a threshold
  collapses onto it in native float32 comparisons, so the deployed model
  could route such values opposite to the IR. Both engines changed
  identically; parity fixtures regenerated.

## unreleased

### Added

- Parser breadth: LightGBM / sklearn (RF, GB, HistGB) / CatBoost parsers, all
  conformance-gated; isolation-forest plausibility as a hard constraint;
  `Target.bands` rating ladder (one compilation, N solves); diverse
  counterfactuals via no-good cuts; infeasibility relaxation hints;
  `suggest_constraints` data mining with transitive reduction and
  near-invariant data-quality findings; `viz` module
  (`plot_changes`/`plot_counterfactuals`/`plot_ladder`).
- Genetic backend: numpy-only constrained GA (Deb ranking, seeded,
  `proof="heuristic"`), vectorized constraint check/repair, cross-backend
  soundness suite.
- Constraint layer: string sugar parser, `Linear`/`Equals`/`Implies`/
  `OneHot`/`AllowMissing`, NaN as a first-class counterfactual value,
  per-feature value policies with cell-safe snapping.
- Vertical slice: XGBoost (object/JSON dump) → tree IR → routing-atomic
  cells → CP-SAT → provably optimal counterfactual, with `Freeze`/`Monotone`/
  `Range` constraints, raw/probability targets, MAD-chain normalizers,
  float-space verification with K×10 retry, and a brute-force exactness oracle
  gating the backend (50-case randomized suite).
- Release engineering: CI conformance matrix over library versions,
  mkdocs-material docs with three executed tutorial notebooks
  (quickstart, credit-risk walkthrough, no-solver environments),
  performance smoke benchmark, clean-venv packaging verification.
- Project skeleton: packaging, CI, docs infrastructure.

### Known limitations

- CP-SAT solve time misses the <1s target at 300+ trees (~40s median on the
  benchmark suite); planned v0.2 optimization via table-constraint encoding.
- Plausibility cannot combine with AllowMissing/NaN factuals.
- `n_counterfactuals > 1` requires the CP-SAT backend.
