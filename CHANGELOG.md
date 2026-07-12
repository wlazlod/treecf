# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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
- **Docs**: long-form ["How treecf finds counterfactuals"](docs/how-it-works.md)
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

### Removed

- **The exact CP-SAT backend, entirely** (per the migration spec's §3.3,
  deferred by D-H1 and now resolved): `backend="cpsat"`, the `[cpsat]`/ortools
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

## [0.0.1] - 2026-07-12

Version deliberately resets BELOW 0.1.0 (which was never published): per the
Rust-migration spec, the Rust-backed rebuild supersedes the prior pure-Python
implementation outright and restarts the version line.

### Changed

- **The genetic backend now runs on a Rust core by default** (44-58x faster
  than the numpy implementation on realistic workloads; 24.6x single-threaded
  — see docs/benchmarks-genetic-rust.md). `backend="genetic"` uses Rust;
  the pure-Python GA remains available as `backend="python"`.
- Packaging switched from hatchling (pure Python) to maturin (single mixed
  Rust/Python package). Installing from source now requires a Rust toolchain;
  platform wheels are built in CI. The numpy-only-core dependency policy ends;
  runtime Python dependencies are unchanged (numpy only).

### Added

- `treecf._treecf_core` extension: tree-IR evaluation (bitwise-identical to
  the Python evaluator), constraint check/repair (bitwise-identical), and the
  genetic algorithm (statistically indistinguishable across 200 seeds x 10
  scenarios; every result float-verified in Python).
- Stage A parity harness: flat-array cross-language contract
  (`treecf.ir.flatten`, `treecf.constraints.flatten`), golden per-seed
  fixtures and 200-seed distributional baselines under tests/fixtures/parity/.

### Unchanged

- CP-SAT backend, constraint layer, targets, mining, viz, docs — the entire
  0.1.0 feature set carries over. CP-SAT's future is a separate decision.

## [0.1.0] - 2026-07-11 (never published)

### Added

- M4 breadth: LightGBM / sklearn (RF, GB, HistGB) / CatBoost parsers, all
  conformance-gated; isolation-forest plausibility as a hard constraint;
  `Target.bands` rating ladder (one compilation, N solves); diverse
  counterfactuals via no-good cuts; infeasibility relaxation hints;
  `suggest_constraints` data mining with transitive reduction and
  near-invariant data-quality findings; `viz` module
  (`plot_changes`/`plot_counterfactuals`/`plot_ladder`).
- M3 genetic backend: numpy-only constrained GA (Deb ranking, seeded,
  `proof="heuristic"`), vectorized constraint check/repair, cross-backend
  soundness suite.
- M2 constraint layer: string sugar parser, `Linear`/`Equals`/`Implies`/
  `OneHot`/`AllowMissing`, NaN as a first-class counterfactual value,
  per-feature value policies with cell-safe snapping.

- M1 vertical slice: XGBoost (object/JSON dump) → tree IR → routing-atomic
  cells → CP-SAT → provably optimal counterfactual, with `Freeze`/`Monotone`/
  `Range` constraints, raw/probability targets, MAD-chain normalizers,
  float-space verification with K×10 retry, and a brute-force exactness oracle
  gating the backend (50-case randomized suite).
- M5 release engineering: CI conformance matrix over library versions,
  mkdocs-material docs with three executed tutorial notebooks
  (quickstart, credit-risk walkthrough, no-solver environments),
  performance smoke benchmark, clean-venv packaging verification.
- Project skeleton: packaging, CI, docs infrastructure (M0).

### Known limitations

- CP-SAT solve time misses the <1s P3 target at 300+ trees (~40s median on
  the §12.8 bench); planned v0.2 optimization via table-constraint encoding.
- Plausibility cannot combine with AllowMissing/NaN factuals.
- `n_counterfactuals > 1` requires the CP-SAT backend.
