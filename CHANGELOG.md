# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-11

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
