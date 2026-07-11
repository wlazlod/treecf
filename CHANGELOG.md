# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Project skeleton: packaging, CI, docs infrastructure (M0).
