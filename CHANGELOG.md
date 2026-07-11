# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- M1 vertical slice: XGBoost (object/JSON dump) → tree IR → routing-atomic
  cells → CP-SAT → provably optimal counterfactual, with `Freeze`/`Monotone`/
  `Range` constraints, raw/probability targets, MAD-chain normalizers,
  float-space verification with K×10 retry, and a brute-force exactness oracle
  gating the backend (50-case randomized suite).
- Project skeleton: packaging, CI, docs infrastructure (M0).
