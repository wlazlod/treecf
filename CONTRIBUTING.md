# Contributing

Notes for anyone changing treecf's source. Issues and PRs are welcome: bug
reports with a reproducing model or dump, parser coverage for a model shape
the current parsers reject, documentation fixes, and benchmark results from
machines unlike the maintainer's are all useful. For security problems, use
the private process in [SECURITY.md](SECURITY.md) instead of an issue.

## Dev setup

```bash
uv sync --all-extras --dev        # Python env, builds the Rust extension via maturin
rustup toolchain install stable   # any toolchain >= the rust-version in rust/Cargo.toml
```

`uv sync` does **not** rebuild the extension for Rust-only source changes; after
editing `rust/`, run:

```bash
uv sync --all-extras --dev --reinstall-package treecf
```

## Test layers, and when each runs

| Layer | Command | When |
|---|---|---|
| Rust unit tests (featureless) | `cargo test --manifest-path rust/Cargo.toml` | every CI run, plus a pinned-MSRV job |
| Python suite | `uv run pytest` | every CI run; excludes `bench` and `slow` markers |
| Docs snippets + structure | `uv run pytest -m slow` | the docs jobs; run it after editing anything under `docs/` |
| Performance smoke | `uv run pytest -m bench` | non-gating; run when touching a search engine |
| Conformance suites | `uv run pytest tests/ir/` | CI matrix rows pinning each supported library version |
| probcal matrix | `uv run pytest tests/test_probcal_matrix.py` | CI job pinning probcal + LightGBM |

Lint and types gate every change: `uv run ruff check .`, `uv run mypy` (strict),
`cargo clippy --all-features -- -D warnings`, `cargo fmt --check`.

## Hard invariants

- **Python↔Rust parity is byte-exact.** The Rust engines are line-for-line
  mirrors of the Python reference, enforced by the fixtures under
  `tests/fixtures/`. Regenerate a fixture only for a documented behavior
  change, only through its `scripts/gen_*_fixtures.py` script, and with a
  changelog entry saying what changed and why.
- **Numeric-model results are stable across releases.** A release that changes
  what `explain` returns for an existing numeric model is a behavior change
  and must say so in the changelog; the invariance tests pin this.
- **Every returned plan is re-verified** in float space through the IR before
  the user sees it. Never bypass the verification step to make a test pass.
- **No silent fallback.** Missing extras raise `MissingExtraError` with the
  pip command; parsers that cannot guarantee parity raise
  `UnsupportedModelError`. Never degrade quietly.

## Code conventions

- numpy-style docstrings; `mypy --strict`; ruff as configured (line length 100).
- Frozen dataclasses for IR, constraints, and results.
- Comments and docstrings describe **current behavior**; release history goes
  in `CHANGELOG.md`, not in code.
- The core package imports numpy only; everything else is a lazily imported
  extra (`tests/test_package.py` enforces this).
- Docs follow the conventions in [`docs/README.md`](docs/README.md).

## Changelog and versioning

The changelog follows [Keep a Changelog](https://keepachangelog.com/); new
entries go under `## [Unreleased]`. The version string lives in several files
and two lockfiles — never edit them by hand. `scripts/bump_version.py NEW`
rewrites every location and `tests/test_version_everywhere.py` fails the
release if any location disagrees.

## PR checklist

- [ ] `uv run pytest`, `uv run ruff check .`, `uv run mypy` green
- [ ] `cargo test`, `cargo clippy --all-features -- -D warnings`,
      `cargo fmt --check` green (if `rust/` changed)
- [ ] behavior changes covered by a test and a `CHANGELOG.md` entry
- [ ] no fixture regenerated without a documented reason
- [ ] docs updated for user-visible changes (`uv run mkdocs build --strict`)
