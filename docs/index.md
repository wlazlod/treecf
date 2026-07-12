# treecf

**Constrained, threshold-aware counterfactual explanations for tree ensembles.**

`treecf` answers: *"what is the minimal, feasible change to this instance such that the
model's raw output lands in a target interval?"* — for XGBoost, LightGBM, CatBoost and
scikit-learn tree ensembles.

!!! warning "Pre-release"
    v0.1 is under active development. The API shown here follows the accepted spec and may
    still shift before the first release.

## Highlights

- **Tree-native counterfactual search** on a **bundled Rust core** — typically
  milliseconds even on 300-tree ensembles (44–58× faster than the equivalent numpy
  implementation; see [benchmarks](benchmarks-genetic-rust.md)), with every result
  float-verified against the model IR before it is returned.
- **Targets as intervals on the raw model output**: custom probability cutoffs, regression
  targets, and rating-grade ladders (`Target.bands`) in one call.
- **Declarative constraints** compiled once for every backend: `Freeze`, `Monotone`, `Range`,
  `OneHot`, and arbitrary linear inter-feature rules like `max_dpd_30d <= max_dpd_12m`.
- **NaN as a first-class counterfactual value** with per-feature opt-in and transition costs.
- **Constraint mining** from background data — candidates are always human-reviewed, never
  auto-applied.
- **Optional plausibility** as a hard isolation-forest constraint.

## Installation

```bash
pip install treecf              # bundled Rust engine; numpy is the only Python dependency
pip install "treecf[viz]"       # matplotlib plots
```

Platform wheels ship the compiled engine — no Rust toolchain needed to install.
Model parsers accept JSON dumps directly, so the training framework does not need to be
installed where explanations are generated.
