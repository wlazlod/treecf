# treecf

**Constrained, threshold-aware counterfactual explanations for tree ensembles.**

`treecf` answers: *"what is the minimal, feasible change to this instance such that the
model's raw output lands in a target interval?"* — for XGBoost, LightGBM, CatBoost and
scikit-learn tree ensembles.

!!! warning "Pre-release"
    v0.1 is under active development. The API shown here follows the accepted spec and may
    still shift before the first release.

## Highlights

- **Exact, tree-native counterfactuals** via CP-SAT, with optimality proofs — plus a
  solver-free genetic backend (numpy only) for restricted environments.
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
pip install treecf              # core: numpy only, genetic backend
pip install "treecf[cpsat]"     # exact CP-SAT backend (ortools)
pip install "treecf[viz]"       # matplotlib plots
```

Model parsers accept JSON dumps directly, so the training framework does not need to be
installed where explanations are generated.
