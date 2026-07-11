# treecf

**Constrained, threshold-aware counterfactual explanations for tree ensembles.**

`treecf` answers the question: *"what is the minimal, feasible change to this instance such
that the model's raw output lands in a target interval?"* — for XGBoost, LightGBM, CatBoost
and scikit-learn tree ensembles.

> Status: pre-release (v0.1). See the [documentation](https://wlazlod.github.io/treecf/) for concepts and tutorials.

## Why another counterfactual package?

- **Tree-native and exact.** Models are parsed into a shared tree IR and encoded for CP-SAT:
  counterfactuals come with an optimality proof, not a heuristic guess. A solver-free genetic
  backend (numpy only) covers environments where `ortools` cannot be installed.
- **Decision thresholds are first-class.** Targets are intervals on the raw model output —
  custom probability cutoffs, regression targets, and whole rating-grade ladders in one call.
- **Real-world constraints.** Declarative layer for immutability, directionality, ranges,
  one-hot consistency, and arbitrary linear inter-feature constraints such as
  `max_dpd_30d <= max_dpd_12m` — compiled once, enforced by every backend.
- **Missing values are values.** NaN can be a legitimate counterfactual state, with
  per-feature opt-in and explicit transition costs.
- **Constraint mining.** Candidate invariants are mined from data and presented for human
  review — never auto-applied.

## Installation

```bash
pip install treecf              # core: numpy only, genetic backend
pip install "treecf[cpsat]"     # exact CP-SAT backend (ortools)
pip install "treecf[xgboost]"   # model parsers as extras; JSON dumps work without them
pip install "treecf[viz]"       # matplotlib plots
```

## Quick look

```python
from treecf import Explainer, Target, constraint, Freeze

exp = Explainer(
    model="model.json",                       # native object or dump file
    background=X_train_sample,
    constraints=[
        constraint("max_dpd_30d <= max_dpd_12m"),
        Freeze("age_of_bureau_file"),
    ],
)
res = exp.explain(x, target=Target.probability(range=(0.0, 0.04)), backend="cpsat")
```

## License

MIT
