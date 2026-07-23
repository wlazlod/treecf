# treecf

**Constrained, threshold-aware counterfactual explanations for tree ensembles.**

`treecf` answers: *"what is the minimal, feasible change to this instance such that the
model's raw output lands in a target interval?"* — for XGBoost, LightGBM, CatBoost and
scikit-learn tree ensembles.

```python
from treecf import Explainer, Freeze, Target

exp = Explainer(model, background=X_train, constraints=[Freeze("age")])
res = exp.explain(x_row, target=Target.probability(range=(0.0, 0.30)), seed=0)
res.changes   # {"utilization": (0.71, 0.419), "max_dpd_12m": (9.0, 3.0)}
```

## Highlights

- **Tree-native search on a bundled Rust core** — typically milliseconds even on 300-tree
  ensembles ([performance](concepts/backends.md#performance)), and every result is
  float-verified against the model before it is returned ([how it works](how-it-works.md)).
- **Targets as intervals on the model output** — probability cutoffs, regression targets,
  and rating-grade ladders in one call ([targets](concepts/targets.md)).
- **Declarative constraints** — `Freeze`, `Monotone`, `Range`, `OneHot`, and linear
  inter-feature rules like `max_dpd_30d <= max_dpd_12m`, compiled once for every engine
  ([constraints](concepts/constraints.md)), with optional mining from background data.
- **NaN as a first-class counterfactual value** with per-feature opt-in and transition
  costs ([missing values](concepts/missing-values.md)).
- **Optional plausibility** as a hard isolation-forest constraint
  ([plausibility](concepts/plausibility.md)).
- **Batch production** — thousands of rows solved in parallel inside the Rust core, with
  portable storage and batch-level plots ([tutorial](notebooks/02-credit-risk-tutorial.ipynb)).

## Where to start

1. [Getting started](getting-started.md) — install and your first counterfactual in five
   minutes.
2. [How it works](how-it-works.md) — the full pipeline, from objective to verified answer.
3. [Tutorials](notebooks/01-quickstart.ipynb) — runnable notebooks, from quickstart to a
   credit-risk batch workflow.
