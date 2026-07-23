# Getting started

## Install

```bash
pip install treecf                       # bundled Rust search engine
pip install "treecf[xgboost,viz]"        # parser extras, matplotlib plots
```

numpy is the only Python dependency; the genetic engine is a compiled Rust core
shipped inside the wheel. Model parsers accept JSON dumps directly, so
explanations can be generated on machines where the training framework (or any
solver) is not installed.

## First counterfactual

```python
import numpy as np
import xgboost as xgb
from treecf import Counterfactual, Explainer, Target, Freeze, Monotone, constraint

# a binary classifier trained on your data
clf = xgb.XGBClassifier(n_estimators=100, max_depth=4).fit(X_train, y_train)

exp = Explainer(
    model=clf,                            # or "model.json", or a dump dict
    background=X_train,                   # fits robust distance normalizers (MAD chain)
    constraints=[
        Freeze("age_of_bureau_file"),     # immutable
        Monotone("age", "increase"),      # can only grow
        constraint("max_dpd_30d <= max_dpd_12m"),   # inter-feature consistency
    ],
)

res = exp.explain(
    x_row,
    target=Target.probability(range=(0.0, 0.04)),   # get under the 4% PD cutoff
    seed=0,
)

if isinstance(res, Counterfactual):
    print(res.changes)      # {"feature": (from, to), ...}
else:
    print(res.reason)       # Infeasible: why no plan was found
```

The search is heuristic (`proof="heuristic"`), feasibility-first, and
seed-deterministic; on toy suites it brackets a brute-force optimum. It runs on
the bundled Rust engine in milliseconds even on 300-tree models;
`backend="python"` runs the reference numpy implementation of the same
algorithm. [How it works](how-it-works.md) walks the whole pipeline.

## Calibrated models

If your pipeline post-hoc calibrates the model's probabilities, express the
target on the *calibrated* scale — `Target.probability` would silently target
the uncalibrated output:

```python
res = exp.explain(
    x_row,
    target=Target.calibrated(cal, range=(0.0, 0.04)),  # calibrated PD ≤ 4%
    seed=0,
)
```

`cal` is any monotone calibrator exposing `interval_inverse` and
`is_monotone_`; see the FAQ for the exact protocol and the `buffer_logit` robustness margin.

## Read the result

| Field | Meaning |
|---|---|
| `x_cf` | counterfactual instance (NaN where a missing state was chosen) |
| `changes` | feature → (factual, counterfactual) for every changed feature |
| `distance`, `n_changed` | weighted L1 distance and L0 count |
| `score_raw`, `score_prob` | raw model output and its sigmoid when applicable |
| `proof` | always `"heuristic"` — the search never claims optimality |
| `snapped` | per-feature outcome of `value_policy` snapping |

Every result is re-verified in float space against the IR before it is returned:
the target and each constraint are checked on the actual returned values.

## Visualize it

```python
from treecf.viz import plot_changes, plot_waterfall, plot_effort

plot_changes(res)                       # dumbbells: from -> to per feature
plot_waterfall(exp, res, target=t)      # SHAP-style: exact score deltas, cutoff line
plot_effort(exp, res)                   # where the applicant's effort goes (J split)
```

## Alternatives for one instance

One plan is rarely the whole story. Ask for several distinct plans for the same
row and compare them side by side:

```python
from treecf.viz import plot_alternatives, plot_tradeoff

batch = exp.explain_batch(x_row.reshape(1, -1), target=t, n_per_example=3, seed=0)
plans = batch.for_id(0)                 # up to 3 distinct plans for this row

plot_alternatives(plans, explainer=exp) # every plan's changes, standardized to Δ/σ
plot_tradeoff(plans, target=t)          # cost vs achieved score: which plan buys what
```

`diversity="lever-blocking"` instead re-solves with each plan's biggest lever
frozen — and reports levers that turn out to be *essential*.

For advice grouped by what a person controls together, ask for one plan per
named feature group — see [Coalitions](concepts/coalitions.md):

```python
result = exp.explain_coalitions(
    x_row, target=t,
    coalitions={"debt": ["max_dpd_30d", "max_dpd_12m"], "income": ["income_monthly"]},
    include_full=True,          # adds the unrestricted "(all levers)" baseline
)
plot_alternatives(result, explainer=exp)   # coalition names label the plans
```

## Scale to a dataset

```python
batch = exp.explain_batch(
    X_declined,                          # e.g. today's declined applications
    target=Target.probability(range=(0.0, 0.30)),
    n_per_example=2,                     # counterfactuals per example
    diversity="seeds",                  # or "lever-blocking" (also finds essential levers)
    ids=app_ids,
    seed=0,
)
batch.save("counterfactuals_today.json")     # compute once, store...
stored = BatchResult.load("counterfactuals_today.json")
stored.for_id("APP-00042")                   # ...look up any time
stored.to_frame()                            # or analyze as a pandas DataFrame
```

Solves run in parallel inside the Rust core. `treecf.viz_batch` plots the whole
batch — lever usage, per-plan effort, cost/sparsity/feasibility — as shown in the
[credit-risk walkthrough](notebooks/02-credit-risk-tutorial.ipynb).

## Where next

- [How it works](how-it-works.md) — the pipeline from objective to verified answer.
- [Concepts](concepts/models.md) — one page per stage: models, targets, constraints,
  missing values, plausibility, backends.
- [Tutorials](notebooks/01-quickstart.ipynb) — runnable notebooks.
- [API reference](api.md).
