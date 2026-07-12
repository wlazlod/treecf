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
from treecf import Explainer, Target, Freeze, Monotone, constraint

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

if hasattr(res, "x_cf"):
    print(res.changes)      # {"feature": (from, to), ...}
    print(res.proof)        # "heuristic" — feasibility-first search, float-verified
else:
    print(res.reason)
```

The search is heuristic (`proof="heuristic"`), feasibility-first, and
seed-deterministic; on toy suites it brackets a brute-force optimum. It runs on
the bundled Rust engine in milliseconds even on 300-tree models;
`backend="python"` runs the reference numpy implementation of the same
algorithm.

## The result object

| Field | Meaning |
|---|---|
| `x_cf` | counterfactual instance (NaN where a missing state was chosen) |
| `changes` | feature → (factual, counterfactual) for every changed feature |
| `distance`, `n_changed` | weighted L1 distance and L0 count |
| `score_raw`, `score_prob` | raw model output and its sigmoid when applicable |
| `proof` | `optimal` / `feasible` (time limit) / `heuristic` |
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

## Mass-produce for a whole dataset

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

Roughly ~100 ms per applicant on a 100-tree model including two alternatives
each (the Rust engine solves in milliseconds; see the tutorial notebook).
