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
