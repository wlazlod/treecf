# Plausibility

An optional hard constraint keeps counterfactuals inside the data manifold: an
isolation forest, parsed through the same tree IR, bounds the anomaly score.

```python
from sklearn.ensemble import IsolationForest
from treecf import Explainer, Plausibility

iso = IsolationForest(n_estimators=100).fit(X_train)
exp = Explainer(model, background=X_train,
                plausibility=Plausibility.isolation_forest(iso, max_anomaly_score=0.55))
```

The bound `s(x') <= theta` is equivalent to a single linear constraint on the
sum of depth-adjusted path lengths, so it composes exactly with everything else.
The genetic backend evaluates the same score directly.

**Cost transparency**: the forest's trees join cell construction and add one
boolean per forest leaf — roughly doubling model size for a typical 100-tree
forest. `plausibility=None` costs nothing.

**v0.1 restriction**: plausibility cannot be combined with `AllowMissing` or
NaN factual values (isolation forests define no NaN routing).
