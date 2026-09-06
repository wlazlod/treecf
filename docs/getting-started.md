# Getting started

## Install

```bash
pip install "treecf[xgboost,viz]"
```

numpy is the only Python dependency; the search engine is a Rust core inside the wheel.
The extras add a parser for your training library and the plots — a JSON model dump
parses without either.

## First counterfactual

Runnable as-is: `credit_demo()` returns a packaged credit model, background rows, and one
declined applicant.

```python
from treecf import Explainer, Freeze, Monotone, Target
from treecf.datasets import credit_demo

model, X, x = credit_demo()
exp = Explainer(model, background=X)
target = Target.probability(range=(0.0, 0.05))    # default probability at most 5%

res = exp.explain(x, target=target, seed=0)
res.changes       # {'income': (4678.0, 6932.4)}
res.distance      # 2.33 — cost in σ-normalized units
res.score_prob    # 0.0443 — the verified output at the plan
res.proof         # 'heuristic' — fast search; see "Need a proof?"
```

Read it: `changes` is what to move and where; `distance` is the cost; `score_prob` is the
model's output at the plan, verified before it is returned. An `Infeasible` result carries
a `reason` instead.

Constrain it: constraints are declared once and every engine honours them.

```python
exp = Explainer(
    model, background=X,
    constraints=[Freeze("occupation"), Monotone("tenure_months", "increase")],
)
res = exp.explain(x, target=target, seed=0)
```

**Need a proof?** `backend="exact"` returns `proof="optimal"` or a certified `Infeasible`;
`region=True` widens the plan into a box every point of which is verified.
→ [Certify and widen](guide/certify.md).

```python
proved = exp.explain(x, target=target, backend="exact", region=True, seed=0)
proved.proof                      # 'optimal'
proved.region.describe()["income"]   # 'in [6.58e+03, 7.81e+03] (data-limited)'
```

**More than one plan?** `exp.explain_diverse(x, target, k=3)` returns the cheapest plans
with distinct lever sets, each with its own proof. → [Run the search](guide/explain.md).

**A whole day's declines?** `exp.explain_batch(X_declined, target)` solves them in parallel.
→ [Run the search](guide/explain.md#a-whole-dataset).

## Where next

- [Deep dive: how it works](how-it-works.md) — the pipeline from objective to verified answer.
- [Glossary](glossary.md) — plan, recourse, lever, cell, region, and the other words used here.
- [API reference](api.md).
