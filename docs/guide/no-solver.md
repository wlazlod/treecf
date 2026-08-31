# Restricted environments

Model-validation and audit hosts often cannot install the training
framework, a MILP solver, or anything beyond a frozen base image. treecf is
built for that host: the core package depends on **numpy only**, parses
**JSON dumps** without any model library, and both search engines — genetic
and exact — run on a **compiled Rust core bundled in the wheel**. There is
no solver to license, install, or explain to IT.

## What runs where

| Environment | What works |
|---|---|
| numpy-only host, dump file shipped in | Everything on this site: parsing, all constraints, genetic and exact backends, regions, certificates, batch files |
| training environment (xgboost/lightgbm/catboost/sklearn installed) | The same, plus passing native model objects directly |
| no compiled wheels allowed | `backend="python"`: the pure-Python genetic engine — same semantics, same seed-determinism, slower |

The split is explicit, never silent: passing a native object without its
library installed raises `MissingExtraError` naming the pip command, and no
backend ever substitutes for another behind your back.

## The workflow

On the modelling side, ship the dump, not the framework
(`model.save_model("model.json")` for XGBoost,
`booster.dump_model()` for LightGBM, `save_model(format="json")` for
CatBoost). On the audit host:

```python
# docs: no-run — model.json / X_sample stand in for the shipped dump and data
from treecf import Explainer, Target, constraint

exp = Explainer("model.json", background=X_sample,
                constraints=[constraint("max_dpd_30d <= max_dpd_12m")])
res = exp.explain(x, target=Target.probability(range=(0.0, 0.30)), seed=0)
```

The docs' own explainer is constructed exactly this way from a committed
LightGBM dump, so every runnable block on this site is also a demonstration
that no training library is needed:

```python
# exp, x, target: the docs explainer, one rejected applicant, the target
res = exp.explain(x, target=target, backend="exact", seed=0)
res.proof   # a full optimality proof, no solver installed
```

## The complete worked session

The [no-solver environments notebook](../notebooks/03-no-solver-environments.ipynb)
runs the whole story end to end — train, dump, ship, explain, verify against
the native model — including the round-trip check that the native model
agrees with every returned plan.

## Related

- [Bring your model](models.md): the dump formats per library.
- [Certify and widen](certify.md): the proofs the bundled exact engine
  produces.
