# treecf

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22069503.svg)](https://doi.org/10.5281/zenodo.22069503)

**Constrained, threshold-aware counterfactual explanations for tree ensembles.**

`treecf` answers the question: *"what is the minimal, feasible change to this instance such
that the model's raw output lands in a target interval?"* — for XGBoost, LightGBM, CatBoost
and scikit-learn tree ensembles.

> On [PyPI](https://pypi.org/project/treecf/). See the [documentation](https://wlazlod.github.io/treecf/) for concepts and tutorials.

## Why another counterfactual package?

- **Tree-native and fast.** Models are parsed into a shared tree IR; the constrained
  genetic search runs on a bundled **Rust core** 44–58× faster than the equivalent numpy
  implementation (see the "Backends and proofs" docs page; the pure-Python engine remains
  available as `backend="python"`), and every result is float-verified against the IR
  before it is returned.
- **Optional optimality proof.** `backend="exact"` branch-and-bounds the same candidate
  grid; on the standard bench model (30-tree/8-feature XGBoost) it proves the cheapest
  counterfactual in a median 0.24s versus 0.005s for the genetic heuristic, closing a
  median 14.33% cost gap the heuristic leaves on the table — measured on a 4-core dev
  machine (`scripts/bench_exact.py`).
- **Certified "no".** A completed exact search returns `Infeasible(proof="certified")` —
  "no recourse exists within these constraints" becomes a provable statement, not a shrug
  after a timeout.
- **Recourse regions.** Any verified counterfactual widens into a certified box — "reduce
  utilization to ≤ 0.40", not "to 0.3972" — with every point in the box provably in-target
  and constraint-feasible; works with every backend.
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
pip install treecf              # bundled Rust engine; numpy is the only Python dep
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
res = exp.explain(x, target=Target.probability(range=(0.0, 0.04)), seed=0)

proved = exp.explain(x, target=t, backend="exact")      # proof="optimal", a certified "no", or a warned degrade
boxed = exp.explain(x, target=t, region=True)            # res.region.describe() -> "utilization <= 0.4"
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the test layers, and the
project's hard invariants; report security issues privately per
[SECURITY.md](SECURITY.md).

## License

MIT
