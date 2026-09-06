# treecf

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22069503.svg)](https://doi.org/10.5281/zenodo.22069503)
[![PyPI](https://img.shields.io/pypi/v/treecf.svg)](https://pypi.org/project/treecf/)
[![Python](https://img.shields.io/pypi/pyversions/treecf.svg)](https://pypi.org/project/treecf/)
[![CI](https://github.com/wlazlod/treecf/actions/workflows/ci.yml/badge.svg)](https://github.com/wlazlod/treecf/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/treecf.svg)](LICENSE)

**Constrained, threshold-aware counterfactual explanations for tree ensembles.**

`treecf` answers the question: *"what is the minimal, feasible change to this instance such
that the model's output lands in a target interval?"* — for XGBoost, LightGBM, CatBoost and
scikit-learn tree ensembles — and can prove the answer is the cheapest, or that none exists.

![Lever-set by feature matrix of a recourse menu: filled cells where a plan changes that lever, a square for a proved-optimal plan, a cross for a lever set certified unable to reach the target](https://raw.githubusercontent.com/wlazlod/treecf/main/docs/guide/img/plot_recourse_menu.png)

> On [PyPI](https://pypi.org/project/treecf/). See the [documentation](https://wlazlod.github.io/treecf/) for concepts and tutorials.

## Why another counterfactual package?

- **Tree-native and fast.** Models are parsed into a shared tree IR and the constrained
  search runs on a bundled Rust core, typically in milliseconds; every result is
  float-verified against the parsed model before it is returned, and the parsers are
  conformance-tested against the native library.
- **Optional proofs, inside a measured envelope.** `backend="exact"` returns
  `proof="optimal"` when no cheaper plan exists under the declared objective (weighted
  distance, plus a per-feature term only if you set `sparsity_weight`), and a completed
  search that finds nothing returns `Infeasible(proof="certified")`. Proofs scale with the
  number of levers the search may move, not with the model's width: on the measured matrix
  the refine search certifies up to 200 trees with 12 free features inside 60 s and nothing
  at 20 features and depth 5 — while on that 300-tree, 50-feature model a coalition of up to
  three levers certifies in under half a second with `search="refine"`, so wide models get
  proofs once the levers are restricted with `Freeze`, coalitions, or a `recourse_menu`. A
  search that runs out of budget returns its best plan labelled `heuristic` and warns; it
  never claims more.
- **Recourse regions.** Any verified counterfactual widens into a certified box — "reduce
  utilization below 0.40", not "to 0.3972" — with every point in the box provably in-target
  and constraint-feasible; works with every backend.
- **Real constraints.** Immutability, directionality, ranges, one-hot consistency, linear
  inter-feature rules such as `max_dpd_30d <= max_dpd_12m`, and NaN as a legitimate value
  with its own transition cost — declared once, enforced by every backend.
- **Menus, diverse plans, certificates.** `recourse_menu` solves every lever set up to a
  size and says which combinations provably cannot work; `explain_diverse` returns the
  cheapest plans with distinct lever sets; `certificate` turns any result into a
  self-contained JSON record a validator re-checks later.

On a 120-tree model and 100 declined rows, treecf's plans cost a seventh of DiCE's at a fifth
of the time; NICE is four times faster per instance, and its plans cost 2.7 times more and
cannot take constraints. The measured tables and the honest reading are on the
[benchmarks page](https://wlazlod.github.io/treecf/concepts/backends/#against-other-cf-libraries).

Not for you if: the model is not a tree ensemble; you want sets of plans diverse by distance
rather than by the levers they use (DiCE does that); you need a proof over dozens of free
levers at once without restricting them (see the [proof envelope](https://wlazlod.github.io/treecf/concepts/certification/#the-proof-envelope-measured));
or you need a frozen API — treecf is in beta, see
[API stability](https://wlazlod.github.io/treecf/api-stability/).

## Installation

```bash
pip install "treecf[xgboost,viz]"   # wheels for Linux, macOS, Windows; no Rust toolchain needed
```

numpy is the only Python dependency; the extras add a parser for your training library and
the plots. JSON model dumps parse without the training library, so explanations can be
generated on a scoring host that has neither it nor a solver.

## Quick look

Runnable as-is: `credit_demo()` returns a packaged credit model, background rows, and one
declined applicant.

```python
from treecf import Explainer, Freeze, Monotone, Target
from treecf.datasets import credit_demo

model, X, x = credit_demo()
target = Target.probability(range=(0.0, 0.05))
exp = Explainer(
    model, background=X,
    constraints=[Freeze("occupation"), Monotone("tenure_months", "increase")],
)

res = exp.explain(x, target=target, seed=0)
res.changes                          # {'income': (4678.0, 6932.4)}
res.proof                            # 'heuristic'

proved = exp.explain(x, target=target, backend="exact", region=True, seed=0)
proved.proof                         # 'optimal'
proved.region.describe()             # {'income': 'in [6.58e+03, 7.81e+03] (data-limited)',
                                     #  'utilization': 'in [0.428, 0.541)', ...}

menu = exp.recourse_menu(x, target=target, max_levers=2)
menu.describe()["dpd_12m"]           # 'no acceptance is reachable by changing only dpd_12m'
```

## Learn more

- [How it works](https://wlazlod.github.io/treecf/how-it-works/) — the pipeline from objective to verified answer.
- [Certification](https://wlazlod.github.io/treecf/concepts/certification/) — what a proof covers and where it stops.
- [Credit-risk walkthrough](https://wlazlod.github.io/treecf/notebooks/02-credit-risk-tutorial/) — a batch workflow end to end.
- [probcal integration](https://wlazlod.github.io/treecf/guide/probcal/) — recourse against calibrated cutoffs.

## Cite

```bibtex
@software{wlazlo_treecf,
  author  = {Wlazło, Daniel},
  title   = {treecf: constrained, threshold-aware counterfactual explanations for tree ensembles},
  doi     = {10.5281/zenodo.22069503},
  url     = {https://github.com/wlazlod/treecf},
  license = {MIT}
}
```

## Contributing and license

MIT. See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the test layers, and the
project's hard invariants; report security issues privately per [SECURITY.md](SECURITY.md).
