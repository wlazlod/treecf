# Certify and widen

The default backend returns a good plan; `backend="exact"` returns a plan
with a *claim* — proved cheapest, cheapest within a stated gap, or certified
impossible — and `region=True` widens a point plan into a certified box.
This page is the workflow; the boundaries of the claims are in
[Certification](../concepts/certification.md).

## Prove optimality

```python
# exp, x, target: the docs explainer, one rejected applicant, the target
from treecf import Counterfactual, Infeasible

res = exp.explain(x, target=target, backend="exact", seed=0)
if isinstance(res, Counterfactual):
    res.proof   # "optimal" — no cheaper feasible row exists in the searched grid
elif isinstance(res, Infeasible):
    res.proof   # "certified" (nothing exists) or "search_exhausted" (budget ran out)
```

Read `proof`, never just the presence of a row: the exact backend reports
`"heuristic"` honestly when a conservative constraint repair had to withdraw
the optimality claim
([the honesty notes](../concepts/certification.md#two-honesty-notes)).

## Budgets, gap, warnings

Three arguments control how hard the search works, and every degraded
outcome warns rather than passing silently:

- `time_budget_s` (default 10.0) and `node_budget` (default 2,000,000) cut
  the search off; a cut always emits a `TreecfWarning` naming the cause and
  the result downgrades to best-found.
- `gap=0.05` accepts a plan provably within 5% of the optimum, reported as
  `proof="optimal_within_gap"` — usually a large speedup for the last few
  percent of proof.
- `warm_start=True` (default) seeds the search with a short genetic pass; it
  changes speed, never the claim.

Presolve — a reachability filter that discards candidate values no tree can
respond to — runs before every exact search and is reported in
`solver_stats["presolve_removed"]`; when it empties a feature's domain
entirely it certifies infeasibility without expanding a single node
([how presolve works](../concepts/certification.md#presolve-pruning-before-the-search-starts)).

## Widen to a region

`region=True` grows a certified box around the plan: per-feature intervals —
and, for categorical features, sets of category codes — within which *every*
row still satisfies the target and constraints:

```python
# exp, x, target: the docs explainer, one rejected applicant, the target
res = exp.explain(x, target=target, backend="exact", region=True, seed=0)
res.region.feature_intervals    # {"income": (lo, hi), ...} — certified intervals
res.region.feature_categories   # {"occupation": (1, 2)} — certified category codes
```

Regions are sound but neither maximal nor monotone in the target interval —
[the fine print](../concepts/certification.md#regions-certified-not-maximal-not-monotone).
`plot_region` in [visualize](visualize.md) draws the box with what stopped
each bound.

## Next

A proof is only worth what a third party can re-check:
[make it auditable](auditability.md). The search itself was set up in
[run the search](explain.md).
