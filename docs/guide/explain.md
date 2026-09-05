# Run the search

With the [model](models.md), [target](targets.md), and
[constraints](constraints.md) in place, this page is the middle of the
workflow: producing plans — for one row, for alternatives per row, and for a
whole dataset.

## One row

```python
# exp, x, target: the docs explainer, one rejected applicant, the target
res = exp.explain(x, target=target, seed=0)
res.changes    # {"feature": (from, to)} — only what changed
res.distance   # the weighted cost of the plan
res.proof      # "heuristic" — the default backend makes no optimality claim
```

The default backend is the genetic search on the bundled Rust core;
`backend="exact"` upgrades the answer to a proof and is the subject of
[certify and widen](certify.md). Either way the result is a
`Counterfactual` or an `Infeasible` — check with `isinstance`, and read
`Infeasible.reason` when nothing was found.

## Alternatives for one row

Two diversity modes produce genuinely different plans instead of one plan
with noise:

- `diversity="seeds"` re-runs the stochastic search from different seeds and
  keeps distinct outcomes.
- `diversity="lever-blocking"` solves once, then re-solves with each used
  lever frozen in turn — "and what if I cannot touch utilization?" answered
  systematically.

```python
# exp, x, target: the docs explainer, one rejected applicant, the target
alts = exp.explain_batch(x[None], target=target,
                         n_per_example=3, diversity="lever-blocking", seed=0)
[(rec.blocked_lever, rec.changes) for rec in alts.records if rec.feasible]
```

## Grouped levers: coalitions

`explain_coalitions` restricts the search to named feature groups and solves
each group independently — the answer to "what can this applicant do through
debt reduction alone?":

```python
# exp, x, target: the docs explainer, one rejected applicant, the target
plans = exp.explain_coalitions(
    x, target=target,
    coalitions={"repayment": ["utilization", "dpd_12m"],
                "profile": ["income", "tenure_months", "occupation"]},
    include_full=True, seed=0,
)
sorted(plans)   # ["(all levers)", "profile", "repayment"]
```

Semantics and comparison plots: [Coalitions](../concepts/coalitions.md).

## Recourse menus and diverse plans

`recourse_menu` solves every lever set up to a size as its own coalition and
returns the whole picture at once — which combinations of levers reach the
target, at what cost, and which provably cannot:

```python
# exp, x, target: the docs explainer, one rejected applicant, the target
menu = exp.recourse_menu(x, target=target, max_levers=2, backend="exact", seed=0)
menu.minimal              # the cheapest lever sets no smaller set can replace
menu.certified_infeasible # sets no acceptance is reachable through, proved
menu.complete             # True: every set up to two levers was settled with a proof
menu.describe()["dpd_12m"]   # "no acceptance is reachable by changing only dpd_12m"
```

The candidate levers are the features the search would branch on for this
applicant (not frozen, more than one candidate value, influential). Entries
are keyed by the features each plan actually changed, sorted and joined
with `+`, so a menu is a mapping in the same shape `explain_coalitions`
returns and every plot that takes one takes a menu. The default
`mode="minimal"` skips any set that contains a set already found feasible
— feasible by monotonicity, and not minimal — and lists the frontier in
`menu.minimal`; `mode="all"` solves every set. `menu.complete` certifies
one precise thing: every enumerated set was settled with a proof (an
optimal plan, or a certified infeasibility) and none was cut off by
`total_budget_s`. The genetic backend never certifies, so its menus are
never complete.

`explain_diverse` reads the `k` cheapest plans off that menu. Diversity
here means distinct lever sets — two plans count as different ways to
reach the target when they change different features — and nothing else:

```python
# exp, x, target: the docs explainer, one rejected applicant, the target
diverse = exp.explain_diverse(x, target=target, k=3, backend="exact", seed=0)
list(diverse)        # plan keys, cheapest first
diverse.jaccard      # pairwise distance between the plans' changed sets
diverse.complete     # False: k plans came back, so more distinct lever sets may exist
```

`diversity="coalitions"` with the `explain_coalitions` group form solves
each declared coalition as itself and reaches for unions of groups only
when fewer than `k` are feasible. The matrix view of a menu is in
[visualize](visualize.md#a-recourse-menu).

## A whole dataset

`explain_batch` runs thousands of rows in parallel inside the Rust core and
returns a `BatchResult` with per-row records, portable JSON storage, and
plotting hooks ([visualize](visualize.md)):

```python
# exp, X_bg, target: the docs explainer, its background rows, the target
batch = exp.explain_batch(X_bg[:20], target=target, seed=0)
sum(r.feasible for r in batch.records)   # rows with a plan
frame = batch.to_frame()                 # one row per (id, plan), pandas
batch.save("batch.json")                 # inert JSON; every 0.x release reads it back
```

`plot_recourse_burden` splits a campaign's outcome by any per-row segment
label — feasibility share and cost distribution together
([visualize](visualize.md#recourse-burden-by-segment)):

![Recourse burden by segment: feasible share and cost distribution per group](img/plot_recourse_burden.png)

### The exact-batch opt-in

`explain_batch(..., backend="exact")` loops the single-row exact solve —
there is no vectorized exact engine — so its wall time is
`rows × plans × time_budget_s` in the worst case, where `plans` is
`n_per_example` (or the coalition count). Because that is easy to
underestimate from one fast `explain` call, the batch path refuses to run
without `allow_exact_batch=True` and names that arithmetic, hours-formatted,
in the error. The full behavior — including how the shared warm start
differs from sequential `explain` calls — is in
[Certification — the exact-batch opt-in](../concepts/certification.md#the-exact-batch-opt-in).

## Next

When a plan needs to carry a proof — optimality, certified infeasibility, or
a whole certified region — move to [certify and widen](certify.md).
