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
