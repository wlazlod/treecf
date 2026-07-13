# Coalitions

A single counterfactual can mix unrelated levers — asking one applicant to raise income,
close credit lines, *and* wait for a delinquency to age out is technically one plan, but
nobody acts on all three fronts at once. **Coalitions** split recourse by what the applicant
actually controls together: you name feature groups, and treecf produces one counterfactual
per group, where each plan may only change features from its own group.

The mode is **opt-in and never the default** — plain `explain` behavior is unchanged.

## One row

```python
result = exp.explain_coalitions(
    applicant,
    target=Target.probability(range=(0.0, 0.30)),
    coalitions={
        "debt history": ["max_dpd_30d", "max_dpd_12m", "months_since_last_delinq"],
        "credit usage": ["utilization", "n_active_loans", "n_loans_total"],
        "income":       ["income_monthly"],
    },
    include_full=True,      # adds the unrestricted "(all levers)" baseline
    seed=0,
)
result   # {"(all levers)": Counterfactual, "debt history": Counterfactual,
         #  "credit usage": Counterfactual, "income": Infeasible}
```

The result is a dict in coalition order (baseline first when requested), one
`Counterfactual` or `Infeasible` per group — the same shape as a `Target.bands` ladder. An
`Infeasible` here is a finding, not a failure: *this group alone cannot reach the target*.
In the example above, no realistic income change gets the applicant under the cutoff — advice
that a mixed plan would have hidden inside one big change-set.

## A whole dataset

```python
batch = exp.explain_batch(
    X_declined, target=t,
    diversity="coalitions",
    coalitions={...},
    include_full=True,
    ids=app_ids, seed=0,
)
batch.to_frame()   # one row per (id, coalition); the `coalition` column names the group
```

Each row gets one record per coalition: feasible plans ranked by distance (`k = 0, 1, …`),
then infeasible coalitions, each carrying its `coalition` name. `n_per_example` is not used
in this mode. Records save/load and `to_frame()` like any other batch.

## Semantics

- **Groups may overlap** — `utilization` can legitimately serve both a "debt" and a
  "behavior" plan.
- **Features in no coalition are never modified** in any coalition solve (the unrestricted
  baseline is the only solve that may touch them).
- **Unknown feature names raise** `TreecfError`, listing the offenders.
- Explicit constraints (`Freeze`, `Monotone`, …) still apply inside every coalition; a
  coalition whose only levers are frozen simply comes back `Infeasible`.
- The reserved name `"(all levers)"` may not be used as a coalition name together with
  `include_full=True`.

## How it works underneath

A coalition solve is the standard, fully verified pipeline with one addition: every feature
outside the group is frozen. Internally each coalition gets an `Explainer` clone with those
extra `Freeze` constraints (the parsed model is shared, so clones are cheap), and every
returned plan passes the same float-space verification against that coalition's constraint
set. In batch mode each coalition's rows solve as one parallel wave in the Rust core — K
coalitions cost K waves. See [How it works](../how-it-works.md#grouped-recourse-coalitions).

## Comparing the plans

The outcome dict plugs straight into the comparison plots — coalition names become labels,
infeasible groups are skipped:

```python
from treecf.viz import plot_alternatives, plot_tradeoff

plot_alternatives(result, explainer=exp)   # per-plan changes, one color per coalition
plot_tradeoff(result, target=t)            # what each group's plan costs and buys
```

## When to use it

Reach for coalitions when recourse is *delivered as advice* — per-department actions,
customer-facing hints, or fairness reviews asking "can this be fixed by behavior alone,
without income changes?". For the single cheapest plan regardless of grouping, plain
`explain` remains the right call; for many stylistically different plans,
`diversity="seeds"` or `"lever-blocking"` ([batch production](../getting-started.md#scale-to-a-dataset)).
