# Certification

Every backend float-verifies its own answer before returning it — that much is true of
`"genetic"`, `"python"`, and `"exact"` alike (see [How it works — nothing ships
unverified](../how-it-works.md#nothing-ships-unverified)). **Certification** is a stronger,
separate claim that only `backend="exact"` and `Explainer.recourse_region` can make: not just
"this row checks out" but "no cheaper row exists" or "every row in this box checks out". This
page is about what that stronger claim covers, what it does not, and where it stops.

## What a certificate covers

A treecf certificate is a statement about the artifact you handed the `Explainer` — this parsed
model, this compiled constraint set, this plausibility threshold (if any), these value-policy
domains (if any) — at the moment the search ran. It is not a statement about the world:

- It says nothing about whether the model itself generalizes, or whether the training data it
  learned from was representative.
- It says nothing about whether the real applicant behind the row can actually execute the
  plan — only that the plan, as a vector of feature values, is feasible against the constraints
  you declared.
- Change the model, the constraints, the plausibility bound, or a value policy, and you have a
  different problem with its own certificate; nothing carries over.

Within that scope the claim is exact: the search enumerates the same kind of candidate grid every
backend shares — the model's cells, refined by the compiled constraints (see [How it works — the
search space](../how-it-works.md#the-search-space-cells-not-real-numbers)) — and every row it
returns is re-verified against the model in float space before you see it, the same as every
other backend.

## Proof taxonomy

`Counterfactual.proof` is always one of three values, and `Infeasible.proof` one of two:

| Field | Value | Meaning |
|---|---|---|
| `Counterfactual.proof` | `"heuristic"` | No optimality claim — the default for `"genetic"`/`"python"`, always; see below for when `"exact"` also reports it |
| `Counterfactual.proof` | `"optimal"` | The exact backend proved no cheaper feasible row exists in the searched grid |
| `Counterfactual.proof` | `"optimal_within_gap"` | `gap > 0` was passed; the exact backend proved no row cheaper by more than that relative fraction exists |
| `Infeasible.proof` | `"search_exhausted"` | The default — a budget ran out, or a heuristic search found nothing; nothing is proven about whether a counterfactual exists at all |
| `Infeasible.proof` | `"certified"` | Exact-backend only — every assignment the searched grid allows was tried and none was feasible; `reason` names the node count behind the proof |

### Two honesty notes

**A feasible exact result can report `proof="heuristic"` without exhausting `node_budget` or
`time_budget_s`.** Two features tied by an inter-feature constraint (`constraint("a <= b")`)
sometimes need a repaired value the search's own candidate grid does not offer; the repair is
conservative by design — see [How it works](../how-it-works.md#the-exact-search-cells-domains-and-branch-and-bound)
— and when it cannot settle every such pair it withdraws the optimality claim rather than risk
overstating it. The row itself is still real, still float-verified, and still the cheapest one
the search happened to find — only the "cheapest possible" claim is dropped. Read `proof`, not
`x_cf`'s presence, to know which claim you got:

```python
res = exp.explain(applicant, target=t, backend="exact", seed=0)
if isinstance(res, Counterfactual):
    res.proof         # "optimal" | "optimal_within_gap" | "heuristic"
    res.solver_stats   # nodes_expanded, nodes_pruned_score, nodes_pruned_cost,
                        # lower_bound, gap, completed, warm_start_used
```

**Certified infeasibility comes only from a completed search.** `Infeasible.proof="certified"`
means the exact backend enumerated the whole reachable grid and rejected every row — it did not
run out of budget, and it did not give up partway through a constraint repair. Every other way of
coming back empty-handed (budget exhausted, a repair withdrawn, the genetic/python backends
finding nothing) reports `proof="search_exhausted"`: a plan was not found, but nothing is proven
about whether one exists.

## Value policies under certification

`value_policy` changes what "optimal" is measured against, and the two backend families read a
policy differently:

- **The exact backend treats a value policy as a hard constraint.** Policy-conforming values are
  the only candidates its search ever builds — a feature under `value_policy={"n_active_loans":
  "integer"}` never gets a fractional candidate to begin with. So `proof="optimal"` on a policy
  run means *optimal among policy-conforming rows*, not optimal over the unrestricted space.
  A callable `value_policy` is rejected outright at exact-backend validation time (it names
  `backend="genetic"` as the fallback): the search has no way to enumerate an arbitrary
  function's conforming values.
- **The genetic backend treats a value policy as a soft preference.** It searches the
  unrestricted space and snaps the winning row onto the policy afterward, reverting features one
  at a time if snapping breaks feasibility. For a changed feature under a policy,
  `snapped[name] = False` can mean either of two things: no conforming value existed inside the
  feature's cell to snap to, or one did and was applied but later reverted because the snapped
  row failed verification — either way, the row you get back may hold a value the policy would
  not have chosen.

The two backends can therefore legitimately return different answers on the same policy run —
not because one is wrong, but because they are certifying against different candidate sets.

## Regions: certified, not maximal, not monotone

`explain(..., region=True)` (or `Explainer.recourse_region` directly) widens a verified
counterfactual into a `RecourseRegion` — a per-feature box around it. The certificate here is
per-point: **every point inside the box, not just the counterfactual itself, is independently a
valid counterfactual** — still in the target interval,
still plausible when plausibility is configured, still constraint-feasible — proven by an
interval-tree walk of every ensemble tree plus a worst-corner check of every linear constraint,
never by sampling.

Two things the certificate does *not* claim:

- **Not maximal.** The box is grown greedily, one joint-grid cell at a time, accepting an
  expansion only when the whole enlarged box still passes the soundness oracle. A larger sound
  box may exist that this growth order did not find.
- **Not monotone in the target.** A strictly narrower target interval can still produce a
  strictly wider region on some feature — growth is greedy and order-dependent, so a feature
  that is forced to stop early on one target can free up room a later feature grows into on a
  narrower one. Do not assume tightening the target only shrinks the region.

`region.describe()` gives one human-readable phrase per non-degenerate feature — two-sided
(`"in [lo, hi]"`) when both endpoints are finite, one-sided (`"≤ v"` / `"≥ v"`) *only* when
the other side is genuinely unbounded, not merely wide, and `"unconstrained"` when both
endpoints are unbounded. `plot_recourse_map(..., schematic=True)`
reads these phrases directly when a plan carries a region, in place of the single-value wording
it otherwise falls back to.

Features an `Implies`, a `OneHot`, or an unsupported multi-feature `Linear` could still break are
never widened at all — they stay pinned at the counterfactual's own value, conservatively, rather
than trust an argument this release has not proven sound for those shapes.

## Scaling guidance

The exact search is a branch-and-bound over a per-feature candidate grid, and its worst case is
exponential in the number of *influential* features — features the search actually branches on,
because more than one candidate value survives constraint pruning — not in the model's total
feature count. In practice:

- `Freeze` and other constraints that pin a feature to one value remove it from branching
  entirely, so a heavily constrained problem searches a much smaller space than the raw feature
  count suggests.
- As a rule of thumb, keep the number of influential features to the low hundreds; well beyond
  that, `node_budget` or `time_budget_s` is likely to cut the search short before it settles the
  space (reported honestly as `proof="heuristic"` or `Infeasible.proof="search_exhausted"`,
  never silently).
- `node_budget` (default 2,000,000 assignments) and `gap` (default `0.0`) are the two pressure
  valves: lowering `node_budget` bounds worst-case wall time at the cost of a less certain
  answer, and a `gap > 0` lets the search settle for — and honestly report, via
  `proof="optimal_within_gap"` — a row provably within that relative fraction of the true
  optimum instead of paying for the last few percent.
- `warm_start` defaults to `True`: a short genetic pass (about a quarter of `time_budget_s`,
  capped at 2 seconds) seeds the exact search with an incumbent before it starts branching,
  which prunes harder from the first node without costing anything from the main budget — the
  exact search still gets the full `time_budget_s` afterward.

## What the exact backend does not certify yet

The exact backend supports single-feature `Linear` constraints and the canonical two-feature
order-pair shape (`constraint("a <= b")`, i.e. `a - b <= 0`) exactly. Any other multi-feature
`Linear` — three or more features, or a two-feature shape other than the canonical order pair —
raises `ConstraintValidationError` and names `backend="genetic"` as the fallback: the genetic
engine has no such restriction, at the cost of the optimality proof.

## Related approaches

treecf's exact backend sits in a line of mixed-integer/constraint work on tree ensembles, though
it depends on no external solver. Kantchelian, Tygar & Joseph, *Evasion and Hardening of Tree
Ensemble Classifiers* (ICML 2016), formulate adversarial evasion of a tree ensemble as a MILP
over the trees' decision paths. Cui, Chen, He & Chen, *Optimal Action Extraction for Random
Forests and Boosted Trees* (KDD 2015), use integer linear programming to find the minimum-cost
action that flips an ensemble's prediction. Tolomei, Silvestri, Haines & Lalmas, *Interpretable
Predictions of Tree-based Ensembles via Actionable Feature Tweaking* (KDD 2017), trade the exact
guarantee for speed with a greedy path-tweaking heuristic on the same actionability question.
Parmentier & Vidal, *Optimal Counterfactual Explanations in Tree Ensembles* (ICML 2021 — the
OCEAN package), generalize furthest: full counterfactual search with plausibility (isolation
forest) and heterogeneous-constraint support, backed by MIP, CP, and MaxSAT solvers.

treecf's own angle is narrower and dependency-free: an exact engine built directly on the same
cell IR the genetic backend already shares, with no solver dependency, paired with the
certificate and region layer this page documents.

## Where to go next

- [How it works — the exact search](../how-it-works.md#the-exact-search-cells-domains-and-branch-and-bound)
  and [— certified regions](../how-it-works.md#certified-regions-grow-and-verify) for the
  mechanisms behind this page.
- [Backends and proofs](backends.md) for the engine-level contract (genetic/python/exact,
  rust-first dispatch, benchmarks).
- [Constraints](constraints.md) for what `Linear`, `Implies`, `OneHot`, `Freeze`, and `Monotone`
  compile to.
- [Plausibility](plausibility.md) for the isolation-forest bound the exact search and the region
  oracle both read.
- The [credit-risk tutorial](../notebooks/02-credit-risk-tutorial.ipynb) runs the exact backend,
  a certified-infeasible case, and a region end to end.
