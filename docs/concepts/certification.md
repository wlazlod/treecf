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
# exp, x, target: the docs explainer, one rejected applicant, the target
from treecf import Counterfactual

res = exp.explain(x, target=target, backend="exact", seed=0)
if isinstance(res, Counterfactual):
    res.proof          # "optimal" | "optimal_within_gap" | "heuristic"
    res.solver_stats   # nodes_expanded, nodes_pruned_score, nodes_pruned_cost,
                       # lower_bound, gap, completed, warm_start_used,
                       # presolve_removed, presolve_certified
```

**Certified infeasibility comes only from a completed search.** `Infeasible.proof="certified"`
means the exact backend enumerated the whole reachable grid and rejected every row — it did not
run out of budget, and it did not give up partway through a constraint repair. Every other way of
coming back empty-handed (budget exhausted, a repair withdrawn, the genetic/python backends
finding nothing) reports `proof="search_exhausted"`: a plan was not found, but nothing is proven
about whether one exists.

## When the budget runs out

Whenever the exact backend returns any result with `solver_stats["completed"] is False` —
`explain`, a `Target.bands` ladder, `explain_coalitions`, or `explain_batch` alike — a
`TreecfWarning` fires, always, naming exactly one of two causes and never conflating them:

- **The search genuinely ran out of budget** (`node_budget` nodes expanded, or `time_budget_s`
  elapsed): the warning says so, states the node count, and — when a row was found — reports it
  as the best found rather than proven optimal, together with a lower-bound/gap parenthetical
  when one is available (`(lower bound X, gap ≤ Y%)`); it also points at raising the budgets or
  passing `gap=` as the next move.
- **A conservative constraint repair withdrew the optimality certificate without touching the
  budget** — the same repair mechanism the first honesty note above describes: the row is still
  real and float-verified, only the "cheapest possible" claim is dropped.

If the search's own warm start (`warm_start=True`, the default) was unseeded (`seed=None`) and
contributed the incumbent, the warning appends a clause noting that a rerun may land on a
different heuristic result and that passing `seed=` fixes it.

`Target.bands`, `explain_coalitions`, and `explain_batch` never emit one warning per degraded
solve — they collapse every degraded solve from one call into a single aggregate `TreecfWarning`
that breaks the count down by cause (`exhausted: N solves; withdrawn: M solves`) and points back
at each result's own `proof`/`solver_stats` for which case applies to it. Read `solver_stats` on
the affected result(s) directly for the full picture: `nodes_expanded`, `nodes_pruned_score`,
`nodes_pruned_cost`, `lower_bound`, `gap`, `completed`, and `warm_start_used`.

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

## Presolve: pruning before the search starts

Before the exact backend expands a single node, a **presolve** pass filters each feature's
candidate list independently: for every candidate state it computes, holding all other features
free, the interval of raw scores the ensemble could still reach (and, when plausibility is
configured, the reachable anomaly-score bracket) and discards the state when even that most
optimistic bracket cannot meet the target. The pass never changes any answer — a discarded state
is one no completion could have made feasible — it only shrinks the space branch-and-bound must
visit, so node counts drop while results stay bit-identical.

Two read-outs land in `solver_stats`:

- `presolve_removed`: how many candidate states the filter discarded across all features.
- `presolve_certified`: `True` when some feature's domain emptied entirely — the target is
  provably unreachable and the search returns `Infeasible.proof="certified"` immediately, with
  zero nodes expanded.

Because each state is tested with all other features free, the filter reaches a fixpoint in one
pass; it deliberately does not fold inter-feature constraints into the brackets, so a state kept
by presolve can still be pruned later by the search itself. Presolve runs on every exact solve,
point or [region](#regions-certified-not-maximal-not-monotone), in both engine implementations,
and its statistics appear in [certificates](#audit-certificates) like the rest of
`solver_stats`.

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
  never silently — see [When the budget runs out](#when-the-budget-runs-out) for the
  `TreecfWarning` this always triggers on `explain`/`explain_batch`/`explain_coalitions`).
- `node_budget` (default 2,000,000 assignments) and `gap` (default `0.0`) are the two pressure
  valves: lowering `node_budget` bounds worst-case wall time at the cost of a less certain
  answer, and a `gap > 0` lets the search settle for — and honestly report, via
  `proof="optimal_within_gap"` — a row provably within that relative fraction of the true
  optimum instead of paying for the last few percent.
- `warm_start` defaults to `True`: a short genetic pass (about a quarter of `time_budget_s`,
  capped at 2 seconds) seeds the exact search with an incumbent before it starts branching,
  which prunes harder from the first node without costing anything from the main budget — the
  exact search still gets the full `time_budget_s` afterward.
- The budgets can be removed entirely: `time_budget_s=math.inf` disables the time cut, and a
  very large `node_budget` (any value up to 2^64 − 1) makes the node cut unreachable, so the
  search runs until it proves optimality or certified infeasibility — practical since Ctrl-C
  now aborts promptly (see [Interrupting a search](#interrupting-a-search)). An unlimited
  budget guarantees the search completes, not that it certifies: the conservative constraint
  repair described above can still return an honestly-warned `proof="heuristic"`.

## The exact-batch opt-in

`explain_batch(..., backend="exact")` has no vectorized population to parallelize the way the
genetic engine does — it loops the single-instance exact solve per row (and per plan, for
`diversity="lever-blocking"`), sequentially, and each row still gets the full, undiminished
`time_budget_s`. That wall time is easy to underestimate from a single
`explain(..., backend="exact")` call, so the batch path requires explicit consent:
`backend="exact"` without `allow_exact_batch=True` raises `ValueError` instead of running,
naming the arithmetic behind the estimate — `rows × plans × time_budget_s`, hours-formatted,
where `plans` is `n_per_example` for `diversity="seeds"`/`"lever-blocking"` or the coalition
count (`len(coalitions) + 1` when `include_full=True`) for `diversity="coalitions"`. The figure
is a floor, not a ceiling: `diversity="seeds"` retries up to three attempts per requested plan
when a draw collides with one already found, so its actual wall time can run past the estimate;
the other two diversity modes never exceed it.

Opting in also changes how `warm_start` (default `True`) behaves. Instead of every row (or, in
`diversity="seeds"`, every attempt) running its own internal genetic warm pass, one vectorized
`Explainer._solve_batch` call warms every row at once, budgeted at
`min(time_budget_s * 0.25, 2.0)` regardless of row count — the saving is entirely in the warm-up,
not in the exact search itself: each row's exact solve afterward still gets the full
`time_budget_s`. A row whose warm draw comes back infeasible, or fails float-space verification,
gets no incumbent and runs unwarmed; there is no per-row genetic fallback.

For `diversity="seeds"`, the shared warm pass has a real trade-off: every attempt of a row starts
from the same one incumbent, rather than each attempt warm-starting its own the way a sequential
run of `explain` would. With `n_per_example=1` that makes no difference — the result matches a
sequential `explain(..., backend="exact")` call exactly — but for `n_per_example > 1` a batch run
is not guaranteed to explore as many distinct warm starts per row as an equivalent sequence of
`explain` calls would. `diversity="lever-blocking"` shares the incumbent for the primary solve
only; the per-lever frozen clones keep their own per-solve `warm_start`, since a `Freeze` changes
the constraint set enough that the primary's incumbent does not necessarily still verify for
them. `diversity="coalitions"` keeps per-coalition-solver `warm_start` throughout, for the same
reason.

## Audit certificates

`Explainer.certificate(x, result, target)` turns any stored result — a `Counterfactual` or an
`Infeasible`, fresh or reloaded years later — into a JSON-serializable audit record. A
certificate is a *reproducibility record plus a fresh verification*: it binds the claim to a
model fingerprint, a constraint fingerprint, and the solve parameters, and re-verifies the
returned plan at issue time — it does not cryptographically prove that a search ran or that a
`proof="optimal"` claim is true; re-running with the recorded seed and budgets on a
fingerprint-matching model is how a validator checks that.

The certificate is a plain `dict` with `"schema_version": 1` that serializes with
`json.dumps(cert, allow_nan=False, sort_keys=True)` — non-finite floats are encoded as the
strings `"NaN"`, `"Infinity"`, and `"-Infinity"` wherever they can occur. Its fields:

| Field | Contents |
|---|---|
| `schema_version`, `created_utc`, `treecf_version` | Schema version (`1`), timezone-aware ISO 8601 issue time, the issuing treecf version |
| `reproducible` (+ `reproducible_reason`) | `false` when a component has no canonical encoding (a callable `value_policy`), with the reason |
| `model` | `ir_fingerprint` (SHA-256 over a canonical byte encoding of the ensemble), feature names, link name; a `plausibility` sub-block (forest fingerprint, `min_total_path`) when configured |
| `constraints` | `fingerprint` (SHA-256 over the constraint set, `sigma`, `weights`, and value policies) plus a human-readable `listing` — the listing is for humans, the fingerprint is for machines |
| `target` | Declared space and bounds (the band's own name and bounds for a `Target.bands` result, passed via `band=`), the resolved raw interval actually used, and `"calibrator": "external — not embedded"` for calibrated targets |
| `solve` | `backend` (recovered from the result's own stats), `proof`, full `solver_stats`, and — when the caller supplies them — `seed`/`node_budget`/`gap`/`time_budget_s`/`warm_start` under `solve.declared` (the result object does not carry them, so their caller-supplied provenance stays explicit) |
| `factual` / `plan` / `infeasible` | The factual `x`; for a `Counterfactual`: `x_cf`, `changes`, `distance`, `snapped`, and region intervals when present; for an `Infeasible`: `reason` and `proof` |
| `verification` | Performed **fresh at issue time**, never copied from the solve: recomputed raw score, target membership, the compiled constraint check, plausibility when configured, and — for a region — a sampled set of re-checked points (each widened feature's endpoints plus the all-lo/all-hi corners; every checked point is recorded). For an `Infeasible` it records only whether the factual itself sits outside the target |

A certificate whose fresh verification fails is still issued, with the failing booleans
recorded — a certificate that refused to print a failure would hide exactly what it exists to
catch — but a `TreecfWarning` names the failed check.

`Explainer.check_certificate(cert)` is the validator's tool: it recomputes both fingerprints
against *this* explainer, re-runs the verification block from the certificate's stored
factual/plan, and reports — it never raises on a mismatch:

```python
# exp, x, target, res: the docs explainer, applicant, target, and solved plan
import json

cert = exp.certificate(x, res, target, seed=0, time_budget_s=10.0)
stored = json.dumps(cert, allow_nan=False, sort_keys=True)   # file it with the decision

# two years later, on the model and constraints the validator was handed:
report = exp.check_certificate(json.loads(stored))
report["model_match"]        # False if the ensemble was swapped
report["constraints_match"]  # False if the constraint set changed
report["verification_ok"]    # False if the stored plan no longer verifies
report["mismatches"]         # one human-readable string per mismatch
```

For batches: every `BatchRecord` now carries its own `proof` and `solver_stats`, mirroring the
single-instance result that produced it, so the aggregate degraded warning's pointer at each
record's own fields is true as written. Build certificates per row post-hoc via
`Explainer.certificate` — there is no batch convenience wrapper in this release.

## Interrupting a search

A `Ctrl-C` during an exact search, a certified-region growth, or a batch genetic solve raises
`KeyboardInterrupt` promptly instead of waiting for the whole search to finish. The Rust core
polls for it from inside its released GIL — for the exact search, about every 2^18 nodes — and
the pure-Python exact fallback raises just as promptly, for a different reason: Python delivers
signals between bytecode instructions, so there is nothing to poll for. Either way nothing is
returned: whatever incumbent the search was holding, or however much of a region it had grown so
far, is discarded rather than handed back partial.

This is reliable only when the call happens on the **main thread** — Python only delivers
`SIGINT` there, so a search launched from a worker thread will not see a `Ctrl-C` this way.

`time_budget_s` is an *anytime* budget, not a deadline the search waits out: an interrupt at any
point during it stops the search immediately rather than continuing to the budget's edge, and a
search that finds nothing before the interrupt lands leaves no result to fall back on — the same
`KeyboardInterrupt` propagates all the way up to caller code.

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
