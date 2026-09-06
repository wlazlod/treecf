# Backends

`explain(..., backend=...)` selects one of three engines:

| backend | engine | proof | when |
|---|---|---|---|
| `"genetic"` (default) | bundled Rust core | `"heuristic"` | 44–58× faster than the numpy engine ([performance](#performance)); typically milliseconds even on 300-tree models |
| `"python"` | pure numpy | `"heuristic"` | reference implementation, kept for cross-checking and as the behavioral baseline the Rust core is tested against |
| `"exact"` | rust-first branch-and-bound, numpy fallback | `"optimal"` / `"optimal_within_gap"` / `"heuristic"` | a proof matters more than solve time — see [the exact backend](#the-exact-backend) below |

All three share one constraint compiler and one notion of the search space (the
[cells](#how-the-search-works) below), and every result — target, every constraint, the
plausibility bound — is re-verified in float space against the IR before being returned; an
invalid candidate is never returned. `"genetic"` and `"python"` are additionally held to
statistical parity (identical outcome distributions across seeds) and are seed-deterministic;
both always report `proof="heuristic"` — feasibility-first and excellent in practice (they
bracket a brute-force oracle on toy suites), but never a claim of optimality. `"exact"` is
covered on its own below; [Certification](certification.md) has the full proof taxonomy for all
three backends, what a proof does and does not cover, and the certified-region layer that sits
on top of any of them.

## How the search works

The model's trees induce, per feature, a set of **cells** — maximal intervals
within which every tree routes identically. The GA seeds its first generation
with the factual instance, one candidate per (feature × cell) move, NaN flips
where `AllowMissing` permits, and background-sample crossovers; evolution uses
feasibility-first (Deb) ranking, uniform crossover, cell-jump/Gaussian/NaN
mutations and a revert-to-factual mutation that drives sparsity.

Candidate values placed next to a decision threshold are kept one *float32*
ulp away from it, so the deployed model (which compares in float32) routes
them the same way the IR does.

## The exact backend

`backend="exact"` searches the same kind of cell grid depth-first with branch-and-bound instead
of evolving a population, and proves what it finds: `proof="optimal"` when no cheaper feasible
row exists in the searched grid, `proof="optimal_within_gap"` when `gap > 0` bounded the proof to
a relative fraction of the optimum instead, and `proof="heuristic"` for a real, verified row it
is not claiming is cheapest — the usual outcome when a budget runs out on a wide model with
every feature free, always accompanied by a warning. `Infeasible.proof="certified"` means the whole
reachable grid was tried and nothing was feasible; the ordinary `"search_exhausted"` means only
that nothing was found. [Certification](certification.md) has the complete taxonomy, the two
honesty notes worth reading before trusting a `"heuristic"` or a `"certified"` result, and how
`value_policy` changes what "optimal" is measured against.

**Rust-first, result-identical fallback.** The search dispatches to a `_treecf_core` Rust
extension when it is importable (bundled in wheel installs; built by `uv sync` in a dev
checkout) and falls back to the pure-Python branch-and-bound otherwise. The fallback is not a
lesser engine: the Rust and Python implementations are proven bit-parity on a fixture set — same
`x_cf` (or both `None`), same `distance`, `proof`, and every `solver_stats` key — so which one
ran changes nothing about the answer, only how fast it arrived.

**Constraint coverage.** Single-feature `Linear` constraints and the canonical two-feature order
pair (`constraint("a <= b")`) are supported exactly. A `Linear` over three or more features, or
any other two-feature shape, raises `ConstraintValidationError` naming `backend="genetic"` as the
fallback — the exact search does not silently solve a smaller problem than the one declared. A
callable `value_policy` is rejected the same way; string and `Grid` policies are supported.

**Cost.** Proof comes at the price of a search that can run to the full `node_budget`
(2,000,000 assignments by default) or `time_budget_s` before answering, unlike the genetic
engine's typical milliseconds. `warm_start=True` (the default) seeds the search with a quick
genetic pass so pruning starts strong; `node_budget` and `gap` are the two levers for trading
proof strength against wall time. See [Certification — scaling
guidance](certification.md#scaling-guidance) for what problem sizes are realistic. Measured
exact-backend solve times are not yet published on this page; the benchmark protocol below
(pre-registered, seeds fixed, backends interleaved on one machine) is what a future exact-backend
number here will follow.

## Performance

The Rust core was gated on a pre-registered benchmark against the numpy
reference before becoming the default: identical seeds per configuration,
infinite time budget (stall/max-generation stopping only), only the solve call
timed, backends interleaved on the same machine. Median results (2026-07-12,
XGBoost binary classifiers, population 80):

| Scenario | numpy | Rust | Speedup |
|---|---|---|---|
| 300 trees, depth 6, 50 features | 4.03 s | 0.070 s | 58× |
| — same, single-threaded (`RAYON_NUM_THREADS=1`) | 3.80 s | 0.154 s | 24.6× |
| 300t/d6/50f + constraint mix + isolation forest | 6.25 s | 0.108 s | 58× |

Most of the gain is single-core: numpy's level-synchronous batch traversal
pays Python/numpy dispatch overhead per tree per depth level on small GA
populations, while the Rust core does a scalar per-row walk over a flat
structure-of-arrays. Rayon over population rows adds the rest.

`explain_batch` additionally fans whole waves of independently seeded solves
across cores in one Rust call — ~1.7× batch throughput on a 4-core machine,
growing with core count — with records identical to solving the rows in a
sequential loop. One caveat: `time_budget_s` stays a per-solve wall-clock
budget, and a solve that actually hits it while sharing cores may stop a
generation earlier than it would alone; stall and max-generation stops, the
common case, are deterministic.

Reproduce with `scripts/bench_genetic.py` and `scripts/bench_batch.py` in the
repository.

### Against other CF libraries

Measured against the pip-installable counterfactual libraries for tree models
— [DiCE](https://github.com/interpretml/DiCE) (its random, genetic, and kdtree
modes) and [NICE](https://github.com/DBrughmans/NICE) — under one protocol: the
same XGBoost model and declined rows per scenario, one counterfactual each, and
the class flip (probability < 0.5) as the goal, since that is the only target
every library expresses natively. Per-instance wall time excludes each method's
one-time setup; validity is re-checked against the model, never taken from the
library. Distance is the σ-normalized L1 over changed features — lower is a
cheaper, more actionable plan.

treecf appears three times: the default genetic backend without constraints, so
no method solves a harder problem; the exact backend in its refine search on
the same problem, attempting a proof within a 10 s wall budget, with the proof
mix of its hundred (or fifty) results in the last column; and the genetic
backend with the scenario's constraints, which no competitor can express. The
competitor rows and the throughput loops were measured on 2026-09-06 against
the package published on PyPI that day; the treecf rows were re-timed on the
same machine against the code in this repository, which no longer spends
seconds sizing the search space for the budget warning after the budget has
ended. DiCE's kdtree mode is skipped where its per-instance time runs to
minutes.

Three scenarios: two synthetic populations, and one public table — OpenML's
default-of-credit-card-clients (30,000 rows, 23 features) — whose constrained
row freezes the demographic columns and puts every column under an integer
value policy.

**medium (120 trees, depth 4, 8 features)** — 100 declined rows:

| Method | Valid | Median / instance | p95 | Features changed | Distance (L1/σ) | Proof mix |
|---|---|---|---|---|---|---|
| treecf (genetic) | 100/100 | 0.017 s | 0.042 s | 1.6 | 1.0 | — |
| treecf (exact, refine) | 100/100 | 0.032 s | 0.234 s | 3.0 | 0.4 | optimal 100 |
| treecf (genetic, constrained) | 100/100 | 0.018 s | 0.035 s | 1.7 | 1.2 | — |
| DiCE (random) | 100/100 | 0.125 s | 0.189 s | 1.7 | 15.5 | — |
| DiCE (genetic) | 100/100 | 0.105 s | 0.120 s | 5.1 | 7.5 | — |
| DiCE (kdtree) | 100/100 | 0.207 s | 0.784 s | 5.3 | 8.4 | — |
| NICE (sparsity) | 100/100 | 0.005 s | 0.010 s | 2.0 | 2.7 | — |

Batch throughput over 500 rows:

| Method | Wall | Rows/s |
|---|---|---|
| treecf explain_batch (one call) | 3.3 s | 149.8 |
| treecf explain loop | 6.9 s | 72.3 |
| NICE loop | 2.9 s | 173.8 |
| DiCE (random) loop | 80.0 s | 6.3 |

**large (300 trees, depth 6, 50 features)** — 50 declined rows:

| Method | Valid | Median / instance | p95 | Features changed | Distance (L1/σ) | Proof mix |
|---|---|---|---|---|---|---|
| treecf (genetic) | 50/50 | 0.073 s | 0.116 s | 1.4 | 3.0 | — |
| treecf (exact, refine) | 50/50 | 10.130 s | 10.205 s | 1.4 | 3.0 | heuristic 50 |
| treecf (genetic, constrained) | 50/50 | 0.072 s | 0.139 s | 2.3 | 5.3 | — |
| DiCE (random) | 50/50 | 1.538 s | 1.601 s | 1.8 | 7.7 | — |
| DiCE (genetic) | 50/50 | 0.712 s | 0.757 s | 50.0 | 62.4 | — |
| NICE (sparsity) | 50/50 | 0.014 s | 0.019 s | 2.2 | 5.2 | — |

Batch throughput over 200 rows:

| Method | Wall | Rows/s |
|---|---|---|
| treecf explain_batch (one call) | 11.8 s | 17.0 |
| treecf explain loop | 14.0 s | 14.3 |
| NICE loop | 3.1 s | 64.1 |
| DiCE (random) loop | 405.9 s | 0.5 |

**public (credit-card default, 200 trees, depth 5, 23 features)** — 100 declined rows:

| Method | Valid | Median / instance | p95 | Features changed | Distance (L1/σ) | Proof mix |
|---|---|---|---|---|---|---|
| treecf (genetic) | 100/100 | 0.029 s | 0.038 s | 0.9 | 0.2 | — |
| treecf (exact, refine) | 100/100 | 0.090 s | 2.962 s | 1.3 | 0.0 | heuristic 4, optimal 96 |
| treecf (genetic, constrained) | 100/100 | 0.036 s | 0.072 s | 1.2 | 0.8 | — |
| DiCE (random) | 100/100 | 0.574 s | 0.825 s | 1.6 | 156.0 | — |
| DiCE (genetic) | 99/100 | 0.580 s | 0.810 s | 14.2 | 20.0 | — |
| NICE (sparsity) | 100/100 | 0.030 s | 0.071 s | 1.6 | 4.6 | — |

Batch throughput over 500 rows:

| Method | Wall | Rows/s |
|---|---|---|
| treecf explain_batch (one call) | 12.8 s | 39.1 |
| treecf explain loop | 15.0 s | 33.4 |
| NICE loop | 14.3 s | 34.9 |
| DiCE (random) loop | 221.4 s | 2.3 |

Honest reading, per row.

*The genetic backend* is the cheapest heuristic on every scenario: 1.0 σ on the medium
model against 2.7 for NICE and 7.5–15.5 for DiCE, 3.0 against 5.2 and 7.7–62 on the large
one, in 17–74 ms per instance with every plan valid. NICE is the faster engine per instance
on all three scenarios (5–30 ms, a lean nearest-neighbour greedy that copies values from
real training rows), and on the large model its loop beats treecf's batch mode too, because
treecf's within-solve parallelism already saturates four cores on 50-feature populations;
treecf's batch call pays on the medium and public tables. What NICE cannot do is the rest
of the job: its plans cost 1.5–5× more, and it has no constraint mechanism, no
probability-interval target, and no float verification. DiCE's genetic mode changes every
feature of the large model and fourteen of the public table's; its random mode is sparse but
lands far from the factual.

*The exact refine row* is where the envelope shows, and the proof column says how much of
each row is proof. On the medium model it proves all 100 rows optimal in 32 ms median — but
those plans change 3.0 features where the heuristic's change 1.6, because the objective is
distance and nothing else ([what optimal means](certification.md#what-optimal-means-and-what-it-does-not)).
On the large model it proves nothing: all 50 solves run to the 10 s wall budget (10.1 s
median, the budget honoured to a tenth of a second) and return the plan the genetic warm
start had already found, labelled `heuristic`. On the public table it reports 96 proofs in
90 ms median with a 3 s tail, and those proofs are cheap for the same reason the
unconstrained plans are: with a distance objective, the optimum is a one-ulp move across a
split XGBoost placed on an observed integer value (0.03 σ), so the proof certifies something
nobody would act on. This is the [proof envelope](certification.md#the-proof-envelope-measured)
in one column: proofs come cheaply at eight features, trivially on a table of integer codes,
and not at all at fifty features with every feature free.

*The constrained row* is the one to read for the public table. Unconstrained, treecf's plans
there move an integer-coded column by one float32 ulp — valid for the model, and useless to a
person. With the demographic columns frozen and an integer value policy on every column, the
constrained row changes 1.2 features by whole units at 0.8 σ, in 36 ms, all valid — still a
fifth of NICE's cost. On the synthetic scenarios the constraints cost little time and some
distance, as constraints should.

alibi's `CounterfactualProto`, measured separately in an earlier run (it needs
TensorFlow and is therefore not in the script), ran in black-box mode with
numerical gradients: about 72 s per instance on the medium model with 3 of 5
attempts succeeding, and 43 s returning **no counterfactual at all** on the
large-model probe — gradient-based methods pay dearly on non-differentiable
ensembles.

Caveats: one machine (4 cores, otherwise idle), default competitor settings,
pure-Python libraries against a compiled core, and a 10 s per-solve budget for
the exact row. Reproduce with `uv run scripts/bench_vs_competitors.py` — its
inline metadata pulls dice-ml and NICEx automatically and downloads the public
table through scikit-learn; `--treecf-only` re-times the treecf rows alone
against a checkout.

## History

Earlier development versions included a different exact backend, `backend="cpsat"`, built on
OR-Tools CP-SAT. It was removed before the first release: it duplicated capability available in
dedicated exact-optimization packages, its solve times missed targets on large ensembles, and
maintaining two backend families doubled the surface of every change.

The current `backend="exact"` is not a revival of that one. It has no solver dependency at all —
branch-and-bound over treecf's own cell grid, in Python with a bit-parity Rust mirror, using the
same constraint compiler and IR every other backend uses — which sidesteps the two reasons the
CP-SAT backend was cut: nothing to duplicate a dedicated package's job, and one backend family
throughout, not two.
