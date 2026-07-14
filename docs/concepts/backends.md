# Backends

Counterfactual search runs on a constrained genetic algorithm with two
interchangeable engines:

| backend | engine | when |
|---|---|---|
| `"genetic"` (default) | bundled Rust core | 44–58× faster than the numpy engine ([performance](#performance)); typically milliseconds even on 300-tree models |
| `"python"` | pure numpy | reference implementation, kept for cross-checking and as the behavioral baseline the Rust core is tested against |

Both engines share one constraint compiler, are held to statistical parity
(identical outcome distributions across seeds), and are seed-deterministic.
Results carry `proof="heuristic"`: the search is feasibility-first and
excellent in practice (it brackets a brute-force oracle on toy suites), but it
does not prove optimality. Every result — target, every constraint, the
plausibility bound — is re-verified in float space against the IR before being
returned; an invalid candidate is never returned.

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
— [DiCE](https://github.com/interpretml/DiCE) (all three model-agnostic modes)
and [NICE](https://github.com/DBrughmans/NICE) — under one protocol: the same
XGBoost model and declined rows per scenario, one counterfactual each, and the
class flip (probability < 0.5) as the goal, since that is the only target
every library expresses natively. Per-instance wall time excludes each
method's one-time setup; validity is re-checked against the model, never taken
from the library; treecf (v0.0.1, from PyPI) runs without constraints so no
method solves a harder problem. Distance is the σ-normalized L1 over changed
features — lower is a cheaper, more actionable plan.

**Medium model** — 120 trees, depth 4, 8 features; 100 declined applicants:

| Method | Valid | Median / instance | p95 | Features changed | Distance (L1/σ) |
|---|---|---|---|---|---|
| treecf | 98/100 | 0.021 s | 0.051 s | 1.7 | **1.0** |
| NICE (sparsity) | 100/100 | **0.008 s** | 0.022 s | 2.0 | 2.7 |
| DiCE (genetic) | 100/100 | 0.172 s | 0.400 s | 5.1 | 7.7 |
| DiCE (random) | 100/100 | 0.229 s | 0.368 s | 1.6 | 14.8 |
| DiCE (kdtree) | 100/100 | 0.305 s | 1.208 s | 5.3 | 8.4 |

**Large model** — 300 trees, depth 6, 50 features; 50 declined rows:

| Method | Valid | Median / instance | p95 | Features changed | Distance (L1/σ) |
|---|---|---|---|---|---|
| treecf | 50/50 | 0.29 s | 0.47 s | 2.5 | **3.6** |
| NICE (sparsity) | 50/50 | **0.015 s** | 0.020 s | 2.2 | 5.2 |
| DiCE (genetic) | 50/50 | 1.94 s | 2.48 s | 50.0 | 62.1 |
| DiCE (random) | 50/50 | 3.49 s | 4.73 s | 1.9 | 8.3 |
| DiCE (kdtree) | 50/50 | 997 s | 2147 s | 50.0 | 63.3 |

Whole-dataset production (500 rows on the medium model, 200 on the large):

| Method | Medium: rows/s | Large: rows/s |
|---|---|---|
| treecf `explain_batch` (one call) | **157** | 16 |
| treecf `explain` loop | 51 | 14 |
| NICE loop | 44 | **59** |
| DiCE (random) loop | 3.5 | 0.2 |

Honest reading. DiCE degrades hard with model size: 12–3400× slower than
treecf per instance on the large model, with its genetic and kdtree modes
returning "plans" that change **all fifty features** (kdtree averages ~17
minutes per instance). NICE — a lean nearest-neighbor greedy — is the genuine
speed rival: fastest per instance on both models and, on the large model,
faster than treecf's batch mode too, because treecf's within-solve
parallelism already saturates a 4-core machine on 50-feature populations
(`explain_batch`'s task-level parallelism pays on wider machines and smaller
models, where it reaches 157 rows/s). What NICE cannot do is the rest of the
job: its plans cost 1.5–2.7× more (it copies values from real training rows
rather than taking threshold-aware minimal steps), and it has no constraint
mechanism, no probability-interval targets, and no float verification.
treecf missed 2 of 100 medium-model instances at the default budget — the
search is heuristic and says so.

alibi's `CounterfactualProto`, measured separately (it needs TensorFlow and is
therefore not in the script), ran in black-box mode with numerical gradients:
~72 s per instance on the medium model with 3 of 5 attempts succeeding, and
43 s returning **no counterfactual at all** on the large-model probe —
gradient-based methods pay dearly on non-differentiable ensembles.

Caveats: one synthetic dataset per scale, one machine (4 cores), default
competitor settings, and pure-Python libraries against a compiled core.
Reproduce with `uv run scripts/bench_vs_competitors.py` — its inline metadata
pulls dice-ml and NICEx automatically.

## History

Earlier development versions included an exact CP-SAT backend (via OR-Tools)
with optimality proofs. It was removed before the first release: it duplicated
capability available in dedicated exact-optimization packages, its solve times
missed targets on large ensembles, and maintaining two backend families
doubled the surface of every change. Users needing provably optimal
counterfactuals can pair treecf's IR with an exact solver directly.
