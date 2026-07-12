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

## History

Earlier development versions included an exact CP-SAT backend (via OR-Tools)
with optimality proofs. It was removed before the first release: it duplicated
capability available in dedicated exact-optimization packages, its solve times
missed targets on large ensembles, and maintaining two backend families
doubled the surface of every change. Users needing provably optimal
counterfactuals can pair treecf's IR with an exact solver directly.
