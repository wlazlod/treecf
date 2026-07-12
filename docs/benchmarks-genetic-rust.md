# Genetic backend: Rust core vs pure-Python (numpy)

Benchmark gate for the Rust migration (2026-07-12). Protocol: identical seeds
per configuration, infinite time budget (stall/max-generation stopping only, so
neither backend gets fewer generations by being slower), only the solve call
timed (model parsing, marshaling and warmup excluded), backends interleaved per
seed on the same machine. `us/eval` normalizes by the number of individual
evaluations (neutralizing the oversized first generation).

XGBoost binary classifiers; population 80 (except where noted); `lam=0.05`;
background sample 2000 rows. 30 seeds on the headline row, 10–20 elsewhere.

| Scenario | Python median | Python us/eval | Rust median | Rust us/eval | Speedup |
|---|---|---|---|---|---|
| **300 trees, depth 6, 50 features (headline)** | 4.03 s | 382 | **0.070 s** | 7.0 | **58x** |
| — same, single-threaded Rust (`RAYON_NUM_THREADS=1`) | 3.80 s | 372 | 0.154 s | 15.2 | 24.6x |
| 100 trees, depth 6, 20 features | 1.53 s | 215 | 0.033 s | 3.5 | 47x |
| 300t/d6/50f + constraint mix + isolation forest | 6.25 s | 599 | 0.108 s | 9.8 | 58x |
| 300t/d6/50f, population 200 | 4.91 s | 302 | 0.112 s | 7.3 | 44x |

**Gate verdict: PASS** — far beyond the pre-registered 2x threshold, so the
Rust engine becomes the default genetic backend (the pure-Python implementation
remains available as `backend="python"`).

Where the speedup comes from: mostly single-core. numpy's level-synchronous
batch traversal pays Python/numpy dispatch overhead per tree per depth level on
small GA populations (80–200 rows), while the Rust core does a scalar per-row
walk over a flat structure-of-arrays — 24.6x before any parallelism. Rayon
over rows adds a further ~2.2x on this machine.

Correctness context: the Rust core is bitwise-identical to Python on tree
evaluation and constraint check/repair, statistically indistinguishable on
end-to-end GA outcomes across 200 seeds x 10 scenarios (feasibility, J via KS,
generations), and every returned counterfactual is float-verified in Python.
Reproduce with `scripts/bench_genetic.py`.

## Batch production (`explain_batch`)

Added 2026-07-12: `explain_batch` fans its independently seeded `(row, seed)`
solves out with rayon inside one Rust call per attempt wave (seeds diversity)
and batches all primary solves (lever-blocking); per-counterfactual
verification scores come from one vectorized IR pass per wave. Records are
identical to the sequential per-row loop (asserted inside the benchmark).

Protocol: medium model (100 trees, depth 6, 20 features), 100 rows,
`n_per_example=3`, warm caches, `time_budget_s=10`. Measured on a 4-core
machine — the sequential baseline already uses rayon inside each solve, so the
batch gain grows with core count.

| Diversity | Sequential loop | Batched | Speedup |
|---|---|---|---|
| seeds (4 rayon threads) | 14.7 s (6.8 rows/s) | **8.6 s (11.7 rows/s)** | **1.7x** |
| seeds (`RAYON_NUM_THREADS=1`) | 23.2 s (4.3 rows/s) | 19.4 s (5.2 rows/s) | 1.2x |
| lever-blocking (4 threads) | 7.6 s (13.2 rows/s) | 6.7 s (15.0 rows/s) | 1.1x |

Determinism caveat: each task keeps its own per-solve `time_budget_s`, but
concurrent tasks share cores, so a task that hits its wall-clock budget under
contention may stop at a different generation than it would sequentially.
Results are bit-identical whenever no task hits its budget (stall and
max-generation stops are deterministic). Reproduce with
`scripts/bench_batch.py`.
