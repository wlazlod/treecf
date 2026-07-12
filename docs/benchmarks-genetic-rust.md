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
