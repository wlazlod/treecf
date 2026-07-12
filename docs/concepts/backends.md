# Backends

Counterfactual search runs on a constrained genetic algorithm with two
interchangeable engines:

| backend | engine | when |
|---|---|---|
| `"genetic"` (default) | bundled Rust core | 44–58× faster than the numpy engine ([benchmarks](../benchmarks-genetic-rust.md)); typically milliseconds even on 300-tree models |
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

## History

Earlier development versions included an exact CP-SAT backend (via OR-Tools)
with optimality proofs. It was removed before the first release: it duplicated
capability available in dedicated exact-optimization packages, its solve times
missed targets on large ensembles, and maintaining two backend families
doubled the surface of every change. Users needing provably optimal
counterfactuals can pair treecf's IR with an exact solver directly.
