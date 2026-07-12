# Backends and proofs

Backend selection is **explicit** — there is no silent fallback.

| backend | dependency | proof | when |
|---|---|---|---|
| `"cpsat"` | `treecf[cpsat]` (ortools) | `optimal` (or `feasible` on timeout, with a gap) | default; exact optimality proofs |
| `"genetic"` | none (bundled Rust core) | `heuristic` | solver-free environments; 44–58× faster than the numpy engine ([benchmarks](../benchmarks-genetic-rust.md)) |
| `"python"` | none (numpy) | `heuristic` | the original pure-Python GA, kept as a reference engine |
| `"highs"` | planned | — | raises `NotImplementedError` |

The Rust and Python genetic engines share one constraint compiler and are held
to statistical parity (identical outcome distributions across seeds); every
result from either engine is float-verified in Python before being returned.

Requesting `backend="cpsat"` without ortools raises
`MissingExtraError("pip install treecf[cpsat]")`.

## How the exact backend works

The model's trees induce, per feature, a set of **cells** — maximal intervals
within which every tree routes identically. The optimization picks one cell per
feature plus an exact value inside it, at integer scale K = 10⁶, minimizing the
weighted L1 distance (normalized per feature by a MAD → IQR → range chain) plus
an optional sparsity term. Every candidate is re-verified in float space; if
fixed-point resolution ever bites, the scale is raised ×10 and re-solved.

## Diverse counterfactuals

```python
results = exp.explain(x, target=t, n_counterfactuals=3)   # CP-SAT only
```

Each solve adds a no-good cut. The default `distinct_changes` mode forbids
repeating an exact change-set (a NaN flip counts as a change);
`distinct_solution` forbids the exact cell assignment. Results come back in
non-decreasing cost order.
