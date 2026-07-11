# Backends and proofs

Backend selection is **explicit** — there is no silent fallback.

| backend | dependency | proof | when |
|---|---|---|---|
| `"cpsat"` | `treecf[cpsat]` (ortools) | `optimal` (or `feasible` on timeout, with a gap) | default; exact optimality proofs |
| `"genetic"` | none (numpy) | `heuristic` | solver-free environments; never claims optimality |
| `"highs"` | planned v0.2 | — | raises `NotImplementedError` |

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
