# Targets

A target is an interval on the **raw model output** — this one abstraction
covers probability cutoffs, regression goals, and rating ladders.

```python
Target.probability(op="<=", value=0.04)      # under the 4% PD cutoff (via logit)
Target.probability(range=(0.2, 0.8))         # inside a probability band
Target.raw(op=">=", value=1.5)               # raw margin / regression units
Target.raw(range=(-1.2, 0.5))
```

Probability targets require a SIGMOID-link model and are converted once via the
logit; open endpoints (`0`/`1`) map to infinities.

## Rating ladders

`Target.bands` solves one model compilation against several intervals — the
"price of each grade":

```python
ladder = exp.explain(x, target=Target.bands({
    "A": (0.00, 0.01),
    "B": (0.01, 0.03),
    "C": (0.03, 0.07),
}))
# {"A": Counterfactual | Infeasible, "B": ..., "C": ...}
```

The AIM is compiled once and only the score bounds are swapped per band, so an
N-band ladder costs one compilation plus N solves.
