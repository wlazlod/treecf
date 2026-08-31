# Set the target

A target is an interval on the model's output: the counterfactual is
feasible when the model's raw score lands inside it. Everything else —
probability cutoffs, rating ladders, calibrated policies — is a way of
constructing that interval. The full semantics are in
[Targets](../concepts/targets.md); this page is the workflow.

## Probability and raw

```python
from treecf import Target

Target.probability(op="<=", value=0.04)   # under a 4% PD cutoff (via logit)
Target.probability(range=(0.0, 0.05))     # inside a probability band
Target.raw(op=">=", value=1.5)            # raw margin / regression units
Target.raw(range=(-1.2, 0.5))
```

`Target.probability` inverts the model's own sigmoid, so it only exists for
models with a sigmoid link; a `RandomForestClassifier` (identity link over
averaged probabilities) takes `Target.raw` with probability-scale numbers
instead, and asking for `Target.probability` there raises `TargetError`
rather than silently targeting the wrong scale.

## Rating ladders: `Target.bands`

One call, one counterfactual (or certified infeasibility) per grade:

```python
# exp, x: the docs explainer and one rejected applicant
from treecf import Target

ladder = exp.explain(x, target=Target.bands({
    "A": (0.00, 0.01),
    "B": (0.01, 0.03),
    "C": (0.03, 0.07),
}), seed=0)
sorted(ladder)   # ["A", "B", "C"], each a Counterfactual or Infeasible
```

## Calibrated policies

When the deployed decision applies a post-hoc calibrator to the model's
probability, the policy lives on the calibrated scale, and
`Target.probability` is the wrong tool — it inverts the model's sigmoid, not
the calibrator. `Target.calibrated` takes any object with the duck-typed
calibrator protocol (`is_monotone_`, `interval_inverse`; every
[probcal](probcal.md) calibrator conforms):

```python
# exp, x, cal: the docs explainer, one rejected applicant, a fitted calibrator
import treecf

res = exp.explain(x, target=treecf.Target.calibrated(cal, op="<=", value=0.02), seed=0)
res.score_calibrated   # the calibrated read-out at the counterfactual
```

`buffer_logit=` shrinks the interval before inversion so a bounded future
recalibration cannot invalidate the plan; `Target.bands(...,
space="calibrated", calibrator=cal)` puts a whole masterscale on the
calibrated scale. [Calibration](../concepts/calibration.md) covers the trap,
the protocol, and provenance in certificates.

## Next

With the target set, declare what may change and by how much:
[constrain the search](constraints.md). Or step back to
[bring your model](models.md).
