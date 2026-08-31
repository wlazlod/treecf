# probcal: recourse on calibrated policies

<!-- docs: requires probcal -->

[probcal](https://github.com/wlazlod/probcal) fits post-hoc probability
calibrators; treecf solves counterfactuals. When the deployed decision is a
cutoff on the *calibrated* probability, the two compose through one
duck-typed protocol: `Target.calibrated` accepts any object with
`is_monotone_` and `interval_inverse` — every probcal calibrator,
`LogitOffset`, `Chain`, and `CalibratedModel` conform. treecf never imports
probcal at runtime; probcal's side of this integration is its own
[treecf guide](https://wlazlod.github.io/probcal/guide/treecf/).

## A calibrated cutoff, end to end

```python
# exp, x, X_bg: the docs explainer, one rejected applicant, background rows
import numpy as np
from probcal import BetaCalibrator
from treecf import Target

# fit on held-out scores and outcomes (synthetic here, yours in practice)
rng = np.random.default_rng(0)
scores = rng.uniform(0.01, 0.6, size=400)
y = (rng.random(400) < scores).astype(np.float64)
pcal = BetaCalibrator().fit(scores, y)

res = exp.explain(x, target=Target.calibrated(pcal, op="<=", value=0.10),
                  seed=0, backend="exact")
res.score_calibrated   # the calibrated probability at the counterfactual
```

treecf resolves the calibrated target once, through
`interval_inverse(..., space="logit")`, which gives bounds on the raw
margin, and optimizes there; the calibrated read-out is exact because the
inverse is.

## Step calibrators and plateaus

Isotonic-family calibrators map whole raw regions to one level. probcal's
generalized inverse returns the largest raw score inside the preimage, so a
counterfactual against a plateau level lands on the block boundary — the
cheapest qualifying raw score — never overshooting into the next block. An
engine-level plateau suite pins that both against probcal's real isotonic
fits and against a counting stub of its inverse contract.

## Drift-robust recourse

Two probcal tools carry over directly:

- `Chain([cal, LogitOffset(...)])` inverts a re-anchored deployment exactly;
  inverting the base calibrator alone answers yesterday's policy.
- `buffer_logit=` on `Target.calibrated` shrinks the interval before
  inversion, so a future central-tendency update up to that magnitude leaves
  the plan valid. The principled value is the offset confidence-sequence
  half-width from probcal's monitor.

## Provenance

A certificate for a calibrated target embeds the calibrator's fingerprint
(probcal objects all provide `fingerprint()`), and
`check_certificate(cert, calibrator=...)` re-checks it and re-inverts the
stored calibrated bounds against the stored raw interval — the certificate
plus the calibrator's probcal JSON is a self-contained, independently
verifiable pair. Details: [calibration](../concepts/calibration.md#calibrator-provenance)
and [auditability](auditability.md).

## Pitfall

`Target.probability` inverts the *model's* sigmoid, not the calibrator: a
"2% PD" policy defined on calibrated probabilities but requested through
`Target.probability(op="<=", value=0.02)` silently targets the wrong
quantity whenever the calibrator is not the identity. Calibrated policies go
through `Target.calibrated` (or `Target.bands(space="calibrated")`), always.
