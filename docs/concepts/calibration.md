# Calibration

Post-hoc calibration inserts a monotone map `g` between the model's probability
output and the number a decision actually uses: `p' = g(predict_proba(x))`.
Cutoffs, rating grades, and recourse policies are then stated on the
*calibrated* scale — and that changes how targets must be built, and nothing
else.

## The trap

`Target.probability` inverts the model's own sigmoid link, not `g`. Once
calibration is deployed, it silently targets the **uncalibrated** probability.
Concretely: suppose an isotonic `g` maps model-p 5% to calibrated 2%. The
intended policy "calibrated PD ≤ 2%" is satisfied by any point with model-p
≤ 5%; but `Target.probability(op="<=", value=0.02)` demands model-p ≤ 2% — a
materially harder, possibly unattainable, target. Nothing errors; the
counterfactuals are just wrong for the policy.

## Why only target construction changes

For a monotone `g`, the preimage identity holds:

```text
{x : g(f(x)) ∈ [lo, hi]}  =  {x : f(x) ∈ [g⁻¹(lo), g⁻¹(hi)]}
```

Calibration never changes counterfactual *geometry* — only the interval the
search must reach. The engine, constraints, pruning, verification, and the
Rust core all consume a raw interval exactly as before. `Target.calibrated`
therefore does one thing: it holds the calibrator and, when the model link is
known, inverts `[lo, hi]` through the calibrator's generalized inverse into
raw-margin bounds.

## The calibrator protocol

treecf imports no calibration library. Any object with these two members
works (e.g. a probcal calibrator):

```python
class SupportsIntervalInverse(Protocol):
    is_monotone_: bool
    def interval_inverse(
        self, lo: float, hi: float, *, space: str = "probability", buffer_logit: float = 0.0
    ) -> tuple[float, float]: ...
```

`space="logit"` (which treecf always uses) returns generalized-inverse bounds
on the logit of the model probability — for a SIGMOID-link ensemble that is
exactly the raw margin. `lo=0.0`/`hi=1.0` map to `-inf`/`+inf`. Non-monotone
calibrators are rejected at construction: the preimage of an interval under a
non-monotone map need not be an interval.

## Usage

```python
target = treecf.Target.calibrated(cal, op="<=", value=0.02)   # calibrated PD ≤ 2%
result = explainer.explain(x, target=target)
```

Masterscales defined on calibrated PD invert per band:

```python
target = treecf.Target.bands(
    {"A": (0.0, 0.005), "B": (0.005, 0.02), "C": (0.02, 0.10)},
    space="calibrated",
    calibrator=cal,
)
```

## Robustness to recalibration drift

Calibrators get refitted — quarterly central-tendency updates are routine in
credit risk. A counterfactual computed today can be invalidated by tomorrow's
recalibration. `buffer_logit=m` shrinks the calibrated interval by `m`
log-odds *before* inversion, so the produced counterfactual survives any
future drift of magnitude ≤ m:

```python
target = treecf.Target.calibrated(cal, op="<=", value=0.02, buffer_logit=0.1)
```

The trade is explicit: robustness paid in recourse difficulty. Two further
practical notes: the calibrator is held by reference (refitting it between
target construction and `explain` changes the inversion — reconstruct the
target after a refit), and step-shaped calibrators (isotonic, binning) make
counterfactuals near a block edge fragile — prefer a continuous calibrator or
a buffer when recourse is downstream.

`explain_batch` resolves the target's raw interval exactly once per call
(once per band for a bands ladder) and threads it through every row — the
calibrator's `interval_inverse` is never re-invoked per row.

## Calibrator provenance (0.2.4)

Certificates for calibrated targets record *which* calibrator produced the
stored raw interval, without embedding it:

```json
"calibrator": {
  "embedded": false,
  "fingerprint": "9f2a…",
  "type": "IsotonicCalibrator",
  "buffer_logit": 0.1
}
```

The `fingerprint` comes from the calibrator's own `fingerprint()` method when
it has one (probcal calibrators do); `null` otherwise. What it proves: a
later auditor holding the same calibrator object can tie the certificate to
it. What it does not prove: that the interval is correct — for that, pass the
calibrator back:

```python
report = explainer.check_certificate(cert, calibrator=cal)
report["calibrator_match"]   # fingerprints agree AND re-inverting the stored
                             # calibrated bounds reproduces the stored raw interval
```

Without `calibrator=`, `check_certificate` behaves exactly as in 0.2.3 (no
`calibrator_match` key). Batch records carry the same fingerprint per row
(`BatchRecord.calibrator_fingerprint`), so each JSON line stays
self-contained. Remember what the engine actually consumed: the raw
*interval*, not the calibrator — provenance ties the interval to its source,
verification always re-runs on the stored interval.

## The calibrated read-out

Results for calibrated targets carry `score_calibrated` — the calibrator's
probability at the counterfactual (and, on certificates, at the factual):

```python
res = explainer.explain(x, target=treecf.Target.calibrated(cal, op="<=", value=0.02))
res.score_raw          # what the engine optimized and verified against
res.score_calibrated   # g(sigmoid(score_raw)) — presentational
```

It is presentational only: the search satisfied the raw interval produced by
`interval_inverse`, and that remains the verified claim. The read-out needs
the calibrator to expose `predict_proba`; calibrators without it (the
protocol requires only `interval_inverse`) leave the field `None`, as do raw
and probability targets.

## Worked examples with probcal

[probcal](https://wlazlod.github.io/probcal/) implements the calibrator
protocol end to end; its [treecf guide](https://wlazlod.github.io/probcal/guide/treecf/)
walks through the pairing from the other side. The trio that covers most
policies:

```python
import probcal, treecf

cal = probcal.BetaCalibrator().fit(scores, y)
exp = treecf.Explainer(model=model, background=X)

# 1. Threshold policy on the calibrated scale
res = exp.explain(x, target=treecf.Target.calibrated(cal, op="<=", value=0.02))

# 2. Masterscale bands, each band inverted through the same calibrator
bands = treecf.Target.bands(
    {"A": (0.0, 0.005), "B": (0.005, 0.02), "C": (0.02, 0.10)},
    space="calibrated", calibrator=cal,
)

# 3. Drift-robust recourse after a macro adjustment
chain = probcal.Chain([cal, probcal.LogitOffset(delta=0.3).fit(cal.predict_proba(scores))])
res = exp.explain(x, target=treecf.Target.calibrated(chain, op="<=", value=0.02,
                                                     buffer_logit=0.2))
```
