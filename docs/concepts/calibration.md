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
