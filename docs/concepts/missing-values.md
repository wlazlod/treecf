# Missing values

NaN is a first-class counterfactual value: "close the delinquency record" can be
a legitimate, priceable recommendation.

```python
from treecf import AllowMissing

AllowMissing("months_since_last_delinquency", delta_miss=2.0)
AllowMissing("bureau_score", delta_miss=3.0, delta_from_miss=1.0)   # asymmetric
```

- Without `AllowMissing`, a feature never becomes NaN, and a NaN factual stays
  fixed (routing follows each node's missing direction).
- With it, the missing state is one more option: flipping value → NaN costs
  `delta_miss` (in the feature's normalized units), NaN → value costs
  `delta_from_miss` (defaults to `delta_miss`). There is deliberately no
  default for these deltas — MAD-based defaults are meaningless for this
  transition.

## Interaction with linear constraints

A `Linear` constraint that references a missing feature is resolved by its
`missing_policy`:

| policy | meaning |
|---|---|
| `"satisfied"` (default) | vacuously true — "no delinquency history" satisfies a DPD-consistency rule |
| `"forbid_missing"` / `"violated"` | the counterfactual may not use NaN for the referenced features |

Mined constraints (`suggest_constraints`) also report missingness links
(`miss(A) => miss(B)`) as joint-`AllowMissing` recommendations.
