# FAQ

!!! info "Shared objects"
    Snippets on this page continue from the objects the [quickstart](getting-started.md) builds with
    `credit_demo()`: `exp`, `x`, `target`, `X_bg`, the solved `res` and `batch`, and `cal`,
    a fitted monotone calibrator (see the [FAQ](faq.md#how-do-i-target-a-calibrated-probability)).

**Why does `Target.probability` fail on my RandomForest?**
Forest classifiers average probabilities; there is no sigmoid link to invert.
Their raw score *is* the averaged probability — use
`Target.raw(range=(0.0, 0.3))`.

**How do I target a *calibrated* probability?**
If model outputs are post-hoc calibrated (`p' = g(predict_proba)`), decisions
are made on the calibrated scale — and `Target.probability` becomes a silent
trap: it inverts the model's own sigmoid link, not `g`. Example: with an
isotonic `g` mapping model-p 5% to calibrated 2%, `Target.probability(op="<=",
value=0.02)` demands model-p ≤ 2% — a materially harder (or unattainable)
target than the intended calibrated-PD ≤ 2%. Use `Target.calibrated` with any
calibrator object (e.g. a probcal calibrator) satisfying the duck-typed
protocol — no calibration library is imported:

```python
from typing import Protocol

class SupportsIntervalInverse(Protocol):
    is_monotone_: bool
    def interval_inverse(
        self, lo: float, hi: float, *, space: str = "probability", buffer_logit: float = 0.0
    ) -> tuple[float, float]: ...
```

```python
import treecf

cal_target = treecf.Target.calibrated(cal, op="<=", value=0.02)   # calibrated PD ≤ 2%
result = exp.explain(x, target=cal_target, seed=0)
```

Pass `buffer_logit=m` to guard the counterfactual against future
recalibration or central-tendency drift of magnitude ≤ m in log-odds. For a
masterscale defined on calibrated PD, bands invert per band:

```python
import treecf

bands = treecf.Target.bands(
    {"A": (0.0, 0.005), "B": (0.005, 0.02), "C": (0.02, 0.10)},
    space="calibrated",
    calibrator=cal,
)
```

**Why is my counterfactual `Infeasible`?**
The search exhausted its budget without a candidate satisfying the target and
every constraint. Check for contradictory constraints (e.g. everything frozen),
an unreachable target interval, or raise `time_budget_s`.

**Can I run treecf where xgboost cannot be installed?**
Yes. Parsers accept JSON dumps (`Booster.save_model("model.json")`,
`dump_model()`, CatBoost `format="json"`), and the genetic backend has no
dependencies beyond the wheel itself: `pip install treecf` on the scoring host,
ship the dump file.

**Is there a wheel for a 32-bit Raspberry Pi?**
Not from this project's CI; 32-bit Raspberry Pi wheels arrive via piwheels'
own builders. The recurring Bookworm build failure there is an upstream
toolchain issue, not a treecf packaging bug. Every other platform ships
from CI as usual.

**What does `(data-limited)` mean in `region.describe()`?**
That side of the certified box stopped at the edge of the explainer's
background data rather than at a constraint or a split of the model. A
feature with no `Range` is grown no further than the data reaches, so an
unconstrained count reads `in [0, 1)` instead of `< 1` over an implicit
minus infinity; `RecourseRegion.data_limited` names the sides. Declare a
`Range` where the domain is known and the side stops there instead. An
explainer built from `normalizers` alone has no data range and lets such a
side run to infinity.

**Why did `proof` come back `"heuristic"` from the exact backend?**
Two causes, and the warning says which. Either the search ran out of its
`node_budget` or `time_budget_s` — the row is the best found, not proven
cheapest; raise the budgets, pass `gap=` to accept a proven tolerance, or
try `search="refine"` — or it withdrew its certificate without spending the
budget: an order-pair constraint (`constraint("a <= b")`) tied a feature
under a value policy, and a completion broke that pair on values the search
could not repair. The row itself is still float-verified; only the
"cheapest possible" claim is dropped. See
[certification](concepts/certification.md#two-honesty-notes).

**How do float32 casts affect routing?**
XGBoost, CatBoost, and scikit-learn round an input to float32 before
comparing it with a split threshold, so a float64 value within half a
float32 ulp of a threshold can route one way in the library and the other
way under a plain float64 comparison. The parsers store each threshold as
the float64 boundary of that cast — the largest value the native model still
routes left — so float64 inputs route as the deployed model routes them and
no pre-rounding is needed on your side. Candidate values placed next to a
threshold are kept one float32 ulp away for the same reason. LightGBM
compares in float64 and needs no adjustment. See
[models](concepts/models.md#float32-pitfalls-handled-for-you).

**What is the Rust core, and do I need a Rust toolchain?**
`backend="genetic"` runs a compiled Rust engine bundled inside the platform
wheel (44–58× faster than the equivalent numpy implementation — see
[backends — performance](concepts/backends.md#performance)). Installing from a wheel needs no
toolchain; only building from the sdist compiles Rust, which requires
rustc 1.86 or newer. The engine is held to
bitwise parity with Python on tree evaluation and constraint checking, and to
statistical parity on end-to-end GA outcomes; every result is float-verified
in Python before being returned.

**When would I use `backend="python"`?**
It is the original numpy implementation of the same genetic algorithm, kept as
a reference engine (and as the behavioral baseline the Rust core is tested
against). Use it to cross-check results or in environments where the compiled
extension cannot load; expect identical result quality, just slower.

**Are mined constraints safe to apply automatically?**
No, by design. They are sample invariants, not domain truths; the API returns
them for review (`as_code()`), and near-invariants are flagged as data-quality
findings instead of constraints.

**Do NaN flips count as "changes" for sparsity and diversity?**
Yes — flipping a value to NaN (or back) increments `n_changed`, pays the
configured delta, and counts in `distinct_changes` diversity cuts.
