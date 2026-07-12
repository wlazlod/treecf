# Constraints

Constraints are declared once, validated against the model's feature space, and
compiled by a single visitor into every backend — the abstract constraint form
and the genetic engines' vectorized check/repair pair share one source of truth.

```python
from treecf import Freeze, Monotone, Range, Linear, Implies, Equals, OneHot, constraint

Freeze("age_of_bureau_file")                    # immutable
Monotone("age", "increase")                     # directional
Range("utilization", 0.0, 1.5)                  # hard bounds
constraint("max_dpd_30d <= max_dpd_12m")        # string sugar -> Linear
constraint("2*a - b <= c + 5")                  # any linear expression
Implies(Equals("has_mortgage", 0), Equals("mortgage_balance", 0))
OneHot(("channel_web", "channel_app", "channel_branch"))
```

Only linear expressions are expressible as strings by design; richer constraints
are objects. Parse errors carry a caret marking the offending token.

## Mining candidates from data

`suggest_constraints` scans a background sample for invariants — pairwise
orders, equalities, binary implications, one-hot groups, missingness links,
integer-valuedness — and returns them **for human review**; nothing is ever
auto-applied:

```python
import treecf

result = treecf.suggest_constraints(X_train, feature_names=names)
for s in result[:20]:
    print(s.as_code())     # constraint("n_active_loans <= n_loans_total")  # support=1.0000, n=48211

accepted = [s.constraint for s in result if s.kind == "order"]
```

Hierarchical time-window features (`dpd_7d <= dpd_30d <= dpd_90d`) are reduced
to a minimal generating set (transitive reduction), and suggestions are ranked
by support and shared name tokens — never filtered.

Near-invariants (support ≥ 99.9% but below 1.0) are returned separately as
`DataQualityFinding` records with the violating rows: a handful of violations
of an otherwise universal rule usually signals an ETL defect, not a domain
exception.


