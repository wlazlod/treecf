# Glossary

The words the documentation uses with a fixed meaning.

**Factual.** The instance being explained: one row, in the model's feature order, as the
model saw it. Written `x` throughout.

**Counterfactual.** A full feature vector `x_cf` the model scores inside the target, differing
from the factual only in the features the plan changes. `Counterfactual` is also the result
type that carries it, with its cost, its verified score, and its proof.

**Plan.** The changes a counterfactual asks for — `changes`, a mapping from feature to
(factual value, counterfactual value). "Plan" and "counterfactual" name the same object from
two sides: the vector, and what to do to reach it.

**Recourse.** The plan read as advice: what the applicant can change to be accepted. A
recourse region, a recourse menu, and a recourse map are all views of plans.

**Lever.** A feature a plan is allowed to change. The set of levers a plan actually changes
is its *lever set*; two plans are different ways to reach the target when their lever sets
differ.

**Target.** The interval the model's output must land in, on the raw score, the model
probability, or a calibrated probability; a `Target.bands` ladder is several intervals.

**Constraint.** A rule every plan must satisfy — `Freeze`, `Monotone`, `Range`, linear
inter-feature rules, one-hot consistency — compiled once and enforced by every backend.

**Cell.** For one feature, an interval between two adjacent split thresholds of the
ensemble, inside which every tree routes identically. The search works over cells, not real
numbers; a counterfactual value is the point of its cell nearest to the factual.

**Category block.** The categorical counterpart of a cell: a set of codes every tree
routes the same way.

**Backend.** The search engine: `"genetic"` (the default, heuristic, on the Rust core),
`"python"` (its numpy reference), or `"exact"` (branch-and-bound with proofs).

**Proof.** The claim a result makes: `"heuristic"` (no optimality claim), `"optimal"`,
`"optimal_within_gap"`, or for an `Infeasible` result `"certified"` versus
`"search_exhausted"`.

**Region.** A per-feature box around a counterfactual, every point of which is verified
in-target and constraint-feasible. Certified, not necessarily maximal, not monotone in the
target; `(data-limited)` marks a side that stopped at the range of the background data.

**Menu.** Every lever set up to a size solved as its own coalition, with the minimal
frontier and the sets certified unable to reach the target.

**Certificate.** A self-contained JSON record of one result — fingerprints of the model and
constraints, the solve parameters, and a fresh verification — that `check_certificate`
re-verifies later.
