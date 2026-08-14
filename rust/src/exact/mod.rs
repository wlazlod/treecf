//! Exact backend — the sequential branch-and-bound search over the candidate grid.
//!
//! This is the Rust mirror of one Python implementation split across five files,
//! and it keeps that split module for module — except that the Python-side
//! split between `exact` and `_exact_bounds` is for file size alone and has no
//! counterpart here, so `search` covers both:
//!
//! | Python | here |
//! |---|---|
//! | `treecf.backends.exact` + `treecf.backends._exact_bounds` | `search` |
//! | `treecf.backends._exact_domains` (+ the `treecf.api._snap` it calls) | `domains` |
//! | `treecf.backends._exact_propagation` | `propagation` |
//! | `treecf.backends._exact_orderpairs` | `orderpairs` |
//!
//! Those five Python files carry the bit-parity contract in their own headers;
//! every module here follows its counterpart line for line, so the operation
//! order of the arithmetic is a compatibility contract rather than a style
//! choice.
//!
//! Three rules make that parity reachable, and they govern all four modules:
//!
//! 1. **No parallelism in the search.** Deliberate: the branch order, the
//!    incumbent that prunes each node, and every counter reported afterwards
//!    all depend on the order nodes are visited in, so the search is
//!    single-threaded by construction and no rayon call may enter it. The
//!    RNG-free stages that `ga.rs` fans out have no analogue here.
//! 2. **Full re-summation.** Assigning a feature re-walks the trees that split
//!    on it from their roots, then the ensemble bracket is re-summed over every
//!    tree in ascending index (`base + tree_0 + tree_1 + ...`), never patched
//!    with an incremental delta. Prune decisions therefore compare the same
//!    float on both sides.
//! 3. **Python's min/max tie behaviour.** `f64::min`/`f64::max` return `-0.0`
//!    on a `0.0`/`-0.0` tie where Python's `min`/`max` return the first
//!    argument. No comparison or sum can differ, but stored brackets and stored
//!    spans would differ in the sign bit, so every place a merge result is
//!    *stored* uses `constraints::py_min`/`constraints::py_max`
//!    — the one shared definition, which reproduces Python's behaviour. Python
//!    is the reference; `f64::min`/`f64::max` appear in none of the four
//!    modules. Zeros a *rounding* produces follow the same reasoning, in
//!    `domains::py_zero`.
//!
//! Two more portability notes:
//!
//! - Python's node masks and the search's assigned mask are arbitrary-precision
//!   ints. Here they are `u64` word bitsets (`search::BitSet`), correct past 64
//!   features.
//! - The completion arbiter is [`crate::constraints::Constraints::check`] (single
//!   row, sequential), which `tests/rust/test_constraints_conformance.py` proves
//!   bit-equal to `CompiledConstraints.check_matrix`, plus the float-space score
//!   through [`crate::ir::Ensemble::raw_score`] and the optional plausibility
//!   bound.
//!
//! Value policies arrive as one optional policy per feature — the marshaled
//! form of Python's name-keyed mapping, with `"raw"` marshaled to `None`.
//! Callable policies cannot cross the boundary; validation rejects them on the
//! Python side before marshaling. Their snapping is the one place parity needed
//! an unusual measure: `treecf.api._snap` orders its candidates by building a
//! `set` first, so two equally distant candidates are separated by CPython's
//! own set-iteration order, which `domains::py_hash_double` and
//! `domains::py_set_order` reproduce.
//!
//! The tests live with the module they cover; their shared builders sit in
//! `test_support` so each golden's own doc comment stays attached to the test
//! it pins.

pub(crate) mod domains;
pub(crate) mod orderpairs;
pub(crate) mod propagation;
pub(crate) mod search;
#[cfg(test)]
pub(crate) mod test_support;

pub use domains::constraint_cells;
pub use search::{solve_exact, ExactParams, ExactResult, ExactStats};

/// Per-feature snapping rule for values that move. Mirrors `treecf.api.ValuePolicy`
/// minus the callable case (rejected before marshaling) and minus `"raw"` (`None`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ValuePolicy {
    Integer,
    Grid { step: f64, anchor: f64 },
}
