//! Exact backend — the sequential branch-and-bound search over the candidate grid.
//!
//! This is the Rust mirror of one Python implementation split across four files:
//! `treecf.backends.exact`, `_exact_domains`, `_exact_propagation` and
//! `_exact_orderpairs`. Those four carry the bit-parity contract in their own
//! headers; everything here follows them line for line, so the operation order
//! of the arithmetic is a compatibility contract rather than a style choice.
//!
//! Three rules make that parity reachable:
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
//!    *stored* uses [`py_min`]/[`py_max`], which reproduce Python's behaviour.
//!    Python is the reference; `f64::min`/`f64::max` appear nowhere below.
//!
//! Two more portability notes:
//!
//! - Python's node masks and the search's assigned mask are arbitrary-precision
//!   ints. Here they are `u64` word bitsets ([`BitSet`]), correct past 64
//!   features.
//! - The completion arbiter is [`Constraints::check`] (single row, sequential),
//!   which `tests/rust/test_constraints_conformance.py` proves bit-equal to
//!   `CompiledConstraints.check_matrix`, plus the float-space score through
//!   [`Ensemble::raw_score`] and the optional plausibility bound.
//!
//! Value policies arrive as one optional policy per feature — the marshaled
//! form of Python's name-keyed mapping, with `"raw"` marshaled to `None`.
//! Callable policies cannot cross the boundary; validation rejects them on the
//! Python side before marshaling. Their snapping is the one place parity needed
//! an unusual measure: `treecf.api._snap` orders its candidates by building a
//! `set` first, so two equally distant candidates are separated by CPython's
//! own set-iteration order, which [`py_hash_double`] and [`py_set_order`]
//! reproduce.

use std::time::Instant;

use crate::cells::{cell_index, feature_cells_joint, Cell};
use crate::constraints::{Constraints, LIN_EQ, LIN_LE, POLICY_SATISFIED};
use crate::ir::Ensemble;

/// The tolerance `check_matrix` allows a linear constraint; an order pair counts
/// as broken exactly when the arbiter would reject it.
const LINEAR_SLACK: f64 = 1e-9;

/// Per-feature snapping rule for values that move. Mirrors `treecf.api.ValuePolicy`
/// minus the callable case (rejected before marshaling) and minus `"raw"` (`None`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ValuePolicy {
    Integer,
    Grid { step: f64, anchor: f64 },
}

/// The exact set of counters `solve_exact` reports — the seven keys of Python's
/// `_stats`. `nodes_pruned_score` counts branches the ensemble can no longer
/// bring into the target (the plausibility bound counts here too); every other
/// cut, cost and feasibility alike, counts under `nodes_pruned_cost`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactStats {
    pub nodes_expanded: u64,
    pub nodes_pruned_score: u64,
    pub nodes_pruned_cost: u64,
    pub lower_bound: f64,
    pub gap: f64,
    pub completed: bool,
    pub warm_start_used: bool,
}

/// Outcome of an exact-backend search. `snapped` lists the feature indices whose
/// winning state was produced by a value policy, in search-order — the marshaled
/// form of Python's name-keyed dict, whose insertion order is the same.
#[derive(Clone, Debug)]
pub struct ExactResult {
    pub x_cf: Option<Vec<f64>>,
    pub proof: &'static str, // "optimal" | "optimal_within_gap" | "heuristic"
    pub stats: ExactStats,
    pub snapped: Vec<usize>,
    pub distance: Option<f64>,
}

/// Budgets and the optimality gap.
#[derive(Clone, Copy, Debug)]
pub struct ExactParams {
    pub node_budget: u64,
    pub gap: f64,
    pub time_budget_s: f64,
}

impl Default for ExactParams {
    fn default() -> Self {
        Self {
            node_budget: 2_000_000,
            gap: 0.0,
            time_budget_s: 10.0,
        }
    }
}

/// Python `min(a, b)`: returns `b` only when `b < a`, so a `0.0`/`-0.0` tie
/// keeps the first argument (`f64::min` would return `-0.0`).
#[inline]
fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Python `max(a, b)`: returns `b` only when `b > a` (see [`py_min`]).
#[inline]
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python's `<`-based ordering: `-0.0` and `0.0` compare equal, so a stable sort
/// leaves them in insertion order. Costs and sort values are never NaN here.
#[inline]
fn py_cmp(a: f64, b: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if a < b {
        Ordering::Less
    } else if b < a {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

// ---------------------------------------------------------------- bitsets ---

/// Fixed-width bitset over features — the mirror of Python's arbitrary-precision
/// int masks, correct for models wider than 64 features.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    fn new(n_features: usize) -> Self {
        Self {
            words: vec![0; n_words(n_features)],
        }
    }

    fn set(&mut self, i: usize) {
        self.words[i / 64] |= 1u64 << (i % 64);
    }

    fn clear(&mut self, i: usize) {
        self.words[i / 64] &= !(1u64 << (i % 64));
    }

    #[cfg(test)]
    fn get(&self, i: usize) -> bool {
        self.words[i / 64] >> (i % 64) & 1 == 1
    }
}

fn n_words(n_features: usize) -> usize {
    n_features.div_ceil(64).max(1)
}

// ------------------------------------------------------------ cost terms ---

/// One feature's contribution to the objective — the per-feature term of
/// `genetic.objective()`, same four cases, same multiply-then-divide order.
fn term_cost(
    x_j: f64,
    r: f64,
    weight_j: f64,
    sigma_j: f64,
    lam: f64,
    to_miss: f64,
    from_miss: f64,
) -> f64 {
    let x_nan = x_j.is_nan();
    let r_nan = r.is_nan();
    if x_nan && r_nan {
        return 0.0;
    }
    if x_nan {
        return (weight_j * from_miss) / sigma_j + lam;
    }
    if r_nan {
        return (weight_j * to_miss) / sigma_j + lam;
    }
    if r == x_j {
        return 0.0;
    }
    let delta = (r - x_j).abs();
    lam + (weight_j * delta) / sigma_j
}

/// Full-row objective, accumulated in ascending feature index.
fn cost_of_row(
    x: &[f64],
    row: &[f64],
    sigma: &[f64],
    weights: &[f64],
    lam: f64,
    deltas: &[(f64, f64)],
) -> f64 {
    let mut total = 0.0;
    for j in 0..x.len() {
        let (to_miss, from_miss) = deltas[j];
        total += term_cost(x[j], row[j], weights[j], sigma[j], lam, to_miss, from_miss);
    }
    total
}

/// (delta_miss, delta_from_miss) per feature, `(0.0, 0.0)` where absent.
fn allow_missing_deltas(cons: &Constraints) -> Vec<(f64, f64)> {
    let mut out = vec![(0.0, 0.0); cons.n_features];
    for &(j, to, from) in &cons.allow_missing {
        out[j as usize] = (to, from);
    }
    out
}

// -------------------------------------------------------- cell arithmetic ---

/// `cell` ∩ `[lo, hi]` (closed bounds); `None` when empty, degenerate open
/// singletons included.
fn intersect_cell(cell: &Cell, lo: f64, hi: f64) -> Option<Cell> {
    let (new_lo, new_lo_open) = if cell.lo > lo {
        (cell.lo, cell.lo_open)
    } else if lo > cell.lo {
        (lo, false)
    } else {
        (cell.lo, cell.lo_open)
    };
    let (new_hi, new_hi_open) = if cell.hi < hi {
        (cell.hi, cell.hi_open)
    } else if hi < cell.hi {
        (hi, false)
    } else {
        (cell.hi, cell.hi_open)
    };
    if new_lo > new_hi {
        return None;
    }
    if new_lo == new_hi && (new_lo_open || new_hi_open) {
        return None;
    }
    Some(Cell {
        lo: new_lo,
        hi: new_hi,
        lo_open: new_lo_open,
        hi_open: new_hi_open,
    })
}

/// `first` ∩ `second`, tighter edge per side, open edge winning at equal values.
fn intersect_cells(first: &Cell, second: &Cell) -> Option<Cell> {
    let (lo, lo_open) = if first.lo > second.lo {
        (first.lo, first.lo_open)
    } else if second.lo > first.lo {
        (second.lo, second.lo_open)
    } else {
        (first.lo, first.lo_open || second.lo_open)
    };
    let (hi, hi_open) = if first.hi < second.hi {
        (first.hi, first.hi_open)
    } else if second.hi < first.hi {
        (second.hi, second.hi_open)
    } else {
        (first.hi, first.hi_open || second.hi_open)
    };
    if lo > hi {
        return None;
    }
    if lo == hi && (lo_open || hi_open) {
        return None;
    }
    Some(Cell {
        lo,
        hi,
        lo_open,
        hi_open,
    })
}

/// Lowest and highest value a cell can actually take: a finite open edge steps
/// one f32 ulp inside (the step `nearest_to` takes), an infinite one stays.
fn achievable_bounds(cell: &Cell) -> (f64, f64) {
    let mut lo = cell.lo;
    if cell.lo_open && lo != f64::NEG_INFINITY {
        lo = cell.nearest_to(cell.lo);
    }
    let mut hi = cell.hi;
    if cell.hi_open && hi != f64::INFINITY {
        hi = cell.nearest_to(cell.hi);
    }
    (lo, hi)
}

/// Values worth trying when a pair `a <= b` has to be pulled onto the boundary
/// `a' == b' == t`: factual of `a`, factual of `b`, the demanded values as
/// given, low end, high end — in that fixed order, non-finite and out-of-interval
/// values dropped, first occurrence winning.
fn boundary_candidates(
    cell_a: &Cell,
    cell_b: &Cell,
    x_a: f64,
    x_b: f64,
    demanded: &[f64],
) -> Vec<f64> {
    let Some(iv) = intersect_cells(cell_a, cell_b) else {
        return Vec::new();
    };
    let (ach_lo, ach_hi) = achievable_bounds(&iv);
    let mut out: Vec<f64> = Vec::new();
    let mut offer = |t: f64| {
        if !t.is_finite() || !iv.contains(t) || out.contains(&t) {
            return;
        }
        out.push(t);
    };
    offer(x_a);
    offer(x_b);
    for &t in demanded {
        offer(t);
    }
    offer(ach_lo);
    offer(ach_hi);
    out
}

// ----------------------------------------------------- the candidate grid ---

/// `cells` with the one holding `value` cut into the part below it, the single
/// point, and the part above — the three pieces `build_cells` emits where an LT
/// and an LE split share a threshold. Idempotent on an existing singleton.
fn split_cell_at(cells: &[Cell], value: f64) -> Vec<Cell> {
    let mut out: Vec<Cell> = Vec::with_capacity(cells.len() + 2);
    for cell in cells {
        if !cell.contains(value) || (cell.lo == value && cell.hi == value) {
            out.push(*cell);
            continue;
        }
        if cell.lo < value {
            out.push(Cell {
                lo: cell.lo,
                hi: value,
                lo_open: cell.lo_open,
                hi_open: true,
            });
        }
        out.push(Cell {
            lo: value,
            hi: value,
            lo_open: false,
            hi_open: false,
        });
        if value < cell.hi {
            out.push(Cell {
                lo: value,
                hi: cell.hi,
                lo_open: true,
                hi_open: cell.hi_open,
            });
        }
    }
    out
}

/// The routing grid, cut finer wherever a constraint can tell two points of one
/// cell apart: every value an implication watches for becomes a boundary of its
/// own. Returns the joint routing grid untouched when there is no implication,
/// which is what keeps non-implication problems bit-stable.
pub fn constraint_cells(cons: &Constraints, ensembles: &[&Ensemble]) -> Vec<Vec<Cell>> {
    let grids = feature_cells_joint(ensembles);
    if cons.implications.is_empty() {
        return grids;
    }
    // per feature, the trigger values in first-seen order (features are
    // independent, so only the per-feature order matters and that is sorted)
    let mut triggers: Vec<Vec<f64>> = vec![Vec::new(); cons.n_features];
    for &(cond_index, cond_value, _, _) in &cons.implications {
        let values = &mut triggers[cond_index as usize];
        if !values.contains(&cond_value) {
            values.push(cond_value);
        }
    }
    let mut refined = grids;
    for (feature, values) in triggers.iter_mut().enumerate() {
        if values.is_empty() {
            continue;
        }
        values.sort_by(|a, b| py_cmp(*a, *b));
        for &value in values.iter() {
            refined[feature] = split_cell_at(&refined[feature], value);
        }
    }
    refined
}

/// Per feature, the exact values some constraint can come to demand of it: an
/// implication's consequence value, and the algebraic solution `rhs / coef` of a
/// single-feature `Linear(op="==")`. Ascending per feature.
fn demanded_values(cons: &Constraints) -> Vec<Vec<f64>> {
    fn offer(demanded: &mut [Vec<f64>], f: usize, value: f64) {
        if !demanded[f].contains(&value) {
            demanded[f].push(value);
        }
    }
    let mut demanded: Vec<Vec<f64>> = vec![Vec::new(); cons.n_features];
    for &(_, _, cons_index, cons_value) in &cons.implications {
        offer(&mut demanded, cons_index as usize, cons_value);
    }
    for lin in &cons.linears {
        if lin.indices.len() == 1 && lin.op == LIN_EQ {
            offer(
                &mut demanded,
                lin.indices[0] as usize,
                lin.rhs / lin.coefs[0],
            );
        }
    }
    for values in demanded.iter_mut() {
        values.sort_by(|a, b| py_cmp(*a, *b));
    }
    demanded
}

// -------------------------------------------------------------- snapping ---

/// CPython's `_Py_HashDouble` for finite values. Needed because `_snap` orders
/// its candidates by building a `set` first: equal-distance candidates come out
/// in set-iteration order, which is hash-driven. Integral floats hash like the
/// equal int, so this also covers `math.floor`/`math.ceil` results.
fn py_hash_double(v: f64) -> i64 {
    const HASH_BITS: u32 = 61;
    const HASH_MODULUS: u64 = (1u64 << 61) - 1;
    if v == 0.0 {
        return 0; // covers -0.0 too, as CPython does
    }
    if !v.is_finite() {
        // snap candidates are always finite; CPython's own answers otherwise
        // (its NaN hash is identity-based, so 0 stands in for the unreachable)
        return if v.is_nan() {
            0
        } else if v > 0.0 {
            314159
        } else {
            -314159
        };
    }
    let (mut m, mut e) = frexp(v);
    let mut sign: i64 = 1;
    if m < 0.0 {
        sign = -1;
        m = -m;
    }
    let mut x: u64 = 0;
    while m != 0.0 {
        x = ((x << 28) & HASH_MODULUS) | (x >> (HASH_BITS - 28));
        m *= 268_435_456.0; // 2**28
        e -= 28;
        let y = m as u64;
        m -= y as f64;
        x += y;
        if x >= HASH_MODULUS {
            x -= HASH_MODULUS;
        }
    }
    let shift = if e >= 0 {
        (e as u32) % HASH_BITS
    } else {
        HASH_BITS - 1 - (((-1 - e) as u32) % HASH_BITS)
    };
    x = ((x << shift) & HASH_MODULUS) | (x >> (HASH_BITS - shift));
    let mut result = (x as i64) * sign;
    if result == -1 {
        result = -2;
    }
    result
}

/// `v = m * 2**e` with `0.5 <= |m| < 1` — C's `frexp`, which Rust has no std
/// equivalent of.
fn frexp(v: f64) -> (f64, i32) {
    if v == 0.0 || !v.is_finite() {
        return (v, 0);
    }
    let bits = v.to_bits();
    let raw_exponent = ((bits >> 52) & 0x7ff) as i32;
    if raw_exponent == 0 {
        let (m, e) = frexp(v * f64::from_bits(0x43f0_0000_0000_0000)); // * 2**64
        return (m, e - 64);
    }
    let mantissa = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1022u64 << 52));
    (mantissa, raw_exponent - 1022)
}

/// Iteration order of a CPython `set` holding up to four items: eight slots, and
/// with `LINEAR_PROBES` (9) above the mask no linear probing ever applies, so a
/// collision goes straight to the perturbed probe. Returns indices into the
/// insertion-ordered input.
fn py_set_order(hashes: &[i64]) -> Vec<usize> {
    const MASK: u64 = 7;
    const PERTURB_SHIFT: u32 = 5;
    let mut table: [Option<usize>; 8] = [None; 8];
    for (k, &hash) in hashes.iter().enumerate() {
        let mut perturb = hash as u64;
        let mut i = (hash as u64) & MASK;
        loop {
            if table[i as usize].is_none() {
                table[i as usize] = Some(k);
                break;
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & MASK;
        }
    }
    table.iter().flatten().copied().collect()
}

/// `sorted(set(values), key=lambda c: abs(c - value))`: duplicates dropped
/// first-wins, the survivors read out in set-iteration order, then stably sorted
/// by distance.
fn snap_candidates(values: &[f64], value: f64) -> Vec<f64> {
    let mut unique: Vec<f64> = Vec::with_capacity(values.len());
    for &v in values {
        if !unique.contains(&v) {
            unique.push(v);
        }
    }
    let hashes: Vec<i64> = unique.iter().map(|&v| py_hash_double(v)).collect();
    let mut ordered: Vec<f64> = py_set_order(&hashes).iter().map(|&k| unique[k]).collect();
    ordered.sort_by(|a, b| py_cmp((a - value).abs(), (b - value).abs()));
    ordered
}

/// Nearest policy-conforming value inside the cell and bounds — port of
/// `treecf.api._snap` (callable policies excluded, see the module header).
fn snap(value: f64, policy: &ValuePolicy, cell: &Cell, lo: f64, hi: f64) -> Option<f64> {
    let candidates = match *policy {
        ValuePolicy::Integer => snap_candidates(&[value.floor(), value.ceil()], value),
        ValuePolicy::Grid { step, anchor } => {
            // Python's round() is round-half-to-even
            let base = anchor + step * ((value - anchor) / step).round_ties_even();
            snap_candidates(&[base, base - step, base + step], value)
        }
    };
    candidates
        .into_iter()
        .find(|&c| cell.contains(c) && lo <= c && c <= hi)
}

// -------------------------------------------------- the branching alphabet ---

/// One candidate value a feature may take in the search. `snapped` is true only
/// for a movement candidate a value policy produced.
#[derive(Clone, Copy, Debug, PartialEq)]
struct State {
    value: f64,
    cost: f64,
    cell_idx: usize,
    is_nan: bool,
    snapped: bool,
}

impl State {
    fn new(value: f64, cost: f64, cell_idx: usize, is_nan: bool) -> Self {
        Self {
            value,
            cost,
            cell_idx,
            is_nan,
            snapped: false,
        }
    }
}

/// Canonical state order: ascending cost, ties by ascending cell index (the NaN
/// state's sentinel index sorts it last), remaining ties by value so `0.0` sorts
/// before `1.0`. Stable, like Python's `sorted`.
fn sort_states(states: &mut [State]) {
    states.sort_by(|a, b| {
        py_cmp(a.cost, b.cost)
            .then(a.cell_idx.cmp(&b.cell_idx))
            .then_with(|| {
                py_cmp(
                    if a.is_nan { 0.0 } else { a.value },
                    if b.is_nan { 0.0 } else { b.value },
                )
            })
    });
}

/// Per-feature candidate states in canonical order — port of `_build_domains`.
/// `grids` must be the grid [`constraint_cells`] builds: every `cell_idx` points
/// into whatever grid is passed here.
#[allow(clippy::too_many_arguments)]
fn build_domains(
    grids: &[Vec<Cell>],
    x: &[f64],
    cons: &Constraints,
    sigma: &[f64],
    weights: &[f64],
    lam: f64,
    policies: &[Option<ValuePolicy>],
) -> Vec<Vec<State>> {
    let (lo, hi, frozen) = cons.instance_bounds(x);
    let lo: Vec<f64> = lo
        .iter()
        .map(|&v| if v.is_nan() { f64::NEG_INFINITY } else { v })
        .collect();
    let hi: Vec<f64> = hi
        .iter()
        .map(|&v| if v.is_nan() { f64::INFINITY } else { v })
        .collect();
    // a single-feature Linear whose missing_policy is forbid_missing/violated
    // (both marshal to the same non-satisfied code) suppresses the NaN state
    let mut suppress_nan = vec![false; cons.n_features];
    for lin in &cons.linears {
        if lin.indices.len() == 1 && lin.policy != POLICY_SATISFIED {
            suppress_nan[lin.indices[0] as usize] = true;
        }
    }
    let mut onehot_member = vec![false; cons.n_features];
    for group in &cons.onehot {
        for &f in group {
            onehot_member[f as usize] = true;
        }
    }
    let demanded = demanded_values(cons);
    let deltas = allow_missing_deltas(cons);

    let mut domains: Vec<Vec<State>> = Vec::with_capacity(x.len());
    for j in 0..x.len() {
        let x_j = x[j];
        let x_nan = x_j.is_nan();
        let allow_j = cons.allows_missing(j);
        let (lo_j, hi_j) = (lo[j], hi[j]);
        let pinned = lo_j == hi_j;
        let cells = &grids[j];
        let (to_miss, from_miss) = deltas[j];
        let (weight_j, sigma_j) = (weights[j], sigma[j]);

        if frozen[j] {
            let idx = if x_nan {
                cells.len()
            } else {
                cell_index(cells, x_j)
            };
            domains.push(vec![State::new(x_j, 0.0, idx, x_nan)]);
            continue;
        }

        if pinned {
            let v = lo_j;
            if !x_nan {
                // the pin fixes the only value the feature may take; going
                // missing is a separate question AllowMissing still answers
                let cost = term_cost(x_j, v, weight_j, sigma_j, lam, to_miss, from_miss);
                let mut states = vec![State::new(v, cost, cell_index(cells, v), false)];
                if allow_j && !suppress_nan[j] {
                    let nan_cost =
                        term_cost(x_j, f64::NAN, weight_j, sigma_j, lam, to_miss, from_miss);
                    states.push(State::new(f64::NAN, nan_cost, cells.len(), true));
                    sort_states(&mut states);
                }
                domains.push(states);
                continue;
            }
            // a NaN factual pinned to v: staying NaN needs no suppressing
            // missing_policy, moving to v needs AllowMissing. Both can fail at
            // once, and the empty domain that leaves is a certified-infeasible
            // signal for the search, not an error.
            let mut states: Vec<State> = Vec::new();
            if !suppress_nan[j] {
                states.push(State::new(f64::NAN, 0.0, cells.len(), true));
            }
            if allow_j {
                let cost = term_cost(x_j, v, weight_j, sigma_j, lam, to_miss, from_miss);
                states.push(State::new(v, cost, cell_index(cells, v), false));
            }
            sort_states(&mut states);
            domains.push(states);
            continue;
        }

        if x_nan && !allow_j {
            // nothing lets this feature become a value; staying missing is its
            // only state, and even that goes away under a suppressing policy
            if suppress_nan[j] {
                domains.push(Vec::new());
            } else {
                domains.push(vec![State::new(x_j, 0.0, cells.len(), true)]);
            }
            continue;
        }

        let policy = policies[j];
        let anchor = if x_nan { 0.0 } else { x_j };
        let is_binary = onehot_member[j];
        let demanded_here = &demanded[j];

        let mut states: Vec<State> = Vec::new();
        let mut keep_added = false;
        if !x_nan && lo_j <= x_j && x_j <= hi_j {
            states.push(State::new(x_j, 0.0, cell_index(cells, x_j), false));
            keep_added = true;
        }

        for (local_idx, cell) in cells.iter().enumerate() {
            let Some(iv) = intersect_cell(cell, lo_j, hi_j) else {
                continue;
            };
            if is_binary {
                for val in [0.0, 1.0] {
                    if !iv.contains(val) {
                        continue;
                    }
                    if keep_added && val == x_j {
                        continue;
                    }
                    let cost = term_cost(x_j, val, weight_j, sigma_j, lam, to_miss, from_miss);
                    states.push(State::new(val, cost, local_idx, false));
                }
                continue;
            }
            let mut added_here: Vec<f64> = Vec::new();
            for &val in demanded_here {
                // a value some constraint may demand of this feature: legal
                // wherever it lands, and never snapped
                if !iv.contains(val) || (keep_added && val == x_j) {
                    continue;
                }
                let cost = term_cost(x_j, val, weight_j, sigma_j, lam, to_miss, from_miss);
                states.push(State::new(val, cost, local_idx, false));
                added_here.push(val);
            }
            let mut r = iv.nearest_to(anchor);
            if keep_added && r == x_j {
                continue;
            }
            let mut snapped = false;
            if let Some(policy) = policy {
                let Some(snapped_r) = snap(r, &policy, &iv, lo_j, hi_j) else {
                    continue;
                };
                r = snapped_r;
                snapped = true;
            }
            if added_here.contains(&r) {
                continue; // the demanded value was this cell's nearest point too
            }
            let cost = term_cost(x_j, r, weight_j, sigma_j, lam, to_miss, from_miss);
            let mut state = State::new(r, cost, local_idx, false);
            state.snapped = snapped;
            states.push(state);
        }

        if allow_j && !suppress_nan[j] {
            let nan_cost = term_cost(x_j, f64::NAN, weight_j, sigma_j, lam, to_miss, from_miss);
            states.push(State::new(f64::NAN, nan_cost, cells.len(), true));
        }

        sort_states(&mut states);
        domains.push(states);
    }
    domains
}

/// Feature indices touched by any constraint, of any kind — the marshaled
/// equivalent of `_referenced_feature_indices`. Derived ranges are folded into
/// `ranges` by the flattener, and every one of them comes from a single-feature
/// Linear whose own index is referenced anyway, so the set is the same.
fn referenced_features(cons: &Constraints) -> Vec<bool> {
    let mut refs = vec![false; cons.n_features];
    for &j in &cons.freeze {
        refs[j as usize] = true;
    }
    for &(j, _, _) in &cons.ranges {
        refs[j as usize] = true;
    }
    for &(j, _) in &cons.equals {
        refs[j as usize] = true;
    }
    for &(j, _) in &cons.monotone {
        refs[j as usize] = true;
    }
    for lin in &cons.linears {
        for &j in &lin.indices {
            refs[j as usize] = true;
        }
    }
    for &(ci, _, si, _) in &cons.implications {
        refs[ci as usize] = true;
        refs[si as usize] = true;
    }
    for group in &cons.onehot {
        for &j in group {
            refs[j as usize] = true;
        }
    }
    for &(j, _, _) in &cons.allow_missing {
        refs[j as usize] = true;
    }
    refs
}

/// Search order: descending split count in the joint grid, ties ascending index.
/// A feature with no split anywhere and no referencing constraint is left out —
/// its domain is a single keep state, so it never needs to branch.
fn feature_order(grids: &[Vec<Cell>], cons: &Constraints) -> Vec<usize> {
    let referenced = referenced_features(cons);
    let split_counts: Vec<usize> = grids.iter().map(|cells| cells.len() - 1).collect();
    let mut included: Vec<usize> = (0..grids.len())
        .filter(|&j| split_counts[j] > 0 || referenced[j])
        .collect();
    included.sort_by_key(|&j| (std::cmp::Reverse(split_counts[j]), j));
    included
}

/// Suffix sums of each ordered feature's cheapest state cost: `h_suffix[k]` is
/// the minimum possible remaining cost once `order[k..]` are still undecided.
fn h_suffix(order: &[usize], domains: &[Vec<State>]) -> Vec<f64> {
    let mut suffix = vec![0.0; order.len() + 1];
    for k in (0..order.len()).rev() {
        suffix[k] = suffix[k + 1] + domains[order[k]][0].cost;
    }
    suffix
}

/// Reject constraint shapes the exact backend cannot handle, and return the
/// canonical order pairs `a - b <= 0` as `(a, b)`, ascending — the order the
/// search repairs them in.
///
/// Single-feature Linears are accepted silently: their bound already lives in
/// the marshaled ranges, and their missing policy still governs the feature's
/// NaN state in [`build_domains`]. Python's `_validate` also rejects callable
/// value policies; those cannot be marshaled at all, so that half of the check
/// stays on the Python side.
fn validate(cons: &Constraints) -> Result<Vec<(usize, usize)>, String> {
    let mut order_pairs: Vec<(usize, usize)> = Vec::new();
    for lin in &cons.linears {
        if lin.indices.len() == 1 {
            continue;
        }
        let mut sorted = lin.coefs.clone();
        sorted.sort_by(|a, b| py_cmp(*a, *b));
        if lin.op == LIN_LE && lin.rhs == 0.0 && sorted == [-1.0, 1.0] {
            let a = lin.indices[lin.coefs.iter().position(|&c| c == 1.0).unwrap()] as usize;
            let b = lin.indices[lin.coefs.iter().position(|&c| c == -1.0).unwrap()] as usize;
            order_pairs.push((a, b));
            continue;
        }
        return Err(
            "Linear constraint over multiple features is not supported by the exact \
             backend yet; use backend=\"genetic\"."
                .to_string(),
        );
    }
    order_pairs.sort_unstable();
    Ok(order_pairs)
}

/// The pairs whose committed cost is a floor on what a set-aside completion
/// could still have been worth.
///
/// Only one kind of pair the argument was proven for qualifies: two plain
/// features, each sitting on the point of its cell nearest to the factual, with
/// no second pair to disturb. A one-hot member sits on its group's 0 or 1
/// instead of that nearest point, a feature some constraint can demand an exact
/// value of carries that value among its candidates, and a value policy moves
/// the candidate onto its own grid; in all three the committed cost can be
/// higher than what a repair would have cost, so it is no floor. Pairs are
/// listed in only if they qualify, never out if they look suspicious: a kind of
/// pair nobody has thought of yet lands on withdrawal, not on silence.
fn g_floor_pairs(
    order_pairs: &[(usize, usize)],
    onehot_member: &[bool],
    demanded: &[Vec<f64>],
    policy_active: &[bool],
) -> Vec<(usize, usize)> {
    let entangled = |pair: (usize, usize)| {
        order_pairs.iter().any(|&other| {
            other != pair
                && (other.0 == pair.0
                    || other.0 == pair.1
                    || other.1 == pair.0
                    || other.1 == pair.1)
        })
    };
    order_pairs
        .iter()
        .copied()
        .filter(|&pair| {
            !entangled(pair)
                && ![pair.0, pair.1]
                    .iter()
                    .any(|&f| onehot_member[f] || !demanded[f].is_empty() || policy_active[f])
        })
        .collect()
}

/// Lowest and highest value a feature can still end up holding: the achievable
/// ends of every cell its states come from, not the state values themselves.
/// `None` when the feature may go missing.
fn domain_span(states: &[State], cells: &[Cell], lo_j: f64, hi_j: f64) -> Option<(f64, f64)> {
    let mut span_lo = f64::INFINITY;
    let mut span_hi = f64::NEG_INFINITY;
    for state in states {
        if state.is_nan {
            return None;
        }
        let Some(iv) = intersect_cell(&cells[state.cell_idx], lo_j, hi_j) else {
            continue; // a state's own cell always survives
        };
        let (cell_lo, cell_hi) = achievable_bounds(&iv);
        span_lo = py_min(span_lo, cell_lo);
        span_hi = py_max(span_hi, cell_hi);
    }
    if span_lo > span_hi {
        return None;
    }
    Some((span_lo, span_hi))
}

// ------------------------------------------------------------ propagation ---

/// What one assignment changed: the features it settled (with their previous
/// setting) and the one-hot counters it moved (with their previous readings) —
/// old values rather than deltas, so putting them back cannot drift.
#[derive(Clone, Debug, Default)]
struct PropFrame {
    settled: Vec<(usize, Option<f64>)>,
    counters: Vec<(usize, usize, usize)>,
}

/// What assigning one feature settles about the features that follow it:
/// an implication's consequence, and a one-hot group's last free member.
/// One-hot bookkeeping only runs for groups whose members hold nothing but 0
/// and 1; anything else is left to the arbiter alone.
struct Propagation<'a> {
    implications: &'a [(u32, f64, u32, f64)],
    groups: &'a [Vec<u32>],
    group_of: Vec<Option<usize>>,
    forced_value: Vec<Option<f64>>,
    ones: Vec<usize>,
    zeros: Vec<usize>,
}

fn force(
    forced_value: &mut [Option<f64>],
    frame: &mut PropFrame,
    assigned: &[bool],
    values: &[f64],
    f: usize,
    value: f64,
) -> bool {
    if assigned[f] {
        return values[f] == value;
    }
    if let Some(current) = forced_value[f] {
        return current == value;
    }
    frame.settled.push((f, None));
    forced_value[f] = Some(value);
    true
}

impl<'a> Propagation<'a> {
    fn new(cons: &'a Constraints, domains: &[Vec<State>]) -> Self {
        let mut group_of: Vec<Option<usize>> = vec![None; cons.n_features];
        for (g_idx, group) in cons.onehot.iter().enumerate() {
            let binary = group.iter().all(|&f| {
                domains[f as usize]
                    .iter()
                    .all(|s| s.value == 0.0 || s.value == 1.0)
            });
            if binary {
                for &f in group {
                    group_of[f as usize] = Some(g_idx);
                }
            }
        }
        Self {
            implications: &cons.implications,
            groups: &cons.onehot,
            group_of,
            forced_value: vec![None; cons.n_features],
            ones: vec![0; cons.onehot.len()],
            zeros: vec![0; cons.onehot.len()],
        }
    }

    /// Settle what follows from `j` taking `v`; report any contradiction. The
    /// changes made before a contradiction are still reported — the caller
    /// restores the frame either way.
    fn apply(&mut self, j: usize, v: f64, assigned: &[bool], values: &[f64]) -> (PropFrame, bool) {
        let mut frame = PropFrame::default();
        if let Some(forced) = self.forced_value[j] {
            if v != forced {
                return (frame, true);
            }
        }
        let groups = self.groups;
        if let Some(g_idx) = self.group_of[j] {
            let group = &groups[g_idx];
            frame
                .counters
                .push((g_idx, self.ones[g_idx], self.zeros[g_idx]));
            if v == 1.0 {
                self.ones[g_idx] += 1;
            } else {
                self.zeros[g_idx] += 1;
            }
            // a second one, or nothing but zeros: the group can no longer sum to
            // one. The all-zeros reading is a backstop — settling the last free
            // member normally catches that case one assignment earlier.
            if self.ones[g_idx] > 1 || self.zeros[g_idx] == group.len() {
                return (frame, true);
            }
            if self.ones[g_idx] == 0 && self.zeros[g_idx] == group.len() - 1 {
                let last = group
                    .iter()
                    .map(|&f| f as usize)
                    .find(|&f| f != j && !assigned[f])
                    .expect("a group one short of all-zeros has a free member");
                if !force(
                    &mut self.forced_value,
                    &mut frame,
                    assigned,
                    values,
                    last,
                    1.0,
                ) {
                    return (frame, true);
                }
            }
        }
        for &(cond_index, cond_value, cons_index, cons_value) in self.implications {
            let triggered = cond_index as usize == j && v == cond_value;
            if triggered
                && !force(
                    &mut self.forced_value,
                    &mut frame,
                    assigned,
                    values,
                    cons_index as usize,
                    cons_value,
                )
            {
                return (frame, true);
            }
        }
        (frame, false)
    }

    /// Put back everything one `apply` settled.
    fn restore(&mut self, frame: &PropFrame) {
        for &(f, previous) in frame.settled.iter().rev() {
            self.forced_value[f] = previous;
        }
        for &(g_idx, ones, zeros) in frame.counters.iter().rev() {
            self.ones[g_idx] = ones;
            self.zeros[g_idx] = zeros;
        }
    }
}

// -------------------------------------------------------- ensemble bounds ---

/// Score bracket of one ensemble under a partial assignment.
///
/// Holds a per-tree `[min, max]` over the leaves still reachable and the
/// ensemble bracket those trees sum to. Assigning a feature recomputes only the
/// trees that split on it, from their roots — the per-tree numbers are always
/// full walks. The ensemble bracket is then re-summed in full over every tree in
/// ascending index, the same additions `raw_score` performs (see rule 2 of the
/// module header).
struct EnsembleBounds<'a> {
    ens: &'a Ensemble,
    n_words: usize,
    sub_min: Vec<f64>,
    sub_max: Vec<f64>,
    mask: Vec<u64>, // n_nodes * n_words
    trees_on_feature: Vec<Vec<usize>>,
    tree_min: Vec<f64>,
    tree_max: Vec<f64>,
    score_min: f64,
    score_max: f64,
}

fn prepare_node(
    ens: &Ensemble,
    idx: usize,
    n_words: usize,
    sub_min: &mut [f64],
    sub_max: &mut [f64],
    mask: &mut [u64],
) {
    if ens.feature[idx] < 0 {
        sub_min[idx] = ens.value[idx];
        sub_max[idx] = ens.value[idx];
        return;
    }
    let (left, right) = (ens.left[idx] as usize, ens.right[idx] as usize);
    prepare_node(ens, left, n_words, sub_min, sub_max, mask);
    prepare_node(ens, right, n_words, sub_min, sub_max, mask);
    sub_min[idx] = py_min(sub_min[left], sub_min[right]);
    sub_max[idx] = py_max(sub_max[left], sub_max[right]);
    for w in 0..n_words {
        mask[idx * n_words + w] = mask[left * n_words + w] | mask[right * n_words + w];
    }
    let f = ens.feature[idx] as usize;
    mask[idx * n_words + f / 64] |= 1u64 << (f % 64);
}

impl<'a> EnsembleBounds<'a> {
    fn new(ens: &'a Ensemble, assigned: &[bool], values: &[f64]) -> Self {
        let n_nodes = ens.feature.len();
        let n_words = n_words(ens.n_features);
        let mut sub_min = vec![0.0; n_nodes];
        let mut sub_max = vec![0.0; n_nodes];
        let mut mask = vec![0u64; n_nodes * n_words];
        for &root in &ens.tree_roots {
            prepare_node(
                ens,
                root as usize,
                n_words,
                &mut sub_min,
                &mut sub_max,
                &mut mask,
            );
        }
        let mut trees_on_feature: Vec<Vec<usize>> = vec![Vec::new(); ens.n_features];
        for (t, &root) in ens.tree_roots.iter().enumerate() {
            let base = root as usize * n_words;
            for f in 0..ens.n_features {
                if mask[base + f / 64] >> (f % 64) & 1 == 1 {
                    trees_on_feature[f].push(t);
                }
            }
        }
        let n_trees = ens.tree_roots.len();
        let mut bounds = Self {
            ens,
            n_words,
            sub_min,
            sub_max,
            mask,
            trees_on_feature,
            tree_min: vec![0.0; n_trees],
            tree_max: vec![0.0; n_trees],
            score_min: 0.0,
            score_max: 0.0,
        };
        bounds.recompute(&BitSet::new(ens.n_features), assigned, values);
        bounds
    }

    /// Walk every tree from its root and re-sum — the from-scratch path.
    fn recompute(&mut self, assigned_mask: &BitSet, assigned: &[bool], values: &[f64]) {
        for t in 0..self.ens.tree_roots.len() {
            let root = self.ens.tree_roots[t] as usize;
            let (low, high) = self.walk(root, assigned_mask, assigned, values);
            self.tree_min[t] = low;
            self.tree_max[t] = high;
        }
        self.resum();
    }

    /// Refresh the trees that split on feature `j`; return their old brackets.
    fn apply(
        &mut self,
        j: usize,
        assigned_mask: &BitSet,
        assigned: &[bool],
        values: &[f64],
    ) -> Vec<(usize, f64, f64)> {
        let mut frame: Vec<(usize, f64, f64)> = Vec::new();
        for k in 0..self.trees_on_feature[j].len() {
            let t = self.trees_on_feature[j][k];
            frame.push((t, self.tree_min[t], self.tree_max[t]));
            let root = self.ens.tree_roots[t] as usize;
            let (low, high) = self.walk(root, assigned_mask, assigned, values);
            self.tree_min[t] = low;
            self.tree_max[t] = high;
        }
        self.resum();
        frame
    }

    /// Put back the brackets an `apply` replaced.
    fn restore(&mut self, frame: &[(usize, f64, f64)]) {
        for &(t, low, high) in frame {
            self.tree_min[t] = low;
            self.tree_max[t] = high;
        }
        self.resum();
    }

    // `low + tree` written out rather than `+=`: it is a full re-sum, not an
    // incremental update (module header, rule 2)
    #[allow(clippy::assign_op_pattern)]
    fn resum(&mut self) {
        let mut low = self.ens.base_score;
        let mut high = self.ens.base_score;
        for t in 0..self.tree_min.len() {
            // written out rather than accumulated in place so that no reader
            // mistakes this for an incremental update: it is a full re-sum
            low = low + self.tree_min[t];
            high = high + self.tree_max[t];
        }
        self.score_min = low;
        self.score_max = high;
    }

    fn touches_assigned(&self, idx: usize, assigned_mask: &BitSet) -> bool {
        let base = idx * self.n_words;
        (0..self.n_words).any(|w| self.mask[base + w] & assigned_mask.words[w] != 0)
    }

    fn walk(
        &self,
        idx: usize,
        assigned_mask: &BitSet,
        assigned: &[bool],
        values: &[f64],
    ) -> (f64, f64) {
        if !self.touches_assigned(idx, assigned_mask) {
            return (self.sub_min[idx], self.sub_max[idx]);
        }
        let f = self.ens.feature[idx] as usize; // a set mask bit means this is a split
        let (left, right) = (self.ens.left[idx] as usize, self.ens.right[idx] as usize);
        if assigned[f] {
            let value = values[f];
            let child = if value.is_nan() {
                if self.ens.missing_left[idx] {
                    left
                } else {
                    right
                }
            } else if self.ens.is_lt[idx] {
                if value < self.ens.threshold[idx] {
                    left
                } else {
                    right
                }
            } else if value <= self.ens.threshold[idx] {
                left
            } else {
                right
            };
            return self.walk(child, assigned_mask, assigned, values);
        }
        let (left_min, left_max) = self.walk(left, assigned_mask, assigned, values);
        let (right_min, right_max) = self.walk(right, assigned_mask, assigned, values);
        (py_min(left_min, right_min), py_max(left_max, right_max))
    }
}

// -------------------------------------------------------- the search core ---

/// The arbiter: constraints, then the float-space score, then plausibility.
#[allow(clippy::too_many_arguments)]
fn accepts(
    ens: &Ensemble,
    if_ens: Option<&Ensemble>,
    min_total_path: f64,
    cons: &Constraints,
    x: &[f64],
    lo_t: f64,
    hi_t: f64,
    row: &[f64],
) -> bool {
    // sequential single row: no rayon anywhere in the search (module header)
    if !cons.check(row, 1, x, false)[0] {
        return false;
    }
    let score = ens.raw_score(row);
    if !(lo_t <= score && score <= hi_t) {
        return false;
    }
    match if_ens {
        None => true,
        Some(if_ens) => if_ens.raw_score(row) >= min_total_path,
    }
}

/// Everything the search reads but never writes.
struct Ctx<'a> {
    ens: &'a Ensemble,
    if_ens: Option<&'a Ensemble>,
    min_total_path: f64,
    x: &'a [f64],
    lo_t: f64,
    hi_t: f64,
    cons: &'a Constraints,
    sigma: &'a [f64],
    weights: &'a [f64],
    lam: f64,
    deltas: Vec<(f64, f64)>,
    grids: Vec<Vec<Cell>>,
    domains: Vec<Vec<State>>,
    order: Vec<usize>,
    level_of: Vec<usize>,
    bounds_lo: Vec<f64>,
    bounds_hi: Vec<f64>,
    order_pairs: Vec<(usize, usize)>,
    bounded_pairs: Vec<(usize, usize)>,
    repairable_pairs: Vec<(usize, usize)>,
    g_floor_pairs: Vec<(usize, usize)>,
    spans: Vec<Option<(f64, f64)>>,
    state_spans: Vec<Vec<(f64, f64)>>,
    demanded: Vec<Vec<f64>>,
}

impl Ctx<'_> {
    fn accepts(&self, row: &[f64]) -> bool {
        accepts(
            self.ens,
            self.if_ens,
            self.min_total_path,
            self.cons,
            self.x,
            self.lo_t,
            self.hi_t,
            row,
        )
    }

    fn cost_of(&self, row: &[f64]) -> f64 {
        cost_of_row(
            self.x,
            row,
            self.sigma,
            self.weights,
            self.lam,
            &self.deltas,
        )
    }

    /// The state index feature `f` currently sits on: the one its level pushed,
    /// or the one being tried right now at the level the stack has not taken yet.
    fn chosen(&self, f: usize, stack: &[usize], next_state: usize) -> usize {
        let level = self.level_of[f];
        if level < stack.len() {
            stack[level]
        } else {
            next_state
        }
    }

    /// The values feature `f` can still end up holding. Undecided, that is every
    /// cell it might yet be put in; decided, the cell it was put in — a boundary
    /// repair may still move it anywhere inside — unless the pair cannot be
    /// repaired at all, and then the value it was given is the only point left.
    fn reach(
        &self,
        f: usize,
        movable: bool,
        assigned: &[bool],
        values: &[f64],
        stack: &[usize],
        next_state: usize,
    ) -> (f64, f64) {
        if !assigned[f] {
            return self.spans[f].expect("a bounded pair's features have spans");
        }
        if !movable {
            return (values[f], values[f]);
        }
        self.state_spans[f][self.chosen(f, stack, next_state)]
    }

    /// True when some pair `a <= b` is already out of reach: the lowest value `a`
    /// can still hold is above the highest `b` can.
    fn unorderable(
        &self,
        assigned: &[bool],
        values: &[f64],
        stack: &[usize],
        next_state: usize,
    ) -> bool {
        for pair in &self.bounded_pairs {
            let (a, b) = *pair;
            let movable = self.repairable_pairs.contains(pair);
            if self
                .reach(a, movable, assigned, values, stack, next_state)
                .0
                > self
                    .reach(b, movable, assigned, values, stack, next_state)
                    .1
            {
                return true;
            }
        }
        false
    }

    /// The cell the current assignment puts `f` in, narrowed to its constraint
    /// bounds; `None` when `f` is currently missing.
    fn intersected_cell(&self, f: usize, stack: &[usize], next_state: usize) -> Option<Cell> {
        let picked = self.domains[f][self.chosen(f, stack, next_state)];
        if picked.is_nan {
            return None;
        }
        intersect_cell(
            &self.grids[f][picked.cell_idx],
            self.bounds_lo[f],
            self.bounds_hi[f],
        )
    }

    /// Exact values a repair of this pair may have to land on: what a constraint
    /// asks of either feature, and what the current assignment has already
    /// settled about them. Feature `a` before feature `b`, its constraint's
    /// demands before what the search settled.
    fn demanded_for(&self, a: usize, b: usize, forced_value: &[Option<f64>]) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();
        for f in [a, b] {
            let settled = forced_value[f];
            for value in self.demanded[f].iter().copied().chain(settled) {
                if !out.contains(&value) {
                    out.push(value);
                }
            }
        }
        out
    }

    fn candidates_for(
        &self,
        a: usize,
        b: usize,
        stack: &[usize],
        next_state: usize,
        forced_value: &[Option<f64>],
    ) -> Vec<f64> {
        let (Some(cell_a), Some(cell_b)) = (
            self.intersected_cell(a, stack, next_state),
            self.intersected_cell(b, stack, next_state),
        ) else {
            return Vec::new();
        };
        boundary_candidates(
            &cell_a,
            &cell_b,
            self.x[a],
            self.x[b],
            &self.demanded_for(a, b, forced_value),
        )
    }

    /// The pairs this row orders the wrong way round, by the arbiter's own
    /// reading. A pair with a missing value on either side is left out: a
    /// missing value cannot be pulled onto a boundary.
    fn broken(&self, row: &[f64]) -> Vec<(usize, usize)> {
        self.order_pairs
            .iter()
            .copied()
            .filter(|&(a, b)| {
                !row[a].is_nan() && !row[b].is_nan() && row[a] - row[b] > LINEAR_SLACK
            })
            .collect()
    }

    /// Remember a completion the repair could not settle, whatever the reason.
    /// The committed cost is a floor on what it could have become for the pairs
    /// listed in `g_floor_pairs`, and nothing at all can be said about any
    /// other, so those withdraw outright.
    fn set_aside(&self, pairs: &[(usize, usize)], g: f64, dropped_floor: &mut f64) {
        if pairs.iter().all(|pair| self.g_floor_pairs.contains(pair)) {
            *dropped_floor = py_min(*dropped_floor, g);
        } else {
            *dropped_floor = f64::NEG_INFINITY;
        }
    }

    /// The row to weigh against the incumbent, or `None` if there is none.
    ///
    /// A completed assignment usually goes straight to the arbiter. When it
    /// orders some pair the wrong way round, both features of that pair first
    /// move onto one shared value inside their cells — the cheapest such value
    /// the arbiter still accepts. Moving inside a cell cannot change how any
    /// tree routes the row, so the repair keeps the score it was pruned on.
    /// Several broken pairs at once are repaired one after another, each on the
    /// cheapest shared value regardless of the arbiter, and the whole completion
    /// is dropped if any pair is left broken.
    fn finish(
        &self,
        row: &[f64],
        stack: &[usize],
        next_state: usize,
        forced_value: &[Option<f64>],
        g: f64,
        dropped_floor: &mut f64,
    ) -> Option<Vec<f64>> {
        let violated = self.broken(row);
        if violated.is_empty() {
            return if self.accepts(row) {
                Some(row.to_vec())
            } else {
                None
            };
        }
        if violated
            .iter()
            .any(|pair| !self.repairable_pairs.contains(pair))
        {
            return None; // a policy-bound pair; the arbiter rejects the row anyway
        }
        if violated.len() == 1 {
            let (a, b) = violated[0];
            let mut best_row: Option<Vec<f64>> = None;
            let mut best_cost = f64::INFINITY;
            for t in self.candidates_for(a, b, stack, next_state, forced_value) {
                let mut variant = row.to_vec();
                variant[a] = t;
                variant[b] = t;
                let cost = self.cost_of(&variant);
                if cost < best_cost && self.accepts(&variant) {
                    best_cost = cost;
                    best_row = Some(variant);
                }
            }
            if best_row.is_none() {
                self.set_aside(&violated, g, dropped_floor);
            }
            return best_row;
        }
        let mut repaired = row.to_vec();
        for &(a, b) in &violated {
            let mut best_t: Option<f64> = None;
            let mut best_cost = f64::INFINITY;
            for t in self.candidates_for(a, b, stack, next_state, forced_value) {
                let mut variant = repaired.clone();
                variant[a] = t;
                variant[b] = t;
                let cost = self.cost_of(&variant);
                if cost < best_cost {
                    best_cost = cost;
                    best_t = Some(t);
                }
            }
            let Some(best_t) = best_t else {
                self.set_aside(&violated, g, dropped_floor);
                return None;
            };
            repaired[a] = best_t;
            repaired[b] = best_t;
        }
        if !self.broken(&repaired).is_empty() || !self.accepts(&repaired) {
            self.set_aside(&violated, g, dropped_floor);
            return None;
        }
        Some(repaired)
    }
}

/// (feature index, model bracket frame, plausibility bracket frame, cost before
/// the move, propagation frame)
struct Frame {
    j: usize,
    model: Vec<(usize, f64, f64)>,
    plausibility: Vec<(usize, f64, f64)>,
    g_before: f64,
    prop: PropFrame,
}

#[allow(clippy::too_many_arguments)]
fn undo(
    frame: &Frame,
    propagation: &mut Propagation,
    model_bounds: &mut EnsembleBounds,
    if_bounds: &mut Option<EnsembleBounds>,
    assigned: &mut [bool],
    assigned_mask: &mut BitSet,
    g: &mut f64,
) {
    propagation.restore(&frame.prop);
    model_bounds.restore(&frame.model);
    if let Some(bounds) = if_bounds.as_mut() {
        bounds.restore(&frame.plausibility);
    }
    assigned[frame.j] = false;
    assigned_mask.clear(frame.j);
    *g = frame.g_before;
}

fn stats(
    nodes_expanded: u64,
    nodes_pruned_score: u64,
    nodes_pruned_cost: u64,
    lower_bound: f64,
    gap: f64,
    completed: bool,
    warm_start_used: bool,
) -> ExactStats {
    ExactStats {
        nodes_expanded,
        nodes_pruned_score,
        nodes_pruned_cost,
        lower_bound,
        gap,
        completed,
        warm_start_used,
    }
}

/// Search the cell grid depth-first for the cheapest counterfactual — port of
/// `treecf.backends.exact.solve_exact`.
///
/// Features are assigned one at a time in a fixed order, each from its own list
/// of candidate states. Two bounds cut the tree: the score bracket the ensemble
/// can still reach, and the cost already committed plus the cheapest possible
/// remainder. A full assignment is accepted only if the compiled constraints
/// admit the row, its score re-computed in float space lands in the target, and
/// — when configured — the isolation forest still calls it plausible.
///
/// `incumbent` is an optional `(cost, row)` warm start already costed *and*
/// verified by the caller: the search takes its feasibility on trust, prunes
/// against its cost, and may hand it straight back.
///
/// The two ways of coming back empty-handed have to be told apart by
/// `stats.completed`, not by `proof`: an `x_cf` of `None` with `completed` true
/// certifies that no counterfactual exists in the searched space, while
/// `completed` false only means the search never settled the whole space.
///
/// `Err` mirrors Python's `ConstraintValidationError` for a multi-feature Linear
/// outside the canonical order-pair shape.
#[allow(clippy::too_many_arguments)]
pub fn solve_exact(
    ens: &Ensemble,
    x: &[f64],
    interval: (f64, f64),
    cons: &Constraints,
    sigma: &[f64],
    weights: &[f64],
    lam: f64,
    value_policies: &[Option<ValuePolicy>],
    plausibility: Option<(&Ensemble, f64)>,
    params: &ExactParams,
    incumbent: Option<(f64, &[f64])>,
) -> Result<ExactResult, String> {
    let start = Instant::now();
    let order_pairs = validate(cons)?;
    let (lo_t, hi_t) = interval;
    let if_ens = plausibility.map(|(ir, _)| ir);
    let min_total_path = plausibility.map_or(0.0, |(_, bound)| bound);
    let gap = params.gap;

    // (a) The factual itself: nothing is ever cheaper than not moving at all.
    if accepts(ens, if_ens, min_total_path, cons, x, lo_t, hi_t, x) {
        return Ok(ExactResult {
            x_cf: Some(x.to_vec()),
            proof: "optimal",
            stats: stats(0, 0, 0, 0.0, gap, true, false),
            snapped: Vec::new(),
            distance: Some(0.0),
        });
    }

    let ensembles: Vec<&Ensemble> = match if_ens {
        None => vec![ens],
        Some(if_ens) => vec![ens, if_ens],
    };
    let grids = constraint_cells(cons, &ensembles);
    let domains = build_domains(&grids, x, cons, sigma, weights, lam, value_policies);
    let order = feature_order(&grids, cons);
    if order.iter().any(|&j| domains[j].is_empty()) {
        // Contradictory constraints left a feature with no legal value at all:
        // nothing to search, and nothing can be feasible.
        return Ok(ExactResult {
            x_cf: None,
            proof: "optimal",
            stats: stats(0, 0, 0, f64::INFINITY, gap, true, false),
            snapped: Vec::new(),
            distance: None,
        });
    }
    let h_suffix = h_suffix(&order, &domains);

    let mut level_of = vec![usize::MAX; x.len()];
    for (level, &f) in order.iter().enumerate() {
        level_of[f] = level;
    }
    // Every feature an implication, a one-hot group or an order pair mentions is
    // constraint-referenced, and feature_order keeps all of those, so the search
    // really does get to decide each of them.
    debug_assert!(
        cons.implications
            .iter()
            .flat_map(|&(ci, _, si, _)| [ci as usize, si as usize])
            .chain(cons.onehot.iter().flatten().map(|&f| f as usize))
            .chain(order_pairs.iter().flat_map(|&(a, b)| [a, b]))
            .all(|f| level_of[f] != usize::MAX),
        "a related feature was left out of the search order"
    );

    let (bounds_lo, bounds_hi, _) = cons.instance_bounds(x);
    let bounds_lo: Vec<f64> = bounds_lo
        .iter()
        .map(|&v| if v.is_nan() { f64::NEG_INFINITY } else { v })
        .collect();
    let bounds_hi: Vec<f64> = bounds_hi
        .iter()
        .map(|&v| if v.is_nan() { f64::INFINITY } else { v })
        .collect();
    let mut spans: Vec<Option<(f64, f64)>> = vec![None; x.len()];
    for &(a, b) in &order_pairs {
        for f in [a, b] {
            spans[f] = domain_span(&domains[f], &grids[f], bounds_lo[f], bounds_hi[f]);
        }
    }
    let bounded_pairs: Vec<(usize, usize)> = order_pairs
        .iter()
        .copied()
        .filter(|&(a, b)| spans[a].is_some() && spans[b].is_some())
        .collect();
    let mut state_spans: Vec<Vec<(f64, f64)>> = vec![Vec::new(); x.len()];
    for &(a, b) in &bounded_pairs {
        for f in [a, b] {
            state_spans[f] = domains[f]
                .iter()
                .map(|st| {
                    match intersect_cell(&grids[f][st.cell_idx], bounds_lo[f], bounds_hi[f]) {
                        None => (st.value, st.value),
                        Some(iv) => achievable_bounds(&iv),
                    }
                })
                .collect();
        }
    }

    // A feature with a value policy has to land on the policy's grid, so it
    // cannot be nudged to an arbitrary point inside its cell and pairs touching
    // one are never repaired. Nothing else is held back: a repair only ever
    // proposes a row, and the arbiter turns down the ones that break something.
    let policy_active = |f: usize| value_policies[f].is_some();
    let repairable_pairs: Vec<(usize, usize)> = order_pairs
        .iter()
        .copied()
        .filter(|&(a, b)| !policy_active(a) && !policy_active(b))
        .collect();
    let policy_bound = !order_pairs.is_empty() && repairable_pairs.len() < order_pairs.len();
    let mut onehot_member = vec![false; cons.n_features];
    for group in &cons.onehot {
        for &f in group {
            onehot_member[f as usize] = true;
        }
    }
    let demanded = demanded_values(cons);
    let policy_flags: Vec<bool> = (0..cons.n_features).map(policy_active).collect();
    let g_floor_pairs = g_floor_pairs(&order_pairs, &onehot_member, &demanded, &policy_flags);

    let ctx = Ctx {
        ens,
        if_ens,
        min_total_path,
        x,
        lo_t,
        hi_t,
        cons,
        sigma,
        weights,
        lam,
        deltas: allow_missing_deltas(cons),
        grids,
        domains,
        order,
        level_of,
        bounds_lo,
        bounds_hi,
        order_pairs,
        bounded_pairs,
        repairable_pairs,
        g_floor_pairs,
        spans,
        state_spans,
        demanded,
    };

    let mut assigned = vec![false; x.len()];
    let mut values = vec![0.0; x.len()];
    let mut assigned_mask = BitSet::new(ens.n_features);
    let mut propagation = Propagation::new(cons, &ctx.domains);
    let mut model_bounds = EnsembleBounds::new(ens, &assigned, &values);
    let mut if_bounds = if_ens.map(|ir| EnsembleBounds::new(ir, &assigned, &values));

    let mut incumbent_cost = f64::INFINITY;
    let mut incumbent_row: Option<Vec<f64>> = None;
    let mut incumbent_states: Option<Vec<State>> = None;
    let mut warm_start_used = false;
    if let Some((cost, row)) = incumbent {
        // (b) A warm start from another backend. Its states are unknown, so a
        // warm winner reports no snapping of its own — the backend that produced
        // the row already applied any value policy to it.
        incumbent_cost = cost;
        incumbent_row = Some(row.to_vec());
        warm_start_used = true;
    }

    let mut nodes_expanded: u64 = 0;
    let mut nodes_pruned_score: u64 = 0;
    let mut nodes_pruned_cost: u64 = 0;
    let mut gap_prune_fired = false;
    let mut completed = true;
    // cheapest committed cost among the completions the repair had to set aside;
    // nothing derived from one of those can cost less than this, so once the
    // incumbent is at least as cheap, setting them aside changed nothing
    let mut dropped_floor = if policy_bound {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };

    let mut stack: Vec<usize> = Vec::new(); // state index chosen at each assigned level
    let mut frames: Vec<Frame> = Vec::new();
    let mut g_stack: Vec<f64> = vec![0.0]; // cost committed before the level of the same index
    let mut g = 0.0;
    let mut next_state: usize = 0;

    while !ctx.order.is_empty() {
        let k = stack.len();
        let n_states = ctx.domains[ctx.order[k]].len();
        if next_state >= n_states {
            if stack.is_empty() {
                break; // the whole space has been enumerated
            }
            let frame = frames.pop().expect("a stacked level owns a frame");
            undo(
                &frame,
                &mut propagation,
                &mut model_bounds,
                &mut if_bounds,
                &mut assigned,
                &mut assigned_mask,
                &mut g,
            );
            g_stack.pop();
            next_state = stack.pop().expect("checked non-empty") + 1;
            continue;
        }
        if nodes_expanded >= params.node_budget
            || start.elapsed().as_secs_f64() > params.time_budget_s
        {
            completed = false;
            break;
        }

        nodes_expanded += 1;
        let j = ctx.order[k];
        let state = ctx.domains[j][next_state];
        let (prop_frame, conflict) = propagation.apply(j, state.value, &assigned, &values);
        assigned[j] = true;
        values[j] = state.value;
        assigned_mask.set(j);
        let frame = Frame {
            j,
            model: model_bounds.apply(j, &assigned_mask, &assigned, &values),
            plausibility: match if_bounds.as_mut() {
                Some(bounds) => bounds.apply(j, &assigned_mask, &assigned, &values),
                None => Vec::new(),
            },
            g_before: g,
            prop: prop_frame,
        };
        g += state.cost;

        if conflict
            || (!ctx.bounded_pairs.is_empty()
                && ctx.unorderable(&assigned, &values, &stack, next_state))
        {
            // No completion below this state can satisfy the constraints, so it
            // is cut on feasibility, counted with the cost prunes.
            nodes_pruned_cost += 1;
            undo(
                &frame,
                &mut propagation,
                &mut model_bounds,
                &mut if_bounds,
                &mut assigned,
                &mut assigned_mask,
                &mut g,
            );
            next_state += 1;
            continue;
        }
        if model_bounds.score_max < lo_t || model_bounds.score_min > hi_t {
            nodes_pruned_score += 1;
            undo(
                &frame,
                &mut propagation,
                &mut model_bounds,
                &mut if_bounds,
                &mut assigned,
                &mut assigned_mask,
                &mut g,
            );
            next_state += 1;
            continue;
        }
        if if_bounds
            .as_ref()
            .is_some_and(|bounds| bounds.score_max < min_total_path)
        {
            nodes_pruned_score += 1;
            undo(
                &frame,
                &mut propagation,
                &mut model_bounds,
                &mut if_bounds,
                &mut assigned,
                &mut assigned_mask,
                &mut g,
            );
            next_state += 1;
            continue;
        }
        let floor = g + h_suffix[k + 1];
        let threshold = if gap == 0.0 {
            incumbent_cost
        } else {
            incumbent_cost / (1.0 + gap)
        };
        if floor >= threshold {
            nodes_pruned_cost += 1;
            if incumbent_cost > floor {
                gap_prune_fired = true; // only the widened threshold cut this branch
            }
            undo(
                &frame,
                &mut propagation,
                &mut model_bounds,
                &mut if_bounds,
                &mut assigned,
                &mut assigned_mask,
                &mut g,
            );
            next_state += 1;
            continue;
        }

        if k + 1 == ctx.order.len() {
            let mut row = x.to_vec();
            for (level, &chosen) in stack.iter().enumerate() {
                row[ctx.order[level]] = ctx.domains[ctx.order[level]][chosen].value;
            }
            row[j] = state.value;
            let accepted = ctx.finish(
                &row,
                &stack,
                next_state,
                &propagation.forced_value,
                g,
                &mut dropped_floor,
            );
            if let Some(accepted) = accepted {
                let cost = ctx.cost_of(&accepted);
                if cost < incumbent_cost {
                    incumbent_cost = cost;
                    incumbent_row = Some(accepted);
                    let mut winning: Vec<State> = stack
                        .iter()
                        .enumerate()
                        .map(|(level, &chosen)| ctx.domains[ctx.order[level]][chosen])
                        .collect();
                    winning.push(state);
                    incumbent_states = Some(winning);
                }
            }
            undo(
                &frame,
                &mut propagation,
                &mut model_bounds,
                &mut if_bounds,
                &mut assigned,
                &mut assigned_mask,
                &mut g,
            );
            next_state += 1;
            continue;
        }

        stack.push(next_state);
        frames.push(frame);
        g_stack.push(g);
        next_state = 0;
    }

    completed = completed && dropped_floor >= incumbent_cost;
    let lower_bound;
    let proof;
    if completed {
        lower_bound = match incumbent_row {
            None => f64::INFINITY,
            Some(_) if gap == 0.0 => incumbent_cost,
            Some(_) => incumbent_cost / (1.0 + gap),
        };
        proof = if gap > 0.0 && gap_prune_fired {
            "optimal_within_gap"
        } else {
            "optimal"
        };
    } else {
        let mut open_view = f64::INFINITY;
        if !ctx.order.is_empty() {
            for level in 0..g_stack.len() {
                open_view = py_min(open_view, g_stack[level] + h_suffix[level]);
            }
        }
        // a completion the repair set aside is worth at least its committed
        // cost, or — where even that does not hold — at least nothing, since the
        // objective is a sum of non-negative terms
        let set_aside_view = if dropped_floor == f64::NEG_INFINITY {
            0.0
        } else {
            dropped_floor
        };
        lower_bound = py_min(py_min(open_view, incumbent_cost), set_aside_view);
        proof = "heuristic";
    }

    let mut snapped: Vec<usize> = Vec::new();
    for (level, chosen_state) in incumbent_states.iter().flatten().enumerate() {
        let f = ctx.order[level];
        // a feature an order-pair repair moved no longer holds the value the
        // policy produced, so it is not reported as snapped either
        if let Some(row) = incumbent_row.as_ref() {
            if row[f] != chosen_state.value {
                continue;
            }
        }
        if chosen_state.snapped && chosen_state.value != x[f] {
            snapped.push(f);
        }
    }

    Ok(ExactResult {
        distance: incumbent_row.as_ref().map(|_| incumbent_cost),
        x_cf: incumbent_row,
        proof,
        stats: stats(
            nodes_expanded,
            nodes_pruned_score,
            nodes_pruned_cost,
            lower_bound,
            gap,
            completed,
            warm_start_used,
        ),
        snapped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{LinearC, LIN_GE};
    use crate::ir::Link;

    // ------------------------------------------------------------ builders ---

    /// One stump per spec `(feature, threshold, is_lt, left_value, right_value)`,
    /// missing routed right unless `missing_left` is asked for.
    fn stumps(specs: &[(i32, f64, bool, f64, f64)], n_features: usize) -> Ensemble {
        stumps_missing(specs, n_features, false)
    }

    fn stumps_missing(
        specs: &[(i32, f64, bool, f64, f64)],
        n_features: usize,
        missing_left: bool,
    ) -> Ensemble {
        let mut feature = Vec::new();
        let mut threshold = Vec::new();
        let mut is_lt = Vec::new();
        let mut miss = Vec::new();
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut value = Vec::new();
        let mut roots = Vec::new();
        for (t, &(f, thr, lt, lv, rv)) in specs.iter().enumerate() {
            let base = (t * 3) as u32;
            roots.push(base);
            feature.extend_from_slice(&[f, -1, -1]);
            threshold.extend_from_slice(&[thr, 0.0, 0.0]);
            is_lt.extend_from_slice(&[lt, false, false]);
            miss.extend_from_slice(&[missing_left, false, false]);
            left.extend_from_slice(&[base + 1, 0, 0]);
            right.extend_from_slice(&[base + 2, 0, 0]);
            value.extend_from_slice(&[0.0, lv, rv]);
        }
        Ensemble::new(
            feature,
            threshold,
            is_lt,
            miss,
            left,
            right,
            value,
            roots,
            0.0,
            Link::Identity,
            n_features,
        )
        .unwrap()
    }

    fn cons_base(p: usize) -> Constraints {
        Constraints {
            n_features: p,
            freeze: vec![],
            ranges: vec![],
            equals: vec![],
            monotone: vec![],
            linears: vec![],
            implications: vec![],
            onehot: vec![],
            allow_missing: vec![],
        }
    }

    fn no_policies(p: usize) -> Vec<Option<ValuePolicy>> {
        vec![None; p]
    }

    fn cell(lo: f64, hi: f64, lo_open: bool, hi_open: bool) -> Cell {
        Cell {
            lo,
            hi,
            lo_open,
            hi_open,
        }
    }

    /// Deterministic LCG — fixed seeds, no external rng in tests.
    struct Lcg(u64);

    impl Lcg {
        fn next_usize(&mut self, bound: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 33) as usize) % bound
        }
    }

    // ------------------------------------------------------- trigger splits ---

    /// The refined grid must still partition the line: neighbours meet at one
    /// value with exactly one closed side, and every probe lands in exactly one
    /// cell.
    fn assert_tiles_the_line(cells: &[Cell], probes: &[f64]) {
        assert_eq!(cells[0].lo, f64::NEG_INFINITY);
        assert!(cells[0].lo_open);
        assert_eq!(cells[cells.len() - 1].hi, f64::INFINITY);
        assert!(cells[cells.len() - 1].hi_open);
        for w in cells.windows(2) {
            assert_eq!(w[0].hi, w[1].lo, "cells must meet");
            assert!(
                w[0].hi_open != w[1].lo_open,
                "exactly one side of a junction is closed"
            );
        }
        for &probe in probes {
            let hits = cells.iter().filter(|c| c.contains(probe)).count();
            assert_eq!(hits, 1, "{probe} landed in {hits} cells");
        }
    }

    #[test]
    fn trigger_split_tiles_the_line_once() {
        let ens = stumps(&[(0, 1.0, true, 0.0, 1.0), (0, 3.0, false, 0.0, 1.0)], 1);
        let mut cons = cons_base(1);
        cons.implications = vec![(0, 2.0, 0, 1.0)];
        let grids = constraint_cells(&cons, &[&ens]);
        assert_tiles_the_line(
            &grids[0],
            &[-5.0, 0.999, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 99.0],
        );
        // the trigger value now owns a singleton cell of its own
        assert!(grids[0]
            .iter()
            .any(|c| c.lo == 2.0 && c.hi == 2.0 && !c.lo_open && !c.hi_open));
    }

    #[test]
    fn splitting_at_an_existing_singleton_is_idempotent() {
        let cells = crate::cells::build_cells(&[(1.0, true), (1.0, false)]);
        let once = split_cell_at(&cells, 1.0);
        let twice = split_cell_at(&once, 1.0);
        assert_eq!(once, cells);
        assert_eq!(twice, cells);
    }

    #[test]
    fn constraint_cells_returns_the_routing_grid_when_no_implication_exists() {
        let ens = stumps(&[(0, 1.0, true, 0.0, 1.0)], 2);
        let cons = cons_base(2);
        assert_eq!(
            constraint_cells(&cons, &[&ens]),
            feature_cells_joint(&[&ens])
        );
    }

    // ------------------------------------------------------ cell arithmetic ---

    #[test]
    fn intersect_cell_keeps_openness_and_drops_degenerate_singletons() {
        let c = cell(0.0, 2.0, true, true);
        assert_eq!(
            intersect_cell(&c, 1.0, 3.0),
            Some(cell(1.0, 2.0, false, true))
        );
        assert_eq!(
            intersect_cell(&c, -1.0, 1.0),
            Some(cell(0.0, 1.0, true, false))
        );
        assert_eq!(intersect_cell(&c, 2.0, 5.0), None); // open edge, empty
        assert_eq!(intersect_cell(&c, 5.0, 6.0), None);
    }

    #[test]
    fn achievable_bounds_step_one_f32_ulp_inside_finite_open_edges() {
        let c = cell(0.0, 1.0, true, true);
        let (lo, hi) = achievable_bounds(&c);
        assert_eq!(lo, f64::from((0.0f32).next_up()));
        assert_eq!(hi, f64::from((1.0f32).next_down()));
        let unbounded = cell(f64::NEG_INFINITY, f64::INFINITY, true, true);
        assert_eq!(
            achievable_bounds(&unbounded),
            (f64::NEG_INFINITY, f64::INFINITY)
        );
    }

    #[test]
    fn boundary_candidates_keep_their_fixed_order_and_drop_repeats() {
        let a = cell(0.0, 10.0, false, false);
        let b = cell(2.0, 6.0, false, false);
        // x_a, x_b, demanded, low end, high end; 2.0 offered twice, kept once
        let got = boundary_candidates(&a, &b, 3.0, 2.0, &[5.0, 2.0]);
        assert_eq!(got, vec![3.0, 2.0, 5.0, 6.0]);
        // out-of-interval and non-finite candidates drop out
        let got = boundary_candidates(&a, &b, 99.0, f64::INFINITY, &[]);
        assert_eq!(got, vec![2.0, 6.0]);
        // disjoint cells: nothing to try
        assert!(
            boundary_candidates(&a, &cell(20.0, 30.0, false, false), 1.0, 25.0, &[]).is_empty()
        );
    }

    // -------------------------------------------------------------- snapping ---

    /// Goldens from CPython:
    /// `uv run python -c 'print([hash(v) for v in (0.1, 2.5, -3.75, 1e300, 2.0, -1.0, 7.0, 8.0, 0.0, 1e-320, 123456789.123)])'`
    #[test]
    fn float_hash_matches_cpython() {
        let goldens: [(f64, i64); 11] = [
            (0.1, 230584300921369408),
            (2.5, 1152921504606846978),
            (-3.75, -1729382256910270467),
            (1e300, 1224995262755759164),
            (2.0, 2),
            (-1.0, -2),
            (7.0, 7),
            (8.0, 8),
            (0.0, 0),
            (1e-320, 33957085184),
            (123456789.123, 283618680910892309),
        ];
        for (value, want) in goldens {
            assert_eq!(py_hash_double(value), want, "hash({value})");
        }
    }

    /// Ties between two equally distant candidates are decided by CPython's set
    /// iteration order, so `_snap`'s candidate list is reproduced through it.
    /// Goldens: `uv run python -c 'import math; print([sorted({math.floor(v), math.ceil(v)}, key=lambda c: abs(c - v)) for v in (7.5, 2.5, -1.5, -2.5, 0.5)])'`
    #[test]
    fn integer_snap_candidates_follow_python_set_order() {
        assert_eq!(snap_candidates(&[7.0, 8.0], 7.5), vec![8.0, 7.0]);
        assert_eq!(snap_candidates(&[2.0, 3.0], 2.5), vec![2.0, 3.0]);
        assert_eq!(snap_candidates(&[-2.0, -1.0], -1.5), vec![-2.0, -1.0]);
        assert_eq!(snap_candidates(&[-3.0, -2.0], -2.5), vec![-3.0, -2.0]);
        assert_eq!(snap_candidates(&[0.0, 1.0], 0.5), vec![0.0, 1.0]);
        // no tie: the nearer candidate wins regardless of set order
        assert_eq!(snap_candidates(&[7.0, 8.0], 7.9), vec![8.0, 7.0]);
        assert_eq!(snap_candidates(&[2.0, 3.0], 2.9), vec![3.0, 2.0]);
    }

    /// Goldens: `uv run python -c 'print([sorted({b, b-s, b+s}, key=lambda c: abs(c-v)) for b, s, v in ((0.0,1.0,0.0),(2.0,2.0,2.0),(-4.0,4.0,-4.0))])'`
    /// The last triple collides in the table (`hash(-8.0) & 7 == hash(0.0) & 7`),
    /// so it also pins the perturbed-probe path.
    #[test]
    fn grid_snap_candidates_follow_python_set_order() {
        assert_eq!(
            snap_candidates(&[0.0, -1.0, 1.0], 0.0),
            vec![0.0, 1.0, -1.0]
        );
        assert_eq!(snap_candidates(&[2.0, 0.0, 4.0], 2.0), vec![2.0, 0.0, 4.0]);
        assert_eq!(
            snap_candidates(&[-4.0, -8.0, 0.0], -4.0),
            vec![-4.0, -8.0, 0.0]
        );
    }

    /// Goldens straight from `treecf.api._snap`:
    /// `uv run python -c 'from treecf.api import _snap, Grid; g = Grid(step=2.0, anchor=0.0); print(_snap(2.4, "integer", lambda c: True, -10, 10), _snap(2.4, "integer", lambda c: True, 3, 10), _snap(2.25, "integer", lambda c: 2.2 <= c <= 2.3, -10, 10), _snap(1.0, g, lambda c: True, -10, 10), _snap(3.0, g, lambda c: True, -10, 10), _snap(2.6, g, lambda c: True, -10, 10))'`
    #[test]
    fn snap_takes_the_first_candidate_inside_cell_and_bounds() {
        let wide = cell(f64::NEG_INFINITY, f64::INFINITY, true, true);
        assert_eq!(
            snap(2.4, &ValuePolicy::Integer, &wide, -10.0, 10.0),
            Some(2.0)
        );
        // 2.0 is outside the bounds, so the further candidate wins
        assert_eq!(
            snap(2.4, &ValuePolicy::Integer, &wide, 3.0, 10.0),
            Some(3.0)
        );
        // nothing on the grid fits inside the cell
        let narrow = cell(2.2, 2.3, false, false);
        assert_eq!(
            snap(2.25, &ValuePolicy::Integer, &narrow, -10.0, 10.0),
            None
        );
        let grid = ValuePolicy::Grid {
            step: 2.0,
            anchor: 0.0,
        };
        // round-half-to-even on the base, like Python's round(): 0.5 -> 0
        assert_eq!(snap(1.0, &grid, &wide, -10.0, 10.0), Some(0.0));
        // 1.5 -> 2, base 4.0, but 2.0 is equally close and comes first out of
        // the candidate set
        assert_eq!(snap(3.0, &grid, &wide, -10.0, 10.0), Some(2.0));
        assert_eq!(snap(2.6, &grid, &wide, -10.0, 10.0), Some(2.0));
    }

    // ---------------------------------------------------- domain goldens ---
    //
    // Every `want` below is the Python reference's own answer, taken as raw f64
    // bit patterns from `_build_domains` over the same inputs:
    //
    //   uv run python -c 'import numpy as np, struct; from treecf.backends._exact_domains
    //   import _build_domains, _constraint_cells; from treecf.constraints.compile import
    //   compile_constraints; from treecf.constraints.objects import *; from treecf.ir.model
    //   import *; ...' — the scenario builders are spelled out per test below.

    /// `(value bits, cost bits, cell index, is_nan, snapped)` per state, in order.
    fn expect_states(got: &[State], want: &[(u64, u64, usize, bool, bool)]) {
        assert_eq!(got.len(), want.len(), "state count: {got:?}");
        for (k, (state, &(value, cost, cell_idx, is_nan, snapped))) in
            got.iter().zip(want).enumerate()
        {
            assert_eq!(state.value.to_bits(), value, "state {k} value: {state:?}");
            assert_eq!(state.cost.to_bits(), cost, "state {k} cost: {state:?}");
            assert_eq!(state.cell_idx, cell_idx, "state {k} cell: {state:?}");
            assert_eq!(state.is_nan, is_nan, "state {k} is_nan: {state:?}");
            assert_eq!(state.snapped, snapped, "state {k} snapped: {state:?}");
        }
    }

    /// The two-feature grid every domain golden below is drawn from:
    /// f0 cells `(-inf,1)`, `[1,3]`, `(3,inf)`; f1 one cell.
    fn golden_ens() -> Ensemble {
        stumps(&[(0, 1.0, true, 0.0, 1.0), (0, 3.0, false, 0.0, 1.0)], 2)
    }

    fn domains_of(
        ens: &Ensemble,
        x: &[f64],
        cons: &Constraints,
        lam: f64,
        policies: &[Option<ValuePolicy>],
    ) -> Vec<Vec<State>> {
        let grids = constraint_cells(cons, &[ens]);
        build_domains(&grids, x, cons, &[1.0, 1.0], &[1.0, 1.0], lam, policies)
    }

    const NAN_BITS: u64 = 0x7ff8000000000000;

    #[test]
    fn domain_keeps_the_factual_and_one_nearest_point_per_cell() {
        let ens = golden_ens();
        let doms = domains_of(&ens, &[2.0, 0.0], &cons_base(2), 0.0, &no_policies(2));
        expect_states(
            &doms[0],
            &[
                (0x4000000000000000, 0x0, 1, false, false), // keep 2.0
                (0x3fefffffe0000000, 0x3ff0000010000000, 0, false, false), // one f32 ulp below 1
                (0x4008000020000000, 0x3ff0000040000000, 2, false, false), // one f32 ulp above 3
            ],
        );
        expect_states(&doms[1], &[(0x0, 0x0, 0, false, false)]);
    }

    #[test]
    fn pinned_feature_keeps_the_pin_and_still_offers_missing() {
        let ens = golden_ens();
        let mut cons = cons_base(2);
        cons.ranges = vec![(0, 2.0, 2.0)];
        cons.allow_missing = vec![(0, 0.5, 0.5)];
        let doms = domains_of(&ens, &[0.0, 0.0], &cons, 0.0, &no_policies(2));
        expect_states(
            &doms[0],
            &[
                (NAN_BITS, 0x3fe0000000000000, 3, true, false), // NaN at delta_miss
                (0x4000000000000000, 0x4000000000000000, 1, false, false), // the pin
            ],
        );
    }

    #[test]
    fn nan_factual_pinned_offers_staying_missing_and_moving_to_the_pin() {
        let ens = golden_ens();
        let mut cons = cons_base(2);
        cons.ranges = vec![(0, 2.0, 2.0)];
        cons.allow_missing = vec![(0, 0.5, 0.5)];
        let doms = domains_of(&ens, &[f64::NAN, 0.0], &cons, 0.0, &no_policies(2));
        expect_states(
            &doms[0],
            &[
                (NAN_BITS, 0x0, 3, true, false),
                (0x4000000000000000, 0x3fe0000000000000, 1, false, false),
            ],
        );
    }

    #[test]
    fn nan_factual_without_allow_missing_can_only_stay_missing() {
        let ens = golden_ens();
        let doms = domains_of(&ens, &[f64::NAN, 0.0], &cons_base(2), 0.0, &no_policies(2));
        expect_states(&doms[0], &[(NAN_BITS, 0x0, 3, true, false)]);
    }

    /// A forbidding `missing_policy` on a NaN factual with no AllowMissing
    /// leaves the feature nothing at all — the certified-infeasible signal.
    #[test]
    fn suppressed_nan_factual_yields_an_empty_domain() {
        let ens = golden_ens();
        let mut cons = cons_base(2);
        cons.linears = vec![LinearC {
            indices: vec![0],
            coefs: vec![1.0],
            op: LIN_LE,
            rhs: 5.0,
            policy: 1, // forbid_missing / violated
        }];
        cons.ranges = vec![(0, f64::NEG_INFINITY, 5.000000001)]; // the derived range
        let doms = domains_of(&ens, &[f64::NAN, 0.0], &cons, 0.0, &no_policies(2));
        assert!(doms[0].is_empty());
    }

    #[test]
    fn frozen_feature_gets_one_free_keep_state() {
        let ens = golden_ens();
        let mut cons = cons_base(2);
        cons.freeze = vec![0];
        let doms = domains_of(&ens, &[2.0, 0.0], &cons, 0.0, &no_policies(2));
        expect_states(&doms[0], &[(0x4000000000000000, 0x0, 1, false, false)]);
    }

    /// A one-hot member offers whichever of 0.0/1.0 each surviving cell holds —
    /// 0.0 before 1.0 when the two share a cell and cost the same — while the
    /// unchanged factual stays available even when it is not binary at all.
    #[test]
    fn onehot_member_is_restricted_to_zero_and_one() {
        let ens = golden_ens();
        let mut cons = cons_base(2);
        cons.onehot = vec![vec![0, 1]];
        let doms = domains_of(&ens, &[0.5, 0.0], &cons, 0.0, &no_policies(2));
        expect_states(
            &doms[0],
            &[
                (0x3fe0000000000000, 0x0, 0, false, false), // keep 0.5
                (0x0, 0x3fe0000000000000, 0, false, false), // 0.0 before ...
                (0x3ff0000000000000, 0x3fe0000000000000, 1, false, false), // ... 1.0
            ],
        );
        // f1 has one wide cell: both binary values come out of it
        expect_states(
            &doms[1],
            &[
                (0x0, 0x0, 0, false, false),
                (0x3ff0000000000000, 0x3ff0000000000000, 0, false, false),
            ],
        );
    }

    /// An implication cuts its trigger value out of the grid and hands its
    /// consequence the demanded value as an extra, never-snapped candidate.
    #[test]
    fn implication_adds_demanded_candidate_and_splits_the_trigger_cell() {
        let ens = golden_ens();
        let mut cons = cons_base(2);
        cons.implications = vec![(1, 1.0, 0, 1.0)]; // b == 1 -> a == 1
        let grids = constraint_cells(&cons, &[&ens]);
        assert_eq!(grids[1].len(), 3); // (-inf,1) [1,1] (1,inf)
        let doms = domains_of(&ens, &[2.0, 0.0], &cons, 0.0, &no_policies(2));
        expect_states(
            &doms[0],
            &[
                (0x4000000000000000, 0x0, 1, false, false), // keep 2.0
                (0x3ff0000000000000, 0x3ff0000000000000, 1, false, false), // demanded 1.0
                (0x3fefffffe0000000, 0x3ff0000010000000, 0, false, false),
                (0x4008000020000000, 0x3ff0000040000000, 2, false, false),
            ],
        );
        expect_states(
            &doms[1],
            &[
                (0x0, 0x0, 0, false, false),
                (0x3ff0000000000000, 0x3ff0000000000000, 1, false, false),
                (0x3ff0000020000000, 0x3ff0000020000000, 2, false, false),
            ],
        );
    }

    /// A single-feature `Linear(op="==")` demands its own algebraic solution,
    /// which sits strictly inside the deliberately widened derived range — the
    /// nearest-point candidate lands on the widened edge instead.
    #[test]
    fn equality_linear_adds_its_algebraic_solution() {
        let ens = golden_ens();
        let mut cons = cons_base(2);
        cons.linears = vec![LinearC {
            indices: vec![0],
            coefs: vec![2.0],
            op: LIN_EQ,
            rhs: 5.0,
            policy: POLICY_SATISFIED,
        }];
        cons.ranges = vec![(0, 2.4999999995, 2.5000000005)]; // the derived range
        let doms = domains_of(&ens, &[2.0, 0.0], &cons, 0.0, &no_policies(2));
        expect_states(
            &doms[0],
            &[
                (0x4003ffffffeed1f4, 0x3fdfffffff768fa0, 1, false, false), // widened edge
                (0x4004000000000000, 0x3fe0000000000000, 1, false, false), // exact 5/2
            ],
        );
    }

    /// The factual's own value is exempt from snapping, so a value policy can
    /// never force a feature that did not need to move; every movement candidate
    /// is snapped or dropped.
    #[test]
    fn value_policy_snaps_movements_and_spares_the_keep_state() {
        let ens = golden_ens();
        let mut policies = no_policies(2);
        policies[0] = Some(ValuePolicy::Integer);
        let doms = domains_of(&ens, &[2.25, 0.0], &cons_base(2), 0.0, &policies);
        expect_states(
            &doms[0],
            &[
                (0x4002000000000000, 0x0, 1, false, false), // keep 2.25, unsnapped
                (0x4010000000000000, 0x3ffc000000000000, 2, false, true), // 4.0 (3 is outside)
                (0x0, 0x4002000000000000, 0, false, true),  // 0.0 (1 is outside)
            ],
        );
        let mut policies = no_policies(2);
        policies[0] = Some(ValuePolicy::Grid {
            step: 2.0,
            anchor: 0.0,
        });
        let doms = domains_of(&ens, &[2.25, 0.0], &cons_base(2), 0.5, &policies);
        expect_states(
            &doms[0],
            &[
                (0x4002000000000000, 0x0, 1, false, false),
                (0x4010000000000000, 0x4002000000000000, 2, false, true),
                (0x0, 0x4006000000000000, 0, false, true),
            ],
        );
    }

    #[test]
    fn feature_order_is_split_count_descending_then_index() {
        let ens = stumps(
            &[
                (1, 1.0, true, 0.0, 1.0),
                (1, 2.0, true, 0.0, 1.0),
                (2, 5.0, true, 0.0, 1.0),
            ],
            4,
        );
        let mut cons = cons_base(4);
        cons.freeze = vec![3]; // referenced but split-free: kept, and last
        let grids = constraint_cells(&cons, &[&ens]);
        assert_eq!(feature_order(&grids, &cons), vec![1, 2, 3]);
    }

    #[test]
    fn h_suffix_sums_the_cheapest_state_per_level() {
        let ens = golden_ens();
        let cons = cons_base(2);
        let grids = constraint_cells(&cons, &[&ens]);
        let domains = build_domains(
            &grids,
            &[2.0, 0.0],
            &cons,
            &[1.0, 1.0],
            &[1.0, 1.0],
            0.0,
            &no_policies(2),
        );
        let order = feature_order(&grids, &cons);
        assert_eq!(h_suffix(&order, &domains), vec![0.0, 0.0]);
    }

    #[test]
    fn validate_recognizes_canonical_order_pairs_and_rejects_the_rest() {
        let mut cons = cons_base(3);
        cons.linears = vec![LinearC {
            indices: vec![2, 0],
            coefs: vec![-1.0, 1.0],
            op: LIN_LE,
            rhs: 0.0,
            policy: POLICY_SATISFIED,
        }];
        assert_eq!(validate(&cons).unwrap(), vec![(0, 2)]);
        cons.linears[0].op = LIN_GE;
        assert!(validate(&cons).is_err());
    }

    #[test]
    fn g_floor_pairs_admit_only_plain_unentangled_pairs() {
        let pairs = vec![(0, 1), (2, 3), (3, 4), (5, 6), (7, 8)];
        let mut onehot = vec![false; 9];
        onehot[5] = true;
        let mut demanded: Vec<Vec<f64>> = vec![Vec::new(); 9];
        demanded[8] = vec![1.0];
        let policy = vec![false; 9];
        // (2,3) and (3,4) share feature 3, so both withdraw; (5,6) has a one-hot
        // member and (7,8) a demanded value
        assert_eq!(
            g_floor_pairs(&pairs, &onehot, &demanded, &policy),
            vec![(0, 1)]
        );
        // a value policy on either side disqualifies the pair too
        let mut policy = vec![false; 9];
        policy[1] = true;
        assert!(g_floor_pairs(&pairs, &onehot, &demanded, &policy).is_empty());
    }

    // ------------------------------------------------------- propagation ---

    fn binary_domains(p: usize) -> Vec<Vec<State>> {
        vec![
            vec![
                State::new(0.0, 0.0, 0, false),
                State::new(1.0, 1.0, 0, false),
            ];
            p
        ]
    }

    #[test]
    fn onehot_counters_force_the_last_member_and_reject_a_second_one() {
        let mut cons = cons_base(3);
        cons.onehot = vec![vec![0, 1, 2]];
        let domains = binary_domains(3);
        let mut prop = Propagation::new(&cons, &domains);
        let mut assigned = vec![false; 3];
        let values = vec![0.0; 3];

        let (frame0, conflict) = prop.apply(0, 0.0, &assigned, &values);
        assert!(!conflict);
        assigned[0] = true;
        // the second zero leaves one free member, which is forced to 1.0
        let (frame1, conflict) = prop.apply(1, 0.0, &assigned, &values);
        assert!(!conflict);
        assert_eq!(prop.forced_value[2], Some(1.0));
        assert_eq!(frame1.settled, vec![(2, None)]);
        // a state contradicting that settlement is cut
        let (frame2, conflict) = prop.apply(2, 0.0, &assigned, &values);
        assert!(conflict);
        prop.restore(&frame2);
        prop.restore(&frame1);
        assert_eq!(prop.forced_value[2], None);
        assert_eq!((prop.ones[0], prop.zeros[0]), (0, 1));
        prop.restore(&frame0);
        assert_eq!((prop.ones[0], prop.zeros[0]), (0, 0));

        // two ones in one group: cut on the spot
        let (_, conflict) = prop.apply(0, 1.0, &assigned, &values);
        assert!(!conflict);
        assigned[0] = true;
        let mut values = values.clone();
        values[0] = 1.0;
        let (_, conflict) = prop.apply(1, 1.0, &assigned, &values);
        assert!(conflict);
    }

    /// A group whose members can hold something other than 0/1 is left to the
    /// arbiter: no counters, no forcing.
    #[test]
    fn non_binary_group_is_not_counted() {
        let mut cons = cons_base(2);
        cons.onehot = vec![vec![0, 1]];
        let mut domains = binary_domains(2);
        domains[1].push(State::new(0.5, 1.0, 0, false));
        let prop = Propagation::new(&cons, &domains);
        assert_eq!(prop.group_of, vec![None, None]);
    }

    #[test]
    fn implication_settles_its_consequence_and_conflicts_with_a_different_value() {
        let mut cons = cons_base(2);
        cons.implications = vec![(0, 1.0, 1, 1.0)];
        let domains = binary_domains(2);
        let mut prop = Propagation::new(&cons, &domains);
        let mut assigned = vec![false; 2];
        let mut values = vec![0.0; 2];

        let (frame, conflict) = prop.apply(0, 0.0, &assigned, &values); // silent
        assert!(!conflict);
        assert_eq!(prop.forced_value[1], None);
        prop.restore(&frame);

        let (frame, conflict) = prop.apply(0, 1.0, &assigned, &values);
        assert!(!conflict);
        assert_eq!(prop.forced_value[1], Some(1.0));
        assigned[0] = true;
        values[0] = 1.0;
        let (deeper, conflict) = prop.apply(1, 0.0, &assigned, &values);
        assert!(conflict);
        prop.restore(&deeper);
        prop.restore(&frame);
        assert_eq!(prop.forced_value[1], None);

        // an already-assigned consequence is checked against, not re-settled
        let mut assigned = vec![false; 2];
        let mut values = vec![0.0; 2];
        assigned[1] = true;
        values[1] = 0.0;
        let (_, conflict) = prop.apply(0, 1.0, &assigned, &values);
        assert!(conflict);
        values[1] = 1.0;
        let (_, conflict) = prop.apply(0, 1.0, &assigned, &values);
        assert!(!conflict);
    }

    // ---------------------------------------------------- ensemble bounds ---

    #[test]
    fn bounds_bracket_narrows_as_features_are_assigned() {
        let ens = stumps(&[(0, 1.0, true, -1.0, 1.0), (1, 2.0, false, 0.0, 0.5)], 2);
        let mut assigned = vec![false; 2];
        let mut values = vec![0.0; 2];
        let mut mask = BitSet::new(2);
        let mut bounds = EnsembleBounds::new(&ens, &assigned, &values);
        assert_eq!((bounds.score_min, bounds.score_max), (-1.0, 1.5));

        assigned[0] = true;
        values[0] = 5.0; // routes right: +1.0
        mask.set(0);
        let frame = bounds.apply(0, &mask, &assigned, &values);
        assert_eq!((bounds.score_min, bounds.score_max), (1.0, 1.5));

        bounds.restore(&frame);
        assert_eq!((bounds.score_min, bounds.score_max), (-1.0, 1.5));
    }

    #[test]
    fn assigned_nan_routes_by_missing_left() {
        for (missing_left, want) in [(true, -1.0), (false, 1.0)] {
            let ens = stumps_missing(&[(0, 1.0, true, -1.0, 1.0)], 2, missing_left);
            let mut assigned = vec![false; 2];
            let mut values = vec![0.0; 2];
            let mut mask = BitSet::new(2);
            let mut bounds = EnsembleBounds::new(&ens, &assigned, &values);
            assigned[0] = true;
            values[0] = f64::NAN;
            mask.set(0);
            bounds.apply(0, &mask, &assigned, &values);
            assert_eq!((bounds.score_min, bounds.score_max), (want, want));
        }
    }

    /// The node masks are word bitsets, so a feature past the first 64 still
    /// invalidates the trees that split on it.
    #[test]
    fn bitset_tracks_features_past_the_first_word() {
        let mut bits = BitSet::new(70);
        assert_eq!(bits.words.len(), 2);
        bits.set(69);
        assert!(bits.get(69) && !bits.get(5));
        bits.set(5);
        bits.clear(69);
        assert!(!bits.get(69) && bits.get(5));

        let ens = stumps(&[(3, 1.0, true, 0.0, 1.0), (69, 2.0, true, 0.0, 0.5)], 70);
        let mut assigned = vec![false; 70];
        let mut values = vec![0.0; 70];
        let mut mask = BitSet::new(70);
        let mut bounds = EnsembleBounds::new(&ens, &assigned, &values);
        assert_eq!((bounds.score_min, bounds.score_max), (0.0, 1.5));
        assigned[69] = true;
        values[69] = 5.0; // routes right in the second tree: +0.5, fixed
        mask.set(69);
        bounds.apply(69, &mask, &assigned, &values);
        assert_eq!((bounds.score_min, bounds.score_max), (0.5, 1.5));
    }

    // ------------------------------------------------------ undo integrity ---

    /// A randomized descent/backtrack walk over a problem with one-hot, an
    /// implication and three trees: every frame must restore the *whole* search
    /// state — brackets, propagation counters, settlements, mask and cost.
    #[test]
    fn every_frame_restores_the_whole_search_state() {
        type Snapshot = (
            Vec<f64>,
            Vec<f64>,
            f64,
            f64,
            Vec<Option<f64>>,
            Vec<usize>,
            Vec<usize>,
            Vec<bool>,
            Vec<u64>,
            f64,
        );

        let ens = stumps(
            &[
                (0, 0.5, true, 0.0, 1.0),
                (1, 0.5, true, 0.0, 0.5),
                (2, 0.5, true, 0.0, 0.25),
            ],
            3,
        );
        let mut cons = cons_base(3);
        cons.onehot = vec![vec![0, 1]];
        cons.implications = vec![(0, 1.0, 2, 1.0)];
        let grids = constraint_cells(&cons, &[&ens]);
        let domains = build_domains(
            &grids,
            &[0.0, 1.0, 0.0],
            &cons,
            &[1.0; 3],
            &[1.0; 3],
            0.0,
            &no_policies(3),
        );
        let order = feature_order(&grids, &cons);
        assert_eq!(order.len(), 3);

        let mut assigned = vec![false; 3];
        let mut values = vec![0.0; 3];
        let mut mask = BitSet::new(3);
        let mut prop = Propagation::new(&cons, &domains);
        let mut bounds = EnsembleBounds::new(&ens, &assigned, &values);
        let mut g = 0.0;

        let snapshot = |prop: &Propagation,
                        bounds: &EnsembleBounds,
                        assigned: &[bool],
                        mask: &BitSet,
                        g: f64|
         -> Snapshot {
            (
                bounds.tree_min.clone(),
                bounds.tree_max.clone(),
                bounds.score_min,
                bounds.score_max,
                prop.forced_value.clone(),
                prop.ones.clone(),
                prop.zeros.clone(),
                assigned.to_vec(),
                mask.words.clone(),
                g,
            )
        };

        let mut rng = Lcg(0x5eed);
        let mut frames: Vec<Frame> = Vec::new();
        let mut snapshots: Vec<Snapshot> = vec![snapshot(&prop, &bounds, &assigned, &mask, g)];
        let mut steps = 0;
        while steps < 4000 {
            steps += 1;
            let depth = frames.len();
            let descend = depth < order.len() && rng.next_usize(4) > 0;
            if descend {
                let j = order[depth];
                let chosen = rng.next_usize(domains[j].len());
                let state = domains[j][chosen];
                let (prop_frame, _conflict) = prop.apply(j, state.value, &assigned, &values);
                assigned[j] = true;
                values[j] = state.value;
                mask.set(j);
                let frame = Frame {
                    j,
                    model: bounds.apply(j, &mask, &assigned, &values),
                    plausibility: Vec::new(),
                    g_before: g,
                    prop: prop_frame,
                };
                g += state.cost;
                frames.push(frame);
                snapshots.push(snapshot(&prop, &bounds, &assigned, &mask, g));
            } else if depth > 0 {
                snapshots.pop();
                let frame = frames.pop().unwrap();
                let mut if_bounds: Option<EnsembleBounds> = None;
                undo(
                    &frame,
                    &mut prop,
                    &mut bounds,
                    &mut if_bounds,
                    &mut assigned,
                    &mut mask,
                    &mut g,
                );
                assert_eq!(
                    snapshot(&prop, &bounds, &assigned, &mask, g),
                    snapshots[snapshots.len() - 1],
                    "frame at depth {depth} did not restore the search state"
                );
            }
        }
    }

    // ------------------------------------------------------ search results ---

    #[allow(clippy::too_many_arguments)]
    fn solve(
        ens: &Ensemble,
        x: &[f64],
        interval: (f64, f64),
        cons: &Constraints,
        lam: f64,
        policies: &[Option<ValuePolicy>],
        params: &ExactParams,
        incumbent: Option<(f64, &[f64])>,
    ) -> ExactResult {
        let p = ens.n_features;
        solve_exact(
            ens,
            x,
            interval,
            cons,
            &vec![1.0; p],
            &vec![1.0; p],
            lam,
            policies,
            None,
            params,
            incumbent,
        )
        .unwrap()
    }

    fn bits_of(row: &[f64]) -> Vec<u64> {
        row.iter().map(|v| v.to_bits()).collect()
    }

    /// PARITY CANARY. Two stumps, one Range, a sparsity penalty; every number
    /// below is what the Python backend answered for the same inputs:
    ///
    /// ```text
    /// uv run python -c '
    /// import struct, numpy as np
    /// from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree
    /// from treecf.constraints.compile import compile_constraints
    /// from treecf.constraints.objects import Range
    /// from treecf.backends.exact import solve_exact
    /// leaf = lambda i, v: Node(i, None, None, None, None, None, None, v)
    /// t0 = Tree((Node(0, 0, 1.0, SplitOp.LT, False, 1, 2, None), leaf(1, 0.0), leaf(2, 1.0)))
    /// t1 = Tree((Node(0, 1, 2.0, SplitOp.LE, False, 1, 2, None), leaf(1, 0.0), leaf(2, 0.5)))
    /// ir = EnsembleIR((t0, t1), 0.0, Link.IDENTITY, 2, ("a", "b"), {})
    /// x = np.array([0.5, 3.0])
    /// compiled = compile_constraints([Range("a", 0.0, 5.0)], ir.feature_names)
    /// r = solve_exact(ir, x, (1.0, 4.0), compiled, np.ones(2), np.ones(2), 0.25)
    /// bits = lambda v: hex(struct.unpack("<Q", struct.pack("<d", v))[0])
    /// print([bits(v) for v in r.x_cf], bits(r.distance), r.proof, r.stats)'
    /// ```
    ///
    /// If this drifts, the full parity suite will too.
    #[test]
    fn parity_canary_matches_the_python_backend() {
        let ens = stumps(&[(0, 1.0, true, 0.0, 1.0), (1, 2.0, false, 0.0, 0.5)], 2);
        let mut cons = cons_base(2);
        cons.ranges = vec![(0, 0.0, 5.0)];
        let result = solve(
            &ens,
            &[0.5, 3.0],
            (1.0, 4.0),
            &cons,
            0.25,
            &no_policies(2),
            &ExactParams::default(),
            None,
        );
        assert_eq!(
            bits_of(result.x_cf.as_ref().unwrap()),
            vec![0x3ff0000000000000, 0x4008000000000000]
        );
        assert_eq!(result.distance.unwrap().to_bits(), 0x3fe8000000000000);
        assert_eq!(result.proof, "optimal");
        assert_eq!(
            result.stats,
            ExactStats {
                nodes_expanded: 4,
                nodes_pruned_score: 1,
                nodes_pruned_cost: 1,
                lower_bound: 0.75,
                gap: 0.0,
                completed: true,
                warm_start_used: false,
            }
        );
        assert!(result.snapped.is_empty());
    }

    /// Exhaustive enumeration over the same alphabet: pruning must not lose the
    /// optimum, and it must actually fire.
    #[test]
    fn pruning_keeps_the_optimum_of_an_exhaustive_enumeration() {
        let ens = stumps(
            &[
                (0, 1.0, true, 0.0, 1.0),
                (1, 2.0, false, 0.0, 0.5),
                (2, 0.5, true, 0.0, 0.25),
            ],
            3,
        );
        let mut cons = cons_base(3);
        cons.ranges = vec![(0, 0.0, 5.0)];
        let x = [0.5, 3.0, 0.0];
        let interval = (1.25, 2.0);
        let lam = 0.1;
        let result = solve(
            &ens,
            &x,
            interval,
            &cons,
            lam,
            &no_policies(3),
            &ExactParams::default(),
            None,
        );

        // the brute-force best over the cartesian product of the domains
        let grids = constraint_cells(&cons, &[&ens]);
        let domains = build_domains(
            &grids,
            &x,
            &cons,
            &[1.0; 3],
            &[1.0; 3],
            lam,
            &no_policies(3),
        );
        let order = feature_order(&grids, &cons);
        let deltas = allow_missing_deltas(&cons);
        let mut best: Option<f64> = None;
        let mut idx = vec![0usize; order.len()];
        loop {
            let mut row = x.to_vec();
            for (level, &f) in order.iter().enumerate() {
                row[f] = domains[f][idx[level]].value;
            }
            if accepts(&ens, None, 0.0, &cons, &x, interval.0, interval.1, &row) {
                let cost = cost_of_row(&x, &row, &[1.0; 3], &[1.0; 3], lam, &deltas);
                if best.is_none_or(|b| cost < b) {
                    best = Some(cost);
                }
            }
            let mut level = order.len();
            loop {
                if level == 0 {
                    assert_eq!(result.distance, best, "the search lost the optimum");
                    assert!(result.stats.completed);
                    assert!(
                        result.stats.nodes_pruned_score + result.stats.nodes_pruned_cost > 0,
                        "the scenario never exercised a prune"
                    );
                    // and the Python backend's own answer for these inputs
                    assert_eq!(
                        bits_of(result.x_cf.as_ref().unwrap()),
                        vec![0x3ff0000000000000, 0x4008000000000000, 0x0]
                    );
                    assert_eq!(result.distance.unwrap().to_bits(), 0x3fe3333333333333);
                    assert_eq!(result.stats.nodes_expanded, 6);
                    assert_eq!(result.stats.nodes_pruned_score, 1);
                    assert_eq!(result.stats.nodes_pruned_cost, 2);
                    return;
                }
                level -= 1;
                idx[level] += 1;
                if idx[level] < domains[order[level]].len() {
                    break;
                }
                idx[level] = 0;
            }
        }
    }

    /// A pair `a <= b` whose candidates land the wrong way round is pulled onto
    /// a shared value inside both cells. Python's answer for these inputs:
    /// `x_cf = [1.0, 1.0]`, distance 1.5, 3 nodes, 1 score prune.
    #[test]
    fn order_pair_completion_is_repaired_onto_the_boundary() {
        let ens = stumps(&[(0, 1.0, true, 0.0, 1.0)], 2);
        let mut cons = cons_base(2);
        cons.linears = vec![LinearC {
            indices: vec![0, 1],
            coefs: vec![1.0, -1.0],
            op: LIN_LE,
            rhs: 0.0,
            policy: POLICY_SATISFIED,
        }];
        let result = solve(
            &ens,
            &[0.0, 0.5],
            (1.0, 2.0),
            &cons,
            0.0,
            &no_policies(2),
            &ExactParams::default(),
            None,
        );
        assert_eq!(
            bits_of(result.x_cf.as_ref().unwrap()),
            vec![0x3ff0000000000000, 0x3ff0000000000000]
        );
        assert_eq!(result.distance.unwrap().to_bits(), 0x3ff8000000000000);
        assert_eq!(result.proof, "optimal");
        assert!(result.stats.completed);
        assert_eq!(result.stats.nodes_expanded, 3);
        assert_eq!(result.stats.nodes_pruned_score, 1);
    }

    /// A value policy on either side makes the pair unrepairable, so the search
    /// withdraws its claim on the whole space (`completed` false) instead of
    /// reporting a certificate.
    #[test]
    fn policy_bound_pair_withdraws_the_completeness_claim() {
        let ens = stumps(&[(0, 1.0, true, 0.0, 1.0)], 2);
        let mut cons = cons_base(2);
        cons.linears = vec![LinearC {
            indices: vec![0, 1],
            coefs: vec![1.0, -1.0],
            op: LIN_LE,
            rhs: 0.0,
            policy: POLICY_SATISFIED,
        }];
        let mut policies = no_policies(2);
        policies[0] = Some(ValuePolicy::Integer);
        let result = solve(
            &ens,
            &[0.0, 0.5],
            (1.0, 2.0),
            &cons,
            0.0,
            &policies,
            &ExactParams::default(),
            None,
        );
        assert!(result.x_cf.is_none());
        assert_eq!(result.proof, "heuristic");
        assert_eq!(
            result.stats,
            ExactStats {
                nodes_expanded: 3,
                nodes_pruned_score: 1,
                nodes_pruned_cost: 1,
                lower_bound: 0.0,
                gap: 0.0,
                completed: false,
                warm_start_used: false,
            }
        );
    }

    /// One-hot and implication together: the group settles its last member and
    /// the implication settles its consequence, both cut before being explored.
    #[test]
    fn onehot_and_implication_search_matches_python() {
        let ens = stumps(&[(0, 0.5, true, 0.0, 1.0), (2, 0.5, true, 0.0, 0.5)], 3);
        let mut cons = cons_base(3);
        cons.onehot = vec![vec![0, 1]];
        cons.implications = vec![(0, 1.0, 2, 1.0)];
        let result = solve(
            &ens,
            &[0.0, 1.0, 0.0],
            (1.0, 2.0),
            &cons,
            0.0,
            &no_policies(3),
            &ExactParams::default(),
            None,
        );
        assert_eq!(
            bits_of(result.x_cf.as_ref().unwrap()),
            vec![0x3ff0000000000000, 0x0, 0x3ff0000000000000]
        );
        assert_eq!(result.distance.unwrap().to_bits(), 0x4008000000000000);
        assert_eq!(
            result.stats,
            ExactStats {
                nodes_expanded: 7,
                nodes_pruned_score: 1,
                nodes_pruned_cost: 3,
                lower_bound: 3.0,
                gap: 0.0,
                completed: true,
                warm_start_used: false,
            }
        );
    }

    /// A NaN factual that AllowMissing lets become a value; the second feature
    /// stays missing-free. Python: `x_cf = [1.0, 0.0]`, distance 0.5.
    #[test]
    fn nan_factual_search_matches_python() {
        let ens = stumps_missing(&[(0, 1.0, true, 0.0, 1.0)], 2, true);
        let mut cons = cons_base(2);
        cons.allow_missing = vec![(0, 0.5, 0.5)];
        let result = solve(
            &ens,
            &[f64::NAN, 0.0],
            (1.0, 2.0),
            &cons,
            0.0,
            &no_policies(2),
            &ExactParams::default(),
            None,
        );
        assert_eq!(
            bits_of(result.x_cf.as_ref().unwrap()),
            vec![0x3ff0000000000000, 0x0]
        );
        assert_eq!(result.distance.unwrap().to_bits(), 0x3fe0000000000000);
        assert_eq!(result.stats.nodes_expanded, 3);
        assert_eq!(result.stats.nodes_pruned_score, 2);
        assert!(result.stats.completed);
    }

    /// A warm start prunes against its cost and is reported through
    /// `warm_start_used`; here the search still beats it.
    #[test]
    fn warm_start_is_reported_and_pruned_against() {
        let ens = stumps(
            &[
                (0, 1.0, true, 0.0, 1.0),
                (1, 2.0, false, 0.0, 0.5),
                (2, 0.5, true, 0.0, 0.25),
            ],
            3,
        );
        let mut cons = cons_base(3);
        cons.ranges = vec![(0, 0.0, 5.0)];
        let warm = [1.0, 3.0, 0.6];
        let result = solve(
            &ens,
            &[0.5, 3.0, 0.0],
            (1.25, 2.0),
            &cons,
            0.1,
            &no_policies(3),
            &ExactParams::default(),
            Some((0.7, &warm)),
        );
        assert!(result.stats.warm_start_used);
        assert_eq!(result.distance.unwrap().to_bits(), 0x3fe3333333333333);
        assert_eq!(result.stats.nodes_expanded, 6);
        assert_eq!(result.stats.nodes_pruned_cost, 2);
    }

    /// A gap only widens the pruning threshold; the proof stays "optimal" until
    /// a branch is actually cut by the widened threshold alone.
    #[test]
    fn gap_widens_the_threshold_and_lowers_the_bound() {
        let ens = stumps(
            &[
                (0, 1.0, true, 0.0, 1.0),
                (1, 2.0, false, 0.0, 0.5),
                (2, 0.5, true, 0.0, 0.25),
            ],
            3,
        );
        let mut cons = cons_base(3);
        cons.ranges = vec![(0, 0.0, 5.0)];
        let params = ExactParams {
            gap: 0.5,
            ..ExactParams::default()
        };
        let result = solve(
            &ens,
            &[0.5, 3.0, 0.0],
            (1.25, 2.0),
            &cons,
            0.1,
            &no_policies(3),
            &params,
            None,
        );
        assert_eq!(result.proof, "optimal");
        assert_eq!(result.distance.unwrap().to_bits(), 0x3fe3333333333333);
        assert_eq!(result.stats.lower_bound, 0.6 / 1.5);
        assert_eq!(result.stats.gap, 0.5);
        assert!(result.stats.completed);
    }

    /// Contradictory constraints leave a feature no legal value: `x_cf` is None
    /// with `completed` true — a certificate, not a failure.
    #[test]
    fn certified_infeasible_reports_an_empty_domain_immediately() {
        let ens = stumps(&[(0, 1.0, true, 0.0, 1.0)], 2);
        let mut cons = cons_base(2);
        cons.linears = vec![LinearC {
            indices: vec![0],
            coefs: vec![1.0],
            op: LIN_LE,
            rhs: 5.0,
            policy: 1,
        }];
        cons.ranges = vec![(0, 2.0, 2.0), (0, f64::NEG_INFINITY, 5.000000001)];
        let result = solve(
            &ens,
            &[f64::NAN, 0.0],
            (1.0, 2.0),
            &cons,
            0.0,
            &no_policies(2),
            &ExactParams::default(),
            None,
        );
        assert!(result.x_cf.is_none());
        assert_eq!(result.proof, "optimal");
        assert!(result.stats.completed);
        assert_eq!(result.stats.lower_bound, f64::INFINITY);
        assert_eq!(result.stats.nodes_expanded, 0);
    }

    /// End to end past 64 features: only the two split features are searched,
    /// and the one in the second mask word is decided correctly.
    #[test]
    fn search_handles_seventy_features() {
        let ens = stumps(&[(3, 1.0, true, 0.0, 1.0), (69, 2.0, true, 0.0, 0.5)], 70);
        let cons = cons_base(70);
        let result = solve(
            &ens,
            &vec![0.0; 70],
            (1.4, 2.0),
            &cons,
            0.0,
            &no_policies(70),
            &ExactParams::default(),
            None,
        );
        let row = result.x_cf.unwrap();
        assert_eq!(row[3].to_bits(), 0x3ff0000000000000);
        assert_eq!(row[69].to_bits(), 0x4000000000000000);
        assert!(row
            .iter()
            .enumerate()
            .all(|(j, &v)| j == 3 || j == 69 || v == 0.0));
        assert_eq!(result.distance.unwrap().to_bits(), 0x4008000000000000);
        assert_eq!(
            result.stats,
            ExactStats {
                nodes_expanded: 4,
                nodes_pruned_score: 2,
                nodes_pruned_cost: 0,
                lower_bound: 3.0,
                gap: 0.0,
                completed: true,
                warm_start_used: false,
            }
        );
    }

    /// The factual itself is accepted before anything is built.
    #[test]
    fn feasible_factual_short_circuits() {
        let ens = stumps(&[(0, 1.0, true, 0.0, 1.0)], 2);
        let result = solve(
            &ens,
            &[5.0, 0.0],
            (0.5, 2.0),
            &cons_base(2),
            0.0,
            &no_policies(2),
            &ExactParams::default(),
            None,
        );
        assert_eq!(result.x_cf.unwrap(), vec![5.0, 0.0]);
        assert_eq!(result.distance, Some(0.0));
        assert_eq!(result.stats.nodes_expanded, 0);
        assert!(result.stats.completed);
    }

    /// The plausibility forest widens the candidate grid and prunes on its own
    /// bracket. Python: `x_cf = [1.0, 0.5]`, distance 1.5, 4 nodes, 2 score prunes.
    #[test]
    fn plausibility_bound_prunes_and_widens_the_grid() {
        let ens = stumps(&[(0, 1.0, true, 0.0, 1.0)], 2);
        let if_ens = stumps(&[(1, 0.5, true, 2.0, 5.0)], 2);
        let cons = cons_base(2);
        let result = solve_exact(
            &ens,
            &[0.0, 0.0],
            (1.0, 2.0),
            &cons,
            &[1.0, 1.0],
            &[1.0, 1.0],
            0.0,
            &no_policies(2),
            Some((&if_ens, 4.0)),
            &ExactParams::default(),
            None,
        )
        .unwrap();
        assert_eq!(
            bits_of(result.x_cf.as_ref().unwrap()),
            vec![0x3ff0000000000000, 0x3fe0000000000000]
        );
        assert_eq!(result.distance.unwrap().to_bits(), 0x3ff8000000000000);
        assert_eq!(
            result.stats,
            ExactStats {
                nodes_expanded: 4,
                nodes_pruned_score: 2,
                nodes_pruned_cost: 0,
                lower_bound: 1.5,
                gap: 0.0,
                completed: true,
                warm_start_used: false,
            }
        );
    }

    /// Two pairs sharing a feature: neither qualifies for the committed-cost
    /// floor, so a repair that comes to nothing withdraws outright.
    #[test]
    fn entangled_pairs_withdraw_when_a_repair_comes_to_nothing() {
        let ens = stumps(&[(0, 1.0, true, 0.0, 1.0)], 3);
        let mut cons = cons_base(3);
        cons.linears = vec![
            LinearC {
                indices: vec![0, 1],
                coefs: vec![1.0, -1.0],
                op: LIN_LE,
                rhs: 0.0,
                policy: POLICY_SATISFIED,
            },
            LinearC {
                indices: vec![1, 2],
                coefs: vec![1.0, -1.0],
                op: LIN_LE,
                rhs: 0.0,
                policy: POLICY_SATISFIED,
            },
        ];
        let result = solve(
            &ens,
            &[0.0, 0.5, 0.75],
            (1.0, 2.0),
            &cons,
            0.0,
            &no_policies(3),
            &ExactParams::default(),
            None,
        );
        assert!(result.x_cf.is_none());
        assert_eq!(result.proof, "heuristic");
        assert_eq!(
            result.stats,
            ExactStats {
                nodes_expanded: 4,
                nodes_pruned_score: 1,
                nodes_pruned_cost: 0,
                lower_bound: 0.0,
                gap: 0.0,
                completed: false,
                warm_start_used: false,
            }
        );
    }

    /// A winning candidate the policy produced is reported as snapped.
    #[test]
    fn snapped_winner_is_reported() {
        let ens = stumps(&[(0, 1.0, true, 0.0, 1.0)], 2);
        let mut cons = cons_base(2);
        cons.ranges = vec![(0, 0.0, 9.0)];
        let mut policies = no_policies(2);
        policies[0] = Some(ValuePolicy::Integer);
        let result = solve(
            &ens,
            &[0.25, 0.0],
            (1.0, 2.0),
            &cons,
            0.0,
            &policies,
            &ExactParams::default(),
            None,
        );
        assert_eq!(
            bits_of(result.x_cf.as_ref().unwrap()),
            vec![0x3ff0000000000000, 0x0]
        );
        assert_eq!(result.distance.unwrap().to_bits(), 0x3fe8000000000000);
        assert_eq!(result.snapped, vec![0]);
        assert_eq!(result.stats.nodes_expanded, 2);
        assert_eq!(result.stats.nodes_pruned_score, 1);
    }

    /// A node budget stops the walk and withdraws the completeness claim.
    #[test]
    fn node_budget_withdraws_the_completeness_claim() {
        let ens = stumps(
            &[
                (0, 1.0, true, 0.0, 1.0),
                (1, 2.0, false, 0.0, 0.5),
                (2, 0.5, true, 0.0, 0.25),
            ],
            3,
        );
        let mut cons = cons_base(3);
        cons.ranges = vec![(0, 0.0, 5.0)];
        let params = ExactParams {
            node_budget: 2,
            ..ExactParams::default()
        };
        let result = solve(
            &ens,
            &[0.5, 3.0, 0.0],
            (1.25, 2.0),
            &cons,
            0.1,
            &no_policies(3),
            &params,
            None,
        );
        assert_eq!(result.stats.nodes_expanded, 2);
        assert!(!result.stats.completed);
        assert_eq!(result.proof, "heuristic");
    }
}
