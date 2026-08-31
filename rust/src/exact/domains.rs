//! The branching alphabet of the exact backend, and what may enter it — port
//! of `treecf.backends._exact_domains` plus the `treecf.api._snap` it calls.
//! Parity rules in the module header of `super` govern this file too; the cost
//! arithmetic here mirrors `genetic.objective()` term for term.

use crate::cells::{cell_index, feature_cells_joint, Cell};
use crate::constraints::{py_max, py_min, Constraints, LIN_EQ, LIN_LE, POLICY_SATISFIED};
use crate::exact::orderpairs::{achievable_bounds, intersect_cell};
use crate::exact::ValuePolicy;
use crate::ir::Ensemble;

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

// ------------------------------------------------------------ cost terms ---

/// One feature's contribution to the objective — the per-feature term of
/// `genetic.objective()`, same four cases, same multiply-then-divide order.
/// A categorical change (`is_cat`) costs one flat unit in place of the
/// absolute code distance; NaN transitions keep their declared deltas.
#[allow(clippy::too_many_arguments)]
fn term_cost(
    x_j: f64,
    r: f64,
    weight_j: f64,
    sigma_j: f64,
    lam: f64,
    to_miss: f64,
    from_miss: f64,
    is_cat: bool,
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
    let delta = if is_cat { 1.0 } else { (r - x_j).abs() };
    lam + (weight_j * delta) / sigma_j
}

/// Full-row objective, accumulated in ascending feature index.
/// `cardinality[j] > 0` marks feature j categorical (flat change cost).
pub(crate) fn cost_of_row(
    x: &[f64],
    row: &[f64],
    sigma: &[f64],
    weights: &[f64],
    lam: f64,
    deltas: &[(f64, f64)],
    cardinality: &[u32],
) -> f64 {
    let mut total = 0.0;
    for j in 0..x.len() {
        let (to_miss, from_miss) = deltas[j];
        total += term_cost(
            x[j],
            row[j],
            weights[j],
            sigma[j],
            lam,
            to_miss,
            from_miss,
            cardinality[j] > 0,
        );
    }
    total
}

/// (delta_miss, delta_from_miss) per feature, `(0.0, 0.0)` where absent.
pub(crate) fn allow_missing_deltas(cons: &Constraints) -> Vec<(f64, f64)> {
    let mut out = vec![(0.0, 0.0); cons.n_features];
    for &(j, to, from) in &cons.allow_missing {
        out[j as usize] = (to, from);
    }
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
pub(crate) fn demanded_values(cons: &Constraints) -> Vec<Vec<f64>> {
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

/// `+0.0` for either zero. Python's `math.floor`/`math.ceil` and its one-argument
/// `round()` all return *ints*, so a zero they produce is always `+0.0` once it
/// meets a float; `f64::ceil` returns `-0.0` over `(-1.0, -0.0]` and
/// `round_ties_even` over `(-0.5, -0.0]`. Nothing compares or sums differently,
/// but the stored sign bit of a snapped candidate would.
#[inline]
fn py_zero(v: f64) -> f64 {
    if v == 0.0 {
        0.0
    } else {
        v
    }
}

/// Nearest policy-conforming value inside the cell and bounds — port of
/// `treecf.api._snap` (callable policies excluded, see the module header).
fn snap(value: f64, policy: &ValuePolicy, cell: &Cell, lo: f64, hi: f64) -> Option<f64> {
    let candidates = match *policy {
        ValuePolicy::Integer => {
            snap_candidates(&[py_zero(value.floor()), py_zero(value.ceil())], value)
        }
        ValuePolicy::Grid { step, anchor } => {
            // Python's round() is round-half-to-even and returns an int, so the
            // multiplier carries no sign of zero into `anchor + step * k`
            let k = py_zero(((value - anchor) / step).round_ties_even());
            let base = anchor + step * k;
            snap_candidates(
                &[py_zero(base), py_zero(base - step), py_zero(base + step)],
                value,
            )
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
pub(crate) struct State {
    pub(crate) value: f64,
    pub(crate) cost: f64,
    pub(crate) cell_idx: usize,
    pub(crate) is_nan: bool,
    pub(crate) snapped: bool,
}

impl State {
    pub(crate) fn new(value: f64, cost: f64, cell_idx: usize, is_nan: bool) -> Self {
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
pub(crate) fn build_domains(
    grids: &[Vec<Cell>],
    x: &[f64],
    cons: &Constraints,
    sigma: &[f64],
    weights: &[f64],
    lam: f64,
    policies: &[Option<ValuePolicy>],
    blocks: &[Vec<Vec<u32>>],
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

        if !blocks[j].is_empty() {
            let allowed = cons
                .allowed_categories
                .iter()
                .find(|(idx, _)| *idx as usize == j)
                .map(|(_, words)| words.as_slice());
            domains.push(categorical_states(
                x_j,
                &blocks[j],
                allowed,
                frozen[j],
                allow_j,
                suppress_nan[j],
                weight_j,
                sigma_j,
                lam,
                to_miss,
                from_miss,
            ));
            continue;
        }

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
                let cost = term_cost(x_j, v, weight_j, sigma_j, lam, to_miss, from_miss, false);
                let mut states = vec![State::new(v, cost, cell_index(cells, v), false)];
                if allow_j && !suppress_nan[j] {
                    let nan_cost = term_cost(
                        x_j,
                        f64::NAN,
                        weight_j,
                        sigma_j,
                        lam,
                        to_miss,
                        from_miss,
                        false,
                    );
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
                let cost = term_cost(x_j, v, weight_j, sigma_j, lam, to_miss, from_miss, false);
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
                    let cost =
                        term_cost(x_j, val, weight_j, sigma_j, lam, to_miss, from_miss, false);
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
                let cost = term_cost(x_j, val, weight_j, sigma_j, lam, to_miss, from_miss, false);
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
            let cost = term_cost(x_j, r, weight_j, sigma_j, lam, to_miss, from_miss, false);
            let mut state = State::new(r, cost, local_idx, false);
            state.snapped = snapped;
            states.push(state);
        }

        if allow_j && !suppress_nan[j] {
            let nan_cost = term_cost(
                x_j,
                f64::NAN,
                weight_j,
                sigma_j,
                lam,
                to_miss,
                from_miss,
                false,
            );
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
    for (j, _) in &cons.allowed_categories {
        refs[*j as usize] = true;
    }
    refs
}

/// Candidate states of a categorical feature — the mirror of
/// `_categorical_states`. A block's representative is its smallest member the
/// declared allowed set admits (cost is flat within a block, so the choice is
/// free; smallest is the determinism rule); a block with no admissible member
/// contributes no state, and an allowed set admitting nothing empties the
/// whole domain — the same certified-infeasible signal a contradictory
/// numeric pin produces. `cell_idx` carries the block index; the NaN state's
/// sentinel index is the block count.
#[allow(clippy::too_many_arguments)]
fn categorical_states(
    x_j: f64,
    blocks_j: &[Vec<u32>],
    allowed: Option<&[u64]>,
    frozen_j: bool,
    allow_j: bool,
    suppressed: bool,
    weight_j: f64,
    sigma_j: f64,
    lam: f64,
    to_miss: f64,
    from_miss: f64,
) -> Vec<State> {
    use crate::constraints::code_allowed;

    let x_nan = x_j.is_nan();
    let n_blocks = blocks_j.len();
    let block_of = |code: f64| -> usize {
        blocks_j
            .iter()
            .position(|block| block.contains(&(code as u32)))
            .expect("code not covered by blocks (should be impossible)")
    };

    if frozen_j {
        let idx = if x_nan { n_blocks } else { block_of(x_j) };
        return vec![State::new(x_j, 0.0, idx, x_nan)];
    }

    if x_nan && !allow_j {
        if suppressed {
            return Vec::new();
        }
        return vec![State::new(x_j, 0.0, n_blocks, true)];
    }

    let is_allowed = |code: f64| -> bool {
        match allowed {
            None => true,
            Some(words) => code_allowed(words, code),
        }
    };

    let mut states: Vec<State> = Vec::new();
    let mut keep_block = usize::MAX;
    if !x_nan && is_allowed(x_j) {
        keep_block = block_of(x_j);
        states.push(State::new(x_j, 0.0, keep_block, false));
    }

    for (block_idx, block) in blocks_j.iter().enumerate() {
        if block_idx == keep_block {
            continue; // the unchanged factual already represents its block
        }
        let member = block.iter().copied().find(|&c| is_allowed(c as f64));
        if let Some(code) = member {
            let rep = code as f64;
            let cost = term_cost(x_j, rep, weight_j, sigma_j, lam, to_miss, from_miss, true);
            states.push(State::new(rep, cost, block_idx, false));
        }
    }

    if allow_j && !suppressed {
        let nan_cost = term_cost(
            x_j,
            f64::NAN,
            weight_j,
            sigma_j,
            lam,
            to_miss,
            from_miss,
            true,
        );
        states.push(State::new(f64::NAN, nan_cost, n_blocks, true));
    }

    sort_states(&mut states);
    states
}

/// Search order: descending split count in the joint grid, ties ascending index.
/// A feature with no split anywhere and no referencing constraint is left out —
/// its domain is a single keep state, so it never needs to branch.
pub(crate) fn feature_order(
    grids: &[Vec<Cell>],
    cons: &Constraints,
    blocks: &[Vec<Vec<u32>>],
) -> Vec<usize> {
    let referenced = referenced_features(cons);
    let split_counts: Vec<usize> = grids
        .iter()
        .enumerate()
        .map(|(j, cells)| {
            if blocks[j].is_empty() {
                cells.len() - 1
            } else {
                blocks[j].len() - 1
            }
        })
        .collect();
    let mut included: Vec<usize> = (0..grids.len())
        .filter(|&j| split_counts[j] > 0 || referenced[j])
        .collect();
    included.sort_by_key(|&j| (std::cmp::Reverse(split_counts[j]), j));
    included
}

/// Suffix sums of each ordered feature's cheapest state cost: `h_suffix[k]` is
/// the minimum possible remaining cost once `order[k..]` are still undecided.
pub(crate) fn h_suffix(order: &[usize], domains: &[Vec<State>]) -> Vec<f64> {
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
pub(crate) fn validate(cons: &Constraints) -> Result<Vec<(usize, usize)>, String> {
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
pub(crate) fn g_floor_pairs(
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
pub(crate) fn domain_span(
    states: &[State],
    cells: &[Cell],
    lo_j: f64,
    hi_j: f64,
) -> Option<(f64, f64)> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{LinearC, LIN_GE};
    use crate::exact::test_support::*;

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
    /// A collision whose probe carries a nonzero perturb (`hash(40) >> 5 == 1`),
    /// so the set-order emulation is pinned past the trivial `perturb == 0` case.
    /// Golden: `uv run python -c 'print(list({36.0, 32.0, 40.0}))'` -> `[32.0, 40.0, 36.0]`
    #[test]
    fn set_order_survives_a_nonzero_perturb_collision() {
        let values = [36.0, 32.0, 40.0];
        let hashes: Vec<i64> = values.iter().map(|&v| py_hash_double(v)).collect();
        let ordered: Vec<f64> = py_set_order(&hashes).iter().map(|&k| values[k]).collect();
        assert_eq!(ordered, vec![32.0, 40.0, 36.0]);
    }

    /// Python's `math.floor`/`math.ceil`/`round()` return ints, so a zero
    /// candidate always reaches the comparison as `+0.0`; Rust's own rounding
    /// hands back `-0.0` just below zero. Goldens:
    /// `uv run python -c 'from treecf.api import _snap, Grid; print(_snap(-0.4, "integer", lambda c: True, -10, 10), _snap(-0.2, Grid(step=1.0, anchor=-0.0), lambda c: True, -10, 10), _snap(-0.2, Grid(step=1.0, anchor=0.0), lambda c: True, -10, 10))'`
    /// — all three are `0.0`, and Python's `0.0` has a clear sign bit.
    #[test]
    fn snap_normalizes_python_zero_signs() {
        let wide = cell(f64::NEG_INFINITY, f64::INFINITY, true, true);
        let got = snap(-0.4, &ValuePolicy::Integer, &wide, -10.0, 10.0).unwrap();
        assert_eq!(got.to_bits(), 0x0, "integer arm produced {got:?}");
        for anchor in [0.0, -0.0] {
            let grid = ValuePolicy::Grid { step: 1.0, anchor };
            let got = snap(-0.2, &grid, &wide, -10.0, 10.0).unwrap();
            assert_eq!(
                got.to_bits(),
                0x0,
                "grid arm (anchor {anchor:?}) produced {got:?}"
            );
        }
        // a genuine negative candidate keeps its sign
        assert_eq!(
            snap(-1.4, &ValuePolicy::Integer, &wide, -10.0, 10.0),
            Some(-1.0)
        );
    }

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
        let blocks = crate::cells::category_blocks_joint(&[&ens]);
        assert_eq!(feature_order(&grids, &cons, &blocks), vec![1, 2, 3]);
    }

    #[test]
    fn h_suffix_sums_the_cheapest_state_per_level() {
        let ens = golden_ens();
        let cons = cons_base(2);
        let grids = constraint_cells(&cons, &[&ens]);
        let blocks = crate::cells::category_blocks_joint(&[&ens]);
        let domains = build_domains(
            &grids,
            &[2.0, 0.0],
            &cons,
            &[1.0, 1.0],
            &[1.0, 1.0],
            0.0,
            &no_policies(2),
            &blocks,
        );
        let order = feature_order(&grids, &cons, &blocks);
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
}
