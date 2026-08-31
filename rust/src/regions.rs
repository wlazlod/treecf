//! Certified recourse regions — port of `treecf.regions`'s sound box oracle
//! and growth loop. Given an already-verified counterfactual `x_cf`, grows a
//! per-feature box outward one joint-grid cell at a time, keeping only
//! expansions the oracle proves sound: every tree's leaf value is bracketed
//! over the whole box (an interval-tree walk), every Linear constraint is
//! checked at its worst corner, and the isolation-forest bound (when
//! configured) is checked the same way. `treecf.regions._degenerate_features`
//! decides which features may grow at all; the caller passes that decision in
//! as `open_set` (Python already needs it to build `feature_intervals`), so
//! this module owns only the oracle and the growth loop — not degeneracy
//! classification or instance-bounds computation, both already done on the
//! Python side of the boundary either way.
//!
//! Two things this port carries over exactly, not just approximately:
//!
//! - **No monotonicity.** A strictly narrower target interval can still grow
//!   a strictly wider region on some feature — growth is greedy and
//!   order-dependent (see `treecf.regions.RecourseRegion`'s own docstring).
//!   No test here or in Python may assert one.
//! - **Unrouted missing splits reject the whole box.** A NaN-degenerate
//!   feature (fixed, never widened) routes a tree split by `missing_left`
//!   alone — unless the node defines no missing direction at all
//!   (`missing_left_defined[node] == false`, the exact case `RustEnsemble`'s
//!   own flat `missing_left: bool` encoding cannot represent, since it
//!   collapses Python's `None` and `False` to the same bit). Widening an
//!   ancestor feature can open a subtree the counterfactual's own verified
//!   path never visits; there is no per-point re-check after a region ships,
//!   so guessing a side would be unsound. `tree_interval_bracket` returns
//!   `None` instead, which `box_feasible` reads as a flat rejection of the
//!   whole box — see `treecf.regions._tree_interval_bracket`'s own doc
//!   comment for the full argument.

use crate::cells::{cell_index, Cell};
use crate::constraints::{py_max, py_min, Constraints, LinearC, LIN_GE, LIN_LE, POLICY_SATISFIED};
use crate::exact::orderpairs::{achievable_bounds, intersect_cell};
use crate::interrupt::{InterruptProbe, SearchOutcome};
use crate::ir::Ensemble;

const LINEAR_SLACK: f64 = 1e-9; // matches Explainer._verify / CompiledConstraints.check_matrix

/// How many growth attempts between two interrupt polls. One attempt is a
/// whole-box soundness check, so this is a far coarser unit of work than a
/// search node and the interval is correspondingly small.
const SIGNAL_CHECK_INTERVAL: u64 = 64;

/// `lo`/`hi` per feature (degenerate coordinates equal `x_cf` there, a single
/// point). `treecf.regions._recourse_region`'s Rust dispatch wraps this into
/// the `RecourseRegion` dataclass — `feature_intervals`/`certified` are
/// presentation, not search state, so they stay on the Python side.
pub struct RegionBox {
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
    /// Certified category sets, aligned with the caller's open categorical
    /// features (ascending member codes); empty when none were requested.
    pub cat_sets: Vec<Vec<u32>>,
}

// --------------------------------------------------------------- oracle ---

/// `[min, max]` leaf value reachable from tree node `node` over the box, or
/// `None` if the box cannot be soundly bracketed at all (see the module doc).
fn tree_interval_bracket(
    ens: &Ensemble,
    missing_defined: &[bool],
    node: u32,
    lo: &[f64],
    hi: &[f64],
    is_nan: &[bool],
    cat_sets: &[Vec<u32>],
) -> Option<(f64, f64)> {
    let i = node as usize;
    if ens.feature[i] < 0 {
        let v = ens.value[i];
        return Some((v, v));
    }
    let f = ens.feature[i] as usize;
    if is_nan[f] {
        if !missing_defined[i] {
            return None;
        }
        let child = if ens.missing_left[i] {
            ens.left[i]
        } else {
            ens.right[i]
        };
        return tree_interval_bracket(ens, missing_defined, child, lo, hi, is_nan, cat_sets);
    }
    // a categorical coordinate holds a SET of codes, not an interval: route
    // each member and take a side only when every member agrees; a set split
    // on a coordinate no set tracks is pinned to one code by the box
    let pinned = [lo[f] as u32];
    let members: Option<&[u32]> = if !cat_sets[f].is_empty() {
        Some(&cat_sets[f])
    } else if ens.node_set[i] >= 0 {
        Some(&pinned)
    } else {
        None
    };
    let (all_left, all_right) = if let Some(members) = members {
        let mut left_any = false;
        let mut right_any = false;
        for &code in members {
            let goes_left = if ens.node_set[i] >= 0 {
                ens.set_contains(ens.node_set[i], code as f64)
            } else if ens.is_lt[i] {
                (code as f64) < ens.threshold[i]
            } else {
                (code as f64) <= ens.threshold[i]
            };
            if goes_left {
                left_any = true;
            } else {
                right_any = true;
            }
        }
        (!right_any, !left_any)
    } else {
        let threshold = ens.threshold[i];
        let (lo_f, hi_f) = (lo[f], hi[f]);
        if ens.is_lt[i] {
            (hi_f < threshold, lo_f >= threshold)
        } else {
            (hi_f <= threshold, lo_f > threshold)
        }
    };
    if all_left {
        return tree_interval_bracket(ens, missing_defined, ens.left[i], lo, hi, is_nan, cat_sets);
    }
    if all_right {
        return tree_interval_bracket(ens, missing_defined, ens.right[i], lo, hi, is_nan, cat_sets);
    }
    let (lmin, lmax) =
        tree_interval_bracket(ens, missing_defined, ens.left[i], lo, hi, is_nan, cat_sets)?;
    let (rmin, rmax) =
        tree_interval_bracket(ens, missing_defined, ens.right[i], lo, hi, is_nan, cat_sets)?;
    Some((py_min(lmin, rmin), py_max(lmax, rmax)))
}

/// `[min, max]` raw score the ensemble can reach anywhere in the box, or
/// `None` if any tree's bracket could not be soundly computed. Summed base +
/// ascending tree index, the same order `Ensemble::raw_score` adds in.
#[cfg(test)] // production growth reads the cached per-tree form instead
fn ensemble_bracket(
    ens: &Ensemble,
    missing_defined: &[bool],
    lo: &[f64],
    hi: &[f64],
    is_nan: &[bool],
    cat_sets: &[Vec<u32>],
) -> Option<(f64, f64)> {
    let mut total_min = ens.base_score;
    let mut total_max = ens.base_score;
    for &root in &ens.tree_roots {
        let (tmin, tmax) =
            tree_interval_bracket(ens, missing_defined, root, lo, hi, is_nan, cat_sets)?;
        total_min += tmin;
        total_max += tmax;
    }
    Some((total_min, total_max))
}

/// Worst-corner feasibility of one Linear constraint over the box. Every
/// degenerate feature has `lo[j] == hi[j]`, so its term reduces to the fixed
/// value `x_cf` already satisfies — one formula covers both a genuine range
/// and a pinned feature, with no special-casing between the two.
fn linear_holds(lin: &LinearC, x_cf: &[f64], lo: &[f64], hi: &[f64]) -> bool {
    if lin.indices.iter().any(|&j| x_cf[j as usize].is_nan()) {
        return lin.policy == POLICY_SATISFIED;
    }
    let mut lo_sum = 0.0;
    let mut hi_sum = 0.0;
    for (k, &j) in lin.indices.iter().enumerate() {
        let j = j as usize;
        let a = lin.coefs[k] * lo[j];
        let b = lin.coefs[k] * hi[j];
        lo_sum += py_min(a, b);
        hi_sum += py_max(a, b);
    }
    match lin.op {
        LIN_LE => hi_sum <= lin.rhs + LINEAR_SLACK,
        LIN_GE => lo_sum >= lin.rhs - LINEAR_SLACK,
        _ => lo_sum >= lin.rhs - LINEAR_SLACK && hi_sum <= lin.rhs + LINEAR_SLACK,
    }
}

/// The soundness oracle: `true` only if EVERY point of the box is provably
/// still in-target, plausible, and Linear-feasible — rejects on any doubt,
/// including an ensemble bracket that could not be soundly computed at all.
#[cfg(test)] // the one-shot oracle the cached growth loop is checked against
#[allow(clippy::too_many_arguments)]
fn box_feasible(
    ens: &Ensemble,
    missing_defined: &[bool],
    if_pair: Option<(&Ensemble, &[bool])>,
    min_total_path: f64,
    interval: (f64, f64),
    linears: &[LinearC],
    x_cf: &[f64],
    lo: &[f64],
    hi: &[f64],
    is_nan: &[bool],
) -> bool {
    let empty: Vec<Vec<u32>> = vec![Vec::new(); ens.n_features];
    let Some((score_min, score_max)) =
        ensemble_bracket(ens, missing_defined, lo, hi, is_nan, &empty)
    else {
        return false;
    };
    if score_min < interval.0 || score_max > interval.1 {
        return false;
    }
    if let Some((if_ens, if_missing_defined)) = if_pair {
        let if_empty: Vec<Vec<u32>> = vec![Vec::new(); if_ens.n_features];
        let Some((if_min, _if_max)) =
            ensemble_bracket(if_ens, if_missing_defined, lo, hi, is_nan, &if_empty)
        else {
            return false;
        };
        if if_min < min_total_path {
            return false;
        }
    }
    linears.iter().all(|lin| linear_holds(lin, x_cf, lo, hi))
}

/// Per-tree brackets cached between growth attempts: widening one feature can
/// only change the brackets of trees that split on it, so only those are
/// re-walked. The ensemble total is still re-summed in full over every tree
/// in ascending index — the same additions `ensemble_bracket` performs — so
/// the accept/reject decisions are identical, just cheaper.
struct BracketCache {
    brackets: Vec<Option<(f64, f64)>>,
    on_feature: Vec<Vec<usize>>,
}

impl BracketCache {
    fn new(
        ens: &Ensemble,
        missing_defined: &[bool],
        lo: &[f64],
        hi: &[f64],
        is_nan: &[bool],
        cat_sets: &[Vec<u32>],
    ) -> Self {
        let brackets = ens
            .tree_roots
            .iter()
            .map(|&root| {
                tree_interval_bracket(ens, missing_defined, root, lo, hi, is_nan, cat_sets)
            })
            .collect();
        let n_trees = ens.tree_roots.len();
        let mut on_feature: Vec<Vec<usize>> = vec![Vec::new(); ens.n_features];
        for t in 0..n_trees {
            let start = ens.tree_roots[t] as usize;
            let end = if t + 1 < n_trees {
                ens.tree_roots[t + 1] as usize
            } else {
                ens.feature.len()
            };
            let mut seen: Vec<usize> = ens.feature[start..end]
                .iter()
                .filter(|&&f| f >= 0)
                .map(|&f| f as usize)
                .collect();
            seen.sort_unstable();
            seen.dedup();
            for f in seen {
                on_feature[f].push(t);
            }
        }
        Self {
            brackets,
            on_feature,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update(
        &mut self,
        ens: &Ensemble,
        missing_defined: &[bool],
        j: usize,
        lo: &[f64],
        hi: &[f64],
        is_nan: &[bool],
        cat_sets: &[Vec<u32>],
    ) -> Vec<(usize, Option<(f64, f64)>)> {
        let mut saved = Vec::with_capacity(self.on_feature[j].len());
        for k in 0..self.on_feature[j].len() {
            let t = self.on_feature[j][k];
            saved.push((t, self.brackets[t]));
            self.brackets[t] = tree_interval_bracket(
                ens,
                missing_defined,
                ens.tree_roots[t],
                lo,
                hi,
                is_nan,
                cat_sets,
            );
        }
        saved
    }

    fn restore(&mut self, saved: &[(usize, Option<(f64, f64)>)]) {
        for &(t, bracket) in saved {
            self.brackets[t] = bracket;
        }
    }

    fn total(&self, base: f64) -> Option<(f64, f64)> {
        let mut total_min = base;
        let mut total_max = base;
        for bracket in &self.brackets {
            let (tmin, tmax) = (*bracket)?;
            total_min += tmin;
            total_max += tmax;
        }
        Some((total_min, total_max))
    }
}

// ------------------------------------------------------------- growth ---

/// The achievable far edge one joint-grid cell beyond `value`, or `None`.
/// Finishes the cell `value` is already inside first, if it has not yet
/// reached that cell's own achievable edge; once it has, claims the whole
/// next cell. Clamped to the instance bounds throughout.
fn next_edge(cells: &[Cell], value: f64, lo_b: f64, hi_b: f64, upper: bool) -> Option<f64> {
    if value.is_infinite() {
        return None;
    }
    let idx = cell_index(cells, value);
    let iv = intersect_cell(&cells[idx], lo_b, hi_b)
        .expect("value is within the instance bounds by construction");
    let (cur_lo, cur_hi) = achievable_bounds(&iv);
    if upper {
        if cur_hi > value {
            return Some(cur_hi);
        }
        if idx + 1 >= cells.len() {
            return None;
        }
        return intersect_cell(&cells[idx + 1], lo_b, hi_b).map(|c| achievable_bounds(&c).1);
    }
    if cur_lo < value {
        return Some(cur_lo);
    }
    if idx == 0 {
        return None;
    }
    intersect_cell(&cells[idx - 1], lo_b, hi_b).map(|c| achievable_bounds(&c).0)
}

/// Attempt one cell of growth on feature `j` in one direction. Mutates
/// `box_lo`/`box_hi` in place; keeps the extension iff the oracle accepts the
/// whole box with it, retracting otherwise.
#[allow(clippy::too_many_arguments)]
fn try_grow(
    j: usize,
    upper: bool,
    box_lo: &mut [f64],
    box_hi: &mut [f64],
    cells: &[Cell],
    lo_b: f64,
    hi_b: f64,
    ens: &Ensemble,
    missing_defined: &[bool],
    if_pair: Option<(&Ensemble, &[bool])>,
    min_total_path: f64,
    interval: (f64, f64),
    linears: &[LinearC],
    x_cf: &[f64],
    is_nan: &[bool],
    cat_sets: &[Vec<u32>],
    model_cache: &mut BracketCache,
    if_cache: &mut Option<BracketCache>,
) -> bool {
    let current = if upper { box_hi[j] } else { box_lo[j] };
    let Some(candidate) = next_edge(cells, current, lo_b, hi_b, upper) else {
        return false;
    };
    if upper {
        box_hi[j] = candidate;
    } else {
        box_lo[j] = candidate;
    }
    let saved = model_cache.update(ens, missing_defined, j, box_lo, box_hi, is_nan, cat_sets);
    let mut saved_if: Vec<(usize, Option<(f64, f64)>)> = Vec::new();
    if let (Some((if_ens, if_missing_defined)), Some(cache)) = (if_pair, if_cache.as_mut()) {
        saved_if = cache.update(
            if_ens,
            if_missing_defined,
            j,
            box_lo,
            box_hi,
            is_nan,
            cat_sets,
        );
    }
    let ok = 'feasible: {
        let Some((score_min, score_max)) = model_cache.total(ens.base_score) else {
            break 'feasible false;
        };
        if score_min < interval.0 || score_max > interval.1 {
            break 'feasible false;
        }
        if let (Some((if_ens, _)), Some(cache)) = (if_pair, if_cache.as_ref()) {
            let Some((if_min, _if_max)) = cache.total(if_ens.base_score) else {
                break 'feasible false;
            };
            if if_min < min_total_path {
                break 'feasible false;
            }
        }
        linears
            .iter()
            .all(|lin| linear_holds(lin, x_cf, box_lo, box_hi))
    };
    if ok {
        return true;
    }
    model_cache.restore(&saved);
    if let Some(cache) = if_cache.as_mut() {
        cache.restore(&saved_if);
    }
    if upper {
        box_hi[j] = current;
    } else {
        box_lo[j] = current;
    }
    false
}

/// Grow a certified box around the verified counterfactual `x_cf`. `open_set`
/// is the ascending, deduplicated list of feature indices `_degenerate_features`
/// (Python) did not exclude — the only features growth may touch; every other
/// coordinate stays pinned at `x_cf`. `lo_b`/`hi_b` are the instance bounds
/// (Freeze/Monotone included, NaN already normalized to `+-inf` by the
/// caller). Growth proceeds one joint-grid cell at a time, per feature in
/// `open_set`'s order, upper endpoint before lower, each accepted only if the
/// box that results still passes the soundness oracle; a feature closes once
/// both directions fail in the same round, and the whole loop stops once a
/// full round accepts nothing. No step reads from hash-map iteration order,
/// so the result is bit-deterministic.
///
/// `probe` is asked every `SIGNAL_CHECK_INTERVAL` growth attempts whether to
/// stop. Answering yes drops the partly grown box and returns
/// `SearchOutcome::Interrupted` — a box is only ever handed back whole.
#[allow(clippy::too_many_arguments)]
pub fn recourse_region(
    ens: &Ensemble,
    missing_defined: &[bool],
    cons: &Constraints,
    x_cf: &[f64],
    interval: (f64, f64),
    lo_b: &[f64],
    hi_b: &[f64],
    open_set: &[usize],
    if_pair: Option<(&Ensemble, &[bool])>,
    min_total_path: f64,
    cat_open: &[usize],
    cat_blocks: &[Vec<Vec<u32>>],
    probe: InterruptProbe<'_>,
) -> SearchOutcome<RegionBox> {
    let ensembles: Vec<&Ensemble> = match if_pair {
        None => vec![ens],
        Some((if_ens, _)) => vec![ens, if_ens],
    };
    let grids = crate::exact::constraint_cells(cons, &ensembles);

    let mut box_lo = x_cf.to_vec();
    let mut box_hi = x_cf.to_vec();
    let is_nan_arr: Vec<bool> = x_cf.iter().map(|v| v.is_nan()).collect();
    // certified member sets, indexed by FEATURE (empty = untracked coordinate)
    let mut cat_sets: Vec<Vec<u32>> = vec![Vec::new(); ens.n_features];
    for &j in cat_open {
        cat_sets[j] = vec![x_cf[j] as u32];
    }
    let mut model_cache = BracketCache::new(
        ens,
        missing_defined,
        &box_lo,
        &box_hi,
        &is_nan_arr,
        &cat_sets,
    );
    let mut if_cache = if_pair.map(|(if_ens, if_missing_defined)| {
        BracketCache::new(
            if_ens,
            if_missing_defined,
            &box_lo,
            &box_hi,
            &is_nan_arr,
            &cat_sets,
        )
    });

    let mut open: Vec<usize> = open_set.to_vec();
    let mut open_cats: Vec<usize> = cat_open.to_vec();
    // Every feature costs exactly two attempts, so counting the pair at once
    // polls on the same attempt numbers as counting them one by one would.
    let mut attempts: u64 = 0;
    while !open.is_empty() || !open_cats.is_empty() {
        let mut still_open: Vec<usize> = Vec::new();
        for &j in &open {
            let cells = &grids[j];
            let grew_up = try_grow(
                j,
                true,
                &mut box_lo,
                &mut box_hi,
                cells,
                lo_b[j],
                hi_b[j],
                ens,
                missing_defined,
                if_pair,
                min_total_path,
                interval,
                &cons.linears,
                x_cf,
                &is_nan_arr,
                &cat_sets,
                &mut model_cache,
                &mut if_cache,
            );
            let grew_down = try_grow(
                j,
                false,
                &mut box_lo,
                &mut box_hi,
                cells,
                lo_b[j],
                hi_b[j],
                ens,
                missing_defined,
                if_pair,
                min_total_path,
                interval,
                &cons.linears,
                x_cf,
                &is_nan_arr,
                &cat_sets,
                &mut model_cache,
                &mut if_cache,
            );
            if grew_up || grew_down {
                still_open.push(j);
            }
            attempts += 2;
            if attempts % SIGNAL_CHECK_INTERVAL == 0 && probe() {
                return SearchOutcome::Interrupted;
            }
        }
        open = still_open;
        // after the numeric pass: one block at a time per categorical feature,
        // ascending feature index, ascending block order
        let mut still_open_cats: Vec<usize> = Vec::new();
        for (k, &j) in cat_open.iter().enumerate() {
            if !open_cats.contains(&j) {
                continue;
            }
            let mut grew_cat = false;
            for members in &cat_blocks[k] {
                let new_members: Vec<u32> = members
                    .iter()
                    .copied()
                    .filter(|c| !cat_sets[j].contains(c))
                    .collect();
                if new_members.is_empty() {
                    continue;
                }
                cat_sets[j].extend(new_members.iter().copied());
                let saved = model_cache.update(
                    ens,
                    missing_defined,
                    j,
                    &box_lo,
                    &box_hi,
                    &is_nan_arr,
                    &cat_sets,
                );
                let mut saved_if: Vec<(usize, Option<(f64, f64)>)> = Vec::new();
                if let (Some((if_ens, if_missing_defined)), Some(cache)) =
                    (if_pair, if_cache.as_mut())
                {
                    saved_if = cache.update(
                        if_ens,
                        if_missing_defined,
                        j,
                        &box_lo,
                        &box_hi,
                        &is_nan_arr,
                        &cat_sets,
                    );
                }
                let ok = 'feasible: {
                    let Some((score_min, score_max)) = model_cache.total(ens.base_score) else {
                        break 'feasible false;
                    };
                    if score_min < interval.0 || score_max > interval.1 {
                        break 'feasible false;
                    }
                    if let (Some((if_ens, _)), Some(cache)) = (if_pair, if_cache.as_ref()) {
                        let Some((if_min, _if_max)) = cache.total(if_ens.base_score) else {
                            break 'feasible false;
                        };
                        if if_min < min_total_path {
                            break 'feasible false;
                        }
                    }
                    cons.linears
                        .iter()
                        .all(|lin| linear_holds(lin, x_cf, &box_lo, &box_hi))
                };
                if ok {
                    grew_cat = true;
                } else {
                    model_cache.restore(&saved);
                    if let Some(cache) = if_cache.as_mut() {
                        cache.restore(&saved_if);
                    }
                    for c in &new_members {
                        cat_sets[j].retain(|m| m != c);
                    }
                }
                attempts += 1;
                if attempts % SIGNAL_CHECK_INTERVAL == 0 && probe() {
                    return SearchOutcome::Interrupted;
                }
            }
            if grew_cat {
                still_open_cats.push(j);
            }
        }
        open_cats = still_open_cats;
        if open.is_empty() && open_cats.is_empty() {
            break;
        }
    }

    let mut grown: Vec<Vec<u32>> = Vec::with_capacity(cat_open.len());
    for &j in cat_open {
        let mut members = cat_sets[j].clone();
        members.sort_unstable();
        grown.push(members);
    }
    SearchOutcome::Done(RegionBox {
        lo: box_lo,
        hi: box_hi,
        cat_sets: grown,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::test_support::{cons_base, stumps};
    use crate::ir::Link;

    /// Three independent levers worth 1.0 / 0.8 / 0.6 on features a/b/c — the
    /// same shape `tests/test_regions.py`'s hand ensemble uses.
    fn levers() -> Ensemble {
        stumps(
            &[
                (0, 1.0, true, 0.0, 1.0),
                (1, 1.0, true, 0.0, 0.8),
                (2, 1.0, true, 0.0, 0.6),
            ],
            3,
        )
    }

    fn all_defined(ens: &Ensemble) -> Vec<bool> {
        vec![true; ens.feature.len()]
    }

    /// Unwrap a growth that was run with a probe that never fires.
    fn done(outcome: SearchOutcome<RegionBox>) -> RegionBox {
        match outcome {
            SearchOutcome::Done(region) => region,
            SearchOutcome::Interrupted => unreachable!("no-op probe never interrupts"),
        }
    }

    // ---------------------------------------------- oracle straddle conservatism ---

    /// A box straddling the split threshold must bracket both children, so a
    /// target interval that only the low leaf reaches rejects a box that also
    /// reaches into the high leaf's cell -- the oracle must not assume the
    /// factual corner is the whole story.
    #[test]
    fn oracle_rejects_a_box_whose_bracket_straddles_outside_the_target() {
        let ens = levers();
        let missing_defined = all_defined(&ens);
        let is_nan = [false, false, false];
        // box on feature 0 spans [0.5, 1.5]: straddles the split at 1.0, so the
        // bracket is [0.0, 1.0] (base 0 from all three trees' left leaves plus
        // the straddling tree's right leaf) -- only the leaf value at exactly a
        // point is knowable, the box must bracket both.
        let lo = [0.5, 0.0, 0.0];
        let hi = [1.5, 0.0, 0.0];
        assert!(!box_feasible(
            &ens,
            &missing_defined,
            None,
            0.0,
            (0.95, 1.05),
            &[],
            &[0.5, 0.0, 0.0],
            &lo,
            &hi,
            &is_nan,
        ));
        // the same box against a target the whole bracket fits inside is fine
        assert!(box_feasible(
            &ens,
            &missing_defined,
            None,
            0.0,
            (-1.0, 2.0),
            &[],
            &[0.5, 0.0, 0.0],
            &lo,
            &hi,
            &is_nan,
        ));
    }

    // ------------------------------------------------------- growth determinism ---

    #[test]
    fn growth_is_deterministic_across_repeated_calls() {
        let ens = levers();
        let missing_defined = all_defined(&ens);
        let cons = cons_base(3);
        let x_cf = [1.0, 0.0, 0.0];
        let lo_b = [f64::NEG_INFINITY; 3];
        let hi_b = [f64::INFINITY; 3];
        let open_set = [0usize, 1, 2];
        let first = done(recourse_region(
            &ens,
            &missing_defined,
            &cons,
            &x_cf,
            (0.9, f64::INFINITY),
            &lo_b,
            &hi_b,
            &open_set,
            None,
            0.0,
            &[],
            &[],
            &mut || false,
        ));
        let second = done(recourse_region(
            &ens,
            &missing_defined,
            &cons,
            &x_cf,
            (0.9, f64::INFINITY),
            &lo_b,
            &hi_b,
            &open_set,
            None,
            0.0,
            &[],
            &[],
            &mut || false,
        ));
        assert_eq!(first.lo, second.lo);
        assert_eq!(first.hi, second.hi);
        // "a" is the only lever the >= 0.9 target needs: it must stay pinned
        // at its cell [1, inf) and b/c are free to grow both directions.
        assert_eq!(first.lo[0], 1.0);
        assert_eq!(first.hi[0], f64::INFINITY);
        assert_eq!(first.lo[1], f64::NEG_INFINITY);
        assert_eq!(first.hi[1], f64::INFINITY);
    }

    // --------------------------------------------------- achievable-edge stepping ---

    #[test]
    fn next_edge_steps_one_cell_at_a_time_then_reports_no_further_cell() {
        // f0 cells: (-inf,1) [1,3] (3,inf) -- see test_support::golden_ens's doc.
        let cells = crate::exact::domains::constraint_cells(
            &cons_base(2),
            &[&crate::exact::test_support::golden_ens()],
        );
        let f0 = &cells[0];
        // starting inside [1,3], growing up first claims this cell's own
        // achievable edge (3.0, closed) ...
        let first = next_edge(f0, 1.0, f64::NEG_INFINITY, f64::INFINITY, true);
        assert_eq!(first, Some(3.0));
        // ... then the next cell (3,inf) is unbounded above, so its achievable
        // edge IS +inf, not a stepped near-3 value -- "no further constraint
        // that way", per the module's own doc comment.
        let second = next_edge(f0, first.unwrap(), f64::NEG_INFINITY, f64::INFINITY, true);
        assert_eq!(second, Some(f64::INFINITY));
        // an already-infinite edge never looks anything up again
        assert_eq!(
            next_edge(f0, second.unwrap(), f64::NEG_INFINITY, f64::INFINITY, true),
            None
        );

        // growing down from inside [1,3] claims 1.0 first, then the clamp at
        // lo_b=0.0 (not the joint grid) stops it -- one more step lands
        // exactly on the clamp, and there is nothing further past it.
        let down_first = next_edge(f0, 1.0, 0.0, f64::INFINITY, false);
        assert_eq!(down_first, Some(0.0));
        assert_eq!(
            next_edge(f0, down_first.unwrap(), 0.0, f64::INFINITY, false),
            None
        );
    }

    // -------------------------------------------------- worst-corner order-pair ---

    /// `a - b <= 0` is the canonical order pair, never pinned by
    /// `_degenerate_features`; the box oracle must still hold it at its worst
    /// corner (`hi[a] - lo[b]`), not just at the factual point.
    #[test]
    fn worst_corner_order_pair_blocks_growth_past_the_boundary() {
        let mut cons = cons_base(2);
        cons.linears = vec![LinearC {
            indices: vec![0, 1],
            coefs: vec![1.0, -1.0],
            op: LIN_LE,
            rhs: 0.0,
            policy: POLICY_SATISFIED,
        }];
        // a <= b at the factual (0.0, 0.0); widening a's hi past b's lo would
        // break the worst corner (hi_a - lo_b > 0) even though the factual
        // corner (a==b==0) still holds.
        let x_cf = [0.0, 0.0];
        assert!(linear_holds(
            &cons.linears[0],
            &x_cf,
            &[0.0, 0.0],
            &[0.0, 0.0]
        ));
        assert!(!linear_holds(
            &cons.linears[0],
            &x_cf,
            &[0.0, 0.0],
            &[1.0, 0.0]
        ));
        assert!(linear_holds(
            &cons.linears[0],
            &x_cf,
            &[0.0, 0.0],
            &[0.0, 1.0]
        ));
    }

    // ---------------------------------------- unrouted missing split rejection ---

    /// Reviewer probe: a model whose root splits on `g`
    /// (routing feature), whose right subtree splits on a NaN-degenerate
    /// feature `f` with an UNDEFINED missing direction (as every split of an
    /// sklearn-parsed model has). Widening `g` past the root threshold opens
    /// that unrouted subtree; the oracle must reject rather than silently pick
    /// a side. Mirrors `tests/test_regions.py::TestUnroutedMissingSplitRejectsGrowth`.
    #[test]
    fn unrouted_missing_split_rejects_the_box_rather_than_guessing_a_side() {
        // node 0: split on g < 1.0 -> {1: leaf 0.0, 2: split on f}
        // node 2: split on f < 0.5, missing UNDEFINED -> {3: leaf 0.0, 4: leaf 5.0}
        let ens = Ensemble::new(
            vec![0, -1, 1, -1, -1],
            vec![1.0, 0.0, 0.5, 0.0, 0.0],
            vec![true, false, true, false, false],
            vec![false, false, false, false, false], // missing_left bit unused: undefined below
            vec![1, 0, 3, 0, 0],
            vec![2, 0, 4, 0, 0],
            vec![0.0, 0.0, 0.0, 0.0, 5.0],
            vec![0],
            0.0,
            Link::Identity,
            2,
        )
        .unwrap();
        let missing_defined = vec![true, true, false, true, true]; // node 2 undefined
        let cons = cons_base(2);
        let x_cf = [0.0, f64::NAN]; // factual path: g<1.0 -> left leaf, never visits node 2
        let lo_b = [f64::NEG_INFINITY, f64::NEG_INFINITY];
        let hi_b = [f64::INFINITY, f64::INFINITY];

        let region = done(recourse_region(
            &ens,
            &missing_defined,
            &cons,
            &x_cf,
            (-1.0, 10.0),
            &lo_b,
            &hi_b,
            &[0],
            None,
            0.0,
            &[],
            &[],
            &mut || false,
        ));
        // g must never cross into the unrouted subtree (g >= 1.0): the (unsound)
        // old behavior -- treating the undefined node as routing right, reaching
        // leaf value 5.0 -- would have accepted a box wide enough to include it.
        assert!(
            region.hi[0] < 1.0,
            "g grew past the unrouted split: hi={}",
            region.hi[0]
        );
        assert!(region.lo[0].is_finite() || region.lo[0] == f64::NEG_INFINITY);

        // directly: a box that DOES straddle into the unrouted subtree is
        // rejected by the oracle itself, not merely never attempted.
        let straddling_hi = [1.5, hi_b[1]];
        assert!(!box_feasible(
            &ens,
            &missing_defined,
            None,
            0.0,
            (-1.0, 10.0),
            &[],
            &x_cf,
            &lo_b,
            &straddling_hi,
            &[false, true],
        ));
    }

    // ------------------------------------------------------- interrupt probe ---

    /// Forty independent levers on forty features, in a target so wide that
    /// every one of them grows all the way out — enough attempts that the
    /// polling interval is crossed several times.
    fn many_levers() -> Ensemble {
        let specs: Vec<(i32, f64, bool, f64, f64)> =
            (0..40).map(|j| (j, 1.0, true, 0.0, 1.0)).collect();
        stumps(&specs, 40)
    }

    fn grow_many(probe: InterruptProbe<'_>) -> SearchOutcome<RegionBox> {
        let ens = many_levers();
        let missing_defined = all_defined(&ens);
        let cons = cons_base(40);
        let x_cf = vec![0.0; 40];
        let lo_b = vec![f64::NEG_INFINITY; 40];
        let hi_b = vec![f64::INFINITY; 40];
        let open_set: Vec<usize> = (0..40).collect();
        recourse_region(
            &ens,
            &missing_defined,
            &cons,
            &x_cf,
            (-1.0, 100.0),
            &lo_b,
            &hi_b,
            &open_set,
            None,
            0.0,
            &[],
            &[],
            probe,
        )
    }

    /// A probe that says stop throws the half-grown box away: nothing partial
    /// is ever handed back.
    #[test]
    fn growth_stops_and_reports_nothing_when_the_probe_says_stop() {
        let mut polls = 0usize;
        let outcome = grow_many(&mut || {
            polls += 1;
            true
        });
        assert!(matches!(outcome, SearchOutcome::Interrupted));
        assert_eq!(polls, 1); // the very first question ends it
    }

    /// The probe is asked once per 64 growth attempts and the box it never
    /// stops is bit-for-bit the box no probe at all produces.
    #[test]
    fn growth_polls_every_64_attempts_and_leaves_the_box_alone() {
        let mut polls = 0usize;
        let probed = done(grow_many(&mut || {
            polls += 1;
            false
        }));
        // 40 features x 2 attempts x 3 rounds = 240 attempts -> 3 questions.
        assert_eq!(polls, 3);
        let plain = done(grow_many(&mut || false));
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
        assert_eq!(bits(&probed.lo), bits(&plain.lo));
        assert_eq!(bits(&probed.hi), bits(&plain.hi));
        assert_eq!(bits(&probed.lo), bits(&vec![f64::NEG_INFINITY; 40]));
        assert_eq!(bits(&probed.hi), bits(&vec![f64::INFINITY; 40]));
    }

    // ------------------------------------------------- Python-derived canary ---

    /// PARITY CANARY. Same three-lever ensemble as `levers()`, target `>= 0.9`
    /// (only "a" matters), factual `x = [0, 0, 0]`, `x_cf = [1, 0, 0]` (a
    /// verified counterfactual: raw_score == 1.0). Every number below is what
    /// `treecf.regions._recourse_region` answered for the same inputs:
    ///
    /// ```text
    /// uv run python -c '
    /// import struct
    /// import numpy as np
    /// from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree
    /// from treecf.constraints.compile import compile_constraints
    /// from treecf.regions import _recourse_region
    /// leaf = lambda i, v: Node(i, None, None, None, None, None, None, v)
    /// stump = lambda f, t, rv: Tree((Node(0, f, t, SplitOp.LT, True, 1, 2, None), leaf(1, 0.0), leaf(2, rv)))
    /// ir = EnsembleIR((stump(0, 1.0, 1.0), stump(1, 1.0, 0.8), stump(2, 1.0, 0.6)), 0.0, Link.IDENTITY, 3, ("a","b","c"), {})
    /// x = np.zeros(3)
    /// x_cf = np.array([1.0, 0.0, 0.0])
    /// compiled = compile_constraints([], ir.feature_names)
    /// region = _recourse_region(ir, x, x_cf, (0.9, float("inf")), compiled, None, 0.0)
    /// bits = lambda v: hex(struct.unpack("<Q", struct.pack("<d", v))[0])
    /// print([bits(v) for v in region.lo], [bits(v) for v in region.hi])'
    /// ```
    ///
    /// If this drifts, the full parity suite will too.
    #[test]
    fn parity_canary_matches_the_python_backend() {
        let ens = levers();
        let missing_defined = all_defined(&ens);
        let cons = cons_base(3);
        let x_cf = [1.0, 0.0, 0.0];
        let lo_b = [f64::NEG_INFINITY; 3];
        let hi_b = [f64::INFINITY; 3];
        let region = done(recourse_region(
            &ens,
            &missing_defined,
            &cons,
            &x_cf,
            (0.9, f64::INFINITY),
            &lo_b,
            &hi_b,
            &[0, 1, 2],
            None,
            0.0,
            &[],
            &[],
            &mut || false,
        ));
        let lo_bits: Vec<u64> = region.lo.iter().map(|v| v.to_bits()).collect();
        let hi_bits: Vec<u64> = region.hi.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            lo_bits,
            vec![0x3ff0000000000000, 0xfff0000000000000, 0xfff0000000000000]
        );
        assert_eq!(
            hi_bits,
            vec![0x7ff0000000000000, 0x7ff0000000000000, 0x7ff0000000000000]
        );
    }
}
