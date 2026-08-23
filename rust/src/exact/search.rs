//! The branch-and-bound search itself — port of `treecf.backends.exact` and
//! `treecf.backends._exact_bounds`. The three parity rules in the module header
//! of `super` are load-bearing here in particular: no rayon call may enter this
//! file, the ensemble brackets are re-summed in full, and every stored merge
//! goes through `py_min`/`py_max`.

use std::time::Instant;

use crate::cells::Cell;
use crate::constraints::{py_max, py_min, Constraints};
use crate::exact::domains::{
    allow_missing_deltas, build_domains, constraint_cells, cost_of_row, demanded_values,
    domain_span, feature_order, g_floor_pairs, h_suffix, validate, State,
};
use crate::exact::orderpairs::{achievable_bounds, boundary_candidates, intersect_cell};
use crate::exact::propagation::{PropFrame, Propagation};
use crate::exact::ValuePolicy;
use crate::interrupt::{InterruptProbe, SearchOutcome};
use crate::ir::Ensemble;

/// The tolerance `check_matrix` allows a linear constraint; an order pair counts
/// as broken exactly when the arbiter would reject it.
const LINEAR_SLACK: f64 = 1e-9;

/// How many expanded nodes between two interrupt polls. Asking is cheap but not
/// free, and a search that gets nowhere near this many nodes finishes fast
/// enough that nobody reaches for the keyboard.
const SIGNAL_CHECK_INTERVAL: u64 = 1 << 18;

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
/// outside the canonical order-pair shape — it stays validation-only, so a
/// caller can keep telling a bad constraint apart from a stopped search.
///
/// `probe` is asked every `SIGNAL_CHECK_INTERVAL` expanded nodes whether to
/// stop. Answering yes throws the search away, incumbent included, and returns
/// `SearchOutcome::Interrupted`; answering no leaves every returned number
/// exactly as it would be without a probe at all.
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
    probe: InterruptProbe<'_>,
) -> Result<SearchOutcome<ExactResult>, String> {
    let start = Instant::now();
    let order_pairs = validate(cons)?;
    let (lo_t, hi_t) = interval;
    let if_ens = plausibility.map(|(ir, _)| ir);
    let min_total_path = plausibility.map_or(0.0, |(_, bound)| bound);
    let gap = params.gap;

    // (a) The factual itself: nothing is ever cheaper than not moving at all.
    if accepts(ens, if_ens, min_total_path, cons, x, lo_t, hi_t, x) {
        return Ok(SearchOutcome::Done(ExactResult {
            x_cf: Some(x.to_vec()),
            proof: "optimal",
            stats: stats(0, 0, 0, 0.0, gap, true, false),
            snapped: Vec::new(),
            distance: Some(0.0),
        }));
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
        return Ok(SearchOutcome::Done(ExactResult {
            x_cf: None,
            proof: "optimal",
            stats: stats(0, 0, 0, f64::INFINITY, gap, true, false),
            snapped: Vec::new(),
            distance: None,
        }));
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
    // Python holds the repairable pairs in a frozenset and weighs its size
    // against the *list* of order pairs, so a pair declared twice — two
    // identical `a - b <= 0` Linears, which the compiler accepts — already
    // trips the withdrawal even with no value policy anywhere. Only this count
    // is deduplicated; membership tests read the same either way.
    let mut unique_repairable: Vec<(usize, usize)> = Vec::new();
    for &pair in &repairable_pairs {
        if !unique_repairable.contains(&pair) {
            unique_repairable.push(pair);
        }
    }
    let policy_bound = !order_pairs.is_empty() && unique_repairable.len() < order_pairs.len();
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
        // Every node reaching this line is expanded below, so each count is
        // asked about at most once and a search of fewer than
        // SIGNAL_CHECK_INTERVAL nodes is never asked at all.
        if nodes_expanded > 0 && nodes_expanded % SIGNAL_CHECK_INTERVAL == 0 && probe() {
            return Ok(SearchOutcome::Interrupted);
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
    let (lower_bound, proof) = if completed {
        let bound = match incumbent_row {
            None => f64::INFINITY,
            Some(_) if gap == 0.0 => incumbent_cost,
            Some(_) => incumbent_cost / (1.0 + gap),
        };
        let label = if gap > 0.0 && gap_prune_fired {
            "optimal_within_gap"
        } else {
            "optimal"
        };
        (bound, label)
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
        (
            py_min(py_min(open_view, incumbent_cost), set_aside_view),
            "heuristic",
        )
    };

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

    Ok(SearchOutcome::Done(ExactResult {
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
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{LinearC, LIN_LE, POLICY_SATISFIED};
    use crate::exact::test_support::*;

    // -------------------------------------------------------- interrupt probe ---

    /// A search that stays under the polling interval is never asked anything,
    /// so a probe that would have said stop changes nothing: the answer is the
    /// same bits the plain solve gives.
    #[test]
    fn a_short_search_is_never_asked_and_answers_the_same_bits() {
        let ens = stumps(&[(0, 1.0, true, 0.0, 1.0), (1, 0.5, true, 0.0, 0.5)], 2);
        let cons = cons_base(2);
        let x = [0.0, 0.0];
        let plain = solve(
            &ens,
            &x,
            (1.0, 2.0),
            &cons,
            0.0,
            &no_policies(2),
            &ExactParams::default(),
            None,
        );
        let mut polls = 0usize;
        let outcome = solve_probed(
            &ens,
            &x,
            (1.0, 2.0),
            &cons,
            0.0,
            &no_policies(2),
            &ExactParams::default(),
            None,
            &mut || {
                polls += 1;
                true // would stop the search — if it were ever asked
            },
        );
        let SearchOutcome::Done(probed) = outcome else {
            panic!("a search of a few nodes must not be interrupted")
        };
        assert_eq!(polls, 0);
        assert!(probed.stats.nodes_expanded > 0);
        assert_eq!(
            bits_of(probed.x_cf.as_ref().unwrap()),
            bits_of(plain.x_cf.as_ref().unwrap())
        );
        assert_eq!(
            probed.distance.unwrap().to_bits(),
            plain.distance.unwrap().to_bits()
        );
        assert_eq!(probed.proof, plain.proof);
        assert_eq!(probed.stats, plain.stats);
        assert_eq!(probed.snapped, plain.snapped);
    }

    /// Twenty levers, each worth one point, and a target halfway between two
    /// whole numbers: no assignment can ever land in it, so no incumbent is
    /// found, nothing is cut on cost, and the search runs well past the
    /// polling interval. The warm start it was handed is a result it could
    /// have returned — a probe that says stop drops that too and reports only
    /// that it stopped.
    #[test]
    fn a_long_search_drops_even_its_warm_start_when_the_probe_says_stop() {
        let specs: Vec<(i32, f64, bool, f64, f64)> =
            (0..20).map(|j| (j, 1.0, true, 0.0, 1.0)).collect();
        let ens = stumps(&specs, 20);
        let cons = cons_base(20);
        let x = vec![0.0; 20];
        let params = ExactParams {
            time_budget_s: 1e9, // the probe, not the clock, must end this one
            ..ExactParams::default()
        };
        let mut polls = 0usize;
        let outcome = solve_probed(
            &ens,
            &x,
            (10.5, 10.5),
            &cons,
            0.0,
            &no_policies(20),
            &params,
            Some((1e12, x.as_slice())), // too dear to prune anything with
            &mut || {
                polls += 1;
                true
            },
        );
        assert!(matches!(outcome, SearchOutcome::Interrupted));
        assert_eq!(polls, 1);
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

    /// The word seam itself: features 63 and 64 sit in different mask words, so
    /// each must invalidate its own tree and neither the other's.
    #[test]
    fn bitset_handles_the_word_seam() {
        let ens = stumps(&[(63, 1.0, true, 0.0, 1.0), (64, 2.0, true, 0.0, 0.5)], 65);
        let mut assigned = vec![false; 65];
        let mut values = vec![0.0; 65];
        let mut mask = BitSet::new(65);
        let mut bounds = EnsembleBounds::new(&ens, &assigned, &values);
        assert_eq!(bounds.mask.len(), ens.feature.len() * 2);
        assert_eq!((bounds.score_min, bounds.score_max), (0.0, 1.5));

        assigned[64] = true;
        values[64] = 5.0; // second tree fixed at +0.5, the first still free
        mask.set(64);
        let frame = bounds.apply(64, &mask, &assigned, &values);
        assert_eq!((bounds.score_min, bounds.score_max), (0.5, 1.5));

        assigned[63] = true;
        values[63] = 5.0; // both fixed now
        mask.set(63);
        bounds.apply(63, &mask, &assigned, &values);
        assert_eq!((bounds.score_min, bounds.score_max), (1.5, 1.5));

        bounds.restore(&frame);
        assert_eq!((bounds.score_min, bounds.score_max), (1.0, 1.5));
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
        let outcome = solve_exact(
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
            &mut || false,
        )
        .unwrap();
        let SearchOutcome::Done(result) = outcome else {
            unreachable!("no-op probe never interrupts")
        };
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

    /// The same pair declared twice — two identical `a - b <= 0` Linears, which
    /// the compiler accepts — collapses in Python's frozenset of repairable
    /// pairs but not in the list of order pairs, so the withdrawal fires even
    /// with no value policy in sight. Python: `x_cf = [1.0, 1.0]`, distance 1.5,
    /// proof "heuristic", completed false (against "optimal"/true for one copy).
    #[test]
    fn duplicate_order_pair_withdraws_the_completeness_claim() {
        let ens = stumps(&[(0, 1.0, true, 0.0, 1.0)], 2);
        let pair = || LinearC {
            indices: vec![0, 1],
            coefs: vec![1.0, -1.0],
            op: LIN_LE,
            rhs: 0.0,
            policy: POLICY_SATISFIED,
        };
        let mut cons = cons_base(2);
        cons.linears = vec![pair(), pair()];
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
        assert_eq!(result.proof, "heuristic");
        assert_eq!(
            result.stats,
            ExactStats {
                nodes_expanded: 3,
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
