//! The coarse-to-fine exact search — port of `treecf.backends._exact_refine`.
//! The parity rules in the module header of `super` govern this file too:
//! every stored merge goes through `py_min`, and the operation order of the
//! cost arithmetic follows the Python reference line for line.
//!
//! A numeric feature's presolved candidate states are grouped by routing cell
//! and arranged in a balanced binary tree over those cells, split at the
//! median. The search holds features to nodes of that tree — whole intervals
//! — and descends only where the score bound forces it; refining all the way
//! down reaches exactly the classic states.

use std::time::Instant;

use crate::cells::Cell;
use crate::constraints::py_min;
use crate::exact::domains::{py_cmp, State};
use crate::exact::orderpairs::{achievable_bounds, intersect_cell};
use crate::exact::propagation::{PropFrame, Propagation};
use crate::exact::search::{
    snapped_of, BitSet, Ctx, EnsembleBounds, ExactParams, ExactResult, ExactStats,
    SIGNAL_CHECK_INTERVAL,
};
use crate::exact::trace::Trace;
use crate::interrupt::{InterruptProbe, SearchOutcome};

/// How many ranges a feature starts the search with, at most: the deepest
/// level of its hierarchy holding no more nodes than this.
const INITIAL_RANGES: usize = 8;

/// One node of a feature's hierarchy: a contiguous run of used cells.
/// `first`/`last` are positions into `Hier::cells_used`; `iv` is the hull of
/// the run's cells intersected with the instance bounds; `span` its achievable
/// `(lo, hi)`; `rep` the index of the cheapest state inside, costing `cost`.
#[derive(Clone, Debug)]
pub(crate) struct HierNode {
    pub(crate) first: usize,
    pub(crate) last: usize,
    pub(crate) iv: Cell,
    pub(crate) span: (f64, f64),
    pub(crate) cost: f64,
    pub(crate) rep: usize,
    pub(crate) n_states: usize,
    pub(crate) left: Option<usize>,
    pub(crate) right: Option<usize>,
}

/// One alternative the search can try on a feature: an atomic state (`node`
/// is `None`) or a hierarchy node it is held to, represented by that node's
/// cheapest state.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Alt {
    pub(crate) state: State,
    pub(crate) state_idx: usize,
    pub(crate) node: Option<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct Hier {
    pub(crate) states: Vec<State>,
    pub(crate) cells_used: Vec<usize>,
    /// state indices per used cell, parallel to `cells_used`, classic order
    pub(crate) ids_by_pos: Vec<Vec<usize>>,
    pub(crate) nodes: Vec<HierNode>, // node 0 is the root
    pub(crate) initial: Vec<Alt>,
}

fn alt_for(nodes: &[HierNode], states: &[State], node_id: usize) -> Alt {
    let node = &nodes[node_id];
    let state = states[node.rep];
    if node.left.is_none() && node.n_states == 1 {
        Alt {
            state,
            state_idx: node.rep,
            node: None,
        }
    } else {
        Alt {
            state,
            state_idx: node.rep,
            node: Some(node_id),
        }
    }
}

/// The hierarchy of a feature's (presolved, cost-sorted) domain, or `None`
/// when its states occupy a single cell.
pub(crate) fn build_hierarchy(
    states: &[State],
    cells: &[Cell],
    lo_b: f64,
    hi_b: f64,
) -> Option<Hier> {
    let mut cells_used: Vec<usize> = Vec::new();
    for state in states {
        if !state.is_nan && !cells_used.contains(&state.cell_idx) {
            cells_used.push(state.cell_idx);
        }
    }
    cells_used.sort_unstable();
    if cells_used.len() < 2 {
        return None;
    }
    let ids_by_pos: Vec<Vec<usize>> = cells_used
        .iter()
        .map(|&c| {
            states
                .iter()
                .enumerate()
                .filter(|(_, s)| !s.is_nan && s.cell_idx == c)
                .map(|(i, _)| i)
                .collect()
        })
        .collect();
    let clamped = |position: usize| -> Cell {
        let cell = cells[cells_used[position]];
        intersect_cell(&cell, lo_b, hi_b).unwrap_or(cell)
    };

    struct Builder<'b> {
        states: &'b [State],
        ids_by_pos: &'b [Vec<usize>],
        clamped: &'b dyn Fn(usize) -> Cell,
        nodes: Vec<Option<HierNode>>,
    }
    impl Builder<'_> {
        fn build(&mut self, first: usize, last: usize) -> usize {
            let node_id = self.nodes.len();
            self.nodes.push(None); // reserve the slot so ids are pre-order
            let (head, tail) = ((self.clamped)(first), (self.clamped)(last));
            let iv = Cell {
                lo: head.lo,
                hi: tail.hi,
                lo_open: head.lo_open,
                hi_open: tail.hi_open,
            };
            let span = (achievable_bounds(&head).0, achievable_bounds(&tail).1);
            let mut rep = usize::MAX;
            let mut n_states = 0;
            for ids in &self.ids_by_pos[first..=last] {
                for &i in ids {
                    rep = rep.min(i);
                    n_states += 1;
                }
            }
            let (mut left, mut right) = (None, None);
            if first < last {
                let mid = (first + last) / 2;
                left = Some(self.build(first, mid));
                right = Some(self.build(mid + 1, last));
            }
            self.nodes[node_id] = Some(HierNode {
                first,
                last,
                iv,
                span,
                cost: self.states[rep].cost,
                rep,
                n_states,
                left,
                right,
            });
            node_id
        }
    }
    let mut builder = Builder {
        states,
        ids_by_pos: &ids_by_pos,
        clamped: &clamped,
        nodes: Vec::new(),
    };
    builder.build(0, cells_used.len() - 1);
    let nodes: Vec<HierNode> = builder
        .nodes
        .into_iter()
        .map(|n| n.expect("every reserved slot is filled"))
        .collect();

    let depth_limit = INITIAL_RANGES.ilog2() as usize;
    let mut frontier: Vec<Alt> = Vec::new();
    fn collect(
        nodes: &[HierNode],
        states: &[State],
        node_id: usize,
        depth: usize,
        limit: usize,
        out: &mut Vec<Alt>,
    ) {
        let node = &nodes[node_id];
        if node.left.is_none() || depth == limit {
            out.push(alt_for(nodes, states, node_id));
            return;
        }
        collect(
            nodes,
            states,
            node.left.expect("internal"),
            depth + 1,
            limit,
            out,
        );
        collect(
            nodes,
            states,
            node.right.expect("internal"),
            depth + 1,
            limit,
            out,
        );
    }
    collect(&nodes, states, 0, 0, depth_limit, &mut frontier);
    for (idx, state) in states.iter().enumerate() {
        if state.is_nan {
            frontier.push(Alt {
                state: *state,
                state_idx: idx,
                node: None,
            });
        }
    }
    let n_cells = cells.len();
    let key = |alt: &Alt| -> (f64, usize, f64) {
        if alt.state.is_nan {
            return (alt.state.cost, n_cells, 0.0);
        }
        let first_cell = match alt.node {
            None => alt.state.cell_idx,
            Some(id) => cells_used[nodes[id].first],
        };
        (alt.state.cost, first_cell, alt.state.value)
    };
    // stable, like Python's sort
    frontier.sort_by(|a, b| {
        let (ka, kb) = (key(a), key(b));
        py_cmp(ka.0, kb.0)
            .then(ka.1.cmp(&kb.1))
            .then_with(|| py_cmp(ka.2, kb.2))
    });
    Some(Hier {
        states: states.to_vec(),
        cells_used,
        ids_by_pos,
        nodes,
        initial: frontier,
    })
}

/// What replaces a range when the search refines it: its two children, the
/// one holding the representative first; for a single cell holding several
/// states, those states in classic order.
pub(crate) fn children_of(hier: &Hier, node_id: usize) -> Vec<Alt> {
    let node = &hier.nodes[node_id];
    let Some(left) = node.left else {
        return hier.ids_by_pos[node.first]
            .iter()
            .map(|&idx| Alt {
                state: hier.states[idx],
                state_idx: idx,
                node: None,
            })
            .collect();
    };
    let right = node.right.expect("internal node");
    let rep_cell = hier.states[node.rep].cell_idx;
    let rep_pos = hier
        .cells_used
        .iter()
        .position(|&c| c == rep_cell)
        .expect("representative sits in a used cell");
    let ordered = if rep_pos <= hier.nodes[left].last {
        [left, right]
    } else {
        [right, left]
    };
    ordered
        .iter()
        .map(|&child| alt_for(&hier.nodes, &hier.states, child))
        .collect()
}

// ------------------------------------------------------------ the search ---

/// What applying one alternative changed, so it can be undone.
struct Applied {
    model: Vec<(usize, f64, f64)>,
    plausibility: Vec<(usize, f64, f64)>,
    prop: PropFrame,
    prev_range: Option<Cell>,
    prev_span: Option<(f64, f64)>,
    prev_node: Option<usize>,
    prev_picked: usize,
    prev_value: f64,
    prev_assigned: bool,
}

/// One level of the coarse-to-fine stack — mirror of Python's `_RFrame`.
struct RFrame {
    j: usize,
    alts: Vec<Alt>,
    next: usize,
    refine_of: Option<usize>,
    level: usize,
    h_rest: f64,
    g_before: f64,
    applied: Option<Applied>,
}

fn frontier_bound(frames: &[RFrame], h_suffix: &[f64]) -> f64 {
    let mut bound = f64::INFINITY;
    for fr in frames {
        if fr.next < fr.alts.len() {
            let rest = if fr.refine_of.is_none() {
                h_suffix[fr.level + 1]
            } else {
                0.0
            };
            bound = py_min(bound, fr.g_before + fr.alts[fr.next].state.cost + rest);
        }
    }
    bound
}

fn sample_bound(frontier: f64, gap: f64, incumbent_cost: f64, dropped_floor: f64) -> f64 {
    let mut bound = frontier;
    if gap > 0.0 {
        bound = py_min(bound, incumbent_cost / (1.0 + gap));
    }
    let set_aside_view = if dropped_floor == f64::NEG_INFINITY {
        0.0
    } else {
        dropped_floor
    };
    py_min(py_min(bound, set_aside_view), incumbent_cost)
}

/// The coarse-to-fine search — mirror of Python's `run_refine`, assembling
/// the `ExactResult` itself. `dropped_floor` starts as the classic search's
/// (negative infinity when some order pair is policy-bound).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    ctx: &Ctx<'_>,
    propagation: &mut Propagation<'_>,
    model_bounds: &mut EnsembleBounds<'_>,
    if_bounds: &mut Option<EnsembleBounds<'_>>,
    assigned: &mut [bool],
    values: &mut [f64],
    h_suffix: &[f64],
    incumbent: Option<(f64, &[f64])>,
    presolve_removed: u64,
    mut dropped_floor: f64,
    start: Instant,
    params: &ExactParams,
    probe: InterruptProbe<'_>,
) -> SearchOutcome<ExactResult> {
    let n = ctx.x.len();
    let order = &ctx.order;
    let n_levels = order.len();
    let gap = params.gap;

    let mut onehot_member = vec![false; n];
    for group in &ctx.cons.onehot {
        for &f in group {
            onehot_member[f as usize] = true;
        }
    }
    let hiers: Vec<Option<Hier>> = (0..n)
        .map(|j| {
            if !order.contains(&j) || ctx.ens.cardinality[j] > 0 || onehot_member[j] {
                None // blocks are coarse already; one-hot members hold 0/1
            } else {
                build_hierarchy(
                    &ctx.domains[j],
                    &ctx.grids[j],
                    ctx.bounds_lo[j],
                    ctx.bounds_hi[j],
                )
            }
        })
        .collect();

    let mut incumbent_cost = f64::INFINITY;
    let mut incumbent_row: Option<Vec<f64>> = None;
    let mut incumbent_states: Option<Vec<State>> = None;
    let mut warm_start_used = false;
    if let Some((cost, row)) = incumbent {
        incumbent_cost = cost;
        incumbent_row = Some(row.to_vec());
        warm_start_used = true;
    }

    let mut nodes_expanded: u64 = 0;
    let mut nodes_pruned_score: u64 = 0;
    let mut nodes_pruned_cost: u64 = 0;
    let mut coarse_accepts: u64 = 0;
    let mut refinements: u64 = 0;
    let mut gap_prune_fired = false;
    let mut completed = true;
    let mut assigned_mask = BitSet::new(ctx.ens.n_features);
    let mut range_mask = BitSet::new(ctx.ens.n_features);
    let mut ranges: Vec<Option<Cell>> = vec![None; n];
    let mut range_span: Vec<Option<(f64, f64)>> = vec![None; n];
    let mut range_node: Vec<Option<usize>> = vec![None; n];
    let mut picked: Vec<usize> = vec![0; n];
    let mut g = 0.0;
    let mut frames: Vec<RFrame> = Vec::new();
    let mut trace = Trace::new();

    let push_feature = |frames: &mut Vec<RFrame>, level: usize, g: f64| {
        let j = order[level];
        let alts: Vec<Alt> = match &hiers[j] {
            Some(hier) => hier.initial.clone(),
            None => ctx.domains[j]
                .iter()
                .enumerate()
                .map(|(i, st)| Alt {
                    state: *st,
                    state_idx: i,
                    node: None,
                })
                .collect(),
        };
        frames.push(RFrame {
            j,
            alts,
            next: 0,
            refine_of: None,
            level,
            h_rest: h_suffix[level],
            g_before: g,
            applied: None,
        });
    };

    if n_levels > 0 {
        push_feature(&mut frames, 0, g);
    }
    while !frames.is_empty() {
        let top = frames.len() - 1;
        if let Some(ap) = frames[top].applied.take() {
            // undo
            let j = frames[top].j;
            propagation.restore(&ap.prop);
            model_bounds.restore(&ap.model);
            if let Some(bounds) = if_bounds.as_mut() {
                bounds.restore(&ap.plausibility);
            }
            ranges[j] = ap.prev_range;
            range_span[j] = ap.prev_span;
            range_node[j] = ap.prev_node;
            picked[j] = ap.prev_picked;
            values[j] = ap.prev_value;
            assigned[j] = ap.prev_assigned;
            if ranges[j].is_none() {
                range_mask.clear(j);
            } else {
                range_mask.set(j);
            }
            if !assigned[j] {
                assigned_mask.clear(j);
            }
        }
        if frames[top].next >= frames[top].alts.len() {
            frames.pop();
            continue;
        }
        if nodes_expanded >= params.node_budget
            || start.elapsed().as_secs_f64() > params.time_budget_s
        {
            completed = false;
            break;
        }
        if nodes_expanded > 0 && nodes_expanded % SIGNAL_CHECK_INTERVAL == 0 && probe() {
            return SearchOutcome::Interrupted;
        }

        nodes_expanded += 1;
        if nodes_expanded & (nodes_expanded - 1) == 0 {
            let bound = sample_bound(
                frontier_bound(&frames, h_suffix),
                gap,
                incumbent_cost,
                dropped_floor,
            );
            trace.record(
                nodes_expanded,
                incumbent_row.as_ref().map(|_| incumbent_cost),
                bound,
                false,
            );
        }
        let fr = &mut frames[top];
        let alt = fr.alts[fr.next];
        fr.next += 1;
        let j = fr.j;
        let g_before = fr.g_before;
        let refine_of = fr.refine_of;
        let level = fr.level;

        // apply
        let prev_range = ranges[j];
        let prev_span = range_span[j];
        let prev_node = range_node[j];
        let prev_picked = picked[j];
        let prev_value = values[j];
        let prev_assigned = assigned[j];
        let (prop_frame, conflict) = match alt.node {
            Some(node_id) => {
                let node = &hiers[j]
                    .as_ref()
                    .expect("range alternatives come from a hierarchy")
                    .nodes[node_id];
                ranges[j] = Some(node.iv);
                range_span[j] = Some(node.span);
                range_node[j] = Some(node_id);
                range_mask.set(j);
                (PropFrame::default(), false)
            }
            None => {
                ranges[j] = None;
                range_span[j] = None;
                range_node[j] = None;
                range_mask.clear(j);
                propagation.apply_with(j, alt.state.value, assigned, values, &ranges)
            }
        };
        assigned[j] = true;
        values[j] = alt.state.value;
        picked[j] = alt.state_idx;
        assigned_mask.set(j);
        let model_frame = model_bounds.apply_with(j, &assigned_mask, assigned, values, &ranges);
        let if_frame = match if_bounds.as_mut() {
            Some(bounds) => bounds.apply_with(j, &assigned_mask, assigned, values, &ranges),
            None => Vec::new(),
        };
        g = g_before + alt.state.cost;
        frames[top].applied = Some(Applied {
            model: model_frame,
            plausibility: if_frame,
            prop: prop_frame,
            prev_range,
            prev_span,
            prev_node,
            prev_picked,
            prev_value,
            prev_assigned,
        });

        if conflict
            || (!ctx.bounded_pairs.is_empty()
                && ctx.unorderable(assigned, values, &picked, &range_span))
        {
            nodes_pruned_cost += 1;
            continue;
        }
        if model_bounds.score_max < ctx.lo_t || model_bounds.score_min > ctx.hi_t {
            nodes_pruned_score += 1;
            continue;
        }
        if if_bounds
            .as_ref()
            .is_some_and(|bounds| bounds.score_max < ctx.min_total_path)
        {
            nodes_pruned_score += 1;
            continue;
        }
        let floor = g + if refine_of.is_none() {
            h_suffix[level + 1]
        } else {
            0.0
        };
        let threshold = if gap == 0.0 {
            incumbent_cost
        } else {
            incumbent_cost / (1.0 + gap)
        };
        if floor >= threshold {
            nodes_pruned_cost += 1;
            if incumbent_cost > floor {
                gap_prune_fired = true;
            }
            continue;
        }

        if refine_of.is_none() && level + 1 < n_levels {
            push_feature(&mut frames, level + 1, g);
            continue;
        }

        // every feature holds something: settle the assignment
        let mut row = ctx.x.to_vec();
        for &f in order {
            row[f] = values[f];
        }
        if range_mask.is_empty() {
            let accepted = ctx.finish(
                &row,
                &picked,
                &propagation.forced_value,
                g,
                &mut dropped_floor,
            );
            if let Some(accepted) = accepted {
                let cost = ctx.cost_of(&accepted);
                if cost < incumbent_cost {
                    incumbent_cost = cost;
                    incumbent_row = Some(accepted);
                    incumbent_states =
                        Some(order.iter().map(|&f| ctx.domains[f][picked[f]]).collect());
                    let bound = sample_bound(
                        frontier_bound(&frames, h_suffix),
                        gap,
                        incumbent_cost,
                        dropped_floor,
                    );
                    trace.record(nodes_expanded, Some(incumbent_cost), bound, true);
                }
            }
            continue;
        }

        // Every closed box ends one of three ways: pruned by a sound bound,
        // accepted here at its true minimum feasible cost, or refined down to
        // atomic cells that the classic completion path decides — so a
        // completed search returns a true optimum and a completed empty search
        // certifies infeasibility, exactly as the classic search does.
        let inside = !(model_bounds.score_min < ctx.lo_t || model_bounds.score_max > ctx.hi_t)
            && if_bounds
                .as_ref()
                .is_none_or(|bounds| bounds.score_min >= ctx.min_total_path);
        if inside && ctx.accepts(&row) {
            coarse_accepts += 1;
            let cost = ctx.cost_of(&row);
            if cost < incumbent_cost {
                incumbent_cost = cost;
                incumbent_row = Some(row);
                incumbent_states = Some(order.iter().map(|&f| ctx.domains[f][picked[f]]).collect());
                let bound = sample_bound(
                    frontier_bound(&frames, h_suffix),
                    gap,
                    incumbent_cost,
                    dropped_floor,
                );
                trace.record(nodes_expanded, Some(incumbent_cost), bound, true);
            }
            continue;
        }

        let mut counts = vec![0u64; n];
        model_bounds.count_unresolved(&ranges, &range_mask, assigned, values, &mut counts);
        if let Some(bounds) = if_bounds.as_ref() {
            bounds.count_unresolved(&ranges, &range_mask, assigned, values, &mut counts);
        }
        let mut target: Option<usize> = None;
        let mut best: i64 = -1;
        for &f in order {
            if ranges[f].is_some() && counts[f] as i64 > best {
                target = Some(f);
                best = counts[f] as i64;
            }
        }
        let target = target.expect("a range is held while range_mask is set");
        let node_id = range_node[target].expect("a held range has a node");
        let hier = hiers[target]
            .as_ref()
            .expect("a held range has a hierarchy");
        let node = &hier.nodes[node_id];
        frames.push(RFrame {
            j: target,
            alts: children_of(hier, node_id),
            next: 0,
            refine_of: Some(node_id),
            level: n_levels,
            h_rest: node.cost,
            g_before: g - node.cost,
            applied: None,
        });
        refinements += 1;
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
        for fr in &frames {
            open_view = py_min(open_view, fr.g_before + fr.h_rest);
        }
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
    trace.record(
        nodes_expanded,
        incumbent_row.as_ref().map(|_| incumbent_cost),
        lower_bound,
        false,
    );
    let snapped = snapped_of(
        order,
        incumbent_states.as_deref(),
        incumbent_row.as_deref(),
        ctx.x,
    );
    SearchOutcome::Done(ExactResult {
        distance: incumbent_row.as_ref().map(|_| incumbent_cost),
        x_cf: incumbent_row,
        proof,
        stats: ExactStats {
            nodes_expanded,
            nodes_pruned_score,
            nodes_pruned_cost,
            lower_bound,
            gap,
            completed,
            warm_start_used,
            presolve_removed,
            presolve_certified: false,
            search: params.search,
            coarse_accepts,
            refinements,
        },
        snapped,
        trace: trace.into_samples(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::build_cells;

    fn states(cells: &[usize]) -> Vec<State> {
        cells
            .iter()
            .map(|&i| State::new(i as f64, i as f64, i, false))
            .collect()
    }

    fn ten_cells() -> Vec<Cell> {
        build_cells(&(1..10).map(|t| (t as f64, true)).collect::<Vec<_>>())
    }

    #[test]
    fn frontier_is_the_depth_three_level() {
        let cells = ten_cells();
        let hier = build_hierarchy(
            &states(&(0..10).collect::<Vec<_>>()),
            &cells,
            f64::NEG_INFINITY,
            f64::INFINITY,
        )
        .unwrap();
        let positions: Vec<(usize, usize)> = hier
            .initial
            .iter()
            .map(|alt| match alt.node {
                None => {
                    let pos = hier
                        .cells_used
                        .iter()
                        .position(|&c| c == alt.state.cell_idx)
                        .unwrap();
                    (pos, pos)
                }
                Some(id) => (hier.nodes[id].first, hier.nodes[id].last),
            })
            .collect();
        assert_eq!(
            positions,
            vec![
                (0, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 6),
                (7, 7),
                (8, 8),
                (9, 9)
            ]
        );
        let root = &hier.nodes[0];
        assert_eq!(
            (root.first, root.last, root.rep, root.n_states),
            (0, 9, 0, 10)
        );
    }

    #[test]
    fn children_lead_with_the_representative_side() {
        let cells = ten_cells();
        let hier = build_hierarchy(
            &states(&(0..10).collect::<Vec<_>>()),
            &cells,
            f64::NEG_INFINITY,
            f64::INFINITY,
        )
        .unwrap();
        let kids = children_of(&hier, 0);
        let first = &hier.nodes[kids[0].node.unwrap()];
        let second = &hier.nodes[kids[1].node.unwrap()];
        assert_eq!((first.first, first.last), (0, 4));
        assert_eq!((second.first, second.last), (5, 9));
    }

    #[test]
    fn a_single_used_cell_has_no_hierarchy() {
        let cells = ten_cells();
        assert!(build_hierarchy(&states(&[3]), &cells, f64::NEG_INFINITY, f64::INFINITY).is_none());
    }
}
