//! The emptiness search behind maximal recourse regions — port of
//! `treecf._region_slab`, node for node.
//!
//! One question per stopped side: does the extension slab (the feature moved
//! into its next routing cell, every other coordinate ranging over the box
//! already certified) contain a point that leaves the target or falls below
//! the plausibility floor? A depth-first search over sub-boxes answers it: a
//! sub-box whose bracket lies inside the target is empty of violators, one
//! whose bracket lies entirely outside yields a witness (the clamp of the
//! counterfactual, re-scored before it is believed), and anything in between
//! is split on the feature with the most straddled tree nodes, the half
//! holding the counterfactual's clamp first. The Python reference and this
//! port visit the same sub-boxes in the same order, so the node counts a
//! region reports agree bit for bit.

use crate::cells::{cell_index, Cell};
use crate::exact::orderpairs::{achievable_bounds, intersect_cell};
use crate::interrupt::InterruptProbe;
use crate::ir::Ensemble;
use crate::regions::tree_interval_bracket;

/// How many search nodes between two interrupt polls.
const SIGNAL_CHECK_INTERVAL: u64 = 4096;

/// Outcome of one emptiness search.
#[derive(Clone, Debug, PartialEq)]
pub enum SlabOutcome {
    Empty,
    Witness(Vec<f64>),
    Unknown,
    Interrupted,
}

/// One emptiness question, fully specified — mirror of Python's `SlabProblem`.
/// `cat_sets` is indexed by feature (empty = untracked), with the slab
/// feature, when categorical, already narrowed to the block under test.
pub struct SlabProblem<'a> {
    pub ens: &'a Ensemble,
    pub missing_defined: &'a [bool],
    pub if_pair: Option<(&'a Ensemble, &'a [bool])>,
    pub min_total_path: f64,
    pub interval: (f64, f64),
    pub grids: &'a [Vec<Cell>],
    pub x_cf: &'a [f64],
    pub is_nan: &'a [bool],
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
    pub cat_sets: Vec<Vec<u32>>,
    /// numeric coordinates allowed to range, ascending
    pub free: Vec<usize>,
    /// categorical coordinates whose set holds more than one code, ascending
    pub free_cats: Vec<usize>,
    /// per feature, its category blocks as member lists (empty when not categorical)
    pub blocks: &'a [Vec<Vec<u32>>],
}

/// The cell holding `value`; an infinite endpoint means the outermost cell.
fn position(cells: &[Cell], value: f64, upper: bool) -> usize {
    if value.is_infinite() {
        return if upper { cells.len() - 1 } else { 0 };
    }
    cell_index(cells, value)
}

fn clamp(value: f64, lo: f64, hi: f64) -> f64 {
    value.max(lo).min(hi)
}

struct SlabSearch<'a, 'p> {
    p: &'p SlabProblem<'a>,
    limit: u64,
    nodes: u64,
    /// per feature: the closed run of cell positions a free numeric feature ranges over
    num_range: Vec<Option<(usize, usize)>>,
    /// per feature: the blocks a free categorical feature's certified set covers
    cat_blocks: Vec<Vec<Vec<u32>>>,
    cat_range: Vec<Option<(usize, usize)>>,
    probe: InterruptProbe<'p>,
}

impl<'a, 'p> SlabSearch<'a, 'p> {
    fn new(p: &'p SlabProblem<'a>, limit: u64, probe: InterruptProbe<'p>) -> Self {
        let n = p.x_cf.len();
        let mut num_range = vec![None; n];
        for &k in &p.free {
            let cells = &p.grids[k];
            num_range[k] = Some((
                position(cells, p.lo[k], false),
                position(cells, p.hi[k], true),
            ));
        }
        let mut cat_blocks: Vec<Vec<Vec<u32>>> = vec![Vec::new(); n];
        let mut cat_range = vec![None; n];
        for &c in &p.free_cats {
            let covered: Vec<Vec<u32>> = p.blocks[c]
                .iter()
                .map(|members| {
                    members
                        .iter()
                        .copied()
                        .filter(|code| p.cat_sets[c].contains(code))
                        .collect::<Vec<u32>>()
                })
                .filter(|members| !members.is_empty())
                .collect();
            cat_range[c] = Some((0, covered.len() - 1));
            cat_blocks[c] = covered;
        }
        Self {
            p,
            limit,
            nodes: 0,
            num_range,
            cat_blocks,
            cat_range,
            probe,
        }
    }

    /// The box the current ranges stand for.
    fn current_box(&self) -> (Vec<f64>, Vec<f64>, Vec<Vec<u32>>) {
        let mut lo = self.p.lo.clone();
        let mut hi = self.p.hi.clone();
        for (k, range) in self.num_range.iter().enumerate() {
            let Some((a, b)) = *range else { continue };
            let cells = &self.p.grids[k];
            let head = intersect_cell(&cells[a], self.p.lo[k], self.p.hi[k])
                .expect("the box's own endpoints are achievable values inside their cells");
            let tail = intersect_cell(&cells[b], self.p.lo[k], self.p.hi[k])
                .expect("the box's own endpoints are achievable values inside their cells");
            lo[k] = achievable_bounds(&head).0;
            hi[k] = achievable_bounds(&tail).1;
        }
        let mut cat_sets = self.p.cat_sets.clone();
        for (c, range) in self.cat_range.iter().enumerate() {
            let Some((a, b)) = *range else { continue };
            let mut members: Vec<u32> = self.cat_blocks[c][a..=b]
                .iter()
                .flatten()
                .copied()
                .collect();
            members.sort_unstable();
            cat_sets[c] = members;
        }
        (lo, hi, cat_sets)
    }

    fn ensemble(
        &self,
        ens: &Ensemble,
        missing_defined: &[bool],
        lo: &[f64],
        hi: &[f64],
        cat_sets: &[Vec<u32>],
        straddles: &mut [u64],
    ) -> Option<(f64, f64)> {
        let mut total_min = ens.base_score;
        let mut total_max = ens.base_score;
        for &root in &ens.tree_roots {
            let (tmin, tmax) = tree_interval_bracket(
                ens,
                missing_defined,
                root,
                lo,
                hi,
                self.p.is_nan,
                cat_sets,
                Some(straddles),
            )?;
            total_min += tmin;
            total_max += tmax;
        }
        Some((total_min, total_max))
    }

    /// The counterfactual clamped into the box, believed only after it
    /// re-scores outside the target or below the plausibility floor.
    fn witness(&self, lo: &[f64], hi: &[f64], cat_sets: &[Vec<u32>]) -> Option<Vec<f64>> {
        let mut point = self.p.x_cf.to_vec();
        for k in 0..point.len() {
            if self.p.is_nan[k] {
                continue;
            }
            if !cat_sets[k].is_empty() {
                let code = point[k] as u32;
                let chosen = if cat_sets[k].contains(&code) {
                    code
                } else {
                    *cat_sets[k].iter().min().expect("non-empty set")
                };
                point[k] = chosen as f64;
                continue;
            }
            point[k] = clamp(point[k], lo[k], hi[k]);
        }
        let score = self.p.ens.raw_score(&point);
        let (lo_t, hi_t) = self.p.interval;
        if score < lo_t || score > hi_t {
            return Some(point);
        }
        if let Some((if_ens, _)) = self.p.if_pair {
            if if_ens.raw_score(&point) < self.p.min_total_path {
                return Some(point);
            }
        }
        None
    }

    fn visit(&mut self) -> SlabOutcome {
        if self.nodes >= self.limit {
            return SlabOutcome::Unknown;
        }
        self.nodes += 1;
        if self.nodes % SIGNAL_CHECK_INTERVAL == 0 && (self.probe)() {
            return SlabOutcome::Interrupted;
        }
        let (lo, hi, cat_sets) = self.current_box();
        let mut straddles = vec![0u64; self.p.x_cf.len()];
        let Some(total) = self.ensemble(
            self.p.ens,
            self.p.missing_defined,
            &lo,
            &hi,
            &cat_sets,
            &mut straddles,
        ) else {
            return SlabOutcome::Unknown;
        };
        let mut if_total: Option<(f64, f64)> = None;
        if let Some((if_ens, if_missing_defined)) = self.p.if_pair {
            let Some(bracket) = self.ensemble(
                if_ens,
                if_missing_defined,
                &lo,
                &hi,
                &cat_sets,
                &mut straddles,
            ) else {
                return SlabOutcome::Unknown;
            };
            if_total = Some(bracket);
        }
        let (lo_t, hi_t) = self.p.interval;
        let mut clean = lo_t <= total.0 && total.1 <= hi_t;
        let mut violates = total.1 < lo_t || total.0 > hi_t;
        if let Some((if_min, if_max)) = if_total {
            clean = clean && if_min >= self.p.min_total_path;
            violates = violates || if_max < self.p.min_total_path;
        }
        if clean {
            return SlabOutcome::Empty;
        }
        if violates {
            return match self.witness(&lo, &hi, &cat_sets) {
                Some(point) => SlabOutcome::Witness(point),
                None => SlabOutcome::Unknown,
            };
        }

        // refine the feature with the most straddled tree nodes, lowest index
        // on ties, numeric features before categorical ones
        let mut target: Option<usize> = None;
        let mut best: u64 = 0;
        for &k in &self.p.free {
            let (a, b) = self.num_range[k].expect("free feature has a range");
            if b > a && straddles[k] > best {
                target = Some(k);
                best = straddles[k];
            }
        }
        let mut categorical_target = false;
        for &c in &self.p.free_cats {
            let (a, b) = self.cat_range[c].expect("free categorical has a range");
            if b > a && straddles[c] > best {
                target = Some(c);
                best = straddles[c];
                categorical_target = true;
            }
        }
        let Some(target) = target else {
            return SlabOutcome::Unknown; // every coordinate atomic yet undecided: cannot happen
        };
        if !categorical_target {
            let (a, b) = self.num_range[target].expect("range");
            let mid = (a + b) / 2;
            let rep = position(
                &self.p.grids[target],
                clamp(self.p.x_cf[target], lo[target], hi[target]),
                false,
            );
            let (first, second) = if rep <= mid {
                ((a, mid), (mid + 1, b))
            } else {
                ((mid + 1, b), (a, mid))
            };
            for child in [first, second] {
                self.num_range[target] = Some(child);
                let outcome = self.visit();
                if outcome != SlabOutcome::Empty {
                    self.num_range[target] = Some((a, b));
                    return outcome;
                }
            }
            self.num_range[target] = Some((a, b));
            return SlabOutcome::Empty;
        }
        let (a, b) = self.cat_range[target].expect("range");
        let mid = (a + b) / 2;
        let code = self.p.x_cf[target] as u32;
        let rep = (a..=b)
            .find(|&pos| self.cat_blocks[target][pos].contains(&code))
            .unwrap_or(a);
        let (first, second) = if rep <= mid {
            ((a, mid), (mid + 1, b))
        } else {
            ((mid + 1, b), (a, mid))
        };
        for child in [first, second] {
            self.cat_range[target] = Some(child);
            let outcome = self.visit();
            if outcome != SlabOutcome::Empty {
                self.cat_range[target] = Some((a, b));
                return outcome;
            }
        }
        self.cat_range[target] = Some((a, b));
        SlabOutcome::Empty
    }
}

/// Answer one emptiness question within `limit` nodes: the outcome and the
/// number of sub-boxes visited.
pub fn search_slab(
    problem: &SlabProblem<'_>,
    limit: u64,
    probe: InterruptProbe<'_>,
) -> (SlabOutcome, u64) {
    if limit == 0 {
        return (SlabOutcome::Unknown, 0);
    }
    let mut search = SlabSearch::new(problem, limit, probe);
    let outcome = search.visit();
    (outcome, search.nodes)
}
