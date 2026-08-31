//! Routing-atomic cells per feature — port of `treecf.aim.cells` (build_cells,
//! Cell::nearest_to with one-ulp stepping inside open bounds).

use crate::ir::Ensemble;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    pub lo: f64,
    pub hi: f64,
    pub lo_open: bool,
    pub hi_open: bool,
}

impl Cell {
    pub fn contains(&self, x: f64) -> bool {
        let above = if self.lo_open {
            x > self.lo
        } else {
            x >= self.lo
        };
        let below = if self.hi_open {
            x < self.hi
        } else {
            x <= self.hi
        };
        above && below
    }

    /// Point of the cell closest to `x`.
    ///
    /// Open bounds step one FLOAT32 ulp inside (float64-ulp fallback for
    /// narrower cells): native GBDTs compare in float32, so a float64-ulp
    /// neighbour of a threshold would collapse onto it in the deployed model.
    /// Must mirror `treecf.aim.cells.Cell.nearest_to` exactly.
    pub fn nearest_to(&self, x: f64) -> f64 {
        if self.contains(x) {
            return x;
        }
        if x <= self.lo {
            if !self.lo_open {
                return self.lo;
            }
            let stepped = (self.lo as f32).next_up() as f64;
            return if self.contains(stepped) {
                stepped
            } else {
                self.lo.next_up()
            };
        }
        if !self.hi_open {
            return self.hi;
        }
        let stepped = (self.hi as f32).next_down() as f64;
        if self.contains(stepped) {
            stepped
        } else {
            self.hi.next_down()
        }
    }
}

/// Split pairs for one feature -> routing-atomic cells (LT+LE collision -> singleton).
pub fn build_cells(pairs: &[(f64, bool)]) -> Vec<Cell> {
    // group ops per threshold; Python dict keys use value equality, so -0.0 == 0.0
    let mut thresholds: Vec<f64> = Vec::new();
    let mut has_lt: Vec<bool> = Vec::new();
    let mut has_le: Vec<bool> = Vec::new();
    for &(raw_t, is_lt) in pairs {
        let t = if raw_t == 0.0 { 0.0 } else { raw_t }; // collapse -0.0 like Python dict keys
        match thresholds.iter().position(|&u| u == t) {
            Some(k) => {
                if is_lt {
                    has_lt[k] = true;
                } else {
                    has_le[k] = true;
                }
            }
            None => {
                thresholds.push(t);
                has_lt.push(is_lt);
                has_le.push(!is_lt);
            }
        }
    }
    let mut order: Vec<usize> = (0..thresholds.len()).collect();
    order.sort_by(|&a, &b| thresholds[a].total_cmp(&thresholds[b]));

    let mut cells = Vec::with_capacity(order.len() + 2);
    let (mut lo, mut lo_open) = (f64::NEG_INFINITY, true);
    for &k in &order {
        let t = thresholds[k];
        if has_lt[k] && has_le[k] {
            cells.push(Cell {
                lo,
                hi: t,
                lo_open,
                hi_open: true,
            });
            cells.push(Cell {
                lo: t,
                hi: t,
                lo_open: false,
                hi_open: false,
            });
            (lo, lo_open) = (t, true);
        } else if has_le[k] {
            cells.push(Cell {
                lo,
                hi: t,
                lo_open,
                hi_open: false,
            });
            (lo, lo_open) = (t, true);
        } else {
            cells.push(Cell {
                lo,
                hi: t,
                lo_open,
                hi_open: true,
            });
            (lo, lo_open) = (t, false);
        }
    }
    cells.push(Cell {
        lo,
        hi: f64::INFINITY,
        lo_open,
        hi_open: true,
    });
    cells
}

/// Cells per feature over the MODEL ensemble only (the GA excludes the IF, as in Python).
pub fn feature_cells(ens: &Ensemble) -> Vec<Vec<Cell>> {
    feature_cells_joint(&[ens])
}

/// Cells per feature across several ensembles — the joint grid of the variadic
/// `treecf.aim.cells.feature_cells(*irs)` (model plus an optional isolation
/// forest). Threshold pairs of every ensemble are merged before `build_cells`,
/// exactly as Python merges them into one per-feature list.
///
/// Panics when the ensembles disagree on the feature-space width, where Python
/// raises `ValueError`; the exact backend only ever pairs a model with an
/// isolation forest already validated against the same feature space.
pub fn feature_cells_joint(ensembles: &[&Ensemble]) -> Vec<Vec<Cell>> {
    let n_features = ensembles[0].n_features;
    assert!(
        ensembles.iter().all(|e| e.n_features == n_features),
        "all ensembles must share the same feature space"
    );
    let mut pairs: Vec<Vec<(f64, bool)>> = vec![Vec::new(); n_features];
    for ens in ensembles {
        for i in 0..ens.feature.len() {
            // set-membership splits carry no threshold; their features partition
            // into category blocks instead of interval cells
            if ens.feature[i] >= 0 && ens.node_set[i] < 0 {
                pairs[ens.feature[i] as usize].push((ens.threshold[i], ens.is_lt[i]));
            }
        }
    }
    pairs.iter().map(|p| build_cells(p)).collect()
}

/// Category blocks per feature across ensembles — port of
/// `treecf.aim.cells.category_blocks`. Indexed by feature; numeric features get
/// an empty vec. Two codes share a block iff no split in any ensemble separates
/// them: set-membership splits partition by membership, numeric splits on a
/// categorical feature (an isolation forest trained on raw codes) partition by
/// threshold side. Blocks are ordered by smallest member (codes scan
/// ascending), members ascend, and a block's representative is its first entry.
///
/// Panics where Python raises `ValueError` (cardinality disagreement), matching
/// the `feature_cells_joint` convention.
pub fn category_blocks_joint(ensembles: &[&Ensemble]) -> Vec<Vec<Vec<u32>>> {
    let n_features = ensembles[0].n_features;
    let mut cardinality = vec![0u32; n_features];
    for ens in ensembles {
        for (j, slot) in cardinality.iter_mut().enumerate() {
            let k = ens.cardinality[j];
            if k > 0 {
                if *slot == 0 {
                    *slot = k;
                } else {
                    assert_eq!(
                        *slot, k,
                        "ensembles disagree on the cardinality of feature {j}"
                    );
                }
            }
        }
    }
    let mut out = Vec::with_capacity(n_features);
    for (j, &card) in cardinality.iter().enumerate() {
        let k = card as usize;
        if k == 0 {
            out.push(Vec::new());
            continue;
        }
        let mut signatures: Vec<Vec<bool>> = vec![Vec::new(); k];
        for ens in ensembles {
            for i in 0..ens.feature.len() {
                if ens.feature[i] != j as i32 {
                    continue;
                }
                if ens.node_set[i] >= 0 {
                    for (code, sig) in signatures.iter_mut().enumerate() {
                        sig.push(ens.set_contains(ens.node_set[i], code as f64));
                    }
                } else if ens.is_lt[i] {
                    for (code, sig) in signatures.iter_mut().enumerate() {
                        sig.push((code as f64) < ens.threshold[i]);
                    }
                } else {
                    for (code, sig) in signatures.iter_mut().enumerate() {
                        sig.push((code as f64) <= ens.threshold[i]);
                    }
                }
            }
        }
        let mut blocks: Vec<Vec<u32>> = Vec::new();
        let mut seen: Vec<usize> = Vec::new(); // block index -> a code carrying its signature
        for code in 0..k {
            match seen.iter().position(|&c| signatures[c] == signatures[code]) {
                Some(b) => blocks[b].push(code as u32),
                None => {
                    seen.push(code);
                    blocks.push(vec![code as u32]);
                }
            }
        }
        out.push(blocks);
    }
    out
}

/// Index of the unique cell containing `x` — port of `treecf.aim.cells.cell_index`.
///
/// Panics where Python raises `ValueError`. The cells partition the whole line,
/// so only a NaN (or a grid belonging to another feature) can miss; callers pass
/// neither — NaN candidates carry the past-the-end sentinel index instead.
pub fn cell_index(cells: &[Cell], x: f64) -> usize {
    cells
        .iter()
        .position(|cell| cell.contains(x))
        .expect("value not covered by cells (should be impossible)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Link;

    #[test]
    fn lt_le_collision_yields_singleton() {
        let cells = build_cells(&[(1.0, true), (1.0, false)]);
        assert_eq!(cells.len(), 3);
        assert_eq!(
            cells[1],
            Cell {
                lo: 1.0,
                hi: 1.0,
                lo_open: false,
                hi_open: false
            }
        );
    }

    #[test]
    fn nearest_to_steps_one_ulp_inside_open_bounds() {
        let c = Cell {
            lo: 0.0,
            hi: 1.0,
            lo_open: true,
            hi_open: true,
        };
        assert_eq!(c.nearest_to(-3.0), (0.0f32).next_up() as f64);
        assert_eq!(c.nearest_to(9.0), (1.0f32).next_down() as f64);
        assert_eq!(c.nearest_to(0.5), 0.5);
    }

    #[test]
    fn closed_bound_is_the_bound() {
        let c = Cell {
            lo: 1.0,
            hi: f64::INFINITY,
            lo_open: false,
            hi_open: true,
        };
        assert_eq!(c.nearest_to(0.0), 1.0);
    }

    /// Two stumps splitting the same feature at different thresholds merge into
    /// one grid, as the variadic Python `feature_cells` does.
    #[test]
    fn joint_grid_merges_thresholds_of_every_ensemble() {
        use crate::ir::{Ensemble, Link};
        let stump = |threshold: f64| {
            Ensemble::new(
                vec![0, -1, -1],
                vec![threshold, 0.0, 0.0],
                vec![true, false, false],
                vec![false; 3],
                vec![1, 0, 0],
                vec![2, 0, 0],
                vec![0.0, -1.0, 1.0],
                vec![0],
                0.0,
                Link::Identity,
                2,
            )
            .unwrap()
        };
        let (a, b) = (stump(1.0), stump(2.0));
        let joint = feature_cells_joint(&[&a, &b]);
        assert_eq!(joint[0].len(), 3); // (-inf,1) [1,2) [2,inf)
        assert_eq!(joint[0][1].lo, 1.0);
        assert_eq!(joint[0][1].hi, 2.0);
        assert_eq!(joint[1].len(), 1); // untouched feature: one cell, no split
        assert_eq!(feature_cells(&a)[0].len(), 2);
    }

    #[test]
    fn cell_index_finds_the_containing_cell() {
        let cells = build_cells(&[(1.0, true), (1.0, false)]);
        assert_eq!(cell_index(&cells, 0.0), 0);
        assert_eq!(cell_index(&cells, 1.0), 1);
        assert_eq!(cell_index(&cells, 1.5), 2);
    }

    /// One set stump per word list, all on feature 0, cardinality `k`.
    fn set_ens(sets: &[u64], k: u32) -> Ensemble {
        let n = sets.len().max(1);
        let mut feature = Vec::new();
        let mut node_set = Vec::new();
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut value = Vec::new();
        let mut roots = Vec::new();
        for (t, _) in sets.iter().enumerate() {
            let base = (t * 3) as u32;
            roots.push(base);
            feature.extend_from_slice(&[0, -1, -1]);
            node_set.extend_from_slice(&[t as i32, -1, -1]);
            left.extend_from_slice(&[base + 1, 0, 0]);
            right.extend_from_slice(&[base + 2, 0, 0]);
            value.extend_from_slice(&[0.0, -1.0, 1.0]);
        }
        if sets.is_empty() {
            roots.push(0);
            feature.push(-1);
            node_set.push(-1);
            left.push(0);
            right.push(0);
            value.push(0.0);
        }
        let n_nodes = feature.len();
        Ensemble::new(
            feature,
            vec![0.0; n_nodes],
            vec![false; n_nodes],
            vec![false; n_nodes],
            left,
            right,
            value,
            roots,
            0.0,
            Link::Identity,
            n,
        )
        .unwrap()
        .with_categories(
            node_set,
            (0..=sets.len() as u32).collect(),
            sets.to_vec(),
            {
                let mut c = vec![0; n];
                c[0] = k;
                c
            },
        )
        .unwrap()
    }

    #[test]
    fn no_splits_one_block() {
        let e = set_ens(&[], 4);
        assert_eq!(e.category_blocks()[0], vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn one_set_two_blocks_numbered_by_smallest_member() {
        let e = set_ens(&[0b101], 4); // {0, 2}
        assert_eq!(e.category_blocks()[0], vec![vec![0, 2], vec![1, 3]]);
    }

    #[test]
    fn two_sets_refine_to_singletons() {
        let e = set_ens(&[0b101, 0b1100], 4); // {0,2} then {2,3}
        assert_eq!(
            e.category_blocks()[0],
            vec![vec![0], vec![1], vec![2], vec![3]]
        );
    }

    #[test]
    fn joint_blocks_refine_and_numeric_splits_partition_codes() {
        let a = set_ens(&[0b101], 4);
        // a numeric stump on the same feature at 1.5 (an isolation forest would)
        let b = {
            let mut e = Ensemble::new(
                vec![0, -1, -1],
                vec![1.5, 0.0, 0.0],
                vec![false, false, false],
                vec![false, false, false],
                vec![1, 0, 0],
                vec![2, 0, 0],
                vec![0.0, -1.0, 1.0],
                vec![0],
                0.0,
                Link::Identity,
                1,
            )
            .unwrap();
            e = e
                .with_categories(vec![-1, -1, -1], vec![0], vec![], vec![4])
                .unwrap();
            e
        };
        let joint = category_blocks_joint(&[&a, &b]);
        assert_eq!(joint[0], vec![vec![0], vec![1], vec![2], vec![3]]);
    }
}
