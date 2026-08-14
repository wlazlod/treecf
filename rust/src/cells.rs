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
            if ens.feature[i] >= 0 {
                pairs[ens.feature[i] as usize].push((ens.threshold[i], ens.is_lt[i]));
            }
        }
    }
    pairs.iter().map(|p| build_cells(p)).collect()
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
}
