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

    /// Point of the cell closest to `x` (open bounds step one ulp inside).
    pub fn nearest_to(&self, x: f64) -> f64 {
        if self.contains(x) {
            return x;
        }
        if x <= self.lo {
            return if self.lo_open {
                self.lo.next_up()
            } else {
                self.lo
            };
        }
        if self.hi_open {
            self.hi.next_down()
        } else {
            self.hi
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
    let mut pairs: Vec<Vec<(f64, bool)>> = vec![Vec::new(); ens.n_features];
    for i in 0..ens.feature.len() {
        if ens.feature[i] >= 0 {
            pairs[ens.feature[i] as usize].push((ens.threshold[i], ens.is_lt[i]));
        }
    }
    pairs.iter().map(|p| build_cells(p)).collect()
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
        assert_eq!(c.nearest_to(-3.0), 0.0f64.next_up());
        assert_eq!(c.nearest_to(9.0), 1.0f64.next_down());
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
}
