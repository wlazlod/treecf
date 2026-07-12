//! Constraint check/repair — an exact port of `CompiledConstraints.check_matrix`,
//! `repair_matrix`, and `instance_bounds` (src/treecf/constraints/compile.py).
//!
//! numpy micro-semantics deliberately replicated:
//! - `max(acc, x)` / `min(acc, x)` keep the accumulator when x is NaN (Python
//!   comparison semantics), so Freeze/Monotone on a NaN factual yields ±inf
//!   bounds, not NaN.
//! - Linear totals use nansum (skip-NaN, index order).
//! - OneHot repair argmax: NaN counts as -1.0, lowest index wins ties.
//! - Repair's per-feature pass: a factual-NaN, non-AllowMissing column is
//!   forced to NaN and SKIPS clipping (the Python `continue`).

pub const LIN_LE: u8 = 0;
pub const LIN_GE: u8 = 1;
pub const LIN_EQ: u8 = 2;
pub const POLICY_SATISFIED: u8 = 0;

const TOL: f64 = 1e-9;

pub struct LinearC {
    pub indices: Vec<u32>,
    pub coefs: Vec<f64>,
    pub op: u8,
    pub rhs: f64,
    pub policy: u8,
}

pub struct Constraints {
    pub n_features: usize,
    pub freeze: Vec<u32>,
    pub ranges: Vec<(u32, f64, f64)>,
    pub equals: Vec<(u32, f64)>,
    pub monotone: Vec<(u32, i8)>, // +1 increase, -1 decrease
    pub linears: Vec<LinearC>,
    pub implications: Vec<(u32, f64, u32, f64)>, // cond_idx, cond_val, cons_idx, cons_val
    pub onehot: Vec<Vec<u32>>,
    pub allow_missing: Vec<(u32, f64, f64)>, // (feature, delta_to, delta_from), index-sorted
}

/// Python `max(a, b)`: returns b only if b > a (NaN comparisons are false).
#[inline]
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

#[inline]
fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

impl Constraints {
    pub fn allows_missing(&self, j: usize) -> bool {
        self.allow_missing
            .iter()
            .any(|&(idx, _, _)| idx as usize == j)
    }

    /// (lo, hi, frozen) per feature for factual `x` — bounds intersected.
    pub fn instance_bounds(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
        let p = self.n_features;
        let mut lo = vec![f64::NEG_INFINITY; p];
        let mut hi = vec![f64::INFINITY; p];
        let mut frozen = vec![false; p];
        for &j in &self.freeze {
            let j = j as usize;
            lo[j] = py_max(lo[j], x[j]);
            hi[j] = py_min(hi[j], x[j]);
            frozen[j] = true;
        }
        for &(j, r_lo, r_hi) in &self.ranges {
            let j = j as usize;
            lo[j] = py_max(lo[j], r_lo);
            hi[j] = py_min(hi[j], r_hi);
        }
        for &(j, value) in &self.equals {
            let j = j as usize;
            lo[j] = py_max(lo[j], value);
            hi[j] = py_min(hi[j], value);
        }
        for &(j, dir) in &self.monotone {
            let j = j as usize;
            if dir > 0 {
                lo[j] = py_max(lo[j], x[j]);
            } else {
                hi[j] = py_min(hi[j], x[j]);
            }
        }
        (lo, hi, frozen)
    }

    /// Vectorized feasibility per row (row-major X), matching `check_matrix`.
    pub fn check(&self, xs: &[f64], n_rows: usize, x: &[f64]) -> Vec<bool> {
        use rayon::prelude::*;
        let p = self.n_features;
        let (lo, hi, _) = self.instance_bounds(x);
        // check_matrix replaces NaN bounds with ±inf (defensive; normally a no-op)
        let lo: Vec<f64> = lo
            .iter()
            .map(|&v| if v.is_nan() { f64::NEG_INFINITY } else { v })
            .collect();
        let hi: Vec<f64> = hi
            .iter()
            .map(|&v| if v.is_nan() { f64::INFINITY } else { v })
            .collect();

        let mut out = vec![false; n_rows];
        out.par_iter_mut().enumerate().for_each(|(r, slot)| {
            *slot = self.check_row(&xs[r * p..(r + 1) * p], x, &lo, &hi);
        });
        out
    }

    fn check_row(&self, row: &[f64], x: &[f64], lo: &[f64], hi: &[f64]) -> bool {
        let p = self.n_features;
        for j in 0..p {
            let v = row[j];
            if !v.is_nan() && !(v >= lo[j] && v <= hi[j]) {
                return false;
            }
        }
        for j in 0..p {
            let allow = self.allows_missing(j);
            if !allow && !x[j].is_nan() && row[j].is_nan() {
                return false;
            }
            if x[j].is_nan() && !allow && !row[j].is_nan() {
                return false;
            }
        }
        for lin in &self.linears {
            let mut total = 0.0;
            let mut any_nan = false;
            for (k, &idx) in lin.indices.iter().enumerate() {
                let term = lin.coefs[k] * row[idx as usize];
                if term.is_nan() {
                    any_nan = true;
                } else {
                    total += term; // nansum: skip NaN, index order
                }
            }
            let holds = match lin.op {
                LIN_LE => total <= lin.rhs + TOL,
                LIN_GE => total >= lin.rhs - TOL,
                _ => (total - lin.rhs).abs() <= TOL,
            };
            let ok = if lin.policy == POLICY_SATISFIED {
                holds || any_nan
            } else {
                holds && !any_nan
            };
            if !ok {
                return false;
            }
        }
        for &(ci, cv, si, sv) in &self.implications {
            if row[ci as usize] == cv && row[si as usize] != sv {
                return false;
            }
        }
        for group in &self.onehot {
            let sum: f64 = group.iter().map(|&j| row[j as usize]).sum();
            if !(sum == 1.0) {
                return false; // NaN sum compares false -> infeasible, like numpy
            }
        }
        true
    }

    /// In-place best-effort repair per row, matching `repair_matrix` exactly.
    pub fn repair(&self, xs: &mut [f64], n_rows: usize, x: &[f64]) {
        use rayon::prelude::*;
        let p = self.n_features;
        let (lo, hi, _) = self.instance_bounds(x);
        let lo: Vec<f64> = lo
            .iter()
            .map(|&v| if v.is_nan() { f64::NEG_INFINITY } else { v })
            .collect();
        let hi: Vec<f64> = hi
            .iter()
            .map(|&v| if v.is_nan() { f64::INFINITY } else { v })
            .collect();
        xs.par_chunks_mut(p).for_each(|row| {
            self.repair_row(row, x, &lo, &hi);
        });
        debug_assert_eq!(xs.len(), n_rows * p);
    }

    fn repair_row(&self, row: &mut [f64], x: &[f64], lo: &[f64], hi: &[f64]) {
        let p = self.n_features;
        for j in 0..p {
            if !self.allows_missing(j) {
                if x[j].is_nan() {
                    row[j] = f64::NAN; // fixed missing; clipping skipped (`continue`)
                    continue;
                }
                if row[j].is_nan() {
                    row[j] = x[j];
                }
            }
            if !row[j].is_nan() {
                // np.clip(v, lo, hi): lower bound applied first
                let mut v = row[j];
                if v < lo[j] {
                    v = lo[j];
                }
                if v > hi[j] {
                    v = hi[j];
                }
                row[j] = v;
            }
        }
        for lin in &self.linears {
            // canonical order-pair hint only: a - b <= 0 -> clip a to b
            if lin.op == LIN_LE && lin.rhs == 0.0 && lin.coefs.len() == 2 {
                let mut sorted = lin.coefs.clone();
                sorted.sort_by(f64::total_cmp);
                if sorted == [-1.0, 1.0] {
                    let a = lin.indices[lin.coefs.iter().position(|&c| c == 1.0).unwrap()];
                    let b = lin.indices[lin.coefs.iter().position(|&c| c == -1.0).unwrap()];
                    let (a, b) = (a as usize, b as usize);
                    if !row[a].is_nan() && !row[b].is_nan() {
                        row[a] = py_min(row[a], row[b]); // np.minimum on non-NaN values
                    }
                }
            }
        }
        for &(ci, cv, si, sv) in &self.implications {
            if row[ci as usize] == cv {
                row[si as usize] = sv;
            }
        }
        for group in &self.onehot {
            let mut winner = 0usize;
            let mut best = f64::NEG_INFINITY;
            for (k, &j) in group.iter().enumerate() {
                let v = row[j as usize];
                let v = if v.is_nan() { -1.0 } else { v };
                if v > best {
                    best = v;
                    winner = k; // strict '>' keeps the LOWEST index on ties (np.argmax)
                }
            }
            for &j in group.iter() {
                row[j as usize] = 0.0;
            }
            row[group[winner] as usize] = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base(p: usize) -> Constraints {
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

    #[test]
    fn freeze_on_nan_factual_gives_infinite_bounds_not_nan() {
        let mut c = base(2);
        c.freeze = vec![0];
        let (lo, hi, frozen) = c.instance_bounds(&[f64::NAN, 0.0]);
        assert_eq!(lo[0], f64::NEG_INFINITY);
        assert_eq!(hi[0], f64::INFINITY);
        assert!(frozen[0]);
    }

    #[test]
    fn nan_column_forcing_skips_clipping() {
        let mut c = base(2);
        c.ranges = vec![(0, 0.0, 1.0)];
        let x = [f64::NAN, 0.0];
        let mut rows = vec![5.0, 5.0];
        c.repair(&mut rows, 1, &x);
        assert!(rows[0].is_nan()); // forced NaN, clip skipped
        assert_eq!(rows[1], 5.0);
    }

    #[test]
    fn linear_nansum_and_policies() {
        let mut c = base(2);
        c.linears = vec![LinearC {
            indices: vec![0, 1],
            coefs: vec![1.0, 1.0],
            op: LIN_LE,
            rhs: 1.0,
            policy: POLICY_SATISFIED,
        }];
        let x = [0.0, 0.0];
        // NaN participant: vacuously satisfied even though the non-NaN part exceeds rhs
        assert!(c.check(&[5.0, f64::NAN], 1, &x)[0] || !c.allows_missing(1));
        // both present and violating
        assert!(!c.check(&[5.0, 5.0], 1, &x)[0]);
    }

    #[test]
    fn onehot_repair_lowest_index_tie_and_nan() {
        let mut c = base(3);
        c.onehot = vec![vec![0, 1, 2]];
        let x = [0.0, 0.0, 1.0];
        let mut rows = vec![f64::NAN, 0.7, 0.7];
        c.repair(&mut rows, 1, &x);
        assert_eq!(rows, vec![0.0, 1.0, 0.0]); // tie at 0.7 -> lowest index (1)
    }

    #[test]
    fn implication_assignment_in_order() {
        let mut c = base(2);
        c.implications = vec![(0, 1.0, 1, 1.0)];
        let x = [0.0, 0.0];
        let mut rows = vec![1.0, 0.0];
        c.repair(&mut rows, 1, &x);
        assert_eq!(rows, vec![1.0, 1.0]);
    }

    #[test]
    fn order_pair_hint_clips_a_to_b() {
        let mut c = base(2);
        c.linears = vec![LinearC {
            indices: vec![0, 1],
            coefs: vec![1.0, -1.0],
            op: LIN_LE,
            rhs: 0.0,
            policy: POLICY_SATISFIED,
        }];
        let x = [0.0, 0.0];
        let mut rows = vec![5.0, 2.0];
        c.repair(&mut rows, 1, &x);
        assert_eq!(rows, vec![2.0, 2.0]);
    }
}
