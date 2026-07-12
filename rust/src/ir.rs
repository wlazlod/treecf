//! Tree-ensemble IR: flat SoA mirroring `treecf.ir.flatten` (the boundary contract).
//!
//! Semantics must match the Python batch evaluator (`raw_score_batch`) exactly:
//! per-node split op (LT: `v < t` -> left; LE: `v <= t` -> left), NaN routed by
//! `missing_left` (false when the node defines no missing direction), and the
//! score accumulated as `base_score + tree_0 + tree_1 + ...` in order — which
//! makes bitwise parity with numpy achievable (identical f64 addition order).

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Link {
    Identity,
    Sigmoid,
}

pub struct Ensemble {
    pub feature: Vec<i32>, // -1 marks a leaf
    pub threshold: Vec<f64>,
    pub is_lt: Vec<bool>,
    pub missing_left: Vec<bool>,
    pub left: Vec<u32>,
    pub right: Vec<u32>,
    pub value: Vec<f64>,
    pub tree_roots: Vec<u32>,
    pub base_score: f64,
    pub link: Link,
    pub n_features: usize,
}

impl Ensemble {
    /// Validate the flat arrays once; traversal afterwards trusts them.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        feature: Vec<i32>,
        threshold: Vec<f64>,
        is_lt: Vec<bool>,
        missing_left: Vec<bool>,
        left: Vec<u32>,
        right: Vec<u32>,
        value: Vec<f64>,
        tree_roots: Vec<u32>,
        base_score: f64,
        link: Link,
        n_features: usize,
    ) -> Result<Self, String> {
        let n = feature.len();
        for (name, len) in [
            ("threshold", threshold.len()),
            ("is_lt", is_lt.len()),
            ("missing_left", missing_left.len()),
            ("left", left.len()),
            ("right", right.len()),
            ("value", value.len()),
        ] {
            if len != n {
                return Err(format!("array {name} has length {len}, expected {n}"));
            }
        }
        for i in 0..n {
            if feature[i] >= 0 {
                if feature[i] as usize >= n_features {
                    return Err(format!("node {i}: feature index out of range"));
                }
                if left[i] as usize >= n || right[i] as usize >= n {
                    return Err(format!("node {i}: child index out of range"));
                }
            }
        }
        for &root in &tree_roots {
            if root as usize >= n && n > 0 {
                return Err("tree root out of range".to_string());
            }
        }
        Ok(Self {
            feature,
            threshold,
            is_lt,
            missing_left,
            left,
            right,
            value,
            tree_roots,
            base_score,
            link,
            n_features,
        })
    }

    /// Leaf value reached by `x` in the tree rooted at `root`.
    #[inline]
    fn leaf_value(&self, root: u32, x: &[f64]) -> f64 {
        let mut i = root as usize;
        while self.feature[i] >= 0 {
            let v = x[self.feature[i] as usize];
            let go_left = if v.is_nan() {
                self.missing_left[i]
            } else if self.is_lt[i] {
                v < self.threshold[i]
            } else {
                v <= self.threshold[i]
            };
            i = if go_left { self.left[i] } else { self.right[i] } as usize;
        }
        self.value[i]
    }

    /// Raw score of one row: base_score + leaf values, trees in order.
    pub fn raw_score(&self, x: &[f64]) -> f64 {
        let mut total = self.base_score;
        for &root in &self.tree_roots {
            total += self.leaf_value(root, x);
        }
        total
    }

    /// Raw scores for a row-major matrix (n_rows x n_features).
    pub fn raw_score_batch(&self, xs: &[f64], n_rows: usize) -> Vec<f64> {
        let mut out = vec![0.0; n_rows];
        self.raw_score_batch_into(xs, n_rows, &mut out);
        out
    }

    pub fn raw_score_batch_into(&self, xs: &[f64], n_rows: usize, out: &mut [f64]) {
        use rayon::prelude::*;
        let p = self.n_features;
        out.par_iter_mut().enumerate().for_each(|(r, slot)| {
            *slot = self.raw_score(&xs[r * p..(r + 1) * p]);
        });
        debug_assert_eq!(out.len(), n_rows);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stump on feature 0 at 1.0: left leaf -1.0, right leaf +1.0.
    fn stump(is_lt: bool, missing_left: bool) -> Ensemble {
        Ensemble::new(
            vec![0, -1, -1],
            vec![1.0, 0.0, 0.0],
            vec![is_lt, false, false],
            vec![missing_left, false, false],
            vec![1, 0, 0],
            vec![2, 0, 0],
            vec![0.0, -1.0, 1.0],
            vec![0],
            0.0,
            Link::Identity,
            2,
        )
        .unwrap()
    }

    #[test]
    fn lt_at_threshold_goes_right() {
        assert_eq!(stump(true, true).raw_score(&[1.0, 0.0]), 1.0);
    }

    #[test]
    fn le_at_threshold_goes_left() {
        assert_eq!(stump(false, true).raw_score(&[1.0, 0.0]), -1.0);
    }

    #[test]
    fn nextafter_sides_route_correctly() {
        let e = stump(true, true);
        assert_eq!(e.raw_score(&[f64::from(1.0f32).next_down(), 0.0]), -1.0);
        assert_eq!(e.raw_score(&[1.0f64.next_up(), 0.0]), 1.0);
    }

    #[test]
    fn nan_routes_by_missing_left() {
        assert_eq!(stump(true, true).raw_score(&[f64::NAN, 0.0]), -1.0);
        assert_eq!(stump(true, false).raw_score(&[f64::NAN, 0.0]), 1.0);
    }

    #[test]
    fn accumulation_is_base_plus_trees_in_order() {
        let mut e = stump(true, true);
        e.base_score = 0.25;
        assert_eq!(e.raw_score(&[0.0, 0.0]), 0.25 + (-1.0));
    }

    #[test]
    fn batch_matches_single_row() {
        let e = stump(true, false);
        let xs = [0.0, 0.0, 2.0, 0.0, f64::NAN, 0.0];
        let batch = e.raw_score_batch(&xs, 3);
        for r in 0..3 {
            assert_eq!(batch[r], e.raw_score(&xs[r * 2..r * 2 + 2]));
        }
    }
}
