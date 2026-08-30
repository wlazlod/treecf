//! Tree-ensemble IR: flat SoA mirroring `treecf.ir.flatten` (the boundary contract).
//!
//! Semantics must match the Python batch evaluator (`raw_score_batch`) exactly:
//! per-node split op (LT: `v < t` -> left; LE: `v <= t` -> left), NaN routed by
//! `missing_left` (false when the node defines no missing direction), and the
//! score accumulated as `base_score + tree_0 + tree_1 + ...` in order — which
//! makes bitwise parity with numpy achievable (identical f64 addition order).

use std::sync::OnceLock;

use crate::cells::Cell;

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
    // Set-membership splits: node_set[i] >= 0 selects a bitset in the interned
    // table (set_offsets CSR into set_words; code c at word c>>6, bit c&63).
    // cardinality[j] > 0 marks feature j categorical with codes 0..cardinality.
    pub node_set: Vec<i32>,
    pub set_offsets: Vec<u32>,
    pub set_words: Vec<u64>,
    pub cardinality: Vec<u32>,
    cells: OnceLock<Vec<Vec<Cell>>>, // lazy: pure function of the split structure
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
        let node_set = vec![-1; n];
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
            node_set,
            set_offsets: vec![0],
            set_words: Vec::new(),
            cardinality: vec![0; n_features],
            cells: OnceLock::new(),
        })
    }

    /// Install set-membership splits and per-feature cardinalities, validated once.
    pub fn with_categories(
        mut self,
        node_set: Vec<i32>,
        set_offsets: Vec<u32>,
        set_words: Vec<u64>,
        cardinality: Vec<u32>,
    ) -> Result<Self, String> {
        if node_set.len() != self.feature.len() {
            return Err("node_set length differs from the node count".to_string());
        }
        if cardinality.len() != self.n_features {
            return Err("cardinality length differs from n_features".to_string());
        }
        if set_offsets.first() != Some(&0) {
            return Err("set_offsets must start at 0".to_string());
        }
        if set_offsets.windows(2).any(|w| w[1] < w[0]) {
            return Err("set_offsets must be non-decreasing".to_string());
        }
        if *set_offsets.last().unwrap() as usize != set_words.len() {
            return Err("set_offsets must end at set_words length".to_string());
        }
        let n_sets = set_offsets.len() - 1;
        for (i, &sid) in node_set.iter().enumerate() {
            if sid >= 0 {
                if self.feature[i] < 0 {
                    return Err(format!("node {i}: a leaf cannot carry a split set"));
                }
                if sid as usize >= n_sets {
                    return Err(format!("node {i}: set index out of range"));
                }
            }
        }
        self.node_set = node_set;
        self.set_offsets = set_offsets;
        self.set_words = set_words;
        self.cardinality = cardinality;
        Ok(self)
    }

    /// Set-membership routing for a non-NaN value: left iff an integral member.
    /// Non-integral values are never members; codes beyond the stored words
    /// (unseen categories) are not members either.
    #[inline]
    pub fn set_contains(&self, set: i32, v: f64) -> bool {
        if !v.is_finite() {
            return false;
        }
        let code = v as i64;
        if code as f64 != v || code < 0 {
            return false;
        }
        let start = self.set_offsets[set as usize] as usize;
        let end = self.set_offsets[set as usize + 1] as usize;
        let word = (code >> 6) as usize;
        if word >= end - start {
            return false;
        }
        (self.set_words[start + word] >> (code & 63)) & 1 == 1
    }

    /// Routing-atomic cells per feature, computed once and cached.
    pub fn feature_cells(&self) -> &[Vec<Cell>] {
        self.cells.get_or_init(|| crate::cells::feature_cells(self))
    }

    /// Leaf value reached by `x` in the tree rooted at `root`.
    #[inline]
    fn leaf_value(&self, root: u32, x: &[f64]) -> f64 {
        let mut i = root as usize;
        while self.feature[i] >= 0 {
            let v = x[self.feature[i] as usize];
            let go_left = if v.is_nan() {
                self.missing_left[i]
            } else if self.node_set[i] >= 0 {
                self.set_contains(self.node_set[i], v)
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

    /// Raw scores for a row-major matrix (n_rows x n_features). `parallel`
    /// only picks the execution strategy; the stage is RNG-free and per-row
    /// independent, so results are identical either way.
    pub fn raw_score_batch(&self, xs: &[f64], n_rows: usize, parallel: bool) -> Vec<f64> {
        let mut out = vec![0.0; n_rows];
        self.raw_score_batch_into(xs, n_rows, &mut out, parallel);
        out
    }

    pub fn raw_score_batch_into(&self, xs: &[f64], n_rows: usize, out: &mut [f64], parallel: bool) {
        use rayon::prelude::*;
        let p = self.n_features;
        if parallel {
            out.par_iter_mut().enumerate().for_each(|(r, slot)| {
                *slot = self.raw_score(&xs[r * p..(r + 1) * p]);
            });
        } else {
            for (r, slot) in out.iter_mut().enumerate() {
                *slot = self.raw_score(&xs[r * p..(r + 1) * p]);
            }
        }
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
        let batch = e.raw_score_batch(&xs, 3, true);
        for r in 0..3 {
            assert_eq!(batch[r], e.raw_score(&xs[r * 2..r * 2 + 2]));
        }
    }

    /// Set stump on feature 0: left leaf -1.0 iff the code is in `words`.
    fn set_stump(words: Vec<u64>, cardinality: u32, missing_left: bool) -> Ensemble {
        Ensemble::new(
            vec![0, -1, -1],
            vec![0.0, 0.0, 0.0],
            vec![false, false, false],
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
        .with_categories(
            vec![0, -1, -1],
            vec![0, words.len() as u32],
            words,
            vec![cardinality, 0],
        )
        .unwrap()
    }

    #[test]
    fn member_code_goes_left_nonmember_right() {
        let e = set_stump(vec![0b101], 4, true); // {0, 2}
        assert_eq!(e.raw_score(&[0.0, 0.0]), -1.0);
        assert_eq!(e.raw_score(&[2.0, 0.0]), -1.0);
        assert_eq!(e.raw_score(&[1.0, 0.0]), 1.0);
        assert_eq!(e.raw_score(&[3.0, 0.0]), 1.0);
    }

    #[test]
    fn unseen_and_non_integral_codes_go_right() {
        let e = set_stump(vec![0b101], 4, true);
        assert_eq!(e.raw_score(&[7.0, 0.0]), 1.0); // beyond the stored words
        assert_eq!(e.raw_score(&[2.5, 0.0]), 1.0); // not a code
        assert_eq!(e.raw_score(&[-1.0, 0.0]), 1.0);
        assert_eq!(e.raw_score(&[f64::INFINITY, 0.0]), 1.0);
    }

    #[test]
    fn set_nan_routes_by_missing_left() {
        assert_eq!(
            set_stump(vec![0b10], 4, true).raw_score(&[f64::NAN, 0.0]),
            -1.0
        );
        assert_eq!(
            set_stump(vec![0b10], 4, false).raw_score(&[f64::NAN, 0.0]),
            1.0
        );
    }

    #[test]
    fn membership_across_word_boundary() {
        // {63, 64, 100} over 128 codes: two words
        let e = set_stump(vec![1u64 << 63, (1u64 << 0) | (1u64 << 36)], 128, true);
        assert_eq!(e.raw_score(&[63.0, 0.0]), -1.0);
        assert_eq!(e.raw_score(&[64.0, 0.0]), -1.0);
        assert_eq!(e.raw_score(&[100.0, 0.0]), -1.0);
        assert_eq!(e.raw_score(&[65.0, 0.0]), 1.0);
        assert_eq!(e.raw_score(&[127.0, 0.0]), 1.0);
    }

    #[test]
    fn with_categories_validates_shapes() {
        let numeric = stump(true, true);
        assert!(numeric
            .with_categories(vec![0, -1], vec![0, 1], vec![1], vec![4, 0])
            .is_err()); // node_set length mismatch
        let numeric = stump(true, true);
        assert!(numeric
            .with_categories(vec![-1, 0, -1], vec![0, 1], vec![1], vec![4, 0])
            .is_err()); // a leaf cannot carry a set
    }
}
