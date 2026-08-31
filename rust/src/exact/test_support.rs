//! Builders and assertions shared by the four modules' tests. Kept in one
//! place so a golden's own doc comment can stay attached to the test it pins.

use crate::cells::Cell;
use crate::constraints::Constraints;
use crate::exact::domains::{build_domains, constraint_cells, State};
use crate::exact::search::solve_exact;
use crate::exact::{ExactParams, ExactResult, ValuePolicy};
use crate::interrupt::{InterruptProbe, SearchOutcome};
use crate::ir::{Ensemble, Link};

// ------------------------------------------------------------ builders ---

/// One stump per spec `(feature, threshold, is_lt, left_value, right_value)`,
/// missing routed right unless `missing_left` is asked for.
pub(crate) fn stumps(specs: &[(i32, f64, bool, f64, f64)], n_features: usize) -> Ensemble {
    stumps_missing(specs, n_features, false)
}

pub(crate) fn stumps_missing(
    specs: &[(i32, f64, bool, f64, f64)],
    n_features: usize,
    missing_left: bool,
) -> Ensemble {
    let mut feature = Vec::new();
    let mut threshold = Vec::new();
    let mut is_lt = Vec::new();
    let mut miss = Vec::new();
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut value = Vec::new();
    let mut roots = Vec::new();
    for (t, &(f, thr, lt, lv, rv)) in specs.iter().enumerate() {
        let base = (t * 3) as u32;
        roots.push(base);
        feature.extend_from_slice(&[f, -1, -1]);
        threshold.extend_from_slice(&[thr, 0.0, 0.0]);
        is_lt.extend_from_slice(&[lt, false, false]);
        miss.extend_from_slice(&[missing_left, false, false]);
        left.extend_from_slice(&[base + 1, 0, 0]);
        right.extend_from_slice(&[base + 2, 0, 0]);
        value.extend_from_slice(&[0.0, lv, rv]);
    }
    Ensemble::new(
        feature,
        threshold,
        is_lt,
        miss,
        left,
        right,
        value,
        roots,
        0.0,
        Link::Identity,
        n_features,
    )
    .unwrap()
}

pub(crate) fn cons_base(p: usize) -> Constraints {
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
        allowed_categories: vec![],
    }
}

pub(crate) fn no_policies(p: usize) -> Vec<Option<ValuePolicy>> {
    vec![None; p]
}

pub(crate) fn cell(lo: f64, hi: f64, lo_open: bool, hi_open: bool) -> Cell {
    Cell {
        lo,
        hi,
        lo_open,
        hi_open,
    }
}

/// Deterministic LCG — fixed seeds, no external rng in tests.
pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    pub(crate) fn next_usize(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as usize) % bound
    }
}

/// `(value bits, cost bits, cell index, is_nan, snapped)` per state, in order.
pub(crate) fn expect_states(got: &[State], want: &[(u64, u64, usize, bool, bool)]) {
    assert_eq!(got.len(), want.len(), "state count: {got:?}");
    for (k, (state, &(value, cost, cell_idx, is_nan, snapped))) in got.iter().zip(want).enumerate()
    {
        assert_eq!(state.value.to_bits(), value, "state {k} value: {state:?}");
        assert_eq!(state.cost.to_bits(), cost, "state {k} cost: {state:?}");
        assert_eq!(state.cell_idx, cell_idx, "state {k} cell: {state:?}");
        assert_eq!(state.is_nan, is_nan, "state {k} is_nan: {state:?}");
        assert_eq!(state.snapped, snapped, "state {k} snapped: {state:?}");
    }
}

/// The two-feature grid every domain golden below is drawn from:
/// f0 cells `(-inf,1)`, `[1,3]`, `(3,inf)`; f1 one cell.
pub(crate) fn golden_ens() -> Ensemble {
    stumps(&[(0, 1.0, true, 0.0, 1.0), (0, 3.0, false, 0.0, 1.0)], 2)
}

pub(crate) fn domains_of(
    ens: &Ensemble,
    x: &[f64],
    cons: &Constraints,
    lam: f64,
    policies: &[Option<ValuePolicy>],
) -> Vec<Vec<State>> {
    let grids = constraint_cells(cons, &[ens]);
    let blocks = crate::cells::category_blocks_joint(&[ens]);
    build_domains(
        &grids,
        x,
        cons,
        &[1.0, 1.0],
        &[1.0, 1.0],
        lam,
        policies,
        &blocks,
    )
}

pub(crate) const NAN_BITS: u64 = 0x7ff8000000000000;

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve(
    ens: &Ensemble,
    x: &[f64],
    interval: (f64, f64),
    cons: &Constraints,
    lam: f64,
    policies: &[Option<ValuePolicy>],
    params: &ExactParams,
    incumbent: Option<(f64, &[f64])>,
) -> ExactResult {
    match solve_probed(
        ens,
        x,
        interval,
        cons,
        lam,
        policies,
        params,
        incumbent,
        &mut || false,
    ) {
        SearchOutcome::Done(result) => result,
        SearchOutcome::Interrupted => unreachable!("no-op probe never interrupts"),
    }
}

/// `solve` with a probe of the caller's own — for the tests that care what the
/// probe is asked and what happens when it says yes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_probed(
    ens: &Ensemble,
    x: &[f64],
    interval: (f64, f64),
    cons: &Constraints,
    lam: f64,
    policies: &[Option<ValuePolicy>],
    params: &ExactParams,
    incumbent: Option<(f64, &[f64])>,
    probe: InterruptProbe<'_>,
) -> SearchOutcome<ExactResult> {
    let p = ens.n_features;
    solve_exact(
        ens,
        x,
        interval,
        cons,
        &vec![1.0; p],
        &vec![1.0; p],
        lam,
        policies,
        None,
        params,
        incumbent,
        probe,
    )
    .unwrap()
}

pub(crate) fn bits_of(row: &[f64]) -> Vec<u64> {
    row.iter().map(|v| v.to_bits()).collect()
}
