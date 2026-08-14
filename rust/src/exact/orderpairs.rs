//! Cell arithmetic the order-pair rules are built on — port of
//! `treecf.backends._exact_orderpairs`. The leaf of the four: it imports
//! nothing from the other three. Parity rules in the module header of `super`
//! govern this file too.

use crate::cells::Cell;

// -------------------------------------------------------- cell arithmetic ---

/// `cell` ∩ `[lo, hi]` (closed bounds); `None` when empty, degenerate open
/// singletons included.
pub(crate) fn intersect_cell(cell: &Cell, lo: f64, hi: f64) -> Option<Cell> {
    let (new_lo, new_lo_open) = if cell.lo > lo {
        (cell.lo, cell.lo_open)
    } else if lo > cell.lo {
        (lo, false)
    } else {
        (cell.lo, cell.lo_open)
    };
    let (new_hi, new_hi_open) = if cell.hi < hi {
        (cell.hi, cell.hi_open)
    } else if hi < cell.hi {
        (hi, false)
    } else {
        (cell.hi, cell.hi_open)
    };
    if new_lo > new_hi {
        return None;
    }
    if new_lo == new_hi && (new_lo_open || new_hi_open) {
        return None;
    }
    Some(Cell {
        lo: new_lo,
        hi: new_hi,
        lo_open: new_lo_open,
        hi_open: new_hi_open,
    })
}

/// `first` ∩ `second`, tighter edge per side, open edge winning at equal values.
fn intersect_cells(first: &Cell, second: &Cell) -> Option<Cell> {
    let (lo, lo_open) = if first.lo > second.lo {
        (first.lo, first.lo_open)
    } else if second.lo > first.lo {
        (second.lo, second.lo_open)
    } else {
        (first.lo, first.lo_open || second.lo_open)
    };
    let (hi, hi_open) = if first.hi < second.hi {
        (first.hi, first.hi_open)
    } else if second.hi < first.hi {
        (second.hi, second.hi_open)
    } else {
        (first.hi, first.hi_open || second.hi_open)
    };
    if lo > hi {
        return None;
    }
    if lo == hi && (lo_open || hi_open) {
        return None;
    }
    Some(Cell {
        lo,
        hi,
        lo_open,
        hi_open,
    })
}

/// Lowest and highest value a cell can actually take: a finite open edge steps
/// one f32 ulp inside (the step `nearest_to` takes), an infinite one stays.
pub(crate) fn achievable_bounds(cell: &Cell) -> (f64, f64) {
    let mut lo = cell.lo;
    if cell.lo_open && lo != f64::NEG_INFINITY {
        lo = cell.nearest_to(cell.lo);
    }
    let mut hi = cell.hi;
    if cell.hi_open && hi != f64::INFINITY {
        hi = cell.nearest_to(cell.hi);
    }
    (lo, hi)
}

/// Values worth trying when a pair `a <= b` has to be pulled onto the boundary
/// `a' == b' == t`: factual of `a`, factual of `b`, the demanded values as
/// given, low end, high end — in that fixed order, non-finite and out-of-interval
/// values dropped, first occurrence winning.
pub(crate) fn boundary_candidates(
    cell_a: &Cell,
    cell_b: &Cell,
    x_a: f64,
    x_b: f64,
    demanded: &[f64],
) -> Vec<f64> {
    let Some(iv) = intersect_cells(cell_a, cell_b) else {
        return Vec::new();
    };
    let (ach_lo, ach_hi) = achievable_bounds(&iv);
    let mut out: Vec<f64> = Vec::new();
    let mut offer = |t: f64| {
        if !t.is_finite() || !iv.contains(t) || out.contains(&t) {
            return;
        }
        out.push(t);
    };
    offer(x_a);
    offer(x_b);
    for &t in demanded {
        offer(t);
    }
    offer(ach_lo);
    offer(ach_hi);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::test_support::cell;

    // ------------------------------------------------------ cell arithmetic ---

    #[test]
    fn intersect_cell_keeps_openness_and_drops_degenerate_singletons() {
        let c = cell(0.0, 2.0, true, true);
        assert_eq!(
            intersect_cell(&c, 1.0, 3.0),
            Some(cell(1.0, 2.0, false, true))
        );
        assert_eq!(
            intersect_cell(&c, -1.0, 1.0),
            Some(cell(0.0, 1.0, true, false))
        );
        assert_eq!(intersect_cell(&c, 2.0, 5.0), None); // open edge, empty
        assert_eq!(intersect_cell(&c, 5.0, 6.0), None);
    }

    #[test]
    fn achievable_bounds_step_one_f32_ulp_inside_finite_open_edges() {
        let c = cell(0.0, 1.0, true, true);
        let (lo, hi) = achievable_bounds(&c);
        assert_eq!(lo, f64::from((0.0f32).next_up()));
        assert_eq!(hi, f64::from((1.0f32).next_down()));
        let unbounded = cell(f64::NEG_INFINITY, f64::INFINITY, true, true);
        assert_eq!(
            achievable_bounds(&unbounded),
            (f64::NEG_INFINITY, f64::INFINITY)
        );
    }

    #[test]
    fn boundary_candidates_keep_their_fixed_order_and_drop_repeats() {
        let a = cell(0.0, 10.0, false, false);
        let b = cell(2.0, 6.0, false, false);
        // x_a, x_b, demanded, low end, high end; 2.0 offered twice, kept once
        let got = boundary_candidates(&a, &b, 3.0, 2.0, &[5.0, 2.0]);
        assert_eq!(got, vec![3.0, 2.0, 5.0, 6.0]);
        // out-of-interval and non-finite candidates drop out
        let got = boundary_candidates(&a, &b, 99.0, f64::INFINITY, &[]);
        assert_eq!(got, vec![2.0, 6.0]);
        // disjoint cells: nothing to try
        assert!(
            boundary_candidates(&a, &cell(20.0, 30.0, false, false), 1.0, 25.0, &[]).is_empty()
        );
    }
}
