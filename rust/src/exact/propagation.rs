//! Domain settling for the exact backend: implications and one-hot groups —
//! port of `treecf.backends._exact_propagation`. Parity rules in the module
//! header of `super` govern this file too.

use crate::constraints::Constraints;
use crate::exact::domains::State;

// ------------------------------------------------------------ propagation ---

/// What one assignment changed: the features it settled (with their previous
/// setting) and the one-hot counters it moved (with their previous readings) —
/// old values rather than deltas, so putting them back cannot drift.
#[derive(Clone, Debug, Default)]
pub(crate) struct PropFrame {
    settled: Vec<(usize, Option<f64>)>,
    counters: Vec<(usize, usize, usize)>,
}

/// What assigning one feature settles about the features that follow it:
/// an implication's consequence, and a one-hot group's last free member.
/// One-hot bookkeeping only runs for groups whose members hold nothing but 0
/// and 1; anything else is left to the arbiter alone.
pub(crate) struct Propagation<'a> {
    implications: &'a [(u32, f64, u32, f64)],
    groups: &'a [Vec<u32>],
    group_of: Vec<Option<usize>>,
    pub(crate) forced_value: Vec<Option<f64>>,
    pub(crate) ones: Vec<usize>,
    pub(crate) zeros: Vec<usize>,
}

fn force(
    forced_value: &mut [Option<f64>],
    frame: &mut PropFrame,
    assigned: &[bool],
    values: &[f64],
    f: usize,
    value: f64,
) -> bool {
    if assigned[f] {
        return values[f] == value;
    }
    if let Some(current) = forced_value[f] {
        return current == value;
    }
    frame.settled.push((f, None));
    forced_value[f] = Some(value);
    true
}

impl<'a> Propagation<'a> {
    pub(crate) fn new(cons: &'a Constraints, domains: &[Vec<State>]) -> Self {
        let mut group_of: Vec<Option<usize>> = vec![None; cons.n_features];
        for (g_idx, group) in cons.onehot.iter().enumerate() {
            let binary = group.iter().all(|&f| {
                domains[f as usize]
                    .iter()
                    .all(|s| s.value == 0.0 || s.value == 1.0)
            });
            if binary {
                for &f in group {
                    group_of[f as usize] = Some(g_idx);
                }
            }
        }
        Self {
            implications: &cons.implications,
            groups: &cons.onehot,
            group_of,
            forced_value: vec![None; cons.n_features],
            ones: vec![0; cons.onehot.len()],
            zeros: vec![0; cons.onehot.len()],
        }
    }

    /// Settle what follows from `j` taking `v`; report any contradiction. The
    /// changes made before a contradiction are still reported — the caller
    /// restores the frame either way.
    pub(crate) fn apply(
        &mut self,
        j: usize,
        v: f64,
        assigned: &[bool],
        values: &[f64],
    ) -> (PropFrame, bool) {
        let mut frame = PropFrame::default();
        if let Some(forced) = self.forced_value[j] {
            if v != forced {
                return (frame, true);
            }
        }
        let groups = self.groups;
        if let Some(g_idx) = self.group_of[j] {
            let group = &groups[g_idx];
            frame
                .counters
                .push((g_idx, self.ones[g_idx], self.zeros[g_idx]));
            if v == 1.0 {
                self.ones[g_idx] += 1;
            } else {
                self.zeros[g_idx] += 1;
            }
            // a second one, or nothing but zeros: the group can no longer sum to
            // one. The all-zeros reading is a backstop — settling the last free
            // member normally catches that case one assignment earlier.
            if self.ones[g_idx] > 1 || self.zeros[g_idx] == group.len() {
                return (frame, true);
            }
            if self.ones[g_idx] == 0 && self.zeros[g_idx] == group.len() - 1 {
                let last = group
                    .iter()
                    .map(|&f| f as usize)
                    .find(|&f| f != j && !assigned[f])
                    .expect("a group one short of all-zeros has a free member");
                if !force(
                    &mut self.forced_value,
                    &mut frame,
                    assigned,
                    values,
                    last,
                    1.0,
                ) {
                    return (frame, true);
                }
            }
        }
        for &(cond_index, cond_value, cons_index, cons_value) in self.implications {
            let triggered = cond_index as usize == j && v == cond_value;
            if triggered
                && !force(
                    &mut self.forced_value,
                    &mut frame,
                    assigned,
                    values,
                    cons_index as usize,
                    cons_value,
                )
            {
                return (frame, true);
            }
        }
        (frame, false)
    }

    /// Put back everything one `apply` settled.
    pub(crate) fn restore(&mut self, frame: &PropFrame) {
        for &(f, previous) in frame.settled.iter().rev() {
            self.forced_value[f] = previous;
        }
        for &(g_idx, ones, zeros) in frame.counters.iter().rev() {
            self.ones[g_idx] = ones;
            self.zeros[g_idx] = zeros;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::test_support::cons_base;

    // ------------------------------------------------------- propagation ---

    fn binary_domains(p: usize) -> Vec<Vec<State>> {
        vec![
            vec![
                State::new(0.0, 0.0, 0, false),
                State::new(1.0, 1.0, 0, false),
            ];
            p
        ]
    }

    #[test]
    fn onehot_counters_force_the_last_member_and_reject_a_second_one() {
        let mut cons = cons_base(3);
        cons.onehot = vec![vec![0, 1, 2]];
        let domains = binary_domains(3);
        let mut prop = Propagation::new(&cons, &domains);
        let mut assigned = vec![false; 3];
        let values = vec![0.0; 3];

        let (frame0, conflict) = prop.apply(0, 0.0, &assigned, &values);
        assert!(!conflict);
        assigned[0] = true;
        // the second zero leaves one free member, which is forced to 1.0
        let (frame1, conflict) = prop.apply(1, 0.0, &assigned, &values);
        assert!(!conflict);
        assert_eq!(prop.forced_value[2], Some(1.0));
        assert_eq!(frame1.settled, vec![(2, None)]);
        // a state contradicting that settlement is cut
        let (frame2, conflict) = prop.apply(2, 0.0, &assigned, &values);
        assert!(conflict);
        prop.restore(&frame2);
        prop.restore(&frame1);
        assert_eq!(prop.forced_value[2], None);
        assert_eq!((prop.ones[0], prop.zeros[0]), (0, 1));
        prop.restore(&frame0);
        assert_eq!((prop.ones[0], prop.zeros[0]), (0, 0));

        // two ones in one group: cut on the spot
        let (_, conflict) = prop.apply(0, 1.0, &assigned, &values);
        assert!(!conflict);
        assigned[0] = true;
        let mut values = values.clone();
        values[0] = 1.0;
        let (_, conflict) = prop.apply(1, 1.0, &assigned, &values);
        assert!(conflict);
    }

    /// A group whose members can hold something other than 0/1 is left to the
    /// arbiter: no counters, no forcing.
    #[test]
    fn non_binary_group_is_not_counted() {
        let mut cons = cons_base(2);
        cons.onehot = vec![vec![0, 1]];
        let mut domains = binary_domains(2);
        domains[1].push(State::new(0.5, 1.0, 0, false));
        let prop = Propagation::new(&cons, &domains);
        assert_eq!(prop.group_of, vec![None, None]);
    }

    #[test]
    fn implication_settles_its_consequence_and_conflicts_with_a_different_value() {
        let mut cons = cons_base(2);
        cons.implications = vec![(0, 1.0, 1, 1.0)];
        let domains = binary_domains(2);
        let mut prop = Propagation::new(&cons, &domains);
        let mut assigned = vec![false; 2];
        let mut values = vec![0.0; 2];

        let (frame, conflict) = prop.apply(0, 0.0, &assigned, &values); // silent
        assert!(!conflict);
        assert_eq!(prop.forced_value[1], None);
        prop.restore(&frame);

        let (frame, conflict) = prop.apply(0, 1.0, &assigned, &values);
        assert!(!conflict);
        assert_eq!(prop.forced_value[1], Some(1.0));
        assigned[0] = true;
        values[0] = 1.0;
        let (deeper, conflict) = prop.apply(1, 0.0, &assigned, &values);
        assert!(conflict);
        prop.restore(&deeper);
        prop.restore(&frame);
        assert_eq!(prop.forced_value[1], None);

        // an already-assigned consequence is checked against, not re-settled
        let mut assigned = vec![false; 2];
        let mut values = vec![0.0; 2];
        assigned[1] = true;
        values[1] = 0.0;
        let (_, conflict) = prop.apply(0, 1.0, &assigned, &values);
        assert!(conflict);
        values[1] = 1.0;
        let (_, conflict) = prop.apply(0, 1.0, &assigned, &values);
        assert!(!conflict);
    }
}
