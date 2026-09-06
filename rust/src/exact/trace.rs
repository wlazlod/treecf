//! The certification trace — port of `treecf.backends._exact_trace`.
//!
//! `(nodes_expanded, incumbent_cost, lower_bound)` at every incumbent update
//! and every power-of-two node count, plus one terminal sample; capped at
//! `CAP` entries by dropping every second non-incumbent sample before the next
//! append. Sample for sample identical to the Python list.

/// One sample; `incumbent` is `None` while no row has been found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceSample {
    pub nodes: u64,
    pub incumbent: Option<f64>,
    pub bound: f64,
}

pub const CAP: usize = 256;

#[derive(Clone, Debug, Default)]
pub struct Trace {
    samples: Vec<TraceSample>,
    incumbent_flags: Vec<bool>,
}

impl Trace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, nodes: u64, incumbent: Option<f64>, bound: f64, is_incumbent: bool) {
        let sample = TraceSample {
            nodes,
            incumbent,
            bound,
        };
        if let Some(last) = self.samples.last_mut() {
            if last.nodes == nodes {
                *last = sample;
                let flag = self.incumbent_flags.last_mut().expect("parallel vectors");
                *flag = *flag || is_incumbent;
                return;
            }
        }
        if self.samples.len() >= CAP {
            let mut kept = Vec::with_capacity(CAP);
            let mut flags = Vec::with_capacity(CAP);
            let mut plain_seen = 0usize;
            for (existing, &flag) in self.samples.iter().zip(self.incumbent_flags.iter()) {
                if flag {
                    kept.push(*existing);
                    flags.push(true);
                    continue;
                }
                if plain_seen % 2 == 0 {
                    kept.push(*existing);
                    flags.push(false);
                }
                plain_seen += 1;
            }
            self.samples = kept;
            self.incumbent_flags = flags;
        }
        self.samples.push(sample);
        self.incumbent_flags.push(is_incumbent);
    }

    pub fn into_samples(self) -> Vec<TraceSample> {
        self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_node_count_replaces_the_last_sample() {
        let mut trace = Trace::new();
        trace.record(4, None, 0.5, false);
        trace.record(4, Some(2.0), 0.5, true);
        assert_eq!(
            trace.into_samples(),
            vec![TraceSample {
                nodes: 4,
                incumbent: Some(2.0),
                bound: 0.5
            }]
        );
    }

    #[test]
    fn cap_thins_plain_samples_and_keeps_incumbents() {
        let mut trace = Trace::new();
        for n in 1..=(CAP as u64) {
            let inc = n % 50 == 0;
            trace.record(n, if inc { Some(n as f64) } else { None }, 0.0, inc);
        }
        trace.record(CAP as u64 + 1, None, 0.0, false);
        let samples = trace.into_samples();
        assert!(samples.len() < CAP);
        let incumbents: Vec<u64> = samples
            .iter()
            .filter(|s| s.incumbent.is_some())
            .map(|s| s.nodes)
            .collect();
        assert_eq!(incumbents, vec![50, 100, 150, 200, 250]);
        assert_eq!(samples.last().unwrap().nodes, CAP as u64 + 1);
    }
}
