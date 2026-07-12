//! Genetic counterfactual search — a faithful structural port of
//! `treecf.backends.genetic.solve_genetic` (see RUST_MIGRATION_AUDIT.md §2-3).
//!
//! RNG: one sequential Pcg64Mcg stream (statistical parity with numpy, D-H6).
//! Rayon parallelizes only RNG-free stages (fitness/check/repair), so results
//! are identical across thread counts by construction — the audit showed child
//! creation is 1-4 % of wall time, so sequential variation costs little.

use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg64Mcg;

use crate::cells::feature_cells;
use crate::constraints::Constraints;
use crate::ir::Ensemble;

pub struct GaParams {
    pub population: usize,
    pub max_generations: usize,
    pub stall_generations: usize,
    pub time_budget_s: f64,
}

pub struct GaResult {
    pub x_cf: Option<Vec<f64>>,
    pub generations: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn solve_genetic(
    ens: &Ensemble,
    x: &[f64],
    lo_t: f64,
    hi_t: f64,
    cons: &Constraints,
    sigma: &[f64],
    weights: &[f64],
    lam: f64,
    background: Option<(&[f64], usize)>, // (row-major data, n_rows)
    plausibility: Option<(&Ensemble, f64)>,
    seed: Option<u64>,
    params: &GaParams,
) -> GaResult {
    let p = ens.n_features;
    let mut rng = match seed {
        Some(s) => Pcg64Mcg::seed_from_u64(s),
        None => Pcg64Mcg::from_os_rng(),
    };
    let normal = Normal::new(0.0, 1.0).unwrap();

    let (_lo_b, _hi_b, frozen) = cons.instance_bounds(x);
    let fixed: Vec<bool> = (0..p)
        .map(|j| frozen[j] || (x[j].is_nan() && !cons.allows_missing(j)))
        .collect();
    let mutable: Vec<usize> = (0..p).filter(|&j| !fixed[j]).collect();
    let can_be_nan: Vec<bool> = (0..p)
        .map(|j| cons.allows_missing(j) && !fixed[j])
        .collect();
    let deltas: Vec<(f64, f64)> = (0..p)
        .map(|j| {
            cons.allow_missing
                .iter()
                .find(|&&(idx, _, _)| idx as usize == j)
                .map(|&(_, to, from)| (to, from))
                .unwrap_or((0.0, 0.0))
        })
        .collect();

    // per-feature candidate pools: nearest point of every MODEL cell to the anchor
    let anchor: Vec<f64> = x
        .iter()
        .map(|&v| if v.is_nan() { 0.0 } else { v })
        .collect();
    let all_cells = feature_cells(ens);
    let pools: Vec<Vec<f64>> = (0..p)
        .map(|j| {
            if fixed[j] {
                Vec::new()
            } else {
                all_cells[j]
                    .iter()
                    .map(|c| c.nearest_to(anchor[j]))
                    .collect()
            }
        })
        .collect();

    let mutate_value =
        |rng: &mut Pcg64Mcg, current: f64, pool: &[f64], sigma_j: f64, nan_allowed: bool| -> f64 {
            let roll: f64 = rng.random();
            if nan_allowed && roll < 0.15 {
                return f64::NAN;
            }
            if !pool.is_empty() && roll < 0.6 {
                return pool[rng.random_range(0..pool.len())];
            }
            let base = if current.is_nan() { 0.0 } else { current };
            base + normal.sample(rng) * sigma_j.max(1e-9)
        };

    // --- initialization: factual + single-feature cell moves + NaN flips + background mixes ---
    let mut pop: Vec<f64> = Vec::new();
    let mut push_row = |pop: &mut Vec<f64>, row: &[f64]| pop.extend_from_slice(row);
    push_row(&mut pop, x);
    for &j in &mutable {
        for &value in &pools[j] {
            let mut row = x.to_vec();
            row[j] = value;
            push_row(&mut pop, &row);
        }
        if can_be_nan[j] {
            let mut row = x.to_vec();
            row[j] = f64::NAN;
            push_row(&mut pop, &row);
        }
    }
    if let Some((bg, n_bg)) = background {
        if n_bg > 0 {
            let take = 20.min(n_bg);
            for _ in 0..take {
                let r = rng.random_range(0..n_bg);
                let bg_row = &bg[r * p..(r + 1) * p];
                let mut row = x.to_vec();
                for &j in &mutable {
                    if rng.random::<f64>() < 0.5 {
                        row[j] = bg_row[j];
                    }
                }
                push_row(&mut pop, &row);
            }
        }
    }
    let n_seeds = pop.len() / p;
    let extra = (params.population.saturating_sub(n_seeds)).max(10);
    for _ in 0..extra {
        let mut row = x.to_vec();
        if !mutable.is_empty() {
            let k = rng.random_range(1..(mutable.len() + 1).max(2));
            let picks = sample_without_replacement(&mut rng, &mutable, k.min(mutable.len()));
            for jj in picks {
                row[jj] = mutate_value(&mut rng, x[jj], &pools[jj], sigma[jj], can_be_nan[jj]);
            }
        }
        push_row(&mut pop, &row);
    }
    let mut n_rows = pop.len() / p;
    cons.repair(&mut pop, n_rows, x);
    pin_fixed(&mut pop, n_rows, p, &fixed, x);

    // --- main loop ---
    let mut best: Option<Vec<f64>> = None;
    let mut best_j = f64::INFINITY;
    let mut stall = 0usize;
    let start = Instant::now();
    let mut generations = 0usize;

    for _gen in 0..params.max_generations {
        generations += 1;
        let (tier, key) = rank_keys(
            ens,
            &pop,
            n_rows,
            x,
            lo_t,
            hi_t,
            cons,
            sigma,
            weights,
            lam,
            &deltas,
            plausibility,
        );
        let mut order: Vec<usize> = (0..n_rows).collect();
        order.sort_by(|&a, &b| tier[a].cmp(&tier[b]).then(key[a].total_cmp(&key[b])));
        let sorted: Vec<f64> = order
            .iter()
            .flat_map(|&r| pop[r * p..(r + 1) * p].iter().copied())
            .collect();
        pop = sorted;
        let best_tier = tier[order[0]];
        let best_key = key[order[0]];

        if best_tier == 0 && best_key < best_j - 1e-12 {
            best = Some(pop[..p].to_vec());
            best_j = best_key;
            stall = 0;
        } else {
            stall += 1;
        }
        if stall >= params.stall_generations || start.elapsed().as_secs_f64() > params.time_budget_s
        {
            break;
        }

        let n_elite = (params.population / 8).max(4);
        let n_children = params.population.saturating_sub(n_elite);
        let mut next = pop[..n_elite.min(n_rows) * p].to_vec();
        while next.len() / p < n_elite.min(n_rows) + n_children {
            let half = (n_rows / 2).max(2);
            let a = rng.random_range(0..half);
            let b = rng.random_range(0..half);
            let mut child = vec![0.0; p];
            for j in 0..p {
                let take_a = rng.random::<f64>() < 0.5;
                child[j] = if take_a {
                    pop[a * p + j]
                } else {
                    pop[b * p + j]
                };
            }
            for &jj in &mutable {
                let roll: f64 = rng.random();
                if roll < 0.15 {
                    child[jj] =
                        mutate_value(&mut rng, child[jj], &pools[jj], sigma[jj], can_be_nan[jj]);
                } else if roll < 0.30 {
                    child[jj] = x[jj]; // revert-to-factual: drives sparsity
                }
            }
            next.extend_from_slice(&child);
        }
        pop = next;
        n_rows = pop.len() / p;
        cons.repair(&mut pop, n_rows, x);
        pin_fixed(&mut pop, n_rows, p, &fixed, x);
    }

    GaResult {
        x_cf: best,
        generations,
    }
}

fn pin_fixed(pop: &mut [f64], n_rows: usize, p: usize, fixed: &[bool], x: &[f64]) {
    for r in 0..n_rows {
        for j in 0..p {
            if fixed[j] {
                pop[r * p + j] = x[j];
            }
        }
    }
}

fn sample_without_replacement(rng: &mut Pcg64Mcg, from: &[usize], k: usize) -> Vec<usize> {
    // Fisher-Yates partial shuffle over a copy (statistical parity; numpy's
    // permutation-based choice(replace=False) is equivalent in distribution)
    let mut items = from.to_vec();
    let n = items.len();
    for i in 0..k.min(n) {
        let j = rng.random_range(i..n);
        items.swap(i, j);
    }
    items.truncate(k.min(n));
    items
}

#[allow(clippy::too_many_arguments)]
fn rank_keys(
    ens: &Ensemble,
    pop: &[f64],
    n_rows: usize,
    x: &[f64],
    lo_t: f64,
    hi_t: f64,
    cons: &Constraints,
    sigma: &[f64],
    weights: &[f64],
    lam: f64,
    deltas: &[(f64, f64)],
    plausibility: Option<(&Ensemble, f64)>,
) -> (Vec<u8>, Vec<f64>) {
    let p = ens.n_features;
    let scores = ens.raw_score_batch(pop, n_rows);
    let mut ok = cons.check(pop, n_rows, x);
    if let Some((if_ens, min_total)) = plausibility {
        let paths = if_ens.raw_score_batch(pop, n_rows);
        for r in 0..n_rows {
            ok[r] = ok[r] && paths[r] >= min_total;
        }
    }
    let mut tier = vec![0u8; n_rows];
    let mut key = vec![0.0f64; n_rows];
    for r in 0..n_rows {
        let s = scores[r];
        let target_ok = s >= lo_t && s <= hi_t;
        tier[r] = if ok[r] && target_ok {
            0
        } else if ok[r] {
            1
        } else {
            2
        };
        if tier[r] == 0 {
            key[r] = objective_row(&pop[r * p..(r + 1) * p], x, sigma, weights, lam, deltas);
        } else {
            let gap = (lo_t - s).max(0.0) + (s - hi_t).max(0.0);
            // np.nan_to_num(gap, posinf=1e18): NaN -> 0.0, +inf -> 1e18
            key[r] = if gap.is_nan() {
                0.0
            } else if gap == f64::INFINITY {
                1e18
            } else {
                gap
            };
        }
    }
    (tier, key)
}

fn objective_row(
    row: &[f64],
    x: &[f64],
    sigma: &[f64],
    weights: &[f64],
    lam: f64,
    deltas: &[(f64, f64)],
) -> f64 {
    let mut total = 0.0;
    for j in 0..row.len() {
        let x_nan = x[j].is_nan();
        let col_nan = row[j].is_nan();
        if x_nan {
            if !col_nan {
                total += weights[j] * deltas[j].1 / sigma[j] + lam; // NaN -> value
            }
        } else {
            if col_nan {
                total += weights[j] * deltas[j].0 / sigma[j] + lam; // value -> NaN
            }
            let moved = !col_nan && row[j] != x[j];
            if moved {
                total += lam + weights[j] * (row[j] - x[j]).abs() / sigma[j];
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Link;

    fn stump() -> Ensemble {
        Ensemble::new(
            vec![0, -1, -1],
            vec![1.0, 0.0, 0.0],
            vec![true, false, false],
            vec![true, false, false],
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

    fn empty_constraints(p: usize) -> Constraints {
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

    fn params() -> GaParams {
        GaParams {
            population: 40,
            max_generations: 100,
            stall_generations: 20,
            time_budget_s: 1e9,
        }
    }

    #[test]
    fn finds_feasible_solution_on_stump() {
        let ens = stump();
        let cons = empty_constraints(2);
        let result = solve_genetic(
            &ens,
            &[0.0, 0.0],
            0.5,
            f64::INFINITY,
            &cons,
            &[1.0, 1.0],
            &[1.0, 1.0],
            0.05,
            None,
            None,
            Some(1),
            &params(),
        );
        let x_cf = result.x_cf.expect("should find a counterfactual");
        assert!(ens.raw_score(&x_cf) >= 0.5);
        assert_eq!(x_cf[1], 0.0); // no reason to touch feature b
    }

    #[test]
    fn same_seed_is_bitwise_deterministic() {
        let ens = stump();
        let cons = empty_constraints(2);
        let run = || {
            solve_genetic(
                &ens,
                &[0.0, 0.0],
                0.5,
                f64::INFINITY,
                &cons,
                &[1.0, 1.0],
                &[1.0, 1.0],
                0.05,
                None,
                None,
                Some(7),
                &params(),
            )
        };
        let (a, b) = (run(), run());
        assert_eq!(a.generations, b.generations);
        assert_eq!(a.x_cf, b.x_cf);
    }

    #[test]
    fn frozen_target_is_infeasible() {
        let ens = stump();
        let mut cons = empty_constraints(2);
        cons.freeze = vec![0, 1];
        let result = solve_genetic(
            &ens,
            &[0.0, 0.0],
            0.5,
            f64::INFINITY,
            &cons,
            &[1.0, 1.0],
            &[1.0, 1.0],
            0.05,
            None,
            None,
            Some(1),
            &params(),
        );
        assert!(result.x_cf.is_none());
    }
}
