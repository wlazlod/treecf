//! PyO3 glue — compiled only with the `python` feature (maturin builds).

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::constraints::{Constraints, LinearC};
use crate::exact::{ExactParams, ValuePolicy};
use crate::ga::GaParams;
use crate::interrupt::SearchOutcome;
use crate::ir::{Ensemble, Link};

/// Per-feature value-policy flat encoding: `0=raw` (`None`), `1=integer`,
/// `2=grid` (reads `step`/`anchor`) — the marshaled form of
/// `Explainer.value_policy` `exact_rust.py` builds from `ir.feature_names`.
///
/// Every failure here is a marshaling bug on the Python side of the
/// boundary, never a user-facing constraint problem, so it raises
/// `PyRuntimeError` — distinct from the `PyValueError`
/// `solve_exact`'s own order-pair validation raises, so `exact_rust.py` can
/// tell the two apart by exception *type* alone, with no text matching on
/// either side.
fn marshal_value_policies(
    code: &[u8],
    step: &[f64],
    anchor: &[f64],
) -> PyResult<Vec<Option<ValuePolicy>>> {
    if code.len() != step.len() || code.len() != anchor.len() {
        return Err(PyRuntimeError::new_err(
            "value policy code/step/anchor arrays must have equal length",
        ));
    }
    code.iter()
        .zip(step)
        .zip(anchor)
        .map(|((&c, &s), &a)| match c {
            0 => Ok(None),
            1 => Ok(Some(ValuePolicy::Integer)),
            2 => Ok(Some(ValuePolicy::Grid { step: s, anchor: a })),
            other => Err(PyRuntimeError::new_err(format!(
                "unknown value policy code {other}"
            ))),
        })
        .collect()
}

#[pyclass(frozen)]
pub struct RustEnsemble {
    pub(crate) inner: Ensemble,
}

#[pymethods]
impl RustEnsemble {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        feature: PyReadonlyArray1<i32>,
        threshold: PyReadonlyArray1<f64>,
        is_lt: PyReadonlyArray1<u8>,
        missing_left: PyReadonlyArray1<u8>,
        left: PyReadonlyArray1<u32>,
        right: PyReadonlyArray1<u32>,
        value: PyReadonlyArray1<f64>,
        tree_roots: PyReadonlyArray1<u32>,
        base_score: f64,
        link: &str,
        n_features: usize,
    ) -> PyResult<Self> {
        let link = match link {
            "identity" => Link::Identity,
            "sigmoid" => Link::Sigmoid,
            other => return Err(PyValueError::new_err(format!("unknown link {other:?}"))),
        };
        let inner = Ensemble::new(
            feature.as_slice()?.to_vec(),
            threshold.as_slice()?.to_vec(),
            is_lt.as_slice()?.iter().map(|&b| b != 0).collect(),
            missing_left.as_slice()?.iter().map(|&b| b != 0).collect(),
            left.as_slice()?.to_vec(),
            right.as_slice()?.to_vec(),
            value.as_slice()?.to_vec(),
            tree_roots.as_slice()?.to_vec(),
            base_score,
            link,
            n_features,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Raw scores for a C-contiguous (n_rows, n_features) float64 matrix.
    fn raw_score_batch<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let shape = x.shape();
        if shape[1] != self.inner.n_features {
            return Err(PyValueError::new_err(format!(
                "expected {} features, got {}",
                self.inner.n_features, shape[1]
            )));
        }
        let xs = x.as_slice()?;
        let scores = self.inner.raw_score_batch(xs, shape[0], true);
        Ok(scores.into_pyarray(py))
    }
}

#[pyclass(frozen)]
pub struct RustConstraints {
    pub(crate) inner: Constraints,
}

#[pymethods]
impl RustConstraints {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_features: usize,
        freeze: PyReadonlyArray1<u32>,
        range_idx: PyReadonlyArray1<u32>,
        range_lo: PyReadonlyArray1<f64>,
        range_hi: PyReadonlyArray1<f64>,
        equals_idx: PyReadonlyArray1<u32>,
        equals_val: PyReadonlyArray1<f64>,
        mono_idx: PyReadonlyArray1<u32>,
        mono_dir: PyReadonlyArray1<i8>,
        lin_offsets: PyReadonlyArray1<u32>,
        lin_indices: PyReadonlyArray1<u32>,
        lin_coefs: PyReadonlyArray1<f64>,
        lin_op: PyReadonlyArray1<u8>,
        lin_rhs: PyReadonlyArray1<f64>,
        lin_policy: PyReadonlyArray1<u8>,
        imp_cond_idx: PyReadonlyArray1<u32>,
        imp_cond_val: PyReadonlyArray1<f64>,
        imp_cons_idx: PyReadonlyArray1<u32>,
        imp_cons_val: PyReadonlyArray1<f64>,
        oh_offsets: PyReadonlyArray1<u32>,
        oh_indices: PyReadonlyArray1<u32>,
        am_idx: PyReadonlyArray1<u32>,
        am_to: PyReadonlyArray1<f64>,
        am_from: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let lin_offsets = lin_offsets.as_slice()?;
        let lin_indices = lin_indices.as_slice()?;
        let lin_coefs = lin_coefs.as_slice()?;
        let lin_op = lin_op.as_slice()?;
        let lin_rhs = lin_rhs.as_slice()?;
        let lin_policy = lin_policy.as_slice()?;
        let mut linears = Vec::with_capacity(lin_op.len());
        for l in 0..lin_op.len() {
            let (start, end) = (lin_offsets[l] as usize, lin_offsets[l + 1] as usize);
            linears.push(LinearC {
                indices: lin_indices[start..end].to_vec(),
                coefs: lin_coefs[start..end].to_vec(),
                op: lin_op[l],
                rhs: lin_rhs[l],
                policy: lin_policy[l],
            });
        }
        let oh_offsets = oh_offsets.as_slice()?;
        let oh_indices = oh_indices.as_slice()?;
        let onehot = (0..oh_offsets.len().saturating_sub(1))
            .map(|g| oh_indices[oh_offsets[g] as usize..oh_offsets[g + 1] as usize].to_vec())
            .collect();
        let implications = imp_cond_idx
            .as_slice()?
            .iter()
            .zip(imp_cond_val.as_slice()?)
            .zip(
                imp_cons_idx
                    .as_slice()?
                    .iter()
                    .zip(imp_cons_val.as_slice()?),
            )
            .map(|((&ci, &cv), (&si, &sv))| (ci, cv, si, sv))
            .collect();
        let allow_missing = am_idx
            .as_slice()?
            .iter()
            .zip(am_to.as_slice()?.iter().zip(am_from.as_slice()?))
            .map(|(&j, (&to, &from))| (j, to, from))
            .collect();
        let ranges = range_idx
            .as_slice()?
            .iter()
            .zip(range_lo.as_slice()?.iter().zip(range_hi.as_slice()?))
            .map(|(&j, (&lo, &hi))| (j, lo, hi))
            .collect();
        let equals = equals_idx
            .as_slice()?
            .iter()
            .zip(equals_val.as_slice()?)
            .map(|(&j, &v)| (j, v))
            .collect();
        let monotone = mono_idx
            .as_slice()?
            .iter()
            .zip(mono_dir.as_slice()?)
            .map(|(&j, &d)| (j, d))
            .collect();
        Ok(Self {
            inner: Constraints {
                n_features,
                freeze: freeze.as_slice()?.to_vec(),
                ranges,
                equals,
                monotone,
                linears,
                implications,
                onehot,
                allow_missing,
            },
        })
    }

    fn check<'py>(
        &self,
        py: Python<'py>,
        x_matrix: PyReadonlyArray2<f64>,
        x: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let shape = x_matrix.shape();
        let ok = self
            .inner
            .check(x_matrix.as_slice()?, shape[0], x.as_slice()?, true);
        Ok(ok.into_pyarray(py))
    }

    fn repair<'py>(
        &self,
        py: Python<'py>,
        x_matrix: PyReadonlyArray2<f64>,
        x: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = x_matrix.shape();
        let mut data = x_matrix.as_slice()?.to_vec();
        self.inner.repair(&mut data, shape[0], x.as_slice()?, true);
        let arr = PyArray1::from_vec(py, data);
        arr.reshape([shape[0], shape[1]])
    }
}

/// Full GA solve.
/// Returns (x_cf | None, generations).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (ensemble, constraints, x, lo_t, hi_t, sigma, weights, lam,
                    background=None, if_ensemble=None, min_total_path=None, seed=None,
                    population=80, max_generations=200, stall_generations=30,
                    time_budget_s=10.0))]
fn solve_genetic_raw<'py>(
    py: Python<'py>,
    ensemble: &RustEnsemble,
    constraints: &RustConstraints,
    x: PyReadonlyArray1<f64>,
    lo_t: f64,
    hi_t: f64,
    sigma: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    lam: f64,
    background: Option<PyReadonlyArray2<f64>>,
    if_ensemble: Option<&RustEnsemble>,
    min_total_path: Option<f64>,
    seed: Option<u64>,
    population: usize,
    max_generations: usize,
    stall_generations: usize,
    time_budget_s: f64,
) -> PyResult<(Option<Bound<'py, PyArray1<f64>>>, usize)> {
    let x_own = x.as_slice()?.to_vec();
    let sigma_own = sigma.as_slice()?.to_vec();
    let weights_own = weights.as_slice()?.to_vec();
    let bg_own: Option<(Vec<f64>, usize)> = match &background {
        Some(bg) => Some((bg.as_slice()?.to_vec(), bg.shape()[0])),
        None => None,
    };
    let params = GaParams {
        population,
        max_generations,
        stall_generations,
        time_budget_s,
        inner_parallel: true,
    };
    let ens = &ensemble.inner;
    let cons = &constraints.inner;
    let plaus = match (if_ensemble, min_total_path) {
        (Some(if_e), Some(bound)) => Some((&if_e.inner, bound)),
        _ => None,
    };
    let result = py.detach(|| {
        crate::ga::solve_genetic(
            ens,
            &x_own,
            lo_t,
            hi_t,
            cons,
            &sigma_own,
            &weights_own,
            lam,
            bg_own.as_ref().map(|(data, n)| (data.as_slice(), *n)),
            plaus,
            seed,
            &params,
        )
    });
    Ok((result.x_cf.map(|v| v.into_pyarray(py)), result.generations))
}

/// Batch GA solve: one independent search per `(task_row, task_seed)` pair,
/// fanned out with rayon under a released GIL. Returns
/// `(x_cf (n_tasks, p) — factual copy where infeasible, feasible mask, generations)`.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyo3(signature = (ensemble, constraints, x_rows, task_row, task_seed, lo_t, hi_t,
                    sigma, weights, lam, background=None, if_ensemble=None,
                    min_total_path=None, population=80, max_generations=200,
                    stall_generations=30, time_budget_s=10.0))]
fn solve_genetic_batch_raw<'py>(
    py: Python<'py>,
    ensemble: &RustEnsemble,
    constraints: &RustConstraints,
    x_rows: PyReadonlyArray2<f64>,
    task_row: PyReadonlyArray1<u64>,
    task_seed: PyReadonlyArray1<u64>,
    lo_t: f64,
    hi_t: f64,
    sigma: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    lam: f64,
    background: Option<PyReadonlyArray2<f64>>,
    if_ensemble: Option<&RustEnsemble>,
    min_total_path: Option<f64>,
    population: usize,
    max_generations: usize,
    stall_generations: usize,
    time_budget_s: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<u64>>,
)> {
    let shape = x_rows.shape();
    let (n_rows, p) = (shape[0], shape[1]);
    if p != ensemble.inner.n_features {
        return Err(PyValueError::new_err(format!(
            "expected {} features, got {}",
            ensemble.inner.n_features, p
        )));
    }
    let xs_own = x_rows.as_slice()?.to_vec();
    let rows = task_row.as_slice()?;
    let seeds = task_seed.as_slice()?;
    if rows.len() != seeds.len() {
        return Err(PyValueError::new_err(
            "task_row and task_seed must have the same length",
        ));
    }
    if rows.iter().any(|&r| r as usize >= n_rows) {
        return Err(PyValueError::new_err("task_row index out of range"));
    }
    let tasks: Vec<(usize, u64)> = rows
        .iter()
        .zip(seeds)
        .map(|(&r, &s)| (r as usize, s))
        .collect();
    let sigma_own = sigma.as_slice()?.to_vec();
    let weights_own = weights.as_slice()?.to_vec();
    let bg_own: Option<(Vec<f64>, usize)> = match &background {
        Some(bg) => Some((bg.as_slice()?.to_vec(), bg.shape()[0])),
        None => None,
    };
    let params = GaParams {
        population,
        max_generations,
        stall_generations,
        time_budget_s,
        // With fewer tasks than threads the per-population rayon stages are
        // the only way to use the spare cores; beyond that, task-level
        // parallelism saturates them and serial inner stages avoid
        // nested-splitting overhead. Results are identical either way.
        inner_parallel: tasks.len() < rayon::current_num_threads(),
    };
    let ens = &ensemble.inner;
    let cons = &constraints.inner;
    let plaus = match (if_ensemble, min_total_path) {
        (Some(if_e), Some(bound)) => Some((&if_e.inner, bound)),
        _ => None,
    };
    let outcome = py.detach(|| {
        let mut noop = || false;
        crate::ga::solve_genetic_batch(
            ens,
            &xs_own,
            &tasks,
            lo_t,
            hi_t,
            cons,
            &sigma_own,
            &weights_own,
            lam,
            bg_own.as_ref().map(|(data, n)| (data.as_slice(), *n)),
            plaus,
            &params,
            &mut noop,
        )
    });
    let results = match outcome {
        SearchOutcome::Done(results) => results,
        SearchOutcome::Interrupted => unreachable!("no-op probe never interrupts"),
    };
    let n_tasks = tasks.len();
    let mut x_cf = vec![0.0f64; n_tasks * p];
    let mut feasible = vec![false; n_tasks];
    let mut generations = vec![0u64; n_tasks];
    for (t, result) in results.iter().enumerate() {
        generations[t] = result.generations as u64;
        match &result.x_cf {
            Some(row) => {
                x_cf[t * p..(t + 1) * p].copy_from_slice(row);
                feasible[t] = true;
            }
            None => {
                let r = tasks[t].0;
                x_cf[t * p..(t + 1) * p].copy_from_slice(&xs_own[r * p..(r + 1) * p]);
            }
        }
    }
    Ok((
        PyArray1::from_vec(py, x_cf).reshape([n_tasks, p])?,
        feasible.into_pyarray(py),
        generations.into_pyarray(py),
    ))
}

/// Full exact-backend solve — port of `treecf.backends.exact.solve_exact`.
/// Returns `(x_cf | None, distance | None, proof, stats, snapped)`: `stats` is
/// the 7-tuple `(nodes_expanded, nodes_pruned_score, nodes_pruned_cost,
/// lower_bound, gap, completed, warm_start_used)`; `snapped` is the winning
/// row's snapped feature indices, in search order — `exact_rust.py` maps them
/// back to names to rebuild `ExactResult` losslessly. A `PyValueError`
/// mirrors Python's `ConstraintValidationError` for a multi-feature Linear
/// outside the canonical order-pair shape (from `solve_exact`'s own
/// `validate`); `exact_rust.py` re-raises that type rather than comparing
/// message text across languages. A `PyRuntimeError` (from the
/// value-policy-array length check, or `marshal_value_policies` itself)
/// means the caller marshaled malformed input — a bug, never a user-facing
/// constraint problem — so it is deliberately a different exception type and
/// propagates as a plain `RuntimeError` instead.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyo3(signature = (ensemble, constraints, x, lo_t, hi_t, sigma, weights, lam,
                    policy_code, policy_step, policy_anchor,
                    if_ensemble=None, min_total_path=None,
                    node_budget=2_000_000, gap=0.0, time_budget_s=10.0,
                    incumbent_cost=None, incumbent_row=None))]
fn solve_exact_raw<'py>(
    py: Python<'py>,
    ensemble: &RustEnsemble,
    constraints: &RustConstraints,
    x: PyReadonlyArray1<f64>,
    lo_t: f64,
    hi_t: f64,
    sigma: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    lam: f64,
    policy_code: PyReadonlyArray1<u8>,
    policy_step: PyReadonlyArray1<f64>,
    policy_anchor: PyReadonlyArray1<f64>,
    if_ensemble: Option<&RustEnsemble>,
    min_total_path: Option<f64>,
    node_budget: u64,
    gap: f64,
    time_budget_s: f64,
    incumbent_cost: Option<f64>,
    incumbent_row: Option<PyReadonlyArray1<f64>>,
) -> PyResult<(
    Option<Bound<'py, PyArray1<f64>>>,
    Option<f64>,
    &'static str,
    (u64, u64, u64, f64, f64, bool, bool),
    Bound<'py, PyArray1<u64>>,
)> {
    let x_own = x.as_slice()?.to_vec();
    let sigma_own = sigma.as_slice()?.to_vec();
    let weights_own = weights.as_slice()?.to_vec();
    let policies = marshal_value_policies(
        policy_code.as_slice()?,
        policy_step.as_slice()?,
        policy_anchor.as_slice()?,
    )?;
    if policies.len() != x_own.len() {
        // a length mismatch between the marshaled policy arrays and the
        // feature count is a marshaling bug, not a user-facing constraint
        // problem — see marshal_value_policies's own doc comment
        return Err(PyRuntimeError::new_err(format!(
            "expected {} value-policy entries, got {}",
            x_own.len(),
            policies.len()
        )));
    }
    let incumbent_row_own: Option<Vec<f64>> = match &incumbent_row {
        Some(r) => Some(r.as_slice()?.to_vec()),
        None => None,
    };
    let incumbent = match (incumbent_cost, &incumbent_row_own) {
        (Some(cost), Some(row)) => Some((cost, row.as_slice())),
        _ => None,
    };
    let ens = &ensemble.inner;
    let cons = &constraints.inner;
    let plaus = match (if_ensemble, min_total_path) {
        (Some(if_e), Some(bound)) => Some((&if_e.inner, bound)),
        _ => None,
    };
    let params = ExactParams {
        node_budget,
        gap,
        time_budget_s,
    };
    let outcome = py
        .detach(|| {
            let mut noop = || false;
            crate::exact::solve_exact(
                ens,
                &x_own,
                (lo_t, hi_t),
                cons,
                &sigma_own,
                &weights_own,
                lam,
                &policies,
                plaus,
                &params,
                incumbent,
                &mut noop,
            )
        })
        .map_err(PyValueError::new_err)?;
    let result = match outcome {
        SearchOutcome::Done(result) => result,
        SearchOutcome::Interrupted => unreachable!("no-op probe never interrupts"),
    };
    let stats = result.stats;
    let stats_tuple = (
        stats.nodes_expanded,
        stats.nodes_pruned_score,
        stats.nodes_pruned_cost,
        stats.lower_bound,
        stats.gap,
        stats.completed,
        stats.warm_start_used,
    );
    let snapped: Vec<u64> = result.snapped.iter().map(|&i| i as u64).collect();
    Ok((
        result.x_cf.map(|v| v.into_pyarray(py)),
        result.distance,
        result.proof,
        stats_tuple,
        snapped.into_pyarray(py),
    ))
}

/// Test-only: per-feature candidate states `_build_domains`/`build_domains`
/// produce, flattened as `(offsets, value, cost, cell_idx, is_nan, snapped)` —
/// `offsets` has `n_features + 1` entries, feature `f`'s states are
/// `offsets[f]..offsets[f + 1]` in every other array. Exists so
/// `test_exact_parity.py` can compare domain construction against the Python
/// reference one feature at a time, independent of the search itself.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyo3(signature = (ensemble, constraints, x, sigma, weights, lam,
                    policy_code, policy_step, policy_anchor, if_ensemble=None))]
fn debug_domains_raw<'py>(
    py: Python<'py>,
    ensemble: &RustEnsemble,
    constraints: &RustConstraints,
    x: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    lam: f64,
    policy_code: PyReadonlyArray1<u8>,
    policy_step: PyReadonlyArray1<f64>,
    policy_anchor: PyReadonlyArray1<f64>,
    if_ensemble: Option<&RustEnsemble>,
) -> PyResult<(
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
)> {
    let x_own = x.as_slice()?.to_vec();
    let sigma_own = sigma.as_slice()?.to_vec();
    let weights_own = weights.as_slice()?.to_vec();
    let policies = marshal_value_policies(
        policy_code.as_slice()?,
        policy_step.as_slice()?,
        policy_anchor.as_slice()?,
    )?;
    if policies.len() != x_own.len() {
        // a length mismatch between the marshaled policy arrays and the
        // feature count is a marshaling bug, not a user-facing constraint
        // problem — see marshal_value_policies's own doc comment
        return Err(PyRuntimeError::new_err(format!(
            "expected {} value-policy entries, got {}",
            x_own.len(),
            policies.len()
        )));
    }
    let ens = &ensemble.inner;
    let cons = &constraints.inner;
    let ensembles: Vec<&Ensemble> = match if_ensemble {
        None => vec![ens],
        Some(if_e) => vec![ens, &if_e.inner],
    };
    let grids = crate::exact::domains::constraint_cells(cons, &ensembles);
    let domains = crate::exact::domains::build_domains(
        &grids,
        &x_own,
        cons,
        &sigma_own,
        &weights_own,
        lam,
        &policies,
    );

    let mut offsets: Vec<u32> = Vec::with_capacity(domains.len() + 1);
    offsets.push(0);
    let mut value: Vec<f64> = Vec::new();
    let mut cost: Vec<f64> = Vec::new();
    let mut cell_idx: Vec<u32> = Vec::new();
    let mut is_nan: Vec<u8> = Vec::new();
    let mut snapped: Vec<u8> = Vec::new();
    for states in &domains {
        for st in states {
            value.push(st.value);
            cost.push(st.cost);
            cell_idx.push(st.cell_idx as u32);
            is_nan.push(u8::from(st.is_nan));
            snapped.push(u8::from(st.snapped));
        }
        offsets.push(value.len() as u32);
    }
    Ok((
        offsets.into_pyarray(py),
        value.into_pyarray(py),
        cost.into_pyarray(py),
        cell_idx.into_pyarray(py),
        is_nan.into_pyarray(py),
        snapped.into_pyarray(py),
    ))
}

/// Certified recourse-region growth — port of `treecf.regions._recourse_region`.
/// Returns `(lo, hi)` per-feature arrays (degenerate coordinates equal
/// `x_cf` there); `regions_rust.py` builds the `RecourseRegion` dataclass
/// from them (`feature_intervals`/`certified` are presentation, not search
/// state, so they stay on the Python side).
///
/// `missing_defined`/`if_missing_defined` carry the `node.missing_left is
/// not None` bit that `RustEnsemble`'s own flat `missing_left: bool`
/// encoding collapses into `false` for both an explicit "route right" and an
/// undefined missing direction — this is the one caller that needs the
/// distinction (see `crate::regions`'s module doc). `lo_b`/`hi_b` are the
/// instance bounds and `open_set` the non-degenerate feature indices;
/// `regions.py` already computes both to build `feature_intervals` either
/// way, so this binding does not re-derive them.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyo3(signature = (ensemble, missing_defined, constraints, x_cf, lo_t, hi_t,
                    lo_b, hi_b, open_set,
                    if_ensemble=None, if_missing_defined=None, min_total_path=None))]
fn compute_region_raw<'py>(
    py: Python<'py>,
    ensemble: &RustEnsemble,
    missing_defined: PyReadonlyArray1<u8>,
    constraints: &RustConstraints,
    x_cf: PyReadonlyArray1<f64>,
    lo_t: f64,
    hi_t: f64,
    lo_b: PyReadonlyArray1<f64>,
    hi_b: PyReadonlyArray1<f64>,
    open_set: PyReadonlyArray1<u32>,
    if_ensemble: Option<&RustEnsemble>,
    if_missing_defined: Option<PyReadonlyArray1<u8>>,
    min_total_path: Option<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let x_cf_own = x_cf.as_slice()?.to_vec();
    let lo_b_own = lo_b.as_slice()?.to_vec();
    let hi_b_own = hi_b.as_slice()?.to_vec();
    let open_set_own: Vec<usize> = open_set.as_slice()?.iter().map(|&j| j as usize).collect();
    let missing_defined_own: Vec<bool> = missing_defined
        .as_slice()?
        .iter()
        .map(|&b| b != 0)
        .collect();
    let if_missing_defined_own: Option<Vec<bool>> = match &if_missing_defined {
        Some(arr) => Some(arr.as_slice()?.iter().map(|&b| b != 0).collect()),
        None => None,
    };
    let ens = &ensemble.inner;
    let cons = &constraints.inner;
    let if_pair = match (if_ensemble, &if_missing_defined_own) {
        (Some(if_e), Some(md)) => Some((&if_e.inner, md.as_slice())),
        _ => None,
    };
    let outcome = py.detach(|| {
        let mut noop = || false;
        crate::regions::recourse_region(
            ens,
            &missing_defined_own,
            cons,
            &x_cf_own,
            (lo_t, hi_t),
            &lo_b_own,
            &hi_b_own,
            &open_set_own,
            if_pair,
            min_total_path.unwrap_or(0.0),
            &mut noop,
        )
    });
    let result = match outcome {
        SearchOutcome::Done(result) => result,
        SearchOutcome::Interrupted => unreachable!("no-op probe never interrupts"),
    };
    Ok((result.lo.into_pyarray(py), result.hi.into_pyarray(py)))
}

#[pymodule]
fn _treecf_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustEnsemble>()?;
    m.add_class::<RustConstraints>()?;
    m.add_function(wrap_pyfunction!(solve_genetic_raw, m)?)?;
    m.add_function(wrap_pyfunction!(solve_genetic_batch_raw, m)?)?;
    m.add_function(wrap_pyfunction!(solve_exact_raw, m)?)?;
    m.add_function(wrap_pyfunction!(debug_domains_raw, m)?)?;
    m.add_function(wrap_pyfunction!(compute_region_raw, m)?)?;
    Ok(())
}
