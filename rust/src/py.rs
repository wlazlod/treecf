//! PyO3 glue — compiled only with the `python` feature (maturin builds).

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::constraints::{Constraints, LinearC};
use crate::ga::GaParams;
use crate::ir::{Ensemble, Link};

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
        let scores = self.inner.raw_score_batch(xs, shape[0]);
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
            .check(x_matrix.as_slice()?, shape[0], x.as_slice()?);
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
        self.inner.repair(&mut data, shape[0], x.as_slice()?);
        let arr = PyArray1::from_vec(py, data);
        arr.reshape([shape[0], shape[1]])
    }
}

/// Full GA solve (migration P4 test path; wired into the public API in P5).
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

#[pymodule]
fn _treecf_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustEnsemble>()?;
    m.add_class::<RustConstraints>()?;
    m.add_function(wrap_pyfunction!(solve_genetic_raw, m)?)?;
    Ok(())
}
