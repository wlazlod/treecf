//! PyO3 glue — compiled only with the `python` feature (maturin builds).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

#[pymodule]
fn _treecf_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustEnsemble>()?;
    Ok(())
}
