// Deferred style/lint items (tracked for follow-up, not behavioural):
// - the simulation entry points are inherently multi-parameter;
// - several hot loops index parallel arrays by one shared counter;
// - the pyo3 0.23 -> IntoPyObject migration (ToPyObject/new_bound/to_object) is
//   a separate modernization task, so the deprecations are allowed for now.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::unwrap_or_default)]
#![allow(deprecated)]

use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;

pub mod correlation;
pub mod gbm;
pub mod math_utils;
pub mod portfolio;
pub mod random;
pub mod storage;

use gbm::{
    simulate_gbm_correlated, simulate_gbm_single, simulate_gbm_time_varying_correlated,
    simulate_gbm_time_varying_single,
};
use portfolio::{simulate_portfolio_losses, AssetData, PortfolioConfig};

/// Flatten a `Vec<Vec<f64>>` (rows x cols) into a contiguous NumPy 2-D array.
/// Returning a NumPy array instead of a Python list-of-lists avoids boxing every
/// element into a `PyFloat`, which dominated the GBM marshalling cost.
fn vec2_to_pyarray(py: Python<'_>, vv: Vec<Vec<f64>>) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let nrows = vv.len();
    let ncols = vv.first().map_or(0, |r| r.len());
    let mut flat = Vec::with_capacity(nrows * ncols);
    for row in &vv {
        if row.len() != ncols {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "ragged GBM path matrix",
            ));
        }
        flat.extend_from_slice(row);
    }
    ndarray::Array2::from_shape_vec((nrows, ncols), flat)
        .map(|arr| arr.into_pyarray_bound(py))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Flatten a `Vec<Vec<Vec<f64>>>` (d0 x d1 x d2) into a contiguous NumPy 3-D array.
fn vec3_to_pyarray(py: Python<'_>, vvv: Vec<Vec<Vec<f64>>>) -> PyResult<Bound<'_, PyArray3<f64>>> {
    let d0 = vvv.len();
    let d1 = vvv.first().map_or(0, |a| a.len());
    let d2 = vvv.first().and_then(|a| a.first()).map_or(0, |p| p.len());
    let mut flat = Vec::with_capacity(d0 * d1 * d2);
    for a in &vvv {
        if a.len() != d1 {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "ragged GBM path tensor",
            ));
        }
        for p in a {
            if p.len() != d2 {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "ragged GBM path tensor",
                ));
            }
            flat.extend_from_slice(p);
        }
    }
    ndarray::Array3::from_shape_vec((d0, d1, d2), flat)
        .map(|arr| arr.into_pyarray_bound(py))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pyfunction]
#[pyo3(name = "simulate_gbm", signature = (mu, sigma, s0, n_paths, n_steps, dt, seed=None))]
fn py_simulate_gbm<'py>(
    py: Python<'py>,
    mu: f64,
    sigma: f64,
    s0: f64,
    n_paths: usize,
    n_steps: usize,
    dt: f64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    vec2_to_pyarray(
        py,
        simulate_gbm_single(mu, sigma, s0, n_paths, n_steps, dt, seed),
    )
}

#[pyfunction]
#[pyo3(name = "simulate_gbm_multi", signature = (mu, sigma, s0, correlation_matrix, n_paths, n_steps, dt, seed=None))]
fn py_simulate_gbm_multi<'py>(
    py: Python<'py>,
    mu: Vec<f64>,
    sigma: Vec<f64>,
    s0: Vec<f64>,
    correlation_matrix: Vec<Vec<f64>>,
    n_paths: usize,
    n_steps: usize,
    dt: f64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let paths = simulate_gbm_correlated(
        mu,
        sigma,
        s0,
        correlation_matrix,
        n_paths,
        n_steps,
        dt,
        seed,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    // `simulate_gbm_correlated` already produced a contiguous Array3, so this is a
    // move into NumPy with no extra copy.
    Ok(paths.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(name = "simulate_gbm_tv", signature = (mu_times, mu_values, sigma_times, sigma_values, s0, n_paths, n_steps, dt, t_start, seed=None))]
fn py_simulate_gbm_time_varying<'py>(
    py: Python<'py>,
    mu_times: Vec<f64>,
    mu_values: Vec<f64>,
    sigma_times: Vec<f64>,
    sigma_values: Vec<f64>,
    s0: f64,
    n_paths: usize,
    n_steps: usize,
    dt: f64,
    t_start: f64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    vec2_to_pyarray(
        py,
        simulate_gbm_time_varying_single(
            &mu_times,
            &mu_values,
            &sigma_times,
            &sigma_values,
            s0,
            n_paths,
            n_steps,
            dt,
            t_start,
            seed,
        ),
    )
}

#[pyfunction]
#[pyo3(name = "simulate_gbm_tv_multi", signature = (mu_times, mu_values, sigma_times, sigma_values, s0, correlation_matrix, n_paths, n_steps, dt, t_start, seed=None))]
fn py_simulate_gbm_time_varying_correlated<'py>(
    py: Python<'py>,
    mu_times: Vec<Vec<f64>>,
    mu_values: Vec<Vec<f64>>,
    sigma_times: Vec<Vec<f64>>,
    sigma_values: Vec<Vec<f64>>,
    s0: Vec<f64>,
    correlation_matrix: Vec<Vec<f64>>,
    n_paths: usize,
    n_steps: usize,
    dt: f64,
    t_start: f64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let paths = simulate_gbm_time_varying_correlated(
        mu_times,
        mu_values,
        sigma_times,
        sigma_values,
        s0,
        correlation_matrix,
        n_paths,
        n_steps,
        dt,
        t_start,
        seed,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    vec3_to_pyarray(py, paths)
}

#[pyfunction]
#[pyo3(name = "simulate_portfolio", signature = (config, assets, n_simulations, kernel, factor_phi, thresholds, n_periods=1, period_length=1.0, seed=None, store_interim=None, output_path=None, batch_size=None))]
fn py_simulate_portfolio(
    py: Python<'_>,
    config: PortfolioConfig,
    assets: Vec<AssetData>,
    n_simulations: usize,
    kernel: String,
    factor_phi: f64,
    thresholds: PyReadonlyArray2<f64>,
    n_periods: usize,
    period_length: f64,
    seed: Option<u64>,
    store_interim: Option<bool>,
    output_path: Option<String>,
    batch_size: Option<usize>,
) -> PyResult<PyObject> {
    // The timing plan crosses the FFI as a NumPy array ([asset][period]) to
    // avoid boxing every threshold; rebuild the per-asset rows the trial loop
    // indexes. The plan is REQUIRED for both kernels — the Rust backend never
    // derives thresholds (see python/simflux/portfolio/default_timing.py).
    let thresholds_view = thresholds.as_array();
    let thresholds: Vec<Vec<f64>> = thresholds_view
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();

    let results = simulate_portfolio_losses(
        &config,
        &assets,
        n_simulations,
        n_periods,
        period_length,
        &kernel,
        factor_phi,
        &thresholds,
        seed,
        store_interim.unwrap_or(false),
        output_path,
        batch_size,
    );

    match results {
        Ok(simulation_results) => Ok(simulation_results.to_object(py)),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "{}",
            e
        ))),
    }
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_simulate_gbm, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_gbm_multi, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_gbm_time_varying, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_simulate_gbm_time_varying_correlated,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_simulate_portfolio, m)?)?;

    m.add_class::<PortfolioConfig>()?;
    m.add_class::<AssetData>()?;

    Ok(())
}
