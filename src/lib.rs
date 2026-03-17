use pyo3::prelude::*;

pub mod random;
pub mod gbm;
pub mod portfolio;
pub mod correlation;
pub mod storage;
pub mod math_utils;

use gbm::{
    simulate_gbm_single, simulate_gbm_correlated,
    simulate_gbm_time_varying_single, simulate_gbm_time_varying_correlated,
};
use portfolio::{simulate_portfolio_losses, PortfolioConfig, AssetData};

#[pyfunction]
#[pyo3(name = "simulate_gbm", signature = (mu, sigma, s0, n_paths, n_steps, dt, seed=None))]
fn py_simulate_gbm(
    mu: f64,
    sigma: f64,
    s0: f64,
    n_paths: usize,
    n_steps: usize,
    dt: f64,
    seed: Option<u64>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(simulate_gbm_single(mu, sigma, s0, n_paths, n_steps, dt, seed))
}

#[pyfunction]
#[pyo3(name = "simulate_gbm_multi", signature = (mu, sigma, s0, correlation_matrix, n_paths, n_steps, dt, seed=None))]
fn py_simulate_gbm_multi(
    mu: Vec<f64>,
    sigma: Vec<f64>,
    s0: Vec<f64>,
    correlation_matrix: Vec<Vec<f64>>,
    n_paths: usize,
    n_steps: usize,
    dt: f64,
    seed: Option<u64>,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    match simulate_gbm_correlated(
        mu, sigma, s0, correlation_matrix,
        n_paths, n_steps, dt, seed,
    ) {
        Ok(paths) => Ok(paths),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())),
    }
}

#[pyfunction]
#[pyo3(name = "simulate_gbm_tv", signature = (mu_times, mu_values, sigma_times, sigma_values, s0, n_paths, n_steps, dt, t_start, seed=None))]
fn py_simulate_gbm_time_varying(
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
) -> PyResult<Vec<Vec<f64>>> {
    Ok(simulate_gbm_time_varying_single(
        &mu_times, &mu_values,
        &sigma_times, &sigma_values,
        s0, n_paths, n_steps, dt, t_start, seed,
    ))
}

#[pyfunction]
#[pyo3(name = "simulate_gbm_tv_multi", signature = (mu_times, mu_values, sigma_times, sigma_values, s0, correlation_matrix, n_paths, n_steps, dt, t_start, seed=None))]
fn py_simulate_gbm_time_varying_correlated(
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
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    match simulate_gbm_time_varying_correlated(
        mu_times, mu_values,
        sigma_times, sigma_values,
        s0, correlation_matrix,
        n_paths, n_steps, dt, t_start, seed,
    ) {
        Ok(paths) => Ok(paths),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())),
    }
}

#[pyfunction]
#[pyo3(name = "simulate_portfolio", signature = (config, assets, n_simulations, n_periods=1, period_length=1.0, seed=None, store_interim=None, output_path=None))]
fn py_simulate_portfolio(
    py: Python<'_>,
    config: PortfolioConfig,
    assets: Vec<AssetData>,
    n_simulations: usize,
    n_periods: usize,
    period_length: f64,
    seed: Option<u64>,
    store_interim: Option<bool>,
    output_path: Option<String>,
) -> PyResult<PyObject> {
    let results = simulate_portfolio_losses(
        &config,
        &assets,
        n_simulations,
        n_periods,
        period_length,
        seed,
        store_interim.unwrap_or(false),
        output_path
    );

    match results {
        Ok(simulation_results) => Ok(simulation_results.to_object(py)),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
    }
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_simulate_gbm, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_gbm_multi, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_gbm_time_varying, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_gbm_time_varying_correlated, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_portfolio, m)?)?;

    m.add_class::<PortfolioConfig>()?;
    m.add_class::<AssetData>()?;

    Ok(())
}
