use pyo3::prelude::*;

pub mod random;
pub mod gbm;
pub mod portfolio;
pub mod correlation;
pub mod storage;
pub mod math_utils;

use gbm::{simulate_gbm_single, simulate_gbm_correlated};
use portfolio::{simulate_portfolio_losses, PortfolioConfig, AssetData};

#[pyfunction]
#[pyo3(name = "simulate_gbm")]
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
#[pyo3(name = "simulate_gbm_multi")]
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
        mu,
        sigma,
        s0,
        correlation_matrix,
        n_paths,
        n_steps,
        dt,
        seed,
    ) {
        Ok(paths) => Ok(paths),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())),
    }
}

#[pyfunction]
#[pyo3(name = "simulate_portfolio")]
fn py_simulate_portfolio(
    config: PortfolioConfig,
    assets: Vec<AssetData>,
    n_simulations: usize,
    seed: Option<u64>,
    store_interim: Option<bool>,
    output_path: Option<String>,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let results = simulate_portfolio_losses(
            &config, 
            &assets, 
            n_simulations, 
            seed, 
            store_interim.unwrap_or(false),
            output_path
        );
        
        match results {
            Ok(simulation_results) => Ok(simulation_results.to_object(py)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        }
    })
}

#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_simulate_gbm, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_gbm_multi, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_portfolio, m)?)?;
    
    // Add classes
    m.add_class::<PortfolioConfig>()?;
    m.add_class::<AssetData>()?;
    
    Ok(())
}
