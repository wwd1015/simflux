use pyo3::prelude::*;
use crate::random::{create_rng, sample_standard_normal};
use crate::correlation::{TwoFactorCorrelationStructure, SystematicFactors};
use crate::math_utils::{calculate_default_threshold, beta_mean_var_to_params, PortfolioStatistics, calculate_portfolio_loss_statistics};
use rayon::prelude::*;
use std::collections::HashMap;
use thiserror::Error;
use statrs::distribution::{Beta as StatrsBeta, ContinuousCDF};
use rand::rngs::StdRng;

#[derive(Error, Debug)]
pub enum PortfolioError {
    #[error("Invalid asset data: {0}")]
    InvalidAssetData(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Simulation error: {0}")]
    SimulationError(String),
    #[error("Storage error: {0}")]
    StorageError(String),
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct AssetData {
    #[pyo3(get, set)]
    pub asset_id: u32,
    #[pyo3(get, set)]
    pub sector_id: u32,
    #[pyo3(get, set)]
    pub pd: f64,                    // Probability of default
    #[pyo3(get, set)]
    pub lgd_mean: f64,              // Loss given default mean
    #[pyo3(get, set)]
    pub lgd_std: f64,               // Loss given default standard deviation
    #[pyo3(get, set)]
    pub exposure: f64,              // Exposure at default
    #[pyo3(get, set)]
    pub sector_name: String,        // Sector name
}

#[pymethods]
impl AssetData {
    #[new]
    pub fn new(
        asset_id: u32,
        sector_id: u32,
        pd: f64,
        lgd_mean: f64,
        lgd_std: f64,
        exposure: f64,
        sector_name: String,
    ) -> PyResult<Self> {
        if pd < 0.0 || pd > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "PD must be between 0 and 1"
            ));
        }
        
        if lgd_mean < 0.0 || lgd_mean > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "LGD mean must be between 0 and 1"
            ));
        }
        
        if lgd_std <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "LGD std must be positive"
            ));
        }
        
        if exposure < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Exposure must be non-negative"
            ));
        }
        
        Ok(AssetData {
            asset_id,
            sector_id,
            pd,
            lgd_mean,
            lgd_std,
            exposure,
            sector_name,
        })
    }
    
    pub fn get_lgd_beta_params(&self) -> PyResult<(f64, f64)> {
        let variance = self.lgd_std.powi(2);
        match beta_mean_var_to_params(self.lgd_mean, variance) {
            Ok(params) => Ok(params),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        }
    }
    
    pub fn get_default_threshold(&self) -> f64 {
        calculate_default_threshold(self.pd)
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    #[pyo3(get, set)]
    pub inter_sector_correlation: f64,
    #[pyo3(get, set)]
    pub intra_sector_correlations: Vec<f64>,
    #[pyo3(get, set)]
    pub systematic_lgd_correlation: f64,
    #[pyo3(get, set)]
    pub sector_names: Vec<String>,
    #[pyo3(get, set)]
    pub sector_correlation_matrix: Vec<Vec<f64>>,
}

#[pymethods]
impl PortfolioConfig {
    #[new]
    pub fn new(
        inter_sector_correlation: f64,
        intra_sector_correlations: Vec<f64>,
        systematic_lgd_correlation: f64,
        sector_names: Vec<String>,
        sector_correlation_matrix: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        if inter_sector_correlation.abs() > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Inter-sector correlation must be between -1 and 1"
            ));
        }
        
        if systematic_lgd_correlation.abs() > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Systematic LGD correlation must be between -1 and 1"
            ));
        }
        
        for &corr in &intra_sector_correlations {
            if corr.abs() > 1.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Intra-sector correlations must be between -1 and 1"
                ));
            }
        }
        
        if intra_sector_correlations.len() != sector_names.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Intra-sector correlations and sector names must have same length"
            ));
        }
        
        let matrix = sector_correlation_matrix.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "sector_correlation_matrix must be provided"
            )
        })?;

        if matrix.len() != sector_names.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "sector_correlation_matrix must match number of sectors"
            ));
        }
        for row in &matrix {
            if row.len() != sector_names.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "sector_correlation_matrix must be square"
                ));
            }
        }

        Ok(PortfolioConfig {
            inter_sector_correlation,
            intra_sector_correlations,
            systematic_lgd_correlation,
            sector_names,
            sector_correlation_matrix: matrix,
        })
    }
}

#[derive(Debug, Clone)]
pub struct AssetSimulationResult {
    pub asset_id: u32,
    pub sector_id: u32,
    pub defaulted: bool,
    pub time_to_default: Option<f64>,
    pub loss_amount: f64,
    pub recovery_rate: f64,
    pub asset_value: f64,
    pub systematic_factor_global: f64,
    pub systematic_factor_sector: f64,
    pub pd: f64,
    pub lgd_mean: f64,
    pub exposure: f64,
}

#[derive(Debug, Clone)]
pub struct TrialResult {
    pub trial_id: usize,
    pub total_loss: f64,
    pub total_defaults: u32,
    pub systematic_factors: SystematicFactors,
    pub asset_results: Vec<AssetSimulationResult>,
}

pub struct SimulationResults {
    pub trial_results: Vec<TrialResult>,
    pub portfolio_statistics: PortfolioStatistics,
    pub sector_statistics: HashMap<String, PortfolioStatistics>,
}

impl ToPyObject for SimulationResults {
    fn to_object(&self, py: Python) -> PyObject {
        let dict = pyo3::types::PyDict::new(py);
        
        // Add portfolio statistics
        let portfolio_stats = pyo3::types::PyDict::new(py);
        portfolio_stats.set_item("mean", self.portfolio_statistics.mean).unwrap();
        portfolio_stats.set_item("std_dev", self.portfolio_statistics.std_dev).unwrap();
        portfolio_stats.set_item("var_95", self.portfolio_statistics.var_95).unwrap();
        portfolio_stats.set_item("var_99", self.portfolio_statistics.var_99).unwrap();
        portfolio_stats.set_item("var_999", self.portfolio_statistics.var_999).unwrap();
        portfolio_stats.set_item("expected_shortfall_95", self.portfolio_statistics.expected_shortfall_95).unwrap();
        portfolio_stats.set_item("expected_shortfall_99", self.portfolio_statistics.expected_shortfall_99).unwrap();
        portfolio_stats.set_item("max_loss", self.portfolio_statistics.max_loss).unwrap();
        
        dict.set_item("portfolio_statistics", portfolio_stats).unwrap();
        dict.set_item("n_trials", self.trial_results.len()).unwrap();
        
        // Add sector statistics
        let sector_stats_dict = pyo3::types::PyDict::new(py);
        for (sector, stats) in &self.sector_statistics {
            let sector_dict = pyo3::types::PyDict::new(py);
            sector_dict.set_item("mean", stats.mean).unwrap();
            sector_dict.set_item("std_dev", stats.std_dev).unwrap();
            sector_dict.set_item("var_95", stats.var_95).unwrap();
            sector_dict.set_item("var_99", stats.var_99).unwrap();
            sector_stats_dict.set_item(sector, sector_dict).unwrap();
        }
        dict.set_item("sector_statistics", sector_stats_dict).unwrap();
        
        dict.to_object(py)
    }
}

pub fn simulate_portfolio_losses(
    config: &PortfolioConfig,
    assets: &[AssetData],
    n_simulations: usize,
    seed: Option<u64>,
    store_interim: bool,
    output_path: Option<String>,
) -> Result<SimulationResults, PortfolioError> {
    // Validate inputs
    validate_portfolio_inputs(config, assets)?;
    
    // Group assets by sector
    let mut sector_assets: HashMap<u32, Vec<&AssetData>> = HashMap::new();
    for asset in assets {
        sector_assets.entry(asset.sector_id).or_insert_with(Vec::new).push(asset);
    }
    
    // Calculate sector sizes for correlation structure
    let sector_sizes: Vec<usize> = (0..config.sector_names.len())
        .map(|i| sector_assets.get(&(i as u32)).map_or(0, |v| v.len()))
        .collect();
    
    // Create correlation structure
    let correlation_structure = TwoFactorCorrelationStructure::new(
        config.inter_sector_correlation,
        config.intra_sector_correlations.clone(),
        sector_sizes,
        config.sector_correlation_matrix.clone(),
    ).map_err(|e| PortfolioError::ConfigError(e.to_string()))?;
    
    // Generate systematic factors for all simulations
    let all_systematic_factors = correlation_structure.generate_factors(n_simulations, seed);
    
    // Run simulations in parallel
    let trial_results: Vec<TrialResult> = (0..n_simulations)
        .into_par_iter()
        .map(|trial_id| {
            simulate_single_trial(
                trial_id,
                config,
                assets,
                &correlation_structure,
                &all_systematic_factors[trial_id],
                seed,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    
    // Calculate portfolio statistics
    let total_losses: Vec<f64> = trial_results.iter().map(|t| t.total_loss).collect();
    let portfolio_statistics = calculate_portfolio_loss_statistics(&total_losses);
    
    // Calculate sector statistics
    let mut sector_statistics = HashMap::new();
    for (sector_id, sector_name) in config.sector_names.iter().enumerate() {
        let sector_losses: Vec<f64> = trial_results
            .iter()
            .map(|trial| {
                trial.asset_results
                    .iter()
                    .filter(|asset| asset.sector_id == sector_id as u32)
                    .map(|asset| asset.loss_amount)
                    .sum()
            })
            .collect();
        
        if !sector_losses.is_empty() {
            let sector_stats = calculate_portfolio_loss_statistics(&sector_losses);
            sector_statistics.insert(sector_name.clone(), sector_stats);
        }
    }
    
    // Store interim results if requested
    if store_interim {
        if let Some(path) = output_path {
            store_interim_results(&trial_results, &path)?;
        }
    }
    
    Ok(SimulationResults {
        trial_results,
        portfolio_statistics,
        sector_statistics,
    })
}

fn simulate_single_trial(
    trial_id: usize,
    config: &PortfolioConfig,
    assets: &[AssetData],
    correlation_structure: &TwoFactorCorrelationStructure,
    systematic_factors: &SystematicFactors,
    seed: Option<u64>,
) -> Result<TrialResult, PortfolioError> {
    let mut rng = create_rng(seed, trial_id as u64);
    let mut asset_results = Vec::with_capacity(assets.len());
    let mut total_loss = 0.0;
    let mut total_defaults = 0;
    
    for asset in assets {
        let result = simulate_single_asset(
            asset,
            correlation_structure,
            systematic_factors,
            config.systematic_lgd_correlation,
            &mut rng,
        )?;
        
        total_loss += result.loss_amount;
        if result.defaulted {
            total_defaults += 1;
        }
        
        asset_results.push(result);
    }
    
    Ok(TrialResult {
        trial_id,
        total_loss,
        total_defaults,
        systematic_factors: systematic_factors.clone(),
        asset_results,
    })
}

fn simulate_single_asset(
    asset: &AssetData,
    correlation_structure: &TwoFactorCorrelationStructure,
    systematic_factors: &SystematicFactors,
    systematic_lgd_correlation: f64,
    rng: &mut StdRng,
) -> Result<AssetSimulationResult, PortfolioError> {
    let sector_index = asset.sector_id as usize;
    let sector_factor = systematic_factors.get_sector_factor(sector_index);
    
    let intra_corr = correlation_structure.intra_sector_correlations[sector_index];
    let sector_loading = intra_corr.sqrt();
    let idio_loading = (1.0 - intra_corr).max(0.0).sqrt();

    // Generate idiosyncratic factor
    let idiosyncratic_factor = sample_standard_normal(rng);
    
    // Calculate asset value (standardized)
    let asset_value = sector_loading * sector_factor + idio_loading * idiosyncratic_factor;
    
    // Check for default
    let default_threshold = asset.get_default_threshold();
    let defaulted = asset_value <= default_threshold;
    
    let mut loss_amount = 0.0;
    let mut recovery_rate = 0.0;
    
    if defaulted {
        // Generate correlated LGD
        let (alpha, beta) = asset.get_lgd_beta_params()
            .map_err(|e| PortfolioError::SimulationError(e.to_string()))?;
        
        // Correlate LGD with systematic factor
        let lgd_systematic_component = systematic_lgd_correlation * sector_factor;
        let lgd_idiosyncratic = (1.0 - systematic_lgd_correlation.powi(2)).max(0.0).sqrt()
            * sample_standard_normal(rng);
        let lgd_normal = lgd_systematic_component + lgd_idiosyncratic;
        
        // Convert to beta distribution via normal CDF
        use crate::math_utils::normal_cdf;
        let lgd_uniform = normal_cdf(lgd_normal);
        let beta_dist = StatrsBeta::new(alpha, beta)
            .map_err(|e| PortfolioError::SimulationError(e.to_string()))?;
        let clamped_uniform = lgd_uniform.clamp(1e-12, 1.0 - 1e-12);
        let lgd_realized = beta_dist.inverse_cdf(clamped_uniform);
        
        recovery_rate = 1.0 - lgd_realized;
        loss_amount = lgd_realized * asset.exposure;
    }
    
    Ok(AssetSimulationResult {
        asset_id: asset.asset_id,
        sector_id: asset.sector_id,
        defaulted,
        time_to_default: if defaulted { Some(1.0) } else { None }, // Simplified: assume 1 year
        loss_amount,
        recovery_rate,
        asset_value,
        systematic_factor_global: sector_factor,
        systematic_factor_sector: sector_factor,
        pd: asset.pd,
        lgd_mean: asset.lgd_mean,
        exposure: asset.exposure,
    })
}

fn validate_portfolio_inputs(
    config: &PortfolioConfig,
    assets: &[AssetData],
) -> Result<(), PortfolioError> {
    if assets.is_empty() {
        return Err(PortfolioError::InvalidAssetData("No assets provided".to_string()));
    }
    
    // Check that all sector IDs are valid
    let max_sector_id = config.sector_names.len() as u32;
    for asset in assets {
        if asset.sector_id >= max_sector_id {
            return Err(PortfolioError::InvalidAssetData(
                format!("Asset {} has invalid sector_id {}", asset.asset_id, asset.sector_id)
            ));
        }
    }
    
    Ok(())
}

fn store_interim_results(
    _trial_results: &[TrialResult],
    output_path: &str,
) -> Result<(), PortfolioError> {
    // Parquet storage temporarily disabled due to dependency conflicts
    // For now, just validate the path
    if output_path.is_empty() {
        return Err(PortfolioError::StorageError("Empty output path".to_string()));
    }
    
    Err(PortfolioError::StorageError(
        "Interim storage via Rust backend is temporarily unavailable".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_asset_data_creation() {
        let asset = AssetData::new(
            1,
            0,
            0.05,
            0.6,
            0.2,
            1000000.0,
            "Technology".to_string(),
        ).unwrap();
        
        assert_eq!(asset.asset_id, 1);
        assert_eq!(asset.pd, 0.05);
        assert!(asset.get_default_threshold() < 0.0); // Low PD should have negative threshold
    }
    
    #[test]
    fn test_portfolio_config_creation() {
        let config = PortfolioConfig::new(
            0.2,
            vec![0.4, 0.3],
            0.1,
            vec!["Tech".to_string(), "Finance".to_string()],
            None,
        ).unwrap();
        
        assert_eq!(config.inter_sector_correlation, 0.2);
        assert_eq!(config.intra_sector_correlations.len(), 2);
    }
}
