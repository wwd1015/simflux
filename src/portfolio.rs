use crate::correlation::{SystematicFactors, TwoFactorCorrelationStructure};
use crate::math_utils::{
    beta_mean_var_to_params, calculate_default_threshold, calculate_portfolio_loss_statistics,
    PortfolioStatistics,
};
use crate::random::{create_rng, sample_standard_normal};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use statrs::distribution::{Beta as StatrsBeta, ContinuousCDF};
use std::collections::HashMap;
use thiserror::Error;

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
    pub pd: f64,
    #[pyo3(get, set)]
    pub lgd_mean: f64,
    #[pyo3(get, set)]
    pub lgd_std: f64,
    #[pyo3(get, set)]
    pub exposure: f64,
    #[pyo3(get, set)]
    pub sector_name: String,
    /// Cumulative PD at each period end.  When `None` the flat `pd` field is
    /// used and a constant hazard is assumed across all periods.
    #[pyo3(get, set)]
    pub pd_term_structure: Option<Vec<f64>>,
}

#[pymethods]
impl AssetData {
    #[new]
    #[pyo3(signature = (asset_id, sector_id, pd, lgd_mean, lgd_std, exposure, sector_name, pd_term_structure=None))]
    pub fn new(
        asset_id: u32,
        sector_id: u32,
        pd: f64,
        lgd_mean: f64,
        lgd_std: f64,
        exposure: f64,
        sector_name: String,
        pd_term_structure: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        if pd < 0.0 || pd > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "PD must be between 0 and 1",
            ));
        }
        if lgd_mean < 0.0 || lgd_mean > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "LGD mean must be between 0 and 1",
            ));
        }
        if lgd_std <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "LGD std must be positive",
            ));
        }
        if exposure < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Exposure must be non-negative",
            ));
        }
        // Validate Beta distribution feasibility
        let max_lgd_var = lgd_mean * (1.0 - lgd_mean);
        if lgd_std * lgd_std >= max_lgd_var {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "lgd_std ({:.4}) must be less than sqrt(lgd_mean*(1-lgd_mean)) = {:.4}",
                lgd_std,
                max_lgd_var.sqrt()
            )));
        }

        if let Some(ref ts) = pd_term_structure {
            for (i, &p) in ts.iter().enumerate() {
                if p < 0.0 || p > 1.0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "pd_term_structure[{}] must be between 0 and 1",
                        i
                    )));
                }
            }
            // Cumulative PDs must be non-decreasing
            for i in 1..ts.len() {
                if ts[i] < ts[i - 1] - 1e-10 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "pd_term_structure must be non-decreasing (cumulative PDs)",
                    ));
                }
            }
        }

        Ok(AssetData {
            asset_id,
            sector_id,
            pd,
            lgd_mean,
            lgd_std,
            exposure,
            sector_name,
            pd_term_structure,
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

// ---------------------------------------------------------------------------
// Helpers – conditional PD derivation
// ---------------------------------------------------------------------------

/// Derive the conditional (forward) PD for a specific period.
///
/// If the asset has a cumulative PD term structure, the forward PD for period
/// `k` is `(cum[k] - cum[k-1]) / (1 - cum[k-1])`.  Otherwise a constant
/// hazard rate is derived from the flat `pd` field spread across `n_periods`.
///
/// Retained as the documented forward-PD reference: it equals the `phi = 0`
/// limit of the frailty barrier (calibrated Python-side); kept under test.
#[allow(dead_code)]
fn get_conditional_pd(asset: &AssetData, period: usize, n_periods: usize) -> f64 {
    if let Some(ref ts) = asset.pd_term_structure {
        if !ts.is_empty() {
            let idx = period.min(ts.len() - 1);
            if period == 0 {
                return ts[0].clamp(0.0, 1.0);
            }
            let prev_idx = (period - 1).min(ts.len() - 1);
            let cum_prev = ts[prev_idx];
            let cum_curr = ts[idx];
            if cum_prev >= 1.0 {
                return 0.0;
            }
            return ((cum_curr - cum_prev) / (1.0 - cum_prev)).clamp(0.0, 1.0);
        }
    }
    // Flat PD – derive per-period conditional PD from constant hazard
    if n_periods == 1 {
        asset.pd
    } else {
        1.0 - (1.0 - asset.pd).powf(1.0 / n_periods as f64)
    }
}

/// Cumulative PD by the end of `period` (0-indexed).  From the term structure
/// directly when present, else from the flat `pd` under a constant hazard so the
/// final period returns `pd` exactly.  Used by the "copula" default-timing model.
fn cumulative_pd(asset: &AssetData, period: usize, n_periods: usize) -> f64 {
    if let Some(ref ts) = asset.pd_term_structure {
        if !ts.is_empty() {
            let idx = period.min(ts.len() - 1);
            return ts[idx].clamp(0.0, 1.0);
        }
    }
    1.0 - (1.0 - asset.pd).powf((period as f64 + 1.0) / n_periods as f64)
}

// ---------------------------------------------------------------------------
// Portfolio config
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    #[pyo3(get, set)]
    pub intra_sector_correlations: Vec<f64>,
    /// LGD-systematic correlation, one value per sector (parity with
    /// `intra_sector_correlations`).
    #[pyo3(get, set)]
    pub systematic_lgd_correlations: Vec<f64>,
    #[pyo3(get, set)]
    pub sector_names: Vec<String>,
    /// Full sector correlation matrix — the single source of truth for
    /// cross-sector coupling.  No scalar summary is carried on the seam.
    #[pyo3(get, set)]
    pub sector_correlation_matrix: Vec<Vec<f64>>,
}

#[pymethods]
impl PortfolioConfig {
    #[new]
    #[pyo3(signature = (intra_sector_correlations, systematic_lgd_correlations, sector_names, sector_correlation_matrix=None))]
    pub fn new(
        intra_sector_correlations: Vec<f64>,
        systematic_lgd_correlations: Vec<f64>,
        sector_names: Vec<String>,
        sector_correlation_matrix: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        for &corr in &systematic_lgd_correlations {
            if corr.abs() > 1.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Systematic LGD correlations must be between -1 and 1",
                ));
            }
        }
        if systematic_lgd_correlations.len() != sector_names.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Systematic LGD correlations and sector names must have same length",
            ));
        }
        for &corr in &intra_sector_correlations {
            if corr.abs() > 1.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Intra-sector correlations must be between -1 and 1",
                ));
            }
        }
        if intra_sector_correlations.len() != sector_names.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Intra-sector correlations and sector names must have same length",
            ));
        }
        let matrix = sector_correlation_matrix.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "sector_correlation_matrix must be provided",
            )
        })?;
        if matrix.len() != sector_names.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "sector_correlation_matrix must match number of sectors",
            ));
        }
        for row in &matrix {
            if row.len() != sector_names.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "sector_correlation_matrix must be square",
                ));
            }
        }
        Ok(PortfolioConfig {
            intra_sector_correlations,
            systematic_lgd_correlations,
            sector_names,
            sector_correlation_matrix: matrix,
        })
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

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
        let dict = pyo3::types::PyDict::new_bound(py);

        let portfolio_stats = pyo3::types::PyDict::new_bound(py);
        portfolio_stats
            .set_item("mean", self.portfolio_statistics.mean)
            .unwrap();
        portfolio_stats
            .set_item("std_dev", self.portfolio_statistics.std_dev)
            .unwrap();
        portfolio_stats
            .set_item("var_95", self.portfolio_statistics.var_95)
            .unwrap();
        portfolio_stats
            .set_item("var_99", self.portfolio_statistics.var_99)
            .unwrap();
        portfolio_stats
            .set_item("var_999", self.portfolio_statistics.var_999)
            .unwrap();
        portfolio_stats
            .set_item(
                "expected_shortfall_95",
                self.portfolio_statistics.expected_shortfall_95,
            )
            .unwrap();
        portfolio_stats
            .set_item(
                "expected_shortfall_99",
                self.portfolio_statistics.expected_shortfall_99,
            )
            .unwrap();
        portfolio_stats
            .set_item("max_loss", self.portfolio_statistics.max_loss)
            .unwrap();

        dict.set_item("portfolio_statistics", portfolio_stats)
            .unwrap();
        dict.set_item("n_trials", self.trial_results.len()).unwrap();

        let sector_stats_dict = pyo3::types::PyDict::new_bound(py);
        for (sector, stats) in &self.sector_statistics {
            let sector_dict = pyo3::types::PyDict::new_bound(py);
            sector_dict.set_item("mean", stats.mean).unwrap();
            sector_dict.set_item("std_dev", stats.std_dev).unwrap();
            sector_dict.set_item("var_95", stats.var_95).unwrap();
            sector_dict.set_item("var_99", stats.var_99).unwrap();
            sector_dict.set_item("var_999", stats.var_999).unwrap();
            sector_dict
                .set_item("expected_shortfall_95", stats.expected_shortfall_95)
                .unwrap();
            sector_dict
                .set_item("expected_shortfall_99", stats.expected_shortfall_99)
                .unwrap();
            sector_dict.set_item("max_loss", stats.max_loss).unwrap();
            sector_stats_dict.set_item(sector, sector_dict).unwrap();
        }
        dict.set_item("sector_statistics", sector_stats_dict)
            .unwrap();

        dict.to_object(py)
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

pub fn simulate_portfolio_losses(
    config: &PortfolioConfig,
    assets: &[AssetData],
    n_simulations: usize,
    n_periods: usize,
    period_length: f64,
    default_timing: &str,
    factor_phi: f64,
    barriers: &[Vec<f64>],
    seed: Option<u64>,
    store_interim: bool,
    output_path: Option<String>,
    batch_size: Option<usize>,
) -> Result<SimulationResults, PortfolioError> {
    validate_portfolio_inputs(config, assets)?;

    if n_periods == 0 {
        return Err(PortfolioError::ConfigError(
            "n_periods must be positive".to_string(),
        ));
    }
    if period_length <= 0.0 {
        return Err(PortfolioError::ConfigError(
            "period_length must be positive".to_string(),
        ));
    }
    // "copula": single frozen latent vs cumulative thresholds (default times,
    // Li 2000). "frailty": persistent AR(1) factor with fresh idiosyncratic and
    // pre-calibrated barriers. Both reproduce the marginal term structure exactly.
    let copula = match default_timing {
        "copula" => true,
        "frailty" => false,
        other => {
            return Err(PortfolioError::ConfigError(format!(
                "default_timing must be 'copula' or 'frailty', got '{}'",
                other
            )))
        }
    };
    if !copula {
        if barriers.len() != assets.len() {
            return Err(PortfolioError::ConfigError(
                "frailty mode requires one barrier row per asset".to_string(),
            ));
        }
        // Guard the inner dimension too: simulate_single_trial indexes
        // barriers[idx][period] for period in 0..n_periods, so a too-short row
        // would panic across the FFI boundary instead of erroring cleanly.
        if barriers.iter().any(|row| row.len() < n_periods) {
            return Err(PortfolioError::ConfigError(
                "frailty mode requires n_periods barriers per asset".to_string(),
            ));
        }
    }

    // Group assets by sector
    let mut sector_assets: HashMap<u32, Vec<&AssetData>> = HashMap::new();
    for asset in assets {
        sector_assets
            .entry(asset.sector_id)
            .or_insert_with(Vec::new)
            .push(asset);
    }

    let sector_sizes: Vec<usize> = (0..config.sector_names.len())
        .map(|i| sector_assets.get(&(i as u32)).map_or(0, |v| v.len()))
        .collect();

    let correlation_structure = TwoFactorCorrelationStructure::new(
        config.intra_sector_correlations.clone(),
        sector_sizes,
        Some(config.sector_correlation_matrix.clone()),
    )
    .map_err(|e| PortfolioError::ConfigError(e.to_string()))?;

    // Generate systematic factors.  The "copula" model freezes a single set
    // across the horizon (one generated); "frailty" draws an independent
    // innovation set per period and then couples them with AR(1) persistence.
    // factors[period][trial].
    let n_factor_sets = if copula { 1 } else { n_periods };
    let mut all_factors: Vec<Vec<SystematicFactors>> = (0..n_factor_sets)
        .map(|period| {
            let period_seed =
                seed.map(|s| s.wrapping_add((period * n_simulations + 1_000_000) as u64));
            correlation_structure.generate_factors(n_simulations, period_seed)
        })
        .collect();

    // Frailty AR(1): F_k = phi*F_{k-1} + sqrt(1-phi^2)*innovation_k, applied to
    // the pre-generated independent innovations.  Each F_k stays N(0, Sigma).
    if !copula && factor_phi > 0.0 && n_periods > 1 {
        let sqrt_innov = (1.0 - factor_phi * factor_phi).max(0.0).sqrt();
        let n_sectors = config.sector_names.len();
        for trial in 0..n_simulations {
            for period in 1..n_periods {
                for s in 0..n_sectors {
                    let prev = all_factors[period - 1][trial].sector_factors[s];
                    let innov = all_factors[period][trial].sector_factors[s];
                    all_factors[period][trial].sector_factors[s] =
                        factor_phi * prev + sqrt_innov * innov;
                }
            }
        }
    }

    // Run simulations in parallel
    let trial_results: Vec<TrialResult> = (0..n_simulations)
        .into_par_iter()
        .map(|trial_id| {
            simulate_single_trial(
                trial_id,
                config,
                assets,
                &correlation_structure,
                &all_factors,
                barriers,
                n_periods,
                period_length,
                copula,
                seed,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Calculate statistics
    let total_losses: Vec<f64> = trial_results.iter().map(|t| t.total_loss).collect();
    let portfolio_statistics = calculate_portfolio_loss_statistics(&total_losses);

    let mut sector_statistics = HashMap::new();
    for (sector_id, sector_name) in config.sector_names.iter().enumerate() {
        let sector_losses: Vec<f64> = trial_results
            .iter()
            .map(|trial| {
                trial
                    .asset_results
                    .iter()
                    .filter(|a| a.sector_id == sector_id as u32)
                    .map(|a| a.loss_amount)
                    .sum()
            })
            .collect();
        if !sector_losses.is_empty() {
            sector_statistics.insert(
                sector_name.clone(),
                calculate_portfolio_loss_statistics(&sector_losses),
            );
        }
    }

    if store_interim {
        if let Some(ref path) = output_path {
            store_interim_results(&trial_results, path, &config.sector_names, batch_size)?;
        }
    }

    Ok(SimulationResults {
        trial_results,
        portfolio_statistics,
        sector_statistics,
    })
}

// ---------------------------------------------------------------------------
// Trial simulation (multi-period aware)
// ---------------------------------------------------------------------------

fn simulate_single_trial(
    trial_id: usize,
    config: &PortfolioConfig,
    assets: &[AssetData],
    correlation_structure: &TwoFactorCorrelationStructure,
    all_factors: &[Vec<SystematicFactors>], // [period][trial]
    barriers: &[Vec<f64>],                  // [asset][period], frailty only
    n_periods: usize,
    period_length: f64,
    copula: bool,
    seed: Option<u64>,
) -> Result<TrialResult, PortfolioError> {
    let mut rng = create_rng(seed, trial_id as u64);
    let n_assets = assets.len();

    // Per-asset mutable state
    let mut defaulted = vec![false; n_assets];
    let mut default_time: Vec<Option<f64>> = vec![None; n_assets];
    let mut loss_amounts = vec![0.0_f64; n_assets];
    let mut recovery_rates = vec![0.0_f64; n_assets];
    let mut asset_values = vec![0.0_f64; n_assets];
    let mut factor_global = vec![0.0_f64; n_assets];
    let mut factor_sector = vec![0.0_f64; n_assets];

    if copula {
        // One-factor Gaussian copula of default times: a single frozen latent
        // per obligor vs the cumulative-PD staircase; default = first crossing.
        let factors = &all_factors[0][trial_id];
        for (idx, asset) in assets.iter().enumerate() {
            let sector_index = asset.sector_id as usize;
            let sector_factor = factors.get_sector_factor(sector_index);

            let intra_corr = correlation_structure.intra_sector_correlations[sector_index];
            let sector_loading = intra_corr.sqrt();
            let idio_loading = (1.0 - intra_corr).max(0.0).sqrt();
            let idio = sample_standard_normal(&mut rng);

            let value = sector_loading * sector_factor + idio_loading * idio;
            asset_values[idx] = value;
            factor_global[idx] = sector_factor;
            factor_sector[idx] = sector_factor;

            let threshold_final =
                calculate_default_threshold(cumulative_pd(asset, n_periods - 1, n_periods));
            if value <= threshold_final {
                defaulted[idx] = true;
                // Default period = first k whose cumulative threshold is crossed.
                let mut dperiod = n_periods - 1;
                for k in 0..n_periods {
                    if value <= calculate_default_threshold(cumulative_pd(asset, k, n_periods)) {
                        dperiod = k;
                        break;
                    }
                }
                default_time[idx] = Some((dperiod as f64 + 1.0) * period_length);

                let lgd_corr = config.systematic_lgd_correlations[sector_index];
                match compute_loss(asset, sector_factor, lgd_corr, &mut rng) {
                    Ok((loss, recovery)) => {
                        loss_amounts[idx] = loss;
                        recovery_rates[idx] = recovery;
                    }
                    Err(e) => return Err(e),
                }
            }
        }
    } else {
        // Frailty: persistent (already AR(1)-coupled) factor, fresh idiosyncratic
        // each period, first-passage against the pre-calibrated per-period barriers.
        for period in 0..n_periods {
            let factors = &all_factors[period][trial_id];

            for (idx, asset) in assets.iter().enumerate() {
                if defaulted[idx] {
                    continue; // already defaulted in an earlier period
                }

                let sector_index = asset.sector_id as usize;
                let sector_factor = factors.get_sector_factor(sector_index);

                let intra_corr = correlation_structure.intra_sector_correlations[sector_index];
                let sector_loading = intra_corr.sqrt();
                let idio_loading = (1.0 - intra_corr).max(0.0).sqrt();
                let idio = sample_standard_normal(&mut rng);

                let value = sector_loading * sector_factor + idio_loading * idio;
                asset_values[idx] = value;
                factor_global[idx] = sector_factor;
                factor_sector[idx] = sector_factor;

                let threshold = barriers[idx][period];

                if value <= threshold {
                    defaulted[idx] = true;
                    default_time[idx] = Some((period as f64 + 1.0) * period_length);

                    let lgd_corr = config.systematic_lgd_correlations[sector_index];
                    match compute_loss(asset, sector_factor, lgd_corr, &mut rng) {
                        Ok((loss, recovery)) => {
                            loss_amounts[idx] = loss;
                            recovery_rates[idx] = recovery;
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
        }
    }

    // Build results
    let mut total_loss = 0.0;
    let mut total_defaults = 0u32;
    let mut asset_results = Vec::with_capacity(n_assets);

    for (idx, asset) in assets.iter().enumerate() {
        total_loss += loss_amounts[idx];
        if defaulted[idx] {
            total_defaults += 1;
        }
        asset_results.push(AssetSimulationResult {
            asset_id: asset.asset_id,
            sector_id: asset.sector_id,
            defaulted: defaulted[idx],
            time_to_default: default_time[idx],
            loss_amount: loss_amounts[idx],
            recovery_rate: recovery_rates[idx],
            asset_value: asset_values[idx],
            systematic_factor_global: factor_global[idx],
            systematic_factor_sector: factor_sector[idx],
            pd: asset.pd,
            lgd_mean: asset.lgd_mean,
            exposure: asset.exposure,
        });
    }

    // Use the first period's factors as the "representative" for the trial
    let representative_factors = all_factors[0][trial_id].clone();

    Ok(TrialResult {
        trial_id,
        total_loss,
        total_defaults,
        systematic_factors: representative_factors,
        asset_results,
    })
}

/// Compute loss amount and recovery rate for a defaulted asset.
fn compute_loss(
    asset: &AssetData,
    sector_factor: f64,
    systematic_lgd_correlation: f64,
    rng: &mut StdRng,
) -> Result<(f64, f64), PortfolioError> {
    let (alpha, beta) = asset
        .get_lgd_beta_params()
        .map_err(|e| PortfolioError::SimulationError(e.to_string()))?;

    // Wrong-way risk: a downturn is a LOW sector factor (defaults fire in the
    // low tail of the latent value), so LGD must load on the NEGATIVE of the
    // factor for a positive `systematic_lgd_correlation` to mean "recoveries
    // fall — LGD rises — exactly when defaults cluster".
    let lgd_sys = -systematic_lgd_correlation * sector_factor;
    let lgd_idio =
        (1.0 - systematic_lgd_correlation.powi(2)).max(0.0).sqrt() * sample_standard_normal(rng);
    let lgd_normal = lgd_sys + lgd_idio;

    use crate::math_utils::normal_cdf;
    let lgd_uniform = normal_cdf(lgd_normal).clamp(1e-12, 1.0 - 1e-12);

    let beta_dist =
        StatrsBeta::new(alpha, beta).map_err(|e| PortfolioError::SimulationError(e.to_string()))?;
    let lgd_realized = beta_dist.inverse_cdf(lgd_uniform);

    Ok((lgd_realized * asset.exposure, 1.0 - lgd_realized))
}

// ---------------------------------------------------------------------------
// Validation & storage
// ---------------------------------------------------------------------------

fn validate_portfolio_inputs(
    config: &PortfolioConfig,
    assets: &[AssetData],
) -> Result<(), PortfolioError> {
    if assets.is_empty() {
        return Err(PortfolioError::InvalidAssetData(
            "No assets provided".to_string(),
        ));
    }
    let max_sector_id = config.sector_names.len() as u32;
    for asset in assets {
        if asset.sector_id >= max_sector_id {
            return Err(PortfolioError::InvalidAssetData(format!(
                "Asset {} has invalid sector_id {}",
                asset.asset_id, asset.sector_id
            )));
        }
    }
    Ok(())
}

fn store_interim_results(
    trial_results: &[TrialResult],
    output_path: &str,
    sector_names: &[String],
    batch_size: Option<usize>,
) -> Result<(), PortfolioError> {
    if output_path.is_empty() {
        return Err(PortfolioError::StorageError(
            "Empty output path".to_string(),
        ));
    }
    crate::storage::write_simulation_results_to_parquet(
        trial_results,
        output_path,
        sector_names,
        batch_size,
    )
    .map_err(|e| PortfolioError::StorageError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_data_creation() {
        let asset =
            AssetData::new(1, 0, 0.05, 0.6, 0.2, 1e6, "Technology".to_string(), None).unwrap();
        assert_eq!(asset.asset_id, 1);
        assert!(asset.get_default_threshold() < 0.0);
    }

    #[test]
    fn test_asset_data_with_term_structure() {
        let ts = vec![0.01, 0.025, 0.04, 0.06];
        let asset = AssetData::new(
            1,
            0,
            0.06,
            0.6,
            0.2,
            1e6,
            "Tech".to_string(),
            Some(ts.clone()),
        )
        .unwrap();
        assert_eq!(asset.pd_term_structure.as_ref().unwrap().len(), 4);
    }

    #[test]
    fn test_conditional_pd_flat() {
        let asset = AssetData {
            asset_id: 0,
            sector_id: 0,
            pd: 0.10,
            lgd_mean: 0.5,
            lgd_std: 0.1,
            exposure: 1e6,
            sector_name: "A".into(),
            pd_term_structure: None,
        };
        // Single period: conditional PD == flat PD
        assert!((get_conditional_pd(&asset, 0, 1) - 0.10).abs() < 1e-12);
        // 4 periods: per-period PD < cumulative PD
        let cpd = get_conditional_pd(&asset, 0, 4);
        assert!(cpd > 0.0 && cpd < 0.10);
    }

    #[test]
    fn test_conditional_pd_term_structure() {
        let asset = AssetData {
            asset_id: 0,
            sector_id: 0,
            pd: 0.10,
            lgd_mean: 0.5,
            lgd_std: 0.1,
            exposure: 1e6,
            sector_name: "A".into(),
            pd_term_structure: Some(vec![0.02, 0.05, 0.08, 0.10]),
        };
        // Period 0: forward PD = cum[0] = 0.02
        assert!((get_conditional_pd(&asset, 0, 4) - 0.02).abs() < 1e-12);
        // Period 1: forward PD = (0.05 - 0.02) / (1 - 0.02)
        let expected = (0.05 - 0.02) / (1.0 - 0.02);
        assert!((get_conditional_pd(&asset, 1, 4) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_config_creation() {
        let config = PortfolioConfig::new(
            vec![0.4, 0.3],
            vec![0.1, 0.1],
            vec!["Tech".to_string(), "Finance".to_string()],
            Some(vec![vec![1.0, 0.2], vec![0.2, 1.0]]),
        )
        .unwrap();
        assert_eq!(config.intra_sector_correlations.len(), 2);
        assert_eq!(config.systematic_lgd_correlations.len(), 2);
    }
}
