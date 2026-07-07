use crate::correlation::TwoFactorCorrelationStructure;
use crate::math_utils::{
    beta_inverse_cdf_prepared, beta_mean_var_to_params, calculate_default_threshold,
    calculate_portfolio_loss_statistics, PortfolioStatistics,
};
use crate::random::{create_rng, sample_standard_normal};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use statrs::function::beta::ln_beta;
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
    /// Per-obligor intra-sector correlation (the share of this obligor's latent
    /// variance carried by its sector factor).  When `None` the sector-level
    /// value from `PortfolioConfig.intra_sector_correlations` is used, so
    /// obligors in a sector may have heterogeneous loadings.
    #[pyo3(get, set)]
    pub intra_sector_correlation: Option<f64>,
    /// Per-period LGD Beta mean.  When `None` the flat `lgd_mean` is used; when
    /// present, an obligor that defaults in period `k` draws LGD from a Beta with
    /// mean `lgd_term_structure[k]` (clamped to the last entry) and the constant
    /// `lgd_std`.
    #[pyo3(get, set)]
    pub lgd_term_structure: Option<Vec<f64>>,
}

#[pymethods]
impl AssetData {
    #[new]
    #[pyo3(signature = (asset_id, sector_id, pd, lgd_mean, lgd_std, exposure, sector_name, pd_term_structure=None, intra_sector_correlation=None, lgd_term_structure=None))]
    pub fn new(
        asset_id: u32,
        sector_id: u32,
        pd: f64,
        lgd_mean: f64,
        lgd_std: f64,
        exposure: f64,
        sector_name: String,
        pd_term_structure: Option<Vec<f64>>,
        intra_sector_correlation: Option<f64>,
        lgd_term_structure: Option<Vec<f64>>,
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

        if let Some(rho) = intra_sector_correlation {
            if !(0.0..=1.0).contains(&rho) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "intra_sector_correlation must be between 0 and 1",
                ));
            }
        }

        if let Some(ref ts) = lgd_term_structure {
            let max_var = lgd_std * lgd_std;
            for (i, &m) in ts.iter().enumerate() {
                if !(0.0..=1.0).contains(&m) {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "lgd_term_structure[{}] must be between 0 and 1",
                        i
                    )));
                }
                if max_var >= m * (1.0 - m) {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "lgd_std is infeasible for lgd_term_structure[{}] mean {:.4}",
                        i, m
                    )));
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
            intra_sector_correlation,
            lgd_term_structure,
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

// (The cumulative-PD staircase derivation that used to live here moved behind
// the Python-side timing plan — see python/simflux/portfolio/default_timing.py —
// so both backends consume identical thresholds instead of re-deriving them.)

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

/// One row of the interim store: a single defaulted obligor in a single trial.
/// Only defaults are materialized (defaults are rare relative to the full
/// asset x trial grid), and only when `store_interim` is requested, so the
/// per-trial memory is O(defaults) instead of O(n_assets). Carries the
/// systematic and idiosyncratic factors at the moment of default for debugging.
#[derive(Debug, Clone)]
pub struct DefaultEvent {
    pub asset_id: u32,
    pub sector_id: u32,
    pub default_period: u32,
    pub time_to_default: f64,
    pub loss_amount: f64,
    pub recovery_rate: f64,
    pub asset_value: f64,
    pub systematic_factor: f64,
    pub idiosyncratic_factor: f64,
    pub pd: f64,
    pub lgd_mean: f64,
    pub exposure: f64,
}

/// Per-trial summary. `sector_losses` (length n_sectors) is accumulated inside
/// the trial so the aggregate statistics never need the per-asset detail.
/// `default_events` is empty unless interim storage was requested.
#[derive(Debug, Clone)]
pub struct TrialResult {
    pub trial_id: usize,
    pub total_loss: f64,
    pub total_defaults: u32,
    pub sector_losses: Vec<f64>,
    pub default_events: Vec<DefaultEvent>,
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
    kernel: &str,
    factor_phi: f64,
    thresholds: &[Vec<f64>],
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
    // The kernel names the simulation dynamic; the thresholds come from the
    // Python-side timing plan for BOTH kernels (staircase quantiles for
    // "copula", calibrated barriers for "frailty") — this backend never derives
    // them. "copula": single frozen latent, first crossing (Li 2000).
    // "frailty": persistent AR(1) factor with fresh idiosyncratic shocks,
    // first passage (Duffie et al. 2009).
    let copula = match kernel {
        "copula" => true,
        "frailty" => false,
        other => {
            return Err(PortfolioError::ConfigError(format!(
                "kernel must be 'copula' or 'frailty', got '{}'",
                other
            )))
        }
    };
    // Guard both dimensions for both kernels: simulate_single_trial indexes
    // thresholds[idx][period] for period in 0..n_periods, so a missing row or
    // too-short row would panic across the FFI boundary instead of erroring.
    if thresholds.len() != assets.len() {
        return Err(PortfolioError::ConfigError(
            "thresholds must have one row per asset (n_assets x n_periods)".to_string(),
        ));
    }
    if thresholds.iter().any(|row| row.len() < n_periods) {
        return Err(PortfolioError::ConfigError(
            "thresholds must have n_periods entries per asset".to_string(),
        ));
    }
    // Content guards: NaN thresholds make `value <= threshold` always false
    // (silent zero defaults) and a NaN/out-of-range factor_phi silently zeroes
    // the factor path — error loudly instead. ±inf thresholds are legitimate
    // (cumulative PD of exactly 0 or 1).
    if thresholds.iter().any(|row| row.iter().any(|v| v.is_nan())) {
        return Err(PortfolioError::ConfigError(
            "thresholds must not contain NaN".to_string(),
        ));
    }
    if factor_phi.is_nan() || !(0.0..=1.0).contains(&factor_phi) {
        return Err(PortfolioError::ConfigError(format!(
            "factor_phi must be in [0, 1], got {}",
            factor_phi
        )));
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
        sector_sizes.clone(),
        Some(config.sector_correlation_matrix.clone()),
    )
    .map_err(|e| PortfolioError::ConfigError(e.to_string()))?;

    // Generate systematic factors.  The "copula" model freezes a single set
    // across the horizon (one generated); "frailty" draws an independent
    // innovation set per period and then couples them with AR(1) persistence.
    // all_factors[period] is a flat trial-major buffer (n_trials x n_sectors).
    let n_sectors = config.sector_names.len();
    let n_factor_sets = if copula { 1 } else { n_periods };
    let mut all_factors: Vec<Vec<f64>> = (0..n_factor_sets)
        .map(|period| {
            let period_seed =
                seed.map(|s| s.wrapping_add((period * n_simulations + 1_000_000) as u64));
            correlation_structure.generate_factors(n_simulations, period_seed)
        })
        .collect();

    // Frailty AR(1): F_k = phi*F_{k-1} + sqrt(1-phi^2)*innovation_k, applied to
    // the pre-generated independent innovations.  Each F_k stays N(0, Sigma).
    // Walking periods in order preserves the recursion; within a period the
    // (trial, sector) updates are elementwise over the flat buffers.
    if !copula && factor_phi > 0.0 && n_periods > 1 {
        let sqrt_innov = (1.0 - factor_phi * factor_phi).max(0.0).sqrt();
        for period in 1..n_periods {
            let (done, rest) = all_factors.split_at_mut(period);
            let prev = &done[period - 1];
            for (curr, &prev) in rest[0].iter_mut().zip(prev.iter()) {
                *curr = factor_phi * prev + sqrt_innov * *curr;
            }
        }
    }

    // Hoist every per-asset constant out of the trial loop; the loop then only
    // samples, compares, and accumulates.
    let precomp = precompute_assets(config, assets, &correlation_structure, n_periods)?;

    // Run simulations in parallel
    let trial_results: Vec<TrialResult> = (0..n_simulations)
        .into_par_iter()
        .map(|trial_id| {
            simulate_single_trial(
                trial_id,
                assets,
                &precomp,
                &all_factors,
                thresholds,
                n_periods,
                period_length,
                copula,
                seed,
                store_interim,
                n_sectors,
            )
        })
        .collect();

    // Calculate statistics
    let total_losses: Vec<f64> = trial_results.iter().map(|t| t.total_loss).collect();
    let portfolio_statistics = calculate_portfolio_loss_statistics(&total_losses);

    let mut sector_statistics = HashMap::new();
    for (sector_id, sector_name) in config.sector_names.iter().enumerate() {
        // Per-sector loss per trial is precomputed on each TrialResult.
        let sector_losses: Vec<f64> = trial_results
            .iter()
            .map(|trial| trial.sector_losses.get(sector_id).copied().unwrap_or(0.0))
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
            // Per-sector asset counts let the analyzer recover default rates from
            // the defaults-only store (non-defaulted asset-trials are not written).
            let meta = InterimMetadata {
                n_trials: n_simulations,
                n_assets: assets.len(),
                sector_names: config.sector_names.clone(),
                sector_asset_counts: sector_sizes.clone(),
            };
            store_interim_results(
                &trial_results,
                path,
                &config.sector_names,
                batch_size,
                &meta,
            )?;
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

/// LGD Beta parameters for one (asset, period), with `ln B(alpha, beta)`
/// precomputed so each quantile draw skips the two `ln_gamma` evaluations.
struct LgdBeta {
    alpha: f64,
    beta: f64,
    ln_beta_ab: f64,
}

/// Per-asset constants hoisted out of the trial loop: loadings, LGD coupling,
/// and the LGD Beta parameters per period.  Deriving the Beta once per
/// (asset, period) instead of once per default event removes repeated
/// parameter validation from the hot path; `AssetData::new` already guarantees
/// feasible Beta parameters, so setup-time failure is the only failure left.
struct AssetPrecomp {
    sector_index: usize,
    sector_loading: f64,
    idio_loading: f64,
    lgd_corr: f64,
    lgd_idio_loading: f64,
    exposure: f64,
    /// LGD Beta per period (index clamped by construction via `lgd_mean_at`).
    betas: Vec<LgdBeta>,
}

fn precompute_assets(
    config: &PortfolioConfig,
    assets: &[AssetData],
    correlation_structure: &TwoFactorCorrelationStructure,
    n_periods: usize,
) -> Result<Vec<AssetPrecomp>, PortfolioError> {
    assets
        .iter()
        .map(|asset| {
            let sector_index = asset.sector_id as usize;
            let intra_corr = asset
                .intra_sector_correlation
                .unwrap_or(correlation_structure.intra_sector_correlations[sector_index]);
            let lgd_corr = config.systematic_lgd_correlations[sector_index];
            let variance = asset.lgd_std * asset.lgd_std;
            let betas = (0..n_periods)
                .map(|period| {
                    let (alpha, beta) =
                        beta_mean_var_to_params(lgd_mean_at(asset, period), variance)
                            .map_err(|e| PortfolioError::SimulationError(e.to_string()))?;
                    Ok(LgdBeta {
                        alpha,
                        beta,
                        ln_beta_ab: ln_beta(alpha, beta),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(AssetPrecomp {
                sector_index,
                sector_loading: intra_corr.sqrt(),
                idio_loading: (1.0 - intra_corr).max(0.0).sqrt(),
                lgd_corr,
                lgd_idio_loading: (1.0 - lgd_corr.powi(2)).max(0.0).sqrt(),
                exposure: asset.exposure,
                betas,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn simulate_single_trial(
    trial_id: usize,
    assets: &[AssetData],
    precomp: &[AssetPrecomp],
    all_factors: &[Vec<f64>], // [period] -> flat trial-major (n_trials x n_sectors)
    thresholds: &[Vec<f64>],  // [asset][period], from the timing plan (both kernels)
    n_periods: usize,
    period_length: f64,
    copula: bool,
    seed: Option<u64>,
    store_interim: bool,
    n_sectors: usize,
) -> TrialResult {
    let mut rng = create_rng(seed, trial_id as u64);
    let n_assets = assets.len();

    let mut total_loss = 0.0;
    let mut total_defaults = 0u32;
    let mut sector_losses = vec![0.0_f64; n_sectors];
    let mut default_events: Vec<DefaultEvent> = Vec::new();

    if copula {
        // One-factor Gaussian copula of default times: a single frozen latent
        // per obligor vs the plan's cumulative staircase; default = first
        // crossing. The staircase is non-decreasing per asset (enforced by the
        // TimingPlan), so the final period's threshold is the loosest.
        // Defaults are discovered in asset order, so losses accumulate directly
        // (no per-asset scratch arrays) in the same float order as ever.
        let factors = &all_factors[0][trial_id * n_sectors..(trial_id + 1) * n_sectors];
        for (idx, pc) in precomp.iter().enumerate() {
            let sector_factor = factors[pc.sector_index];
            let idio = sample_standard_normal(&mut rng);
            let value = pc.sector_loading * sector_factor + pc.idio_loading * idio;

            if value <= thresholds[idx][n_periods - 1] {
                // Default period = first k whose cumulative threshold is crossed.
                let mut dperiod = n_periods - 1;
                for k in 0..n_periods {
                    if value <= thresholds[idx][k] {
                        dperiod = k;
                        break;
                    }
                }

                let (loss, recovery) = compute_loss(pc, dperiod, sector_factor, &mut rng);
                total_loss += loss;
                sector_losses[pc.sector_index] += loss;
                total_defaults += 1;
                if store_interim {
                    let asset = &assets[idx];
                    default_events.push(DefaultEvent {
                        asset_id: asset.asset_id,
                        sector_id: asset.sector_id,
                        default_period: dperiod as u32,
                        time_to_default: (dperiod as f64 + 1.0) * period_length,
                        loss_amount: loss,
                        recovery_rate: recovery,
                        asset_value: value,
                        systematic_factor: sector_factor,
                        idiosyncratic_factor: idio,
                        pd: asset.pd,
                        lgd_mean: asset.lgd_mean,
                        exposure: asset.exposure,
                    });
                }
            }
        }
    } else {
        // Frailty: persistent (already AR(1)-coupled) factor, fresh idiosyncratic
        // each period, first-passage against the plan's calibrated barriers.
        // Defaults fire in (period, asset) order but are reduced in asset order
        // below, keeping the accumulation float-order (and the event order)
        // identical to the copula branch and to prior releases.
        let mut defaulted = vec![false; n_assets];
        let mut loss_amounts = vec![0.0_f64; n_assets];
        // Event detail is only materialized when interim storage asked for it.
        let mut detail: Option<FrailtyEventDetail> =
            store_interim.then(|| FrailtyEventDetail::new(n_assets));

        for period in 0..n_periods {
            let factors = &all_factors[period][trial_id * n_sectors..(trial_id + 1) * n_sectors];

            for (idx, pc) in precomp.iter().enumerate() {
                if defaulted[idx] {
                    continue; // already defaulted in an earlier period
                }

                let sector_factor = factors[pc.sector_index];
                let idio = sample_standard_normal(&mut rng);
                let value = pc.sector_loading * sector_factor + pc.idio_loading * idio;

                if value <= thresholds[idx][period] {
                    defaulted[idx] = true;
                    let (loss, recovery) = compute_loss(pc, period, sector_factor, &mut rng);
                    loss_amounts[idx] = loss;
                    if let Some(d) = detail.as_mut() {
                        d.default_period[idx] = period as u32;
                        d.recovery_rate[idx] = recovery;
                        d.asset_value[idx] = value;
                        d.factor_sector[idx] = sector_factor;
                        d.factor_idio[idx] = idio;
                    }
                }
            }
        }

        for (idx, pc) in precomp.iter().enumerate() {
            total_loss += loss_amounts[idx];
            sector_losses[pc.sector_index] += loss_amounts[idx];
            if defaulted[idx] {
                total_defaults += 1;
                if let Some(d) = detail.as_ref() {
                    let asset = &assets[idx];
                    default_events.push(DefaultEvent {
                        asset_id: asset.asset_id,
                        sector_id: asset.sector_id,
                        default_period: d.default_period[idx],
                        time_to_default: (d.default_period[idx] as f64 + 1.0) * period_length,
                        loss_amount: loss_amounts[idx],
                        recovery_rate: d.recovery_rate[idx],
                        asset_value: d.asset_value[idx],
                        systematic_factor: d.factor_sector[idx],
                        idiosyncratic_factor: d.factor_idio[idx],
                        pd: asset.pd,
                        lgd_mean: asset.lgd_mean,
                        exposure: asset.exposure,
                    });
                }
            }
        }
    }

    TrialResult {
        trial_id,
        total_loss,
        total_defaults,
        sector_losses,
        default_events,
    }
}

/// Per-asset default detail for the frailty kernel, kept only while a trial
/// runs and only when interim storage was requested.
struct FrailtyEventDetail {
    default_period: Vec<u32>,
    recovery_rate: Vec<f64>,
    asset_value: Vec<f64>,
    factor_sector: Vec<f64>,
    factor_idio: Vec<f64>,
}

impl FrailtyEventDetail {
    fn new(n_assets: usize) -> Self {
        Self {
            default_period: vec![0; n_assets],
            recovery_rate: vec![0.0; n_assets],
            asset_value: vec![0.0; n_assets],
            factor_sector: vec![0.0; n_assets],
            factor_idio: vec![0.0; n_assets],
        }
    }
}

/// LGD Beta mean for an obligor defaulting in `period`: the term-structure value
/// (clamped to its last entry) when present, else the flat `lgd_mean`.
fn lgd_mean_at(asset: &AssetData, period: usize) -> f64 {
    match &asset.lgd_term_structure {
        Some(ts) if !ts.is_empty() => ts[period.min(ts.len() - 1)],
        _ => asset.lgd_mean,
    }
}

/// Compute loss amount and recovery rate for an asset defaulting in `period`,
/// using the asset's precomputed loadings and per-period LGD Beta.
#[inline]
fn compute_loss(
    pc: &AssetPrecomp,
    period: usize,
    sector_factor: f64,
    rng: &mut StdRng,
) -> (f64, f64) {
    // Wrong-way risk: a downturn is a LOW sector factor (defaults fire in the
    // low tail of the latent value), so LGD must load on the NEGATIVE of the
    // factor for a positive `systematic_lgd_correlation` to mean "recoveries
    // fall — LGD rises — exactly when defaults cluster".
    let lgd_sys = -pc.lgd_corr * sector_factor;
    let lgd_idio = pc.lgd_idio_loading * sample_standard_normal(rng);
    let lgd_normal = lgd_sys + lgd_idio;

    use crate::math_utils::normal_cdf;
    let lgd_uniform = normal_cdf(lgd_normal).clamp(1e-12, 1.0 - 1e-12);

    let lb = &pc.betas[period];
    let lgd_realized = beta_inverse_cdf_prepared(lb.alpha, lb.beta, lb.ln_beta_ab, lgd_uniform);

    (lgd_realized * pc.exposure, 1.0 - lgd_realized)
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

/// Portfolio-shape metadata written into the interim Parquet file so the
/// analyzer can recover totals (default rates, trial/asset counts) that the
/// defaults-only rows no longer carry directly.
#[derive(Debug, Clone)]
pub struct InterimMetadata {
    pub n_trials: usize,
    pub n_assets: usize,
    pub sector_names: Vec<String>,
    pub sector_asset_counts: Vec<usize>,
}

fn store_interim_results(
    trial_results: &[TrialResult],
    output_path: &str,
    sector_names: &[String],
    batch_size: Option<usize>,
    meta: &InterimMetadata,
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
        meta,
    )
    .map_err(|e| PortfolioError::StorageError(e.to_string()))
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
            1e6,
            "Technology".to_string(),
            None,
            None,
            None,
        )
        .unwrap();
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
            None,
            None,
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
            intra_sector_correlation: None,
            lgd_term_structure: None,
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
            intra_sector_correlation: None,
            lgd_term_structure: None,
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
