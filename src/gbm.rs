use crate::correlation::{cholesky_decomposition, generate_correlated_normals, CorrelationError};
use crate::random::{create_rng, sample_standard_normal};
use rayon::prelude::*;

/// Linear interpolation with clamping outside the provided time range.
pub fn linear_interp(t: f64, times: &[f64], values: &[f64]) -> f64 {
    if times.is_empty() {
        return 0.0;
    }
    if t <= times[0] {
        return values[0];
    }
    let n = times.len();
    if t >= times[n - 1] {
        return values[n - 1];
    }
    // Binary search for the interval containing t
    let mut lo = 0;
    let mut hi = n - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if times[mid] <= t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let frac = (t - times[lo]) / (times[hi] - times[lo]);
    values[lo] + frac * (values[hi] - values[lo])
}

pub fn simulate_gbm_single(
    mu: f64,
    sigma: f64,
    s0: f64,
    n_paths: usize,
    n_steps: usize,
    dt: f64,
    seed: Option<u64>,
) -> Vec<Vec<f64>> {
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let vol_sqrt_dt = sigma * dt.sqrt();

    (0..n_paths)
        .into_par_iter()
        .map(|path_idx| {
            let mut rng = create_rng(seed, path_idx as u64);
            let mut path = Vec::with_capacity(n_steps + 1);
            path.push(s0);
            let mut current_value = s0;

            for _ in 0..n_steps {
                let dw = sample_standard_normal(&mut rng);
                current_value *= (drift + vol_sqrt_dt * dw).exp();
                path.push(current_value);
            }

            path
        })
        .collect()
}

pub fn simulate_gbm_correlated(
    mu: Vec<f64>,
    sigma: Vec<f64>,
    s0: Vec<f64>,
    correlation_matrix: Vec<Vec<f64>>,
    n_paths: usize,
    n_steps: usize,
    dt: f64,
    seed: Option<u64>,
) -> Result<Vec<Vec<Vec<f64>>>, CorrelationError> {
    let n_assets = mu.len();

    if sigma.len() != n_assets {
        return Err(CorrelationError::InvalidMatrix(format!(
            "sigma length {} does not match mu length {}",
            sigma.len(),
            n_assets
        )));
    }
    if s0.len() != n_assets {
        return Err(CorrelationError::InvalidMatrix(format!(
            "s0 length {} does not match mu length {}",
            s0.len(),
            n_assets
        )));
    }
    if correlation_matrix.len() != n_assets {
        return Err(CorrelationError::InvalidMatrix(format!(
            "correlation_matrix size {} does not match n_assets {}",
            correlation_matrix.len(),
            n_assets
        )));
    }

    // Precompute constants
    let drift: Vec<f64> = mu
        .iter()
        .zip(sigma.iter())
        .map(|(&m, &s)| (m - 0.5 * s * s) * dt)
        .collect();

    let vol_sqrt_dt: Vec<f64> = sigma.iter().map(|&s| s * dt.sqrt()).collect();

    // Generate all correlated random numbers at once for efficiency
    let total_randoms = n_paths * n_steps;
    let correlated_randoms = generate_correlated_normals(total_randoms, correlation_matrix, seed)?;

    // Simulate paths
    let paths = (0..n_paths)
        .into_par_iter()
        .map(|path_idx| {
            let mut paths = vec![Vec::with_capacity(n_steps + 1); n_assets];
            let mut current_values = s0.clone();

            // Initialize paths with starting values
            for asset_idx in 0..n_assets {
                paths[asset_idx].push(s0[asset_idx]);
            }

            // Generate path steps. `correlated_randoms` is a flat sample-major
            // buffer; sample (path, step) starts at `(path*n_steps + step)*n_assets`.
            for step in 0..n_steps {
                let base = (path_idx * n_steps + step) * n_assets;

                for asset_idx in 0..n_assets {
                    let dw = correlated_randoms[base + asset_idx];
                    current_values[asset_idx] *=
                        (drift[asset_idx] + vol_sqrt_dt[asset_idx] * dw).exp();
                    paths[asset_idx].push(current_values[asset_idx]);
                }
            }

            paths
        })
        .collect();

    Ok(paths)
}

// ---------------------------------------------------------------------------
// Time-varying GBM
// ---------------------------------------------------------------------------

pub fn simulate_gbm_time_varying_single(
    mu_times: &[f64],
    mu_values: &[f64],
    sigma_times: &[f64],
    sigma_values: &[f64],
    s0: f64,
    n_paths: usize,
    n_steps: usize,
    dt: f64,
    t_start: f64,
    seed: Option<u64>,
) -> Vec<Vec<f64>> {
    // Pre-compute mu and sigma at each simulation time step
    let mu_interp: Vec<f64> = (0..n_steps)
        .map(|i| linear_interp(t_start + i as f64 * dt, mu_times, mu_values))
        .collect();
    let sigma_interp: Vec<f64> = (0..n_steps)
        .map(|i| linear_interp(t_start + i as f64 * dt, sigma_times, sigma_values))
        .collect();

    let sqrt_dt = dt.sqrt();

    (0..n_paths)
        .into_par_iter()
        .map(|path_idx| {
            let mut rng = create_rng(seed, path_idx as u64);
            let mut path = Vec::with_capacity(n_steps + 1);
            path.push(s0);
            let mut current = s0;

            for i in 0..n_steps {
                let mu = mu_interp[i];
                let sigma = sigma_interp[i];
                let drift = (mu - 0.5 * sigma * sigma) * dt;
                let vol_sqrt_dt = sigma * sqrt_dt;
                let dw = sample_standard_normal(&mut rng);
                current *= (drift + vol_sqrt_dt * dw).exp();
                path.push(current);
            }

            path
        })
        .collect()
}

pub fn simulate_gbm_time_varying_correlated(
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
) -> Result<Vec<Vec<Vec<f64>>>, CorrelationError> {
    let n_assets = s0.len();

    if mu_times.len() != n_assets || mu_values.len() != n_assets {
        return Err(CorrelationError::InvalidMatrix(
            "mu_times/mu_values length must match number of assets".to_string(),
        ));
    }
    if sigma_times.len() != n_assets || sigma_values.len() != n_assets {
        return Err(CorrelationError::InvalidMatrix(
            "sigma_times/sigma_values length must match number of assets".to_string(),
        ));
    }

    // Pre-compute parameters at each time step for each asset
    let mu_interp: Vec<Vec<f64>> = (0..n_assets)
        .map(|a| {
            (0..n_steps)
                .map(|i| linear_interp(t_start + i as f64 * dt, &mu_times[a], &mu_values[a]))
                .collect()
        })
        .collect();
    let sigma_interp: Vec<Vec<f64>> = (0..n_assets)
        .map(|a| {
            (0..n_steps)
                .map(|i| linear_interp(t_start + i as f64 * dt, &sigma_times[a], &sigma_values[a]))
                .collect()
        })
        .collect();

    let cholesky = cholesky_decomposition(correlation_matrix)?;
    let sqrt_dt = dt.sqrt();

    let paths = (0..n_paths)
        .into_par_iter()
        .map(|path_idx| {
            let mut rng = create_rng(seed, path_idx as u64);
            let mut asset_paths = vec![Vec::with_capacity(n_steps + 1); n_assets];
            let mut current_values = s0.clone();

            for a in 0..n_assets {
                asset_paths[a].push(s0[a]);
            }

            for i in 0..n_steps {
                // Generate independent normals and apply Cholesky
                let independent: Vec<f64> = (0..n_assets)
                    .map(|_| sample_standard_normal(&mut rng))
                    .collect();

                let mut correlated = vec![0.0; n_assets];
                for a in 0..n_assets {
                    for j in 0..=a {
                        correlated[a] += cholesky[a][j] * independent[j];
                    }
                }

                for a in 0..n_assets {
                    let mu = mu_interp[a][i];
                    let sigma = sigma_interp[a][i];
                    let drift = (mu - 0.5 * sigma * sigma) * dt;
                    let vol_sqrt_dt = sigma * sqrt_dt;
                    current_values[a] *= (drift + vol_sqrt_dt * correlated[a]).exp();
                    asset_paths[a].push(current_values[a]);
                }
            }

            asset_paths
        })
        .collect();

    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gbm_single_basic() {
        let paths = simulate_gbm_single(0.05, 0.2, 100.0, 10, 252, 1.0 / 252.0, Some(42));

        assert_eq!(paths.len(), 10);
        assert_eq!(paths[0].len(), 253); // n_steps + 1
        assert_eq!(paths[0][0], 100.0); // Starting value

        // Check that all values are positive
        for path in &paths {
            for &value in path {
                assert!(value > 0.0);
            }
        }
    }

    #[test]
    fn test_gbm_correlated_basic() {
        let mu = vec![0.05, 0.03];
        let sigma = vec![0.2, 0.15];
        let s0 = vec![100.0, 50.0];
        let correlation_matrix = vec![vec![1.0, 0.3], vec![0.3, 1.0]];

        let paths = simulate_gbm_correlated(
            mu,
            sigma,
            s0,
            correlation_matrix,
            5,
            10,
            1.0 / 252.0,
            Some(42),
        )
        .unwrap();

        assert_eq!(paths.len(), 5); // n_paths
        assert_eq!(paths[0].len(), 2); // n_assets
        assert_eq!(paths[0][0].len(), 11); // n_steps + 1

        // Check starting values
        assert_eq!(paths[0][0][0], 100.0);
        assert_eq!(paths[0][1][0], 50.0);
    }

    #[test]
    fn test_gbm_correlated_dimension_mismatch() {
        let result = simulate_gbm_correlated(
            vec![0.05, 0.03],
            vec![0.2], // wrong length
            vec![100.0, 50.0],
            vec![vec![1.0, 0.3], vec![0.3, 1.0]],
            5,
            10,
            1.0 / 252.0,
            Some(42),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_interp() {
        let times = vec![0.0, 0.5, 1.0];
        let values = vec![0.04, 0.08, 0.04];

        assert!((linear_interp(0.0, &times, &values) - 0.04).abs() < 1e-12);
        assert!((linear_interp(0.25, &times, &values) - 0.06).abs() < 1e-12);
        assert!((linear_interp(0.5, &times, &values) - 0.08).abs() < 1e-12);
        assert!((linear_interp(1.0, &times, &values) - 0.04).abs() < 1e-12);

        // Clamping
        assert!((linear_interp(-1.0, &times, &values) - 0.04).abs() < 1e-12);
        assert!((linear_interp(5.0, &times, &values) - 0.04).abs() < 1e-12);
    }

    #[test]
    fn test_time_varying_single() {
        let paths = simulate_gbm_time_varying_single(
            &[0.0, 1.0],
            &[0.05, 0.05],
            &[0.0, 1.0],
            &[0.2, 0.2],
            100.0,
            10,
            50,
            1.0 / 50.0,
            0.0,
            Some(42),
        );
        assert_eq!(paths.len(), 10);
        assert_eq!(paths[0].len(), 51);
        assert_eq!(paths[0][0], 100.0);
        for path in &paths {
            for &v in path {
                assert!(v > 0.0);
            }
        }
    }
}
