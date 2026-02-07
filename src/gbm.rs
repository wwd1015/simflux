use crate::random::{create_rng, sample_standard_normal};
use crate::correlation::{generate_correlated_normals, CorrelationError};
use rayon::prelude::*;

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
    
    // Validate inputs
    assert_eq!(sigma.len(), n_assets);
    assert_eq!(s0.len(), n_assets);
    assert_eq!(correlation_matrix.len(), n_assets);
    
    // Precompute constants
    let drift: Vec<f64> = mu.iter()
        .zip(sigma.iter())
        .map(|(&m, &s)| (m - 0.5 * s * s) * dt)
        .collect();
    
    let vol_sqrt_dt: Vec<f64> = sigma.iter()
        .map(|&s| s * dt.sqrt())
        .collect();
    
    // Generate all correlated random numbers at once for efficiency
    let total_randoms = n_paths * n_steps;
    let correlated_randoms = generate_correlated_normals(
        total_randoms,
        correlation_matrix,
        seed,
    )?;
    
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

            // Generate path steps
            for step in 0..n_steps {
                let random_idx = path_idx * n_steps + step;
                let randoms = &correlated_randoms[random_idx];

                for asset_idx in 0..n_assets {
                    let dw = randoms[asset_idx];
                    current_values[asset_idx] *= (drift[asset_idx] + vol_sqrt_dt[asset_idx] * dw).exp();
                    paths[asset_idx].push(current_values[asset_idx]);
                }
            }

            paths
        })
        .collect();

    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gbm_single_basic() {
        let paths = simulate_gbm_single(0.05, 0.2, 100.0, 10, 252, 1.0/252.0, Some(42));
        
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
        let correlation_matrix = vec![
            vec![1.0, 0.3],
            vec![0.3, 1.0]
        ];
        
        let paths = simulate_gbm_correlated(
            mu, sigma, s0, correlation_matrix, 
            5, 10, 1.0/252.0, Some(42)
        ).unwrap();
        
        assert_eq!(paths.len(), 5); // n_paths
        assert_eq!(paths[0].len(), 2); // n_assets
        assert_eq!(paths[0][0].len(), 11); // n_steps + 1
        
        // Check starting values
        assert_eq!(paths[0][0][0], 100.0);
        assert_eq!(paths[0][1][0], 50.0);
    }
}
