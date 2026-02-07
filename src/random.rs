use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Normal, StandardNormal};
use rayon::prelude::*;

/// Create a reproducible RNG stream for a given index.
pub fn create_rng(seed: Option<u64>, stream: u64) -> StdRng {
    match seed {
        Some(base) => StdRng::seed_from_u64(base.wrapping_add(stream)),
        None => StdRng::from_entropy(),
    }
}

pub fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    rng.sample(StandardNormal)
}

pub fn sample_normal(rng: &mut StdRng, mean: f64, std: f64) -> f64 {
    let normal = Normal::new(mean, std).unwrap();
    rng.sample(normal)
}

pub fn generate_independent_normals(n: usize, seed: Option<u64>) -> Vec<f64> {
    (0..n)
        .into_par_iter()
        .map(|idx| {
            let mut rng = create_rng(seed, idx as u64);
            sample_standard_normal(&mut rng)
        })
        .collect()
}

pub fn generate_correlated_normals_batch(
    n_samples: usize,
    n_factors: usize,
    cholesky: &[Vec<f64>],
    seed: Option<u64>,
) -> Vec<Vec<f64>> {
    (0..n_samples)
        .into_par_iter()
        .map(|sample_idx| {
            let mut rng = create_rng(seed, sample_idx as u64);

            // Generate independent standard normals
            let independent: Vec<f64> = (0..n_factors)
                .map(|_| sample_standard_normal(&mut rng))
                .collect();

            // Apply Cholesky transformation
            let mut correlated = vec![0.0; n_factors];
            for i in 0..n_factors {
                for j in 0..=i {
                    correlated[i] += cholesky[i][j] * independent[j];
                }
            }
            correlated
        })
        .collect()
}
