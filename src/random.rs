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

pub fn sample_normal(rng: &mut StdRng, mean: f64, std: f64) -> Result<f64, String> {
    let normal = Normal::new(mean, std)
        .map_err(|e| format!("Invalid normal distribution parameters: {}", e))?;
    Ok(rng.sample(normal))
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
