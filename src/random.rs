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

/// Generate `n_samples` correlated normal vectors as a single flat, sample-major
/// buffer: element (sample `s`, factor `i`) lives at `s * n_factors + i`.
///
/// Returning one contiguous `Vec<f64>` instead of a `Vec<Vec<f64>>` avoids
/// `n_samples` tiny heap allocations, and seeding ONE RNG per chunk (rather than
/// per sample — the dominant cost) keeps the result deterministic for a given
/// seed while making generation O(chunks) RNG initializations instead of
/// O(n_samples).
pub fn generate_correlated_normals_batch(
    n_samples: usize,
    n_factors: usize,
    cholesky: &[Vec<f64>],
    seed: Option<u64>,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_samples * n_factors];
    if n_samples == 0 || n_factors == 0 {
        return out;
    }

    // One RNG per chunk of samples (seeded by chunk index) — deterministic across
    // runs and across thread scheduling, since each chunk owns a fixed sample range.
    const SAMPLES_PER_CHUNK: usize = 4096;
    out.par_chunks_mut(SAMPLES_PER_CHUNK * n_factors)
        .enumerate()
        .for_each(|(chunk_idx, slice)| {
            let mut rng = create_rng(seed, chunk_idx as u64);
            let mut independent = vec![0.0_f64; n_factors];
            let n_in_chunk = slice.len() / n_factors;
            for s in 0..n_in_chunk {
                for z in independent.iter_mut() {
                    *z = sample_standard_normal(&mut rng);
                }
                let base = s * n_factors;
                for i in 0..n_factors {
                    let mut acc = 0.0;
                    for j in 0..=i {
                        acc += cholesky[i][j] * independent[j];
                    }
                    slice[base + i] = acc;
                }
            }
        });
    out
}
