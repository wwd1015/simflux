use crate::random::{create_rng, generate_correlated_normals_batch, sample_standard_normal};
use nalgebra::{Cholesky, DMatrix};
use rayon::prelude::*;

#[derive(Debug)]
pub enum CorrelationError {
    InvalidMatrix(String),
    DecompositionFailed,
    DimensionMismatch,
}

impl std::fmt::Display for CorrelationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CorrelationError::InvalidMatrix(msg) => {
                write!(f, "Invalid correlation matrix: {}", msg)
            }
            CorrelationError::DecompositionFailed => write!(f, "Cholesky decomposition failed"),
            CorrelationError::DimensionMismatch => write!(f, "Dimension mismatch"),
        }
    }
}

impl std::error::Error for CorrelationError {}

pub fn validate_correlation_matrix(matrix: &[Vec<f64>]) -> Result<(), CorrelationError> {
    let n = matrix.len();

    // Check if square
    for row in matrix {
        if row.len() != n {
            return Err(CorrelationError::InvalidMatrix(
                "Matrix is not square".to_string(),
            ));
        }
    }

    // Check diagonal elements
    for i in 0..n {
        if (matrix[i][i] - 1.0).abs() > 1e-8 {
            return Err(CorrelationError::InvalidMatrix(
                "Diagonal elements must be 1.0".to_string(),
            ));
        }
    }

    // Check symmetry
    for i in 0..n {
        for j in 0..n {
            if (matrix[i][j] - matrix[j][i]).abs() > 1e-8 {
                return Err(CorrelationError::InvalidMatrix(
                    "Matrix is not symmetric".to_string(),
                ));
            }
        }
    }

    // Check correlation bounds
    for i in 0..n {
        for j in 0..n {
            if matrix[i][j].abs() > 1.0 {
                return Err(CorrelationError::InvalidMatrix(
                    "Correlation values must be between -1 and 1".to_string(),
                ));
            }
        }
    }

    Ok(())
}

pub fn cholesky_decomposition(matrix: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, CorrelationError> {
    validate_correlation_matrix(&matrix)?;

    let n = matrix.len();
    let flat: Vec<f64> = matrix.into_iter().flatten().collect();
    let nalgebra_matrix = DMatrix::from_row_slice(n, n, &flat);

    match Cholesky::new(nalgebra_matrix) {
        Some(chol) => {
            let l = chol.l();
            let mut result = vec![vec![0.0; n]; n];

            for i in 0..n {
                for j in 0..n {
                    result[i][j] = l[(i, j)];
                }
            }

            Ok(result)
        }
        None => Err(CorrelationError::DecompositionFailed),
    }
}

/// Correlated standard normals as a flat, sample-major `Vec<f64>` of length
/// `n_samples * n_factors` (factor `i` of sample `s` at `s * n_factors + i`).
pub fn generate_correlated_normals(
    n_samples: usize,
    correlation_matrix: Vec<Vec<f64>>,
    seed: Option<u64>,
) -> Result<Vec<f64>, CorrelationError> {
    let cholesky = cholesky_decomposition(correlation_matrix)?;
    let n_factors = cholesky.len();

    Ok(generate_correlated_normals_batch(
        n_samples, n_factors, &cholesky, seed,
    ))
}

pub struct TwoFactorCorrelationStructure {
    pub intra_sector_correlations: Vec<f64>,
    pub sector_sizes: Vec<usize>,
    sector_cholesky: Vec<Vec<f64>>,
}

impl TwoFactorCorrelationStructure {
    pub fn new(
        intra_sector_correlations: Vec<f64>,
        sector_sizes: Vec<usize>,
        sector_correlation_matrix: Option<Vec<Vec<f64>>>,
    ) -> Result<Self, CorrelationError> {
        let n_sectors = sector_sizes.len();
        if n_sectors == 0 {
            return Err(CorrelationError::InvalidMatrix(
                "At least one sector is required".to_string(),
            ));
        }

        if intra_sector_correlations.len() != sector_sizes.len() {
            return Err(CorrelationError::DimensionMismatch);
        }

        for &corr in &intra_sector_correlations {
            if !(0.0..=1.0).contains(&corr) {
                return Err(CorrelationError::InvalidMatrix(
                    "Intra-sector correlations must be between 0 and 1".to_string(),
                ));
            }
        }

        let matrix = sector_correlation_matrix.ok_or_else(|| {
            CorrelationError::InvalidMatrix(
                "sector correlation matrix must be provided".to_string(),
            )
        })?;

        if matrix.len() != n_sectors {
            return Err(CorrelationError::DimensionMismatch);
        }
        for row in &matrix {
            if row.len() != n_sectors {
                return Err(CorrelationError::InvalidMatrix(
                    "sector correlation matrix must be square".to_string(),
                ));
            }
        }

        let sector_matrix = matrix;

        let sector_cholesky = cholesky_decomposition(sector_matrix)?;

        Ok(Self {
            intra_sector_correlations,
            sector_sizes,
            sector_cholesky,
        })
    }

    pub fn generate_factors(
        &self,
        n_simulations: usize,
        seed: Option<u64>,
    ) -> Vec<SystematicFactors> {
        let n_sectors = self.sector_sizes.len();
        let cholesky = &self.sector_cholesky;

        (0..n_simulations)
            .into_par_iter()
            .map(|sim_idx| {
                let mut rng = create_rng(seed, sim_idx as u64);
                let independent: Vec<f64> = (0..n_sectors)
                    .map(|_| sample_standard_normal(&mut rng))
                    .collect();

                let mut sector_factors = vec![0.0; n_sectors];
                for i in 0..n_sectors {
                    for j in 0..=i {
                        sector_factors[i] += cholesky[i][j] * independent[j];
                    }
                }

                SystematicFactors { sector_factors }
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct SystematicFactors {
    pub sector_factors: Vec<f64>,
}

impl SystematicFactors {
    pub fn get_sector_factor(&self, sector_id: usize) -> f64 {
        self.sector_factors[sector_id]
    }
}
