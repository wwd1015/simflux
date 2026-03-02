"""Random number generation utilities and correlation matrix helpers."""

import numpy as np
from typing import List, Optional
import random


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducible results across Python and NumPy.
    
    Parameters:
    -----------
    seed : int
        Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def generate_correlation_matrix(n: int, 
                               correlation_strength: float = 0.3,
                               random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate a valid correlation matrix with specified average correlation strength.
    
    Parameters:
    -----------
    n : int
        Size of correlation matrix (n x n)
    correlation_strength : float, default=0.3
        Target average off-diagonal correlation (0 to 1)
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray
        Valid correlation matrix (positive definite, symmetric)
    """
    rng = np.random.default_rng(random_state)

    if not 0 <= correlation_strength <= 1:
        raise ValueError("correlation_strength must be between 0 and 1")

    if n < 2:
        raise ValueError("n must be at least 2")

    # Generate random matrix
    A = rng.normal(0, 1, (n, n))
    
    # Make it symmetric
    A = (A + A.T) / 2
    
    # Ensure positive definiteness by adding to diagonal
    eigenvals = np.linalg.eigvals(A)
    min_eigenval = np.min(eigenvals)
    if min_eigenval <= 0:
        A += np.eye(n) * (abs(min_eigenval) + 0.1)
    
    # Convert to correlation matrix
    D = np.sqrt(np.diag(A))
    correlation_matrix = A / np.outer(D, D)
    
    # Adjust correlation strength
    correlation_matrix = adjust_correlation_strength(
        correlation_matrix, 
        target_strength=correlation_strength
    )
    
    # Final validation and adjustment
    correlation_matrix = ensure_valid_correlation_matrix(correlation_matrix)
    
    return correlation_matrix


def adjust_correlation_strength(matrix: np.ndarray, 
                              target_strength: float) -> np.ndarray:
    """
    Adjust a correlation matrix to have a target average correlation strength.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input correlation matrix
    target_strength : float
        Target average off-diagonal correlation
        
    Returns:
    --------
    np.ndarray
        Adjusted correlation matrix
    """
    n = matrix.shape[0]
    
    # Calculate current average correlation (off-diagonal)
    off_diagonal_mask = ~np.eye(n, dtype=bool)
    current_strength = np.mean(np.abs(matrix[off_diagonal_mask]))
    
    if current_strength == 0:
        return np.eye(n)
    
    # Scale off-diagonal elements
    scaling_factor = target_strength / current_strength
    adjusted = matrix.copy()
    adjusted[off_diagonal_mask] *= scaling_factor
    
    # Ensure diagonal is 1
    np.fill_diagonal(adjusted, 1.0)
    
    return adjusted


def ensure_valid_correlation_matrix(matrix: np.ndarray, 
                                   max_iterations: int = 100) -> np.ndarray:
    """
    Ensure a matrix is a valid correlation matrix through eigenvalue adjustment.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix
    max_iterations : int, default=100
        Maximum iterations for adjustment
        
    Returns:
    --------
    np.ndarray
        Valid correlation matrix
    """
    adjusted = matrix.copy()
    
    for _ in range(max_iterations):
        # Check if already valid
        eigenvals = np.linalg.eigvals(adjusted)
        if np.all(eigenvals > 1e-8):
            break
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(adjusted)
        
        # Adjust negative eigenvalues
        eigenvals = np.maximum(eigenvals, 1e-8)
        
        # Reconstruct matrix
        adjusted = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Normalize to correlation matrix
        diag_sqrt = np.sqrt(np.diag(adjusted))
        adjusted = adjusted / np.outer(diag_sqrt, diag_sqrt)
        
        # Ensure diagonal is exactly 1
        np.fill_diagonal(adjusted, 1.0)
    
    return adjusted


def create_block_correlation_matrix(block_sizes: List[int],
                                   intra_block_corr: float,
                                   inter_block_corr: float = 0.0) -> np.ndarray:
    """
    Create a block correlation matrix with different correlations within/between blocks.
    
    Useful for sector-based correlation structures.
    
    Parameters:
    -----------
    block_sizes : list
        Size of each block (e.g., [10, 5, 8] for 3 blocks)
    intra_block_corr : float
        Correlation within blocks
    inter_block_corr : float, default=0.0
        Correlation between blocks
        
    Returns:
    --------
    np.ndarray
        Block correlation matrix
    """
    total_size = sum(block_sizes)
    correlation_matrix = np.full((total_size, total_size), inter_block_corr)
    
    # Fill diagonal blocks
    start_idx = 0
    for block_size in block_sizes:
        end_idx = start_idx + block_size
        
        # Set intra-block correlations
        correlation_matrix[start_idx:end_idx, start_idx:end_idx] = intra_block_corr
        
        start_idx = end_idx
    
    # Set diagonal to 1
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Validate and adjust if necessary
    correlation_matrix = ensure_valid_correlation_matrix(correlation_matrix)
    
    return correlation_matrix


def correlation_from_factor_loadings(factor_loadings: np.ndarray) -> np.ndarray:
    """
    Create correlation matrix from factor loadings (factor model approach).
    
    Parameters:
    -----------
    factor_loadings : np.ndarray
        Matrix of shape (n_assets, n_factors) with factor loadings
        
    Returns:
    --------
    np.ndarray
        Implied correlation matrix
    """
    # Correlation = L @ L.T + D where D is diagonal of specific variances
    correlation_matrix = factor_loadings @ factor_loadings.T
    
    # Add specific variance to make diagonal 1
    diagonal_adjustment = 1.0 - np.diag(correlation_matrix)
    correlation_matrix += np.diag(diagonal_adjustment)
    
    return correlation_matrix


# Validation functions
def validate_correlation_matrix_strict(
    matrix: np.ndarray,
    name: str = "correlation_matrix",
    check_positive_definite: bool = True,
    pd_tolerance: float = 1e-12,
) -> None:
    """Validate a correlation matrix, raising ValueError on failure.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to validate.
    name : str
        Label used in error messages.
    check_positive_definite : bool
        If True, require strictly positive definite (eigenvalues > pd_tolerance).
        If False, allow positive semi-definite.
    pd_tolerance : float
        Threshold for eigenvalue positivity check.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")

    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric")

    if not np.allclose(np.diag(matrix), 1.0):
        raise ValueError(f"{name} diagonal must be 1.0")

    if np.any(np.abs(matrix) > 1.0):
        raise ValueError(f"{name} values must be between -1 and 1")

    eigenvals = np.linalg.eigvals(matrix)
    if check_positive_definite:
        if np.any(eigenvals <= pd_tolerance):
            raise ValueError(f"{name} must be positive definite")
    else:
        if np.any(eigenvals < -pd_tolerance):
            raise ValueError(f"{name} must be positive semi-definite")


def validate_correlation_matrix(matrix: np.ndarray, tolerance: float = 1e-8) -> bool:
    """
    Validate if a matrix is a proper correlation matrix.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Matrix to validate
    tolerance : float, default=1e-8
        Numerical tolerance for checks
        
    Returns:
    --------
    bool
        True if valid correlation matrix
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Check symmetry
    if not np.allclose(matrix, matrix.T, atol=tolerance):
        return False
    
    # Check diagonal
    if not np.allclose(np.diag(matrix), 1.0, atol=tolerance):
        return False
    
    # Check bounds
    if np.any(matrix < -1 - tolerance) or np.any(matrix > 1 + tolerance):
        return False
    
    # Check positive semi-definite
    eigenvals = np.linalg.eigvals(matrix)
    if np.any(eigenvals < -tolerance):
        return False
    
    return True