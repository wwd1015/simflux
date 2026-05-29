"""Random number generation utilities and correlation matrix helpers."""

import numpy as np
from typing import List, Optional
import random

from ..core.backend import CORRELATION_TOLERANCE


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducible results across Python and NumPy.

    Parameters
    ----------
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

    Parameters
    ----------
    n : int
        Size of correlation matrix (n x n)
    correlation_strength : float, default=0.3
        Target average off-diagonal correlation (0 to 1)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Valid correlation matrix (positive definite, symmetric)
    """
    rng = np.random.default_rng(random_state)

    if not 0 <= correlation_strength <= 1:
        raise ValueError("correlation_strength must be between 0 and 1")

    if n < 2:
        raise ValueError("n must be at least 2")

    A = rng.normal(0, 1, (n, n))
    A = (A + A.T) / 2

    eigenvals = np.linalg.eigvals(A)
    min_eigenval = np.min(eigenvals)
    if min_eigenval <= 0:
        A += np.eye(n) * (abs(min_eigenval) + 0.1)

    D = np.sqrt(np.diag(A))
    correlation_matrix = A / np.outer(D, D)

    correlation_matrix = adjust_correlation_strength(
        correlation_matrix,
        target_strength=correlation_strength
    )

    correlation_matrix = ensure_valid_correlation_matrix(correlation_matrix)

    return correlation_matrix


def adjust_correlation_strength(matrix: np.ndarray,
                              target_strength: float) -> np.ndarray:
    """
    Adjust a correlation matrix to have a target average correlation strength.

    Parameters
    ----------
    matrix : np.ndarray
        Input correlation matrix
    target_strength : float
        Target average off-diagonal correlation

    Returns
    -------
    np.ndarray
        Adjusted correlation matrix
    """
    n = matrix.shape[0]

    off_diagonal_mask = ~np.eye(n, dtype=bool)
    current_strength = np.mean(np.abs(matrix[off_diagonal_mask]))

    if current_strength == 0:
        return np.eye(n)

    scaling_factor = target_strength / current_strength
    adjusted = matrix.copy()
    adjusted[off_diagonal_mask] *= scaling_factor

    np.fill_diagonal(adjusted, 1.0)

    return adjusted


def ensure_valid_correlation_matrix(matrix: np.ndarray,
                                   max_iterations: int = 100) -> np.ndarray:
    """
    Ensure a matrix is a valid correlation matrix through eigenvalue adjustment.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix
    max_iterations : int, default=100
        Maximum iterations for adjustment

    Returns
    -------
    np.ndarray
        Valid correlation matrix
    """
    adjusted = matrix.copy()

    for _ in range(max_iterations):
        eigenvals, eigenvecs = np.linalg.eigh(adjusted)
        if np.all(eigenvals > CORRELATION_TOLERANCE):
            break

        eigenvals = np.maximum(eigenvals, CORRELATION_TOLERANCE)

        adjusted = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        diag_sqrt = np.sqrt(np.diag(adjusted))
        adjusted = adjusted / np.outer(diag_sqrt, diag_sqrt)

        np.fill_diagonal(adjusted, 1.0)

    return adjusted


def safe_cholesky(matrix: np.ndarray, name: str = "correlation_matrix") -> np.ndarray:
    """Decompose ``matrix`` into a sampling factor ``L`` with ``L @ L.T ≈ matrix``.

    This is the single decompose-or-repair primitive used by every correlated
    sampler (correlated GBM, time-varying correlated GBM, the portfolio NumPy
    fallback, and the pure-Python correlation structure).  It defines one repair
    policy and one error mode so the running decomposition has a single home:

    1. Attempt a plain Cholesky decomposition (exact for positive-definite input).
    2. If that fails (near-singular / positive-semi-definite), repair the matrix
       by clamping its eigenvalues to ``CORRELATION_TOLERANCE`` and return the
       resulting square-root factor.
    3. If even the repaired factor is not finite, raise ``ValueError`` — this
       primitive never lets a raw ``numpy.linalg.LinAlgError`` leak to callers.

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric correlation/covariance matrix.
    name : str
        Label used in the error message.

    Returns
    -------
    np.ndarray
        A factor ``L`` such that ``L @ L.T`` reproduces ``matrix``.  For the
        repaired path ``L`` is a square root rather than lower-triangular, which
        is equivalent for generating correlated draws via ``samples @ L.T``.
    """
    matrix = np.asarray(matrix, dtype=float)
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, CORRELATION_TOLERANCE)
        factor = eigenvecs @ np.diag(np.sqrt(eigenvals))
        if not np.all(np.isfinite(factor)):
            raise ValueError(
                f"{name} could not be decomposed: it is not positive semi-definite"
            )
        return factor


def create_block_correlation_matrix(block_sizes: List[int],
                                   intra_block_corr: float,
                                   inter_block_corr: float = 0.0) -> np.ndarray:
    """
    Create a block correlation matrix with different correlations within/between blocks.

    Parameters
    ----------
    block_sizes : list
        Size of each block (e.g., [10, 5, 8] for 3 blocks)
    intra_block_corr : float
        Correlation within blocks
    inter_block_corr : float, default=0.0
        Correlation between blocks

    Returns
    -------
    np.ndarray
        Block correlation matrix
    """
    total_size = sum(block_sizes)
    correlation_matrix = np.full((total_size, total_size), inter_block_corr)

    start_idx = 0
    for block_size in block_sizes:
        end_idx = start_idx + block_size
        correlation_matrix[start_idx:end_idx, start_idx:end_idx] = intra_block_corr
        start_idx = end_idx

    np.fill_diagonal(correlation_matrix, 1.0)

    correlation_matrix = ensure_valid_correlation_matrix(correlation_matrix)

    return correlation_matrix


def correlation_from_factor_loadings(factor_loadings: np.ndarray) -> np.ndarray:
    """
    Create correlation matrix from factor loadings (factor model approach).

    Parameters
    ----------
    factor_loadings : np.ndarray
        Matrix of shape (n_assets, n_factors) with factor loadings

    Returns
    -------
    np.ndarray
        Implied correlation matrix
    """
    correlation_matrix = factor_loadings @ factor_loadings.T

    diagonal_adjustment = 1.0 - np.diag(correlation_matrix)
    correlation_matrix += np.diag(diagonal_adjustment)

    return correlation_matrix


# Validation functions
def approx_norm_ppf(p: np.ndarray) -> np.ndarray:
    """Vectorized inverse normal CDF using Acklam's algorithm.

    Accurate to ~1e-8 across the full (0, 1) range.  Used as a fallback
    when scipy is not installed.
    """
    p = np.asarray(p, dtype=float)
    result = np.empty_like(p)
    lo = p <= 0; hi = p >= 1; mid = ~lo & ~hi
    result[lo] = -10.0; result[hi] = 10.0
    if not np.any(mid):
        return result
    pm = p[mid]
    out = np.empty_like(pm)

    # Central region coefficients
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    # Tail region coefficients
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]

    p_low = 0.02425
    central = (pm >= p_low) & (pm <= 1.0 - p_low)
    tail_lo = pm < p_low
    tail_hi = pm > 1.0 - p_low

    if np.any(central):
        q = pm[central] - 0.5
        r = q * q
        out[central] = (
            (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
        )
    if np.any(tail_lo):
        q = np.sqrt(-2.0 * np.log(pm[tail_lo]))
        out[tail_lo] = (
            (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        )
    if np.any(tail_hi):
        q = np.sqrt(-2.0 * np.log(1.0 - pm[tail_hi]))
        out[tail_hi] = -(
            (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        )

    result[mid] = out
    return result


def validate_correlation_matrix_strict(
    matrix: np.ndarray,
    name: str = "correlation_matrix",
    check_positive_definite: bool = True,
    pd_tolerance: float = CORRELATION_TOLERANCE,
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

    if not np.allclose(matrix, matrix.T, atol=CORRELATION_TOLERANCE):
        raise ValueError(f"{name} must be symmetric")

    if not np.allclose(np.diag(matrix), 1.0, atol=CORRELATION_TOLERANCE):
        raise ValueError(f"{name} diagonal must be 1.0")

    if np.any(np.abs(matrix) > 1.0 + CORRELATION_TOLERANCE):
        raise ValueError(f"{name} values must be between -1 and 1")

    eigenvals = np.linalg.eigvals(matrix)
    if check_positive_definite:
        if np.any(eigenvals <= pd_tolerance):
            raise ValueError(f"{name} must be positive definite")
    else:
        if np.any(eigenvals < -pd_tolerance):
            raise ValueError(f"{name} must be positive semi-definite")


def validate_correlation_matrix(matrix: np.ndarray, tolerance: float = CORRELATION_TOLERANCE) -> bool:
    """
    Validate if a matrix is a proper (positive semi-definite) correlation matrix.

    This is the boolean counterpart of :func:`validate_correlation_matrix_strict`
    and is implemented as a thin ``try/except`` over it, so the two share one
    check body. The structural checks (symmetry, unit diagonal, value range) use
    the shared ``CORRELATION_TOLERANCE``; the ``tolerance`` argument here governs
    only the positive-semi-definiteness check.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to validate
    tolerance : float
        Numerical tolerance for the positive-semi-definiteness check

    Returns
    -------
    bool
        True if valid correlation matrix
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    try:
        validate_correlation_matrix_strict(
            matrix, check_positive_definite=False, pd_tolerance=tolerance
        )
        return True
    except ValueError:
        return False
