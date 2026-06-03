"""Dynamic frailty default timing: a persistent systematic factor with a
calibrated, time-dependent default barrier.

Model (Duffie, Eckner, Horel & Saita, 2009 — "Frailty Correlated Default"):
the systematic sector factor follows an AR(1) across periods,

    F_k = phi * F_{k-1} + sqrt(1 - phi^2) * eta_k,     F_k ~ N(0, 1),

and, conditional on the factor *path*, defaults are independent (doubly
stochastic).  An obligor's per-period latent is
``V_k = sqrt(rho) F_k + sqrt(1 - rho) eps_k`` with a *fresh* idiosyncratic eps_k,
and it defaults at the first period ``V_k <= b_k``.

The barriers ``b_k`` are **calibrated** so the marginal cumulative PD term
structure is preserved exactly for *any* persistence ``phi``:

    P(default by k) = cum_k        for every k.

This is the non-negotiable step.  A naive AR(1) on the independence-derived
*forward* thresholds biases the cumulative PD downward (because the forward-PD
calibration assumes independence); calibrating the barrier removes that bias.

Because the factor is Markov, the joint survival probability collapses to a 1-D
forward recursion on a grid of factor values, so calibration is a cheap
sequential 1-D root-find:

    psi_0(f)  = s_0(f)
    psi_k(f)  = s_k(f) * E[psi_{k-1}(F_{k-1}) | F_k = f]
    S_k       = E[psi_k(F_k)]   :=  1 - cum_k          (solve for b_k)

where ``s_k(f) = Phi((sqrt(rho) f - b_k) / sqrt(1 - rho))`` is the one-period
survival probability given the factor.  At ``phi = 0`` this reduces exactly to
the independent-period forward-PD thresholds; at ``phi = 1`` the factor is frozen.
"""

from __future__ import annotations

from math import erf, sqrt
from typing import Dict, Tuple

import numpy as np


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Vectorized standard-normal CDF (scipy if present, else erf — no hard dep)."""
    try:
        from scipy.special import erf as _erf  # type: ignore
    except ImportError:
        _erf = np.vectorize(erf)
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / sqrt(2.0)))


def per_period_phi(factor_persistence: float, period_length: float) -> float:
    """Per-period AR(1) coefficient from an *annual* autocorrelation.

    ``phi = factor_persistence ** period_length`` keeps the persistence
    grid-consistent: the same annual cycle correlation maps to the right
    per-period coefficient whatever the step size.
    """
    fp = min(max(float(factor_persistence), 0.0), 1.0)
    return fp ** float(period_length)


def calibrate_barriers(
    cum_pds: np.ndarray,
    rho: float,
    phi: float,
    grid_points: int = 301,
    grid_limit: float = 8.0,
    bisection_iters: int = 50,
) -> np.ndarray:
    """Solve per-period barriers so ``P(default by k) = cum_pds[k]`` exactly.

    Parameters
    ----------
    cum_pds : array
        Cumulative PD by the end of each period (non-decreasing, in [0, 1]).
    rho : float
        Intra-sector (asset) correlation for this obligor.
    phi : float
        Per-period AR(1) persistence of the systematic factor.
    grid_points, grid_limit, bisection_iters :
        Quadrature/root-find controls.

    Returns
    -------
    np.ndarray
        Barriers ``b_0, ..., b_{n-1}`` on the standard-normal latent scale.
    """
    cum = np.clip(np.asarray(cum_pds, dtype=float), 0.0, 1.0)
    cum = np.maximum.accumulate(cum)  # defensive: cumulative PD is non-decreasing
    n = len(cum)
    if n == 0:
        return np.empty(0)

    rho = float(np.clip(rho, 0.0, 1.0 - 1e-9))
    sq_rho, sq_1mrho = sqrt(rho), sqrt(1.0 - rho)
    phi = float(np.clip(phi, 0.0, 1.0))

    f = np.linspace(-grid_limit, grid_limit, grid_points)
    w = np.exp(-0.5 * f ** 2)
    w /= w.sum()  # standard-normal quadrature weights

    # Reverse AR(1) transition kernel T[i, j] = P(F_{k-1}=f_j | F_k=f_i).
    if phi <= 0.0:
        T = np.broadcast_to(w, (grid_points, grid_points))      # independent
    elif phi >= 1.0 - 1e-12:
        T = np.eye(grid_points)                                 # frozen
    else:
        var = 1.0 - phi * phi
        diff = f[None, :] - phi * f[:, None]
        T = np.exp(-0.5 * diff ** 2 / var)
        T /= T.sum(axis=1, keepdims=True)

    def survival(b: float) -> np.ndarray:
        return _norm_cdf((sq_rho * f - b) / sq_1mrho)

    barriers = np.empty(n)
    psi = np.ones(grid_points)  # psi_{-1} = 1
    for k in range(n):
        k_psi = T @ psi
        target = 1.0 - cum[k]
        # S_k(b) = sum_i w_i * survival_i(b) * k_psi_i, monotone DECREASING in b.
        lo, hi = -grid_limit, grid_limit
        for _ in range(bisection_iters):
            mid = 0.5 * (lo + hi)
            s_k = float(np.dot(w, survival(mid) * k_psi))
            if s_k > target:
                lo = mid  # survival too high -> raise the barrier
            else:
                hi = mid
        b_k = 0.5 * (lo + hi)
        barriers[k] = b_k
        psi = survival(b_k) * k_psi

    return barriers


def survival_curve(
    barriers: np.ndarray,
    rho: float,
    phi: float,
    grid_points: int = 301,
    grid_limit: float = 8.0,
) -> np.ndarray:
    """Forward survival probabilities ``S_k = P(survive through period k)`` under
    the AR(1) factor and the given barriers — the deterministic inverse of
    :func:`calibrate_barriers`.  ``1 - survival_curve(calibrate_barriers(cum, ...))``
    recovers ``cum`` (to grid accuracy), with no Monte Carlo.
    """
    b = np.asarray(barriers, dtype=float)
    n = len(b)
    if n == 0:
        return np.empty(0)
    rho = float(np.clip(rho, 0.0, 1.0 - 1e-9))
    sq_rho, sq_1mrho = sqrt(rho), sqrt(1.0 - rho)
    phi = float(np.clip(phi, 0.0, 1.0))

    f = np.linspace(-grid_limit, grid_limit, grid_points)
    w = np.exp(-0.5 * f ** 2)
    w /= w.sum()
    if phi <= 0.0:
        T = np.broadcast_to(w, (grid_points, grid_points))
    elif phi >= 1.0 - 1e-12:
        T = np.eye(grid_points)
    else:
        var = 1.0 - phi * phi
        diff = f[None, :] - phi * f[:, None]
        T = np.exp(-0.5 * diff ** 2 / var)
        T /= T.sum(axis=1, keepdims=True)

    out = np.empty(n)
    psi = np.ones(grid_points)
    for k in range(n):
        psi = _norm_cdf((sq_rho * f - b[k]) / sq_1mrho) * (T @ psi)
        out[k] = float(np.dot(w, psi))
    return out


def barrier_matrix(
    cum_matrix: np.ndarray,
    asset_rhos: np.ndarray,
    phi: float,
) -> np.ndarray:
    """Calibrate barriers for a whole book, caching by ``(rho, cum-curve)``.

    Parameters
    ----------
    cum_matrix : array, shape (n_periods, n_assets)
        Cumulative PD by period end for each asset.
    asset_rhos : array, shape (n_assets,)
        Per-asset intra-sector correlation.
    phi : float
        Per-period AR(1) persistence.

    Returns
    -------
    np.ndarray, shape (n_periods, n_assets)
        Per-asset, per-period barriers.
    """
    n_periods, n_assets = cum_matrix.shape
    out = np.empty((n_periods, n_assets))
    cache: Dict[Tuple, np.ndarray] = {}
    for i in range(n_assets):
        curve = cum_matrix[:, i]
        key = (round(float(asset_rhos[i]), 10), tuple(np.round(curve, 12)))
        b = cache.get(key)
        if b is None:
            b = calibrate_barriers(curve, float(asset_rhos[i]), phi)
            cache[key] = b
        out[:, i] = b
    return out
