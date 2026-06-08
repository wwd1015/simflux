"""NumPy adapter for two-factor portfolio loss simulation.

This is the pure-Python implementation behind the portfolio backend seam — the
fallback used when the Rust extension is not installed.  It is a free function
(no simulator object) so it can be tested directly, and so the model module
(:mod:`simflux.portfolio.two_factor_model`) holds the *model* — construction,
validation, PD/LGD assembly, dispatch — without also carrying a second
simulation engine inline.

It mirrors the Rust backend's memory shape: each trial is reduced on the fly to
its total loss and per-sector loss, so retained memory is
``O(n_simulations * n_sectors)``.  Simulations are walked in bounded chunks, so
dense scratch is ``O(chunk * n_assets)`` rather than a full
``n_simulations * n_assets`` grid.  The per-simulation math is identical to the
Rust path; the two agree in distribution, which is what cross-validation asserts.
"""

from __future__ import annotations

import warnings
from math import erf
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.base import SimulationConfig, check_memory
from ..utils.random_utils import approx_norm_ppf, safe_cholesky

# Max f64 elements in one dense (chunk, n_assets) scratch array. The reduction
# below summarizes each chunk of simulations, so this caps peak scratch instead
# of letting it scale with n_simulations. ~32K elements keeps the dozen-odd live
# intermediates to a few MB total with no measurable throughput loss (NumPy
# vectorization saturates well below this batch size); larger caps only inflate
# peak memory.
FALLBACK_CHUNK_ELEMENTS = 32_768


def simulate_portfolio_numpy(
    *,
    config: SimulationConfig,
    sector_correlation_matrix: np.ndarray,
    intra_sector_correlations: Sequence[float],
    systematic_lgd_correlations: Sequence[float],
    sector_names: Sequence[str],
    asset_sector_ids: np.ndarray,
    asset_lgd_means: np.ndarray,
    asset_lgd_stds: np.ndarray,
    asset_exposures: np.ndarray,
    final_cumulative_pd: np.ndarray,
    n_simulations: int,
    n_periods: int,
    default_timing: str,
    factor_phi: float,
    barriers: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Run the portfolio loss simulation in vectorized NumPy.

    Parameters mirror the model's resolved state: ``asset_*`` are per-obligor
    arrays (already expanded from :class:`AssetData`), ``final_cumulative_pd`` is
    the cumulative PD by horizon end per obligor (used by the copula model), and
    ``barriers`` is the calibrated per-period barrier matrix (used by frailty).
    Returns ``{"portfolio_statistics", "sector_statistics"}``; the caller stamps
    the rest of the :class:`PortfolioResult` contract.
    """
    try:
        from scipy import stats as scipy_stats
        from scipy.special import erf as scipy_erf

        norm_ppf = scipy_stats.norm.ppf
        beta_ppf = scipy_stats.beta.ppf
        erf_func = scipy_erf
        have_scipy = True
    except ImportError:
        # No scipy: use dependency-free fallbacks. The Beta inverse-CDF is an
        # accurate tabulated inversion (utils/special), so LGD is correct, not
        # approximate; only the normal-quantile uses an approximation.
        warnings.warn(
            "scipy not available; using built-in special-function fallbacks "
            "(accurate Beta inverse-CDF, approximate normal quantile). Install "
            "scipy for the reference implementation.",
            UserWarning,
        )
        from ..utils.special import beta_ppf as _fallback_beta_ppf

        norm_ppf = approx_norm_ppf
        beta_ppf = _fallback_beta_ppf
        erf_func = np.vectorize(erf)
        have_scipy = False

    rng = np.random.default_rng(config.seed)

    n_assets = len(asset_sector_ids)
    n_sectors = len(sector_names)
    if n_sectors == 0:
        raise ValueError("Portfolio must contain at least one sector")

    # Retained outputs are O(n_simulations * n_sectors) (per-trial total and
    # per-sector loss); dense per-chunk scratch is bounded independently of
    # n_simulations (see FALLBACK_CHUNK_ELEMENTS), so the guard no longer scales
    # with n_assets * n_simulations.
    check_memory(config, n_simulations * (n_sectors + 1) + 8 * FALLBACK_CHUNK_ELEMENTS)

    sector_cholesky = safe_cholesky(
        sector_correlation_matrix, name="sector_correlation_matrix"
    )

    asset_sector_ids = np.asarray(asset_sector_ids)
    asset_lgd_means = np.asarray(asset_lgd_means)
    asset_lgd_stds = np.asarray(asset_lgd_stds)
    asset_exposures = np.asarray(asset_exposures)

    intra_corrs = np.asarray(intra_sector_correlations)
    asset_intra_corrs = intra_corrs[asset_sector_ids]
    sector_loadings = np.sqrt(asset_intra_corrs)
    idio_loadings = np.sqrt(np.maximum(0.0, 1 - asset_intra_corrs))

    # Per-sector LGD-systematic correlation, expanded to one value per asset.
    asset_lgd_corrs = np.asarray(systematic_lgd_correlations)[asset_sector_ids]
    lgd_idio_scale = np.sqrt(np.maximum(0.0, 1 - asset_lgd_corrs**2))

    lgd_alpha = asset_lgd_means * (
        (asset_lgd_means * (1 - asset_lgd_means)) / (asset_lgd_stds**2) - 1
    )
    lgd_beta_param = (1 - asset_lgd_means) * (
        (asset_lgd_means * (1 - asset_lgd_means)) / (asset_lgd_stds**2) - 1
    )

    # Build the LGD inverse-CDF once, so the chunk loop below never recomputes it.
    # With scipy, beta.ppf is cheap and stateless, so call it per chunk. Without
    # scipy, tabulate I_x(alpha, beta) once per *distinct* obligor (alpha, beta)
    # up front and invert by interpolation per chunk — otherwise the (expensive)
    # tabulation would repeat on every chunk.
    if have_scipy:

        def _inv_beta(lgd_uniform: np.ndarray) -> np.ndarray:
            return beta_ppf(
                lgd_uniform,
                lgd_alpha[np.newaxis, :],
                lgd_beta_param[np.newaxis, :],
            )

    else:
        from ..utils.special import reg_incomplete_beta as _reg_incomplete_beta

        _lgd_grid = np.linspace(0.0, 1.0, 8192)
        _cdf_cache: Dict[tuple, np.ndarray] = {}
        _col_cdf: List[np.ndarray] = []
        for _a, _b in zip(lgd_alpha, lgd_beta_param):
            _key = (float(_a), float(_b))
            _cdf = _cdf_cache.get(_key)
            if _cdf is None:
                _cdf = _reg_incomplete_beta(_a, _b, _lgd_grid)
                np.maximum.accumulate(_cdf, out=_cdf)  # monotone for np.interp
                _cdf_cache[_key] = _cdf
            _col_cdf.append(_cdf)

        def _inv_beta(lgd_uniform: np.ndarray) -> np.ndarray:
            out = np.empty_like(lgd_uniform)
            for j in range(lgd_uniform.shape[-1]):
                out[..., j] = np.interp(lgd_uniform[..., j], _col_cdf[j], _lgd_grid)
            return out

    def _lgd_from_normal(lgd_normal: np.ndarray) -> np.ndarray:
        """Map a standard-normal LGD driver through the Gaussian copula to a
        Beta-distributed realized LGD, using the precomputed inverse CDF."""
        lgd_uniform = np.clip(
            0.5 * (1 + erf_func(lgd_normal / np.sqrt(2))), 1e-12, 1 - 1e-12
        )
        return _inv_beta(lgd_uniform)

    if default_timing == "copula":
        # One-factor Gaussian copula of default *times* (Li, 2000): a single latent
        # V per obligor for the whole horizon, thresholded against the cumulative
        # PD. Default by horizon <=> V <= Phi^{-1}(cum_final); the marginal
        # cumulative PD is reproduced exactly for any correlation.
        final_threshold = norm_ppf(np.asarray(final_cumulative_pd))

    # Asset column indices per sector, computed once for the chunk reduction.
    sector_asset_cols = [
        np.flatnonzero(asset_sector_ids == s) for s in range(n_sectors)
    ]
    chunk = max(1, min(n_simulations, FALLBACK_CHUNK_ELEMENTS // max(1, n_assets)))

    def _copula_chunk_losses(n_chunk: int) -> np.ndarray:
        sector_factor_all = rng.normal(size=(n_chunk, n_sectors)) @ sector_cholesky.T
        asset_sector_factors = sector_factor_all[:, asset_sector_ids]
        idiosyncratic = rng.normal(size=(n_chunk, n_assets))
        latent = (
            sector_loadings[np.newaxis, :] * asset_sector_factors
            + idio_loadings[np.newaxis, :] * idiosyncratic
        )
        defaulted = latent <= final_threshold[np.newaxis, :]
        if not np.any(defaulted):
            return np.zeros((n_chunk, n_assets))
        lgd_sys = -asset_lgd_corrs[np.newaxis, :] * asset_sector_factors
        lgd_idio = lgd_idio_scale[np.newaxis, :] * rng.normal(size=(n_chunk, n_assets))
        lgd_realized = _lgd_from_normal(lgd_sys + lgd_idio)
        return np.where(defaulted, lgd_realized * asset_exposures[np.newaxis, :], 0.0)

    def _frailty_chunk_losses(n_chunk: int) -> np.ndarray:
        # Dynamic frailty: a persistent (AR(1)) systematic factor with fresh
        # idiosyncratic shocks each period; first-passage against the pre-calibrated
        # barriers (which preserve the marginal PD for any phi).
        assert barriers is not None  # always computed for default_timing="frailty"
        sqrt_innov = (max(0.0, 1.0 - factor_phi**2)) ** 0.5
        prev_factor: Optional[np.ndarray] = None
        ever_defaulted = np.zeros((n_chunk, n_assets), dtype=bool)
        chunk_losses = np.zeros((n_chunk, n_assets))
        for period in range(n_periods):
            innovation = rng.normal(size=(n_chunk, n_sectors)) @ sector_cholesky.T
            if prev_factor is None:
                sector_factor_all = innovation
            else:
                sector_factor_all = factor_phi * prev_factor + sqrt_innov * innovation
            prev_factor = sector_factor_all
            asset_sector_factors = sector_factor_all[:, asset_sector_ids]

            idiosyncratic = rng.normal(size=(n_chunk, n_assets))
            asset_values = (
                sector_loadings[np.newaxis, :] * asset_sector_factors
                + idio_loadings[np.newaxis, :] * idiosyncratic
            )

            thresholds = barriers[period]  # (n_assets,) calibrated barrier
            new_defaults = (asset_values <= thresholds[np.newaxis, :]) & ~ever_defaulted

            if np.any(new_defaults):
                # Wrong-way risk: LGD loads on the NEGATIVE of the (per-sector)
                # factor, so positive systematic_lgd_correlation => higher LGD in
                # stress.
                lgd_sys = -asset_lgd_corrs[np.newaxis, :] * asset_sector_factors
                lgd_idio = lgd_idio_scale[np.newaxis, :] * rng.normal(
                    size=(n_chunk, n_assets)
                )
                lgd_realized = _lgd_from_normal(lgd_sys + lgd_idio)
                period_losses = lgd_realized * asset_exposures[np.newaxis, :]
                chunk_losses = np.where(new_defaults, period_losses, chunk_losses)

            ever_defaulted |= new_defaults
        return chunk_losses

    chunk_losses_fn = (
        _copula_chunk_losses if default_timing == "copula" else _frailty_chunk_losses
    )

    total_losses = np.empty(n_simulations)
    sector_loss_mat = np.zeros((n_simulations, n_sectors))
    for start in range(0, n_simulations, chunk):
        stop = min(start + chunk, n_simulations)
        chunk_losses = chunk_losses_fn(stop - start)
        total_losses[start:stop] = chunk_losses.sum(axis=1)
        for s, cols in enumerate(sector_asset_cols):
            if cols.size:
                sector_loss_mat[start:stop, s] = chunk_losses[:, cols].sum(axis=1)
        del chunk_losses

    sector_losses: Dict[str, np.ndarray] = {
        sector: sector_loss_mat[:, s_idx] for s_idx, sector in enumerate(sector_names)
    }

    def calculate_stats(loss_arr: np.ndarray) -> Dict[str, float]:
        p95, p99, p999 = np.percentile(loss_arr, [95, 99, 99.9])
        return {
            "mean": float(np.mean(loss_arr)),
            "std_dev": float(np.std(loss_arr)),
            "var_95": float(p95),
            "var_99": float(p99),
            "var_999": float(p999),
            "expected_shortfall_95": float(np.mean(loss_arr[loss_arr >= p95])),
            "expected_shortfall_99": float(np.mean(loss_arr[loss_arr >= p99])),
            "max_loss": float(np.max(loss_arr)),
        }

    return {
        "portfolio_statistics": calculate_stats(total_losses),
        "sector_statistics": {
            s: calculate_stats(loss) for s, loss in sector_losses.items()
        },
    }
