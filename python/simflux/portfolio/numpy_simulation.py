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

LGD may be time-varying: ``lgd_means_by_period`` carries one Beta mean per period
per obligor, and an obligor's realized LGD is drawn from the Beta for the period
it defaults in (``lgd_std`` is constant).  When the mean is constant across
periods this reduces exactly to the time-invariant model.
"""

from __future__ import annotations

import warnings
from math import erf
from typing import Any, Dict, Optional

import numpy as np

from ..core.base import check_memory
from .inputs import PortfolioInputs

# Max f64 elements in one dense (chunk, n_assets) scratch array. The reduction
# below summarizes each chunk of simulations, so this caps peak scratch instead
# of letting it scale with n_simulations. ~32K elements keeps the dozen-odd live
# intermediates to a few MB total with no measurable throughput loss (NumPy
# vectorization saturates well below this batch size); larger caps only inflate
# peak memory.
FALLBACK_CHUNK_ELEMENTS = 32_768


def simulate_portfolio_numpy(inputs: PortfolioInputs) -> Dict[str, Any]:
    """Run the portfolio loss simulation in vectorized NumPy.

    ``inputs`` is the backend seam's whole contract — the same value the Rust
    adapter consumes, validated at construction (shapes, plan/period
    consistency).  Its timing plan is the sole timing input: this adapter never
    derives thresholds, it only selects the simulation kernel the plan names
    and compares latents against ``plan.thresholds``.  Returns
    ``{"portfolio_statistics", "sector_statistics"}``; the caller completes the
    :class:`PortfolioResult` contract (one stamping site, next to the TypedDict).
    """
    config = inputs.config
    sector_names = inputs.sector_names
    asset_intra_correlations = inputs.asset_intra_correlations
    systematic_lgd_correlations = inputs.systematic_lgd_correlations
    asset_sector_ids = inputs.asset_sector_ids
    lgd_means_by_period = inputs.lgd_means_by_period
    asset_lgd_stds = inputs.asset_lgd_stds
    asset_exposures = inputs.asset_exposures
    n_simulations = inputs.n_simulations
    plan = inputs.plan
    n_periods = plan.n_periods
    try:
        # betaincinv IS beta.ppf's kernel for a standard (loc=0, scale=1) Beta —
        # identical values — but scipy.special imports in a fraction of
        # scipy.stats's time, which matters on the first simulate() of a session.
        from scipy.special import betaincinv, erf as scipy_erf

        def beta_ppf(q, a, b):  # scipy.stats argument order
            return betaincinv(a, b, q)

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

    # The validated CorrelationMatrix value carries its own (cached) sampling
    # factor — no raw decomposition at this seam.
    sector_cholesky = inputs.sector_correlation.cholesky()

    asset_sector_ids = np.asarray(asset_sector_ids)
    asset_lgd_stds = np.asarray(asset_lgd_stds)
    asset_exposures = np.asarray(asset_exposures)
    lgd_means_by_period = np.asarray(lgd_means_by_period, dtype=float)

    # Per-obligor intra-sector correlation (heterogeneous loadings within a sector
    # are allowed); already resolved to one value per asset by the caller.
    asset_intra_corrs = np.asarray(asset_intra_correlations)
    sector_loadings = np.sqrt(asset_intra_corrs)
    idio_loadings = np.sqrt(np.maximum(0.0, 1 - asset_intra_corrs))

    # Per-sector LGD-systematic correlation, expanded to one value per asset.
    asset_lgd_corrs = np.asarray(systematic_lgd_correlations)[asset_sector_ids]
    lgd_idio_scale = np.sqrt(np.maximum(0.0, 1 - asset_lgd_corrs**2))

    # Beta (alpha, beta) per period per obligor, from the period's mean and the
    # obligor's (constant) lgd_std. Shapes: (n_periods, n_assets).
    _var = asset_lgd_stds[np.newaxis, :] ** 2
    _common = lgd_means_by_period * (1 - lgd_means_by_period) / _var - 1
    lgd_alpha_by_period = lgd_means_by_period * _common
    lgd_beta_by_period = (1 - lgd_means_by_period) * _common

    # LGD inverse-CDF, accepting per-element (alpha, beta). With scipy, beta.ppf is
    # vectorized and exact. Without scipy, tabulate I_x(alpha, beta) once per
    # *distinct* (alpha, beta) pair seen (cached across chunks and periods) and
    # invert by interpolation — so chunking never recomputes a table.
    if have_scipy:

        def _inv_beta(u: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
            return beta_ppf(u, alpha, beta)

    else:
        from ..utils.special import reg_incomplete_beta as _reg_incomplete_beta

        _lgd_grid = np.linspace(0.0, 1.0, 8192)
        _cdf_cache: Dict[tuple, np.ndarray] = {}

        def _beta_table(av: float, bv: float) -> np.ndarray:
            cdf = _cdf_cache.get((av, bv))
            if cdf is None:
                cdf = _reg_incomplete_beta(av, bv, _lgd_grid)
                np.maximum.accumulate(cdf, out=cdf)  # monotone for np.interp
                _cdf_cache[(av, bv)] = cdf
            return cdf

        if np.allclose(lgd_means_by_period, lgd_means_by_period[0:1]):
            # Time-invariant LGD (the common case): the per-element (alpha, beta)
            # passed in are constant down each obligor column, so tabulate once per
            # column and interpolate per column — no per-call np.unique grouping.
            _col_cdf = [
                _beta_table(
                    float(lgd_alpha_by_period[0, j]), float(lgd_beta_by_period[0, j])
                )
                for j in range(n_assets)
            ]

            def _inv_beta(
                u: np.ndarray, alpha: np.ndarray, beta: np.ndarray
            ) -> np.ndarray:
                out = np.empty_like(u)
                for j in range(u.shape[-1]):
                    out[..., j] = np.interp(u[..., j], _col_cdf[j], _lgd_grid)
                return out

        else:
            # Time-varying LGD: (alpha, beta) vary per element (by default period),
            # so group the distinct pairs and interpolate each group.
            def _inv_beta(
                u: np.ndarray, alpha: np.ndarray, beta: np.ndarray
            ) -> np.ndarray:
                u = np.asarray(u, dtype=float)
                alpha = np.broadcast_to(np.asarray(alpha, dtype=float), u.shape)
                beta = np.broadcast_to(np.asarray(beta, dtype=float), u.shape)
                out = np.empty_like(u)
                flat_out = out.ravel()
                flat_u = u.ravel()
                pairs = np.stack([alpha.ravel(), beta.ravel()], axis=1)
                uniq, inv_idx = np.unique(pairs, axis=0, return_inverse=True)
                inv_idx = inv_idx.ravel()
                for p_i in range(len(uniq)):
                    sel = inv_idx == p_i
                    flat_out[sel] = np.interp(
                        flat_u[sel],
                        _beta_table(float(uniq[p_i, 0]), float(uniq[p_i, 1])),
                        _lgd_grid,
                    )
                return out

    def _lgd_from_normal(
        lgd_normal: np.ndarray, alpha: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """Map a standard-normal LGD driver through the Gaussian copula to a
        Beta-distributed realized LGD with the given (per-element) Beta params."""
        lgd_uniform = np.clip(
            0.5 * (1 + erf_func(lgd_normal / np.sqrt(2))), 1e-12, 1 - 1e-12
        )
        return _inv_beta(lgd_uniform, alpha, beta)

    # Per-period thresholds come from the timing plan for BOTH kernels — the
    # staircase/barrier derivation lives in default_timing, not here, and the
    # plan/LGD shape consistency was validated by PortfolioInputs.
    thresholds_by_period = plan.thresholds
    asset_cols = np.arange(n_assets)

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
        # First period whose cumulative threshold is crossed = the default period.
        default_period = np.zeros((n_chunk, n_assets), dtype=np.intp)
        ever = np.zeros((n_chunk, n_assets), dtype=bool)
        for k in range(n_periods):
            newly = (latent <= thresholds_by_period[k][np.newaxis, :]) & ~ever
            if k > 0:
                default_period[newly] = k
            ever |= newly
        if not np.any(ever):
            return np.zeros((n_chunk, n_assets))
        lgd_sys = -asset_lgd_corrs[np.newaxis, :] * asset_sector_factors
        lgd_idio = lgd_idio_scale[np.newaxis, :] * rng.normal(size=(n_chunk, n_assets))
        alpha = lgd_alpha_by_period[default_period, asset_cols[np.newaxis, :]]
        beta = lgd_beta_by_period[default_period, asset_cols[np.newaxis, :]]
        lgd_realized = _lgd_from_normal(lgd_sys + lgd_idio, alpha, beta)
        return np.where(ever, lgd_realized * asset_exposures[np.newaxis, :], 0.0)

    def _frailty_chunk_losses(n_chunk: int) -> np.ndarray:
        # Dynamic frailty: a persistent (AR(1)) systematic factor with fresh
        # idiosyncratic shocks each period; first-passage against the plan's
        # pre-calibrated barriers (which preserve the marginal PD for any phi).
        factor_phi = plan.factor_phi
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

            thresholds = thresholds_by_period[period]  # (n_assets,) calibrated barrier
            new_defaults = (asset_values <= thresholds[np.newaxis, :]) & ~ever_defaulted

            if np.any(new_defaults):
                # Wrong-way risk: LGD loads on the NEGATIVE of the (per-sector)
                # factor, so positive systematic_lgd_correlation => higher LGD in
                # stress.  LGD Beta params are this period's.
                lgd_sys = -asset_lgd_corrs[np.newaxis, :] * asset_sector_factors
                lgd_idio = lgd_idio_scale[np.newaxis, :] * rng.normal(
                    size=(n_chunk, n_assets)
                )
                lgd_realized = _lgd_from_normal(
                    lgd_sys + lgd_idio,
                    lgd_alpha_by_period[period][np.newaxis, :],
                    lgd_beta_by_period[period][np.newaxis, :],
                )
                period_losses = lgd_realized * asset_exposures[np.newaxis, :]
                chunk_losses = np.where(new_defaults, period_losses, chunk_losses)

            ever_defaulted |= new_defaults
        return chunk_losses

    chunk_losses_fn = (
        _copula_chunk_losses if plan.kernel == "copula" else _frailty_chunk_losses
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
