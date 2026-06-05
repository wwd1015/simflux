"""Cross-validation tests: Rust vs NumPy fallback for all simulation types.

These tests run both backends on the same problem and verify that the
distributional statistics converge to the same values.  Because each backend
uses a different RNG stream, individual paths differ — but with enough
samples the means and variances should agree within Monte Carlo tolerance.

SCOPE: this file proves *parity* (the two backends agree), not *correctness*.
A wrong formula shared by both backends passes every assertion here because the
backends still agree with each other.  Model correctness is pinned against
external truth elsewhere: ``test_analytic_vasicek.py`` (ASRF closed form),
``test_behavioral_invariants.py`` (economic invariants), and
``test_lgd_wrong_way.py`` (wrong-way risk sign).  See the
``TestConditionalPDDeterministic`` class below for the deterministic PD pins.
"""

import pytest
import numpy as np
import warnings
from unittest.mock import patch

import simflux as sf
from simflux.core.backend import Backend
from simflux.core.base import SimulationConfig
from simflux.core.engine import SimulationEngine
from simflux.portfolio.two_factor_model import TwoFactorPortfolio

try:
    import scipy  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    # Without scipy the NumPy fallback uses an approximate LGD, so the two
    # backends agree only loosely; with scipy both use the exact Beta LGD.
    _HAS_SCIPY = False


# -----------------------------------------------------------------------
# 1. Single-asset GBM
# -----------------------------------------------------------------------

class TestGBMCrossValidation:
    """Rust vs NumPy for GBM simulation."""

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_single_asset_gbm(self):
        n_paths, n_steps, T = 5000, 252, 1.0
        mu, sigma, S0 = 0.05, 0.2, 100.0

        # Rust
        gbm_rust = sf.GBM(mu=mu, sigma=sigma, S0=S0,
                           config=SimulationConfig(seed=1))
        paths_rust = gbm_rust.simulate(n_paths=n_paths, n_steps=n_steps, T=T)

        # NumPy fallback — cross the SAME public seam real callers use, with the
        # backend forced off, rather than reaching into a private method.
        engine = SimulationEngine(config=SimulationConfig(seed=2))
        with patch.object(Backend, "is_available", return_value=False), \
             patch.object(Backend, "get_rust", return_value=None):
            paths_np = engine.simulate_gbm(mu, sigma, S0, n_paths, n_steps, T)

        # Compare log-return statistics
        log_ret_rust = np.log(paths_rust[:, -1] / S0)
        log_ret_np = np.log(paths_np[:, -1] / S0)

        expected_mean = (mu - 0.5 * sigma**2) * T
        expected_std = sigma * np.sqrt(T)

        # Both should match the theoretical moments
        assert abs(np.mean(log_ret_rust) - expected_mean) < 0.02
        assert abs(np.mean(log_ret_np) - expected_mean) < 0.02
        assert abs(np.std(log_ret_rust) - expected_std) < 0.02
        assert abs(np.std(log_ret_np) - expected_std) < 0.02

        # And agree with each other
        assert abs(np.mean(log_ret_rust) - np.mean(log_ret_np)) < 0.02


# -----------------------------------------------------------------------
# 2. Correlated multi-asset GBM
# -----------------------------------------------------------------------

class TestCorrelatedGBMCrossValidation:
    """Rust vs NumPy for correlated GBM."""

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_correlated_gbm(self):
        n_paths, n_steps, T = 3000, 100, 0.5
        mu = [0.06, 0.04]
        sigma = [0.2, 0.15]
        S0 = [100.0, 50.0]
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])

        # Rust
        gbm_rust = sf.CorrelatedGBM(
            mu=mu, sigma=sigma, S0=S0, correlation_matrix=corr,
            config=SimulationConfig(seed=10),
        )
        paths_rust = gbm_rust.simulate(n_paths=n_paths, n_steps=n_steps, T=T)

        # NumPy — public seam, backend forced off
        engine = SimulationEngine(config=SimulationConfig(seed=20))
        with patch.object(Backend, "is_available", return_value=False), \
             patch.object(Backend, "get_rust", return_value=None):
            paths_np = engine.simulate_gbm_correlated(
                mu, sigma, S0, corr, n_paths, n_steps, T
            )

        # Both should preserve correlation (within MC tolerance)
        for paths, label in [(paths_rust, "rust"), (paths_np, "numpy")]:
            ret0 = (paths[:, 0, -1] / paths[:, 0, 0]) - 1
            ret1 = (paths[:, 1, -1] / paths[:, 1, 0]) - 1
            realized = np.corrcoef(ret0, ret1)[0, 1]
            assert abs(realized - 0.5) < 0.10, f"{label}: realized corr {realized}"


# -----------------------------------------------------------------------
# 3. Time-varying GBM (single + correlated)
# -----------------------------------------------------------------------

class TestTimeVaryingGBMCrossValidation:
    """Rust vs NumPy for time-varying GBM."""

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_time_varying_single(self):
        n_paths, n_steps, T = 5000, 252, 1.0
        mu_t = [0.0, 1.0]; mu_v = [0.05, 0.05]
        sig_t = [0.0, 1.0]; sig_v = [0.2, 0.2]
        S0 = 100.0

        # Rust
        engine_rust = SimulationEngine(config=SimulationConfig(seed=30))
        paths_rust = engine_rust.simulate_gbm_time_varying(
            mu_t, mu_v, sig_t, sig_v, S0, n_paths, n_steps, T
        )

        # NumPy — public seam, backend forced off
        engine_np = SimulationEngine(config=SimulationConfig(seed=40))
        with patch.object(Backend, "is_available", return_value=False), \
             patch.object(Backend, "get_rust", return_value=None):
            paths_np = engine_np.simulate_gbm_time_varying(
                mu_t, mu_v, sig_t, sig_v, S0, n_paths, n_steps, T
            )

        # Constant params → same as regular GBM with mu=0.05, sigma=0.2
        expected_mean = (0.05 - 0.5 * 0.04) * T

        lr_rust = np.log(paths_rust[:, -1] / S0)
        lr_np = np.log(paths_np[:, -1] / S0)

        assert abs(np.mean(lr_rust) - expected_mean) < 0.02
        assert abs(np.mean(lr_np) - expected_mean) < 0.02
        assert abs(np.mean(lr_rust) - np.mean(lr_np)) < 0.02

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_time_varying_single_nonconstant(self):
        """Rust vs NumPy under a NON-constant schedule — exercises interpolation.

        The constant-parameter case above never interpolates, so it cannot catch
        a grid/interpolation disagreement between the two backends.  A ramped
        schedule forces both to interpolate mu and sigma; their terminal
        log-return distributions must still agree, which is only true if both
        interpolate the same schedule onto the same time grid.
        """
        n_paths, n_steps, T = 8000, 252, 1.0
        mu_t = [0.0, 0.5, 1.0]; mu_v = [0.02, 0.12, 0.04]
        sig_t = [0.0, 0.5, 1.0]; sig_v = [0.15, 0.35, 0.20]
        S0 = 100.0

        engine = SimulationEngine(config=SimulationConfig(seed=70))
        paths_rust = engine.simulate_gbm_time_varying(
            mu_t, mu_v, sig_t, sig_v, S0, n_paths, n_steps, T
        )
        with patch.object(Backend, "is_available", return_value=False), \
             patch.object(Backend, "get_rust", return_value=None):
            paths_np = engine.simulate_gbm_time_varying(
                mu_t, mu_v, sig_t, sig_v, S0, n_paths, n_steps, T
            )

        lr_rust = np.log(paths_rust[:, -1] / S0)
        lr_np = np.log(paths_np[:, -1] / S0)

        # Pin the NumPy terminal log-mean to the analytic LINEAR-interpolated
        # integral. A step/nearest interpolation regression shifts this by ~0.005,
        # so this tolerance (well above the deterministic same-seed MC residual)
        # fails on a non-linear interpolation bug.
        grid = np.linspace(0, T, n_steps + 1)[:-1]
        dt = T / n_steps
        mu_lin = np.interp(grid, mu_t, mu_v)
        sig_lin = np.interp(grid, sig_t, sig_v)
        expected_lin = float(np.sum((mu_lin - 0.5 * sig_lin ** 2) * dt))
        assert abs(np.mean(lr_np) - expected_lin) < 0.004

        # Fixed seeds → Rust and NumPy must agree near the MC-noise floor; tight
        # enough to catch a one-sided grid/interpolation disagreement.
        assert abs(np.mean(lr_rust) - np.mean(lr_np)) < 0.005
        assert abs(np.std(lr_rust) - np.std(lr_np)) < 0.005

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_time_varying_correlated(self):
        n_paths, n_steps, T = 3000, 100, 1.0
        corr = np.array([[1.0, 0.4], [0.4, 1.0]])

        engine_rust = SimulationEngine(config=SimulationConfig(seed=50))
        paths_rust = engine_rust.simulate_gbm_time_varying_correlated(
            mu_times=[[0, 1], [0, 1]], mu_values=[[0.05, 0.05], [0.03, 0.03]],
            sigma_times=[[0, 1], [0, 1]], sigma_values=[[0.2, 0.2], [0.15, 0.15]],
            s0=[100, 50], correlation_matrix=corr,
            n_paths=n_paths, n_steps=n_steps, T=T,
        )

        engine_np = SimulationEngine(config=SimulationConfig(seed=60))
        with patch.object(Backend, "is_available", return_value=False), \
             patch.object(Backend, "get_rust", return_value=None):
            paths_np = engine_np.simulate_gbm_time_varying_correlated(
                mu_times=[[0, 1], [0, 1]], mu_values=[[0.05, 0.05], [0.03, 0.03]],
                sigma_times=[[0, 1], [0, 1]], sigma_values=[[0.2, 0.2], [0.15, 0.15]],
                s0=[100, 50], correlation_matrix=corr,
                n_paths=n_paths, n_steps=n_steps, T=T,
            )

        for paths, label in [(paths_rust, "rust"), (paths_np, "numpy")]:
            ret0 = (paths[:, 0, -1] / paths[:, 0, 0]) - 1
            ret1 = (paths[:, 1, -1] / paths[:, 1, 0]) - 1
            realized = np.corrcoef(ret0, ret1)[0, 1]
            assert abs(realized - 0.4) < 0.10, f"{label}: realized corr {realized}"


# -----------------------------------------------------------------------
# 4. Portfolio loss (single-period + multi-period) — also in
#    test_multi_period.py::TestRustNumpyCrossValidation
# -----------------------------------------------------------------------

class TestPortfolioCrossValidation:
    """Rust vs NumPy for portfolio loss simulation."""

    @staticmethod
    def _make_portfolio(seed=None):
        assets = [
            sf.AssetData(i, i % 2, 0.06, 0.5, 0.1, 1_000_000,
                         "A" if i % 2 == 0 else "B")
            for i in range(20)
        ]
        return sf.TwoFactorPortfolio(
            assets=assets, intra_sector_correlations=0.35,
            config=SimulationConfig(seed=seed),
        )

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_single_period_portfolio(self):
        n_sims = 10_000

        r_rust = self._make_portfolio(seed=100).simulate(n_simulations=n_sims)

        with patch.object(Backend, "is_available", return_value=False), \
             patch.object(Backend, "get_rust", return_value=None):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r_np = self._make_portfolio(seed=200).simulate(n_simulations=n_sims)

        # Result-contract parity: both backends return exactly the same key set
        # (this is the seam the dual implementation must keep in lockstep).
        assert set(r_rust) == set(r_np)
        assert r_rust["n_trials"] == r_np["n_trials"] == n_sims
        assert set(r_rust["portfolio_statistics"]) == set(r_np["portfolio_statistics"])
        assert set(r_rust["sector_statistics"]) == set(r_np["sector_statistics"])

        rust_stats = r_rust["portfolio_statistics"]
        np_stats = r_np["portfolio_statistics"]

        # Distributional agreement on more than the mean.  The tail (var_95) and
        # the per-sector means catch a misplaced loading or a dropped sector that
        # a single mean comparison would hide.
        assert abs(rust_stats["mean"] - np_stats["mean"]) / max(np_stats["mean"], 1.0) < 0.30
        assert abs(rust_stats["var_95"] - np_stats["var_95"]) / max(np_stats["var_95"], 1.0) < 0.40
        for sector in r_np["sector_statistics"]:
            rs = r_rust["sector_statistics"][sector]["mean"]
            ns = r_np["sector_statistics"][sector]["mean"]
            assert abs(rs - ns) / max(ns, 1.0) < 0.35

        # With scipy both backends draw the exact Beta LGD, so they agree tightly.
        if _HAS_SCIPY:
            assert abs(rust_stats["mean"] - np_stats["mean"]) / max(np_stats["mean"], 1.0) < 0.15
            assert abs(rust_stats["var_99"] - np_stats["var_99"]) / max(np_stats["var_99"], 1.0) < 0.25

    def test_singular_sector_matrix_rejected_on_both_backends(self):
        """A singular (PSD-but-not-PD) sector matrix must be rejected identically
        regardless of backend — not silently repaired by NumPy while Rust raises.

        Guards the parity of the dual implementation at construction time: the
        full sector matrix must be positive definite (the Rust Cholesky requires
        it), so an in-bounds but rank-deficient matrix is rejected up front on
        either backend rather than diverging at simulate time.
        """
        assets = [
            sf.AssetData(i, i % 2, 0.06, 0.5, 0.1, 1_000_000,
                         "A" if i % 2 == 0 else "B")
            for i in range(4)
        ]
        singular = np.array([[1.0, 1.0], [1.0, 1.0]])  # eigenvalues [0, 2]: PSD, singular

        with pytest.raises(ValueError, match="positive definite"):
            sf.TwoFactorPortfolio(assets=assets, sector_correlation_matrix=singular)

        with patch.object(Backend, "is_available", return_value=False), \
             patch.object(Backend, "get_rust", return_value=None):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(ValueError, match="positive definite"):
                    sf.TwoFactorPortfolio(assets=assets, sector_correlation_matrix=singular)

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_multi_period_portfolio(self):
        n_sims = 10_000
        ts = [0.015, 0.03, 0.045, 0.06]

        assets = [
            sf.AssetData(i, i % 2, 0.06, 0.5, 0.1, 1_000_000,
                         "A" if i % 2 == 0 else "B",
                         pd_term_structure=ts)
            for i in range(20)
        ]

        p_rust = sf.TwoFactorPortfolio(
            assets=assets, intra_sector_correlations=0.35,
            config=SimulationConfig(seed=300),
        )
        r_rust = p_rust.simulate(n_simulations=n_sims, n_periods=4, period_length=0.25)

        with patch.object(Backend, "is_available", return_value=False), \
             patch.object(Backend, "get_rust", return_value=None):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_np = sf.TwoFactorPortfolio(
                    assets=assets, intra_sector_correlations=0.35,
                    config=SimulationConfig(seed=400),
                )
                r_np = p_np.simulate(n_simulations=n_sims, n_periods=4, period_length=0.25)

        rust_mean = r_rust["portfolio_statistics"]["mean"]
        np_mean = r_np["portfolio_statistics"]["mean"]
        assert rust_mean > 0 and np_mean > 0
        assert abs(rust_mean - np_mean) / max(np_mean, 1.0) < 0.30
        # With scipy both backends use the exact Beta LGD → tighter agreement.
        if _HAS_SCIPY:
            assert abs(rust_mean - np_mean) / max(np_mean, 1.0) < 0.15


# -----------------------------------------------------------------------
# 5. Deterministic conditional-PD pin (no Monte Carlo, no backend)
# -----------------------------------------------------------------------

class TestConditionalPDDeterministic:
    """Exact checks of the per-period conditional-PD derivation.

    Both backends share this PD-wiring logic (the Python helpers mirror the
    Rust ``get_conditional_pd`` line for line).  Pinning it with closed-form
    equalities catches a wiring bug deterministically, instead of relying on it
    surfacing below the distributional cross-check tolerance.
    """

    def test_flat_single_period(self):
        assert TwoFactorPortfolio._conditional_pd_flat(0.1, 1) == pytest.approx(0.1)

    def test_flat_constant_hazard_compounds_back(self):
        pd, n = 0.10, 4
        cond = TwoFactorPortfolio._conditional_pd_flat(pd, n)
        # A constant per-period hazard must compound back to the cumulative PD.
        assert 1 - (1 - cond) ** n == pytest.approx(pd)

    def test_term_structure_forward_pd(self):
        ts = [0.02, 0.05, 0.08, 0.10]
        f = TwoFactorPortfolio._conditional_pd_from_term_structure
        assert f(ts, 0) == pytest.approx(0.02)
        assert f(ts, 1) == pytest.approx((0.05 - 0.02) / (1 - 0.02))
        assert f(ts, 3) == pytest.approx((0.10 - 0.08) / (1 - 0.08))

    def test_term_structure_saturated(self):
        # Once cumulative PD reaches 1.0 the forward PD is 0.
        assert TwoFactorPortfolio._conditional_pd_from_term_structure([1.0, 1.0], 1) == 0.0
