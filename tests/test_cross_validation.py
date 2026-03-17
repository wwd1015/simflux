"""Cross-validation tests: Rust vs NumPy fallback for all simulation types.

These tests run both backends on the same problem and verify that the
distributional statistics converge to the same values.  Because each backend
uses a different RNG stream, individual paths differ — but with enough
samples the means and variances should agree within Monte Carlo tolerance.
"""

import pytest
import numpy as np
import warnings
from unittest.mock import patch

import simflux as sf
from simflux.core.backend import Backend
from simflux.core.base import SimulationConfig
from simflux.core.engine import SimulationEngine


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

        # NumPy fallback
        engine = SimulationEngine(config=SimulationConfig(seed=2))
        paths_np = engine._numpy_simulate_gbm(mu, sigma, S0, n_paths, n_steps, T)

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

        # NumPy
        engine = SimulationEngine(config=SimulationConfig(seed=20))
        paths_np = engine._numpy_simulate_gbm_correlated(
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

        # NumPy
        engine_np = SimulationEngine(config=SimulationConfig(seed=40))
        paths_np = engine_np._numpy_simulate_gbm_time_varying(
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
        paths_np = engine_np._numpy_simulate_gbm_time_varying_correlated(
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

        rust_mean = r_rust["portfolio_statistics"]["mean"]
        np_mean = r_np["portfolio_statistics"]["mean"]
        assert abs(rust_mean - np_mean) / max(np_mean, 1.0) < 0.30

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
