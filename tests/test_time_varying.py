"""Tests for simplified time-varying parameter functionality."""

import pytest
import numpy as np
import sys
import os

# Add the python directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from unittest.mock import patch, MagicMock
import simflux as sf
from simflux.processes.time_varying import (
    TimeVaryingGBM,
    TimeVaryingCorrelatedGBM
)


class TestTimeVaryingGBM:
    """Test single-asset time-varying GBM."""

    def test_constant_parameters(self):
        """With constant time series, should behave like regular GBM."""
        gbm = TimeVaryingGBM(
            mu_times=[0.0, 1.0],
            mu_values=[0.05, 0.05],
            sigma_times=[0.0, 1.0],
            sigma_values=[0.2, 0.2],
            S0=100
        )

        paths = gbm.simulate(n_paths=10, n_steps=5, T=1.0)
        assert paths.shape == (10, 6)
        assert np.all(paths[:, 0] == 100.0)
        assert np.all(paths > 0)

    def test_varying_parameters(self):
        """Test with time-varying mu and sigma."""
        gbm = TimeVaryingGBM(
            mu_times=[0.0, 0.5, 1.0],
            mu_values=[0.05, 0.10, 0.03],
            sigma_times=[0.0, 1.0],
            sigma_values=[0.2, 0.3],
            S0=100
        )

        paths = gbm.simulate(n_paths=100, n_steps=252, T=1.0)
        assert paths.shape == (100, 253)
        assert np.all(paths[:, 0] == 100.0)
        assert np.all(paths > 0)

    def test_get_parameters_at_time(self):
        """Test linear interpolation of parameters."""
        gbm = TimeVaryingGBM(
            mu_times=[0.0, 1.0],
            mu_values=[0.04, 0.08],
            sigma_times=[0.0, 1.0],
            sigma_values=[0.20, 0.30],
            S0=100
        )

        mu, sigma = gbm.get_parameters_at_time(0.0)
        assert mu == pytest.approx(0.04)
        assert sigma == pytest.approx(0.20)

        mu, sigma = gbm.get_parameters_at_time(1.0)
        assert mu == pytest.approx(0.08)
        assert sigma == pytest.approx(0.30)

        mu, sigma = gbm.get_parameters_at_time(0.5)
        assert mu == pytest.approx(0.06)
        assert sigma == pytest.approx(0.25)

    def test_extrapolation_clamps(self):
        """Values outside the time range should clamp to boundary values."""
        gbm = TimeVaryingGBM(
            mu_times=[0.5, 1.5],
            mu_values=[0.04, 0.08],
            sigma_times=[0.5, 1.5],
            sigma_values=[0.20, 0.30],
            S0=100
        )

        mu, sigma = gbm.get_parameters_at_time(0.0)
        assert mu == pytest.approx(0.04)
        assert sigma == pytest.approx(0.20)

        mu, sigma = gbm.get_parameters_at_time(5.0)
        assert mu == pytest.approx(0.08)
        assert sigma == pytest.approx(0.30)

    def test_multiple_time_points(self):
        """Test interpolation with many breakpoints."""
        gbm = TimeVaryingGBM(
            mu_times=[0.0, 0.25, 0.75, 1.0],
            mu_values=[0.08, -0.15, 0.12, 0.08],
            sigma_times=[0.0, 0.25, 0.75, 1.0],
            sigma_values=[0.20, 0.45, 0.25, 0.20],
            S0=100
        )

        mu, sigma = gbm.get_parameters_at_time(0.25)
        assert mu == pytest.approx(-0.15)
        assert sigma == pytest.approx(0.45)

        mu, sigma = gbm.get_parameters_at_time(0.5)
        assert mu == pytest.approx((-0.15 + 0.12) / 2)
        assert sigma == pytest.approx((0.45 + 0.25) / 2)

    def test_validation_mismatched_lengths(self):
        """mu_times and mu_values must have same length."""
        with pytest.raises(ValueError, match="mu_times and mu_values must have same length"):
            TimeVaryingGBM(
                mu_times=[0.0, 1.0],
                mu_values=[0.05],
                sigma_times=[0.0, 1.0],
                sigma_values=[0.2, 0.3],
                S0=100
            )

        with pytest.raises(ValueError, match="sigma_times and sigma_values must have same length"):
            TimeVaryingGBM(
                mu_times=[0.0, 1.0],
                mu_values=[0.05, 0.08],
                sigma_times=[0.0],
                sigma_values=[0.2, 0.3],
                S0=100
            )

    def test_validation_simulation_inputs(self):
        """Simulation inputs must be positive."""
        gbm = TimeVaryingGBM(
            mu_times=[0.0, 1.0],
            mu_values=[0.05, 0.05],
            sigma_times=[0.0, 1.0],
            sigma_values=[0.2, 0.2],
            S0=100
        )

        with pytest.raises(ValueError, match="n_paths must be positive"):
            gbm.simulate(n_paths=0, n_steps=10, T=1.0)

        with pytest.raises(ValueError, match="n_steps must be positive"):
            gbm.simulate(n_paths=10, n_steps=0, T=1.0)

        with pytest.raises(ValueError, match="T must be positive"):
            gbm.simulate(n_paths=10, n_steps=10, T=0)


class TestTimeVaryingCorrelatedGBM:
    """Test multi-asset time-varying GBM."""

    def test_constant_parameters(self):
        """Test with constant time series for both assets."""
        gbm = TimeVaryingCorrelatedGBM(
            mu_times=[[0.0, 1.0], [0.0, 1.0]],
            mu_values=[[0.05, 0.05], [0.03, 0.03]],
            sigma_times=[[0.0, 1.0], [0.0, 1.0]],
            sigma_values=[[0.2, 0.2], [0.15, 0.15]],
            S0=[100, 50],
            correlation_matrix=np.array([[1.0, 0.3], [0.3, 1.0]])
        )

        paths = gbm.simulate(n_paths=10, n_steps=5, T=1.0)
        assert paths.shape == (10, 2, 6)
        assert np.all(paths[:, 0, 0] == 100)
        assert np.all(paths[:, 1, 0] == 50)
        assert np.all(paths > 0)

    def test_varying_parameters(self):
        """Test with different time series per asset."""
        gbm = TimeVaryingCorrelatedGBM(
            mu_times=[[0.0, 0.5, 1.0], [0.0, 1.0]],
            mu_values=[[0.15, -0.10, 0.20], [0.06, 0.08]],
            sigma_times=[[0.0, 0.3, 1.0], [0.0, 0.5, 1.0]],
            sigma_values=[[0.25, 0.60, 0.30], [0.18, 0.35, 0.22]],
            S0=[100, 100],
            correlation_matrix=np.array([[1.0, 0.4], [0.4, 1.0]])
        )

        paths = gbm.simulate(n_paths=100, n_steps=20, T=1.0)
        assert paths.shape == (100, 2, 21)
        assert np.all(paths > 0)

    def test_get_parameters_at_time(self):
        """Test parameter interpolation for multi-asset."""
        gbm = TimeVaryingCorrelatedGBM(
            mu_times=[[0.0, 1.0], [0.0, 1.0]],
            mu_values=[[0.04, 0.08], [0.02, 0.06]],
            sigma_times=[[0.0, 1.0], [0.0, 1.0]],
            sigma_values=[[0.20, 0.30], [0.10, 0.20]],
            S0=[100, 50],
            correlation_matrix=np.array([[1.0, 0.3], [0.3, 1.0]])
        )

        mu, sigma = gbm.get_parameters_at_time(0.5)
        np.testing.assert_array_almost_equal(mu, [0.06, 0.04])
        np.testing.assert_array_almost_equal(sigma, [0.25, 0.15])

    def test_correlation_preserved(self):
        """Realized correlation should be close to target with enough paths."""
        target_corr = 0.5
        gbm = TimeVaryingCorrelatedGBM(
            mu_times=[[0.0, 1.0], [0.0, 1.0]],
            mu_values=[[0.05, 0.05], [0.05, 0.05]],
            sigma_times=[[0.0, 1.0], [0.0, 1.0]],
            sigma_values=[[0.2, 0.2], [0.2, 0.2]],
            S0=[100, 100],
            correlation_matrix=np.array([[1.0, target_corr], [target_corr, 1.0]])
        )

        paths = gbm.simulate(n_paths=5000, n_steps=252, T=1.0)
        returns_0 = paths[:, 0, -1] / paths[:, 0, 0] - 1
        returns_1 = paths[:, 1, -1] / paths[:, 1, 0] - 1
        realized = np.corrcoef(returns_0, returns_1)[0, 1]

        assert abs(realized - target_corr) < 0.1

    def test_validation_mismatched_assets(self):
        """Must provide time series for each asset."""
        with pytest.raises(ValueError, match="Must provide mu time series for each asset"):
            TimeVaryingCorrelatedGBM(
                mu_times=[[0.0, 1.0]],
                mu_values=[[0.05, 0.05]],
                sigma_times=[[0.0, 1.0], [0.0, 1.0]],
                sigma_values=[[0.2, 0.2], [0.15, 0.15]],
                S0=[100, 50],
                correlation_matrix=np.array([[1.0, 0.3], [0.3, 1.0]])
            )


class TestRealWorldScenarios:
    """Test realistic time-varying scenarios."""

    def test_market_crash_scenario(self):
        """Normal -> Crash -> Recovery."""
        gbm = TimeVaryingGBM(
            mu_times=[0.0, 0.25, 0.5, 1.0],
            mu_values=[0.08, -0.30, 0.15, 0.08],
            sigma_times=[0.0, 0.25, 0.5, 1.0],
            sigma_values=[0.18, 0.60, 0.30, 0.20],
            S0=100
        )

        paths = gbm.simulate(n_paths=100, n_steps=100, T=1.0)
        assert paths.shape == (100, 101)
        assert np.all(paths > 0)
        assert np.all(paths[:, 0] == 100)

    def test_gradual_transition(self):
        """Slow linear change over 2 years."""
        gbm = TimeVaryingGBM(
            mu_times=[0.0, 2.0],
            mu_values=[0.03, 0.12],
            sigma_times=[0.0, 2.0],
            sigma_values=[0.15, 0.25],
            S0=100
        )

        paths = gbm.simulate(n_paths=50, n_steps=504, T=2.0)
        assert paths.shape == (50, 505)
        assert np.all(paths > 0)

    def test_cyclical_parameters(self):
        """Build cyclical time series from numpy arrays."""
        years = np.linspace(0, 5, 21)
        cycle_mu = 0.06 + 0.04 * np.sin(2 * np.pi * years / 5)
        cycle_sigma = 0.18 - 0.08 * np.sin(2 * np.pi * years / 5)

        gbm = TimeVaryingGBM(
            mu_times=years,
            mu_values=cycle_mu,
            sigma_times=years,
            sigma_values=cycle_sigma,
            S0=100
        )

        paths = gbm.simulate(n_paths=50, n_steps=1260, T=5.0)
        assert paths.shape == (50, 1261)
        assert np.all(paths > 0)

        mu_start, sigma_start = gbm.get_parameters_at_time(0.0)
        mu_end, sigma_end = gbm.get_parameters_at_time(5.0)
        assert mu_start == pytest.approx(mu_end, abs=0.01)
        assert sigma_start == pytest.approx(sigma_end, abs=0.01)

    def test_multi_asset_crisis(self):
        """Tech and utility stocks with different crisis impacts."""
        gbm = TimeVaryingCorrelatedGBM(
            mu_times=[[0.0, 0.3, 1.0], [0.0, 0.4, 1.0]],
            mu_values=[[0.15, -0.10, 0.25], [0.05, 0.02, 0.06]],
            sigma_times=[[0.0, 0.3, 1.0], [0.0, 0.4, 1.0]],
            sigma_values=[[0.30, 0.65, 0.35], [0.12, 0.20, 0.14]],
            S0=[100, 100],
            correlation_matrix=np.array([[1.0, 0.3], [0.3, 1.0]])
        )

        paths = gbm.simulate(n_paths=100, n_steps=252, T=1.0)
        assert paths.shape == (100, 2, 253)
        assert np.all(paths > 0)
