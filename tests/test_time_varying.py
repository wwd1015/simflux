"""Tests for time-varying parameter functionality."""

import pytest
import numpy as np
import warnings
from unittest.mock import patch
import sys
import os

# Add the python directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import simflux as sf
from simflux.processes.time_varying import (
    TimeVaryingParameter,
    FunctionalParameter,
    TimeVaryingGBM,
    TimeVaryingCorrelatedGBM,
    create_regime_gbm,
    create_linear_trend_gbm,
    create_cyclical_gbm
)


class TestTimeVaryingParameter:
    """Test time-varying parameter classes."""

    def test_time_varying_parameter_initialization(self):
        """Test TimeVaryingParameter creation."""
        times = [0, 1, 2]
        values = [0.05, 0.08, 0.06]
        param = TimeVaryingParameter(times, values)

        assert len(param.times) == 3
        assert len(param.values) == 3
        np.testing.assert_array_equal(param.times, [0, 1, 2])
        np.testing.assert_array_equal(param.values, [0.05, 0.08, 0.06])

    def test_time_varying_parameter_sorting(self):
        """Test that parameters are sorted by time."""
        times = [2, 0, 1]
        values = [0.06, 0.05, 0.08]
        param = TimeVaryingParameter(times, values)

        np.testing.assert_array_equal(param.times, [0, 1, 2])
        np.testing.assert_array_equal(param.values, [0.05, 0.08, 0.06])

    def test_time_varying_parameter_validation(self):
        """Test parameter validation."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="times and values must have same length"):
            TimeVaryingParameter([0, 1], [0.05])

        # Empty arrays
        with pytest.raises(ValueError, match="Must specify at least one time point"):
            TimeVaryingParameter([], [])

    def test_step_interpolation(self):
        """Test step function interpolation."""
        param = TimeVaryingParameter([0, 1, 2], [0.05, 0.08, 0.06], method='step')

        # Test individual points
        assert param(0.0) == 0.05
        assert param(0.5) == 0.05
        assert param(1.0) == 0.08
        assert param(1.5) == 0.08
        assert param(2.0) == 0.06
        assert param(3.0) == 0.06  # Extrapolation

        # Test array input
        result = param([0.0, 0.5, 1.0, 1.5, 2.0])
        expected = [0.05, 0.05, 0.08, 0.08, 0.06]
        np.testing.assert_array_equal(result, expected)

    def test_linear_interpolation(self):
        """Test linear interpolation."""
        param = TimeVaryingParameter([0, 1, 2], [0.05, 0.08, 0.06], method='linear')

        assert param(0.0) == 0.05
        assert param(0.5) == 0.065  # (0.05 + 0.08) / 2
        assert param(1.0) == 0.08
        assert param(1.5) == 0.07   # (0.08 + 0.06) / 2
        assert param(2.0) == 0.06

    def test_cubic_interpolation_fallback(self):
        """Test cubic interpolation with scipy fallback."""
        param = TimeVaryingParameter([0, 1, 2, 3], [0.05, 0.08, 0.06, 0.07], method='cubic')

        # Should work (either with scipy or fallback to linear)
        result = param(0.5)
        assert isinstance(result, (float, np.floating))

    def test_invalid_method(self):
        """Test invalid interpolation method."""
        param = TimeVaryingParameter([0, 1], [0.05, 0.08], method='invalid')

        with pytest.raises(ValueError, match="Unknown interpolation method"):
            param(0.5)


class TestFunctionalParameter:
    """Test functional parameter class."""

    def test_functional_parameter(self):
        """Test FunctionalParameter with simple function."""
        def linear_func(t):
            return 0.05 + 0.01 * t

        param = FunctionalParameter(linear_func)

        np.testing.assert_almost_equal(param(0.0), 0.05, decimal=10)
        np.testing.assert_almost_equal(param(1.0), 0.06, decimal=10)
        np.testing.assert_almost_equal(param(2.0), 0.07, decimal=10)

        # Test array input
        result = param([0.0, 1.0, 2.0])
        expected = [0.05, 0.06, 0.07]
        np.testing.assert_array_almost_equal(result, expected)

    def test_sinusoidal_function(self):
        """Test with sinusoidal function."""
        def sin_func(t):
            return 0.05 + 0.02 * np.sin(2 * np.pi * t)

        param = FunctionalParameter(sin_func)

        assert abs(param(0.0) - 0.05) < 1e-10
        assert abs(param(0.25) - 0.07) < 1e-10
        assert abs(param(0.5) - 0.05) < 1e-10


class TestTimeVaryingGBM:
    """Test time-varying GBM simulation."""

    def test_time_varying_gbm_constant_parameters(self):
        """Test TimeVaryingGBM with constant parameters (should behave like regular GBM)."""
        # Create with constant parameters
        gbm = TimeVaryingGBM(mu=0.05, sigma=0.2, S0=100)

        # Test parameter functions
        assert gbm.mu_func(0.0) == 0.05
        assert gbm.mu_func(1.0) == 0.05
        assert gbm.sigma_func(0.0) == 0.2
        assert gbm.sigma_func(1.0) == 0.2

        # Test simulation
        paths = gbm.simulate(n_paths=10, n_steps=5, T=1.0)
        assert paths.shape == (10, 6)
        assert np.all(paths[:, 0] == 100.0)

    def test_time_varying_gbm_with_parameter_objects(self):
        """Test TimeVaryingGBM with TimeVaryingParameter objects."""
        mu_param = TimeVaryingParameter([0, 0.5, 1.0], [0.05, 0.08, 0.06])
        sigma_param = TimeVaryingParameter([0, 1.0], [0.2, 0.3])

        gbm = TimeVaryingGBM(mu=mu_param, sigma=sigma_param, S0=100)

        # Test parameter evaluation
        assert gbm.mu_func(0.0) == 0.05
        assert gbm.mu_func(0.5) == 0.08
        assert gbm.sigma_func(0.0) == 0.2
        assert gbm.sigma_func(1.0) == 0.3

        # Test simulation
        paths = gbm.simulate(n_paths=10, n_steps=252, T=1.0)
        assert paths.shape == (10, 253)
        assert np.all(paths[:, 0] == 100.0)
        assert np.all(paths > 0)  # Prices should stay positive

    def test_time_varying_gbm_validation(self):
        """Test TimeVaryingGBM parameter validation."""
        # Invalid S0
        with pytest.raises(ValueError, match="S0 must be positive"):
            TimeVaryingGBM(mu=0.05, sigma=0.2, S0=0)

        with pytest.raises(ValueError, match="S0 must be positive"):
            TimeVaryingGBM(mu=0.05, sigma=0.2, S0=-100)

        # Invalid parameter type
        with pytest.raises(TypeError, match="Invalid parameter type"):
            TimeVaryingGBM(mu="invalid", sigma=0.2, S0=100)

    def test_time_varying_gbm_simulation_validation(self):
        """Test simulation parameter validation."""
        gbm = TimeVaryingGBM(mu=0.05, sigma=0.2, S0=100)

        with pytest.raises(ValueError, match="n_paths must be positive"):
            gbm.simulate(n_paths=0, n_steps=10)

        with pytest.raises(ValueError, match="n_steps must be positive"):
            gbm.simulate(n_paths=10, n_steps=0)

        with pytest.raises(ValueError, match="T must be positive"):
            gbm.simulate(n_paths=10, n_steps=10, T=0)

    def test_get_parameter_history(self):
        """Test parameter history retrieval."""
        mu_param = TimeVaryingParameter([0, 1], [0.05, 0.08], method='linear')
        gbm = TimeVaryingGBM(mu=mu_param, sigma=0.2, S0=100)

        times, mu_values, sigma_values = gbm.get_parameter_history(T=1.0, n_points=3)

        np.testing.assert_array_equal(times, [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(mu_values, [0.05, 0.065, 0.08])
        np.testing.assert_array_equal(sigma_values, [0.2, 0.2, 0.2])


class TestTimeVaryingCorrelatedGBM:
    """Test time-varying correlated GBM."""

    def test_time_varying_correlated_gbm_constant(self):
        """Test with constant parameters."""
        correlation_matrix = [[1.0, 0.3], [0.3, 1.0]]
        gbm = TimeVaryingCorrelatedGBM(
            mu=[0.05, 0.03],
            sigma=[0.2, 0.15],
            S0=[100, 50],
            correlation_matrix=correlation_matrix
        )

        paths = gbm.simulate(n_paths=10, n_steps=5, T=1.0)
        assert paths.shape == (10, 2, 6)
        assert np.all(paths[:, 0, 0] == 100)
        assert np.all(paths[:, 1, 0] == 50)

    def test_time_varying_correlated_gbm_varying(self):
        """Test with time-varying parameters."""
        mu1 = TimeVaryingParameter([0, 1], [0.05, 0.08])
        mu2 = TimeVaryingParameter([0, 1], [0.03, 0.06])
        sigma1 = TimeVaryingParameter([0, 1], [0.2, 0.25])
        sigma2 = 0.15  # Constant

        correlation_matrix = [[1.0, 0.3], [0.3, 1.0]]
        gbm = TimeVaryingCorrelatedGBM(
            mu=[mu1, mu2],
            sigma=[sigma1, sigma2],
            S0=[100, 50],
            correlation_matrix=correlation_matrix
        )

        paths = gbm.simulate(n_paths=100, n_steps=20, T=1.0)
        assert paths.shape == (100, 2, 21)
        assert np.all(paths > 0)

    def test_time_varying_correlated_validation(self):
        """Test validation for correlated GBM."""
        correlation_matrix = [[1.0, 0.3], [0.3, 1.0]]

        # Mismatched parameter lengths
        with pytest.raises(ValueError, match="sigma must have same length as mu"):
            TimeVaryingCorrelatedGBM(
                mu=[0.05, 0.03],
                sigma=[0.2],  # Wrong length
                S0=[100, 50],
                correlation_matrix=correlation_matrix
            )

        # Wrong correlation matrix size
        with pytest.raises(ValueError, match="correlation_matrix must be 2x2"):
            TimeVaryingCorrelatedGBM(
                mu=[0.05, 0.03],
                sigma=[0.2, 0.15],
                S0=[100, 50],
                correlation_matrix=[[1.0]]  # Wrong size
            )

    def test_get_parameter_history_multi_asset(self):
        """Test parameter history for multiple assets."""
        mu1 = TimeVaryingParameter([0, 1], [0.05, 0.08])
        mu2 = 0.03  # Constant

        gbm = TimeVaryingCorrelatedGBM(
            mu=[mu1, mu2],
            sigma=[0.2, 0.15],
            S0=[100, 50],
            correlation_matrix=[[1.0, 0.3], [0.3, 1.0]]
        )

        times, mu_values, sigma_values = gbm.get_parameter_history(T=1.0, n_points=3)

        assert times.shape == (3,)
        assert mu_values.shape == (3, 2)
        assert sigma_values.shape == (3, 2)

        # Check mu values - step interpolation means middle value = first value
        np.testing.assert_array_equal(mu_values[:, 0], [0.05, 0.05, 0.08])  # Varying (step)
        np.testing.assert_array_equal(mu_values[:, 1], [0.03, 0.03, 0.03])   # Constant


class TestConvenienceFunctions:
    """Test convenience functions for creating time-varying GBMs."""

    def test_create_regime_gbm(self):
        """Test regime change GBM creation."""
        regime_changes = [
            (0.5, 0.08, 0.25),  # At t=0.5, change to mu=0.08, sigma=0.25
            (1.0, 0.06, 0.20),  # At t=1.0, change to mu=0.06, sigma=0.20
        ]

        gbm = create_regime_gbm(regime_changes, S0=100)
        assert isinstance(gbm, TimeVaryingGBM)

        # Test parameter values
        paths = gbm.simulate(n_paths=5, n_steps=10, T=1.5)
        assert paths.shape == (5, 11)

    def test_create_linear_trend_gbm(self):
        """Test linear trend GBM creation."""
        gbm = create_linear_trend_gbm(
            mu_start=0.05, mu_end=0.08,
            sigma_start=0.2, sigma_end=0.25,
            S0=100, T=1.0
        )

        assert isinstance(gbm, TimeVaryingGBM)

        # Test parameter evolution
        times, mu_vals, sigma_vals = gbm.get_parameter_history(T=1.0, n_points=3)
        np.testing.assert_array_almost_equal(mu_vals, [0.05, 0.065, 0.08])
        np.testing.assert_array_almost_equal(sigma_vals, [0.2, 0.225, 0.25])

    def test_create_cyclical_gbm(self):
        """Test cyclical GBM creation."""
        gbm = create_cyclical_gbm(
            base_mu=0.05, mu_amplitude=0.02, mu_period=2.0,
            base_sigma=0.2, sigma_amplitude=0.05, sigma_period=1.0,
            S0=100
        )

        assert isinstance(gbm, TimeVaryingGBM)

        # Test cyclical behavior
        times, mu_vals, sigma_vals = gbm.get_parameter_history(T=2.0, n_points=5)

        # At t=0: should be base + amplitude (sin(0) = 0)
        assert abs(mu_vals[0] - 0.05) < 1e-10

        # At t=0.5 (quarter period for mu): should be base + amplitude (sin(π/2) = 1)
        # Period is 2.0, so at t=0.5, we're at π/2 in the cycle
        assert abs(mu_vals[1] - (0.05 + 0.02)) < 1e-10


class TestRealWorldScenarios:
    """Test realistic time-varying scenarios."""

    def test_market_crash_scenario(self):
        """Test market crash scenario with regime changes."""
        # Normal market -> Crash -> Recovery
        regime_changes = [
            (0.25, -0.30, 0.6),  # Crash: negative return, high volatility
            (0.5, 0.15, 0.3),    # Recovery: high return, moderate volatility
        ]

        gbm = create_regime_gbm(regime_changes, S0=100)
        paths = gbm.simulate(n_paths=100, n_steps=100, T=1.0)

        # Should maintain reasonable properties
        assert paths.shape == (100, 101)
        assert np.all(paths > 0)  # No negative prices
        assert np.all(paths[:, 0] == 100)  # Correct initial price

    def test_economic_cycle_simulation(self):
        """Test economic cycle with cyclical parameters."""
        # Simulate 5-year economic cycle
        gbm = create_cyclical_gbm(
            base_mu=0.06, mu_amplitude=0.04, mu_period=5.0,  # 5-year cycle
            base_sigma=0.18, sigma_amplitude=0.08, sigma_period=5.0,
            S0=100
        )

        paths = gbm.simulate(n_paths=50, n_steps=1260, T=5.0)  # 5 years, daily steps

        assert paths.shape == (50, 1261)
        assert np.all(paths > 0)

        # Check parameter evolution over full cycle
        times, mu_vals, sigma_vals = gbm.get_parameter_history(T=5.0, n_points=21)

        # Should return to approximately initial values after full cycle
        assert abs(mu_vals[0] - mu_vals[-1]) < 0.01
        assert abs(sigma_vals[0] - sigma_vals[-1]) < 0.01

    def test_interest_rate_environment_changes(self):
        """Test changing interest rate environment."""
        # Rising rate environment
        mu_rising = TimeVaryingParameter(
            times=[0, 0.5, 1.0, 1.5, 2.0],
            values=[0.02, 0.03, 0.05, 0.07, 0.08],
            method='linear'
        )

        # Volatility decreases as rates stabilize
        sigma_param = TimeVaryingParameter(
            times=[0, 1.0, 2.0],
            values=[0.25, 0.20, 0.15],
            method='linear'
        )

        gbm = TimeVaryingGBM(mu=mu_rising, sigma=sigma_param, S0=100)
        paths = gbm.simulate(n_paths=100, n_steps=504, T=2.0)

        assert paths.shape == (100, 505)
        assert np.all(paths > 0)