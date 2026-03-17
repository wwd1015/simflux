"""Tests for SimulationEngine – NumPy fallback paths and time-varying dispatch."""

import pytest
import numpy as np
import warnings
from unittest.mock import patch

from simflux.core.engine import SimulationEngine
from simflux.core.base import SimulationConfig
from simflux.core.backend import Backend


# ---------------------------------------------------------------------------
# Time-varying GBM – NumPy fallback (force Rust off)
# ---------------------------------------------------------------------------

class TestTimeVaryingNumPyFallback:
    """Ensure time-varying GBM NumPy paths work when Rust is unavailable."""

    @patch.object(Backend, "is_available", return_value=False)
    def test_single_asset_fallback(self, _mock):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            engine = SimulationEngine()

        paths = engine.simulate_gbm_time_varying(
            mu_times=[0.0, 1.0],
            mu_values=[0.05, 0.05],
            sigma_times=[0.0, 1.0],
            sigma_values=[0.2, 0.2],
            s0=100.0,
            n_paths=50,
            n_steps=20,
            T=1.0,
        )

        assert paths.shape == (50, 21)
        assert np.all(paths[:, 0] == 100.0)
        assert np.all(paths > 0)

    @patch.object(Backend, "is_available", return_value=False)
    def test_correlated_fallback(self, _mock):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            engine = SimulationEngine()

        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        paths = engine.simulate_gbm_time_varying_correlated(
            mu_times=[[0.0, 1.0], [0.0, 1.0]],
            mu_values=[[0.05, 0.05], [0.03, 0.03]],
            sigma_times=[[0.0, 1.0], [0.0, 1.0]],
            sigma_values=[[0.2, 0.2], [0.15, 0.15]],
            s0=[100.0, 50.0],
            correlation_matrix=corr,
            n_paths=50,
            n_steps=20,
            T=1.0,
        )

        assert paths.shape == (50, 2, 21)
        assert np.all(paths[:, 0, 0] == 100.0)
        assert np.all(paths[:, 1, 0] == 50.0)
        assert np.all(paths > 0)


# ---------------------------------------------------------------------------
# Validation in engine
# ---------------------------------------------------------------------------

class TestEngineValidation:
    """Validate that SimulationEngine rejects bad inputs."""

    def test_gbm_negative_sigma(self):
        engine = SimulationEngine()
        with pytest.raises(ValueError, match="sigma must be positive"):
            engine.simulate_gbm(mu=0.05, sigma=-0.1, s0=100, n_paths=10, n_steps=10)

    def test_gbm_zero_s0(self):
        engine = SimulationEngine()
        with pytest.raises(ValueError, match="s0 must be positive"):
            engine.simulate_gbm(mu=0.05, sigma=0.2, s0=0, n_paths=10, n_steps=10)

    def test_correlated_sigma_length_mismatch(self):
        engine = SimulationEngine()
        with pytest.raises(ValueError, match="sigma must have same length as mu"):
            engine.simulate_gbm_correlated(
                mu=[0.05, 0.03],
                sigma=[0.2],
                s0=[100, 50],
                correlation_matrix=np.eye(2),
                n_paths=10,
                n_steps=10,
            )

    def test_time_varying_zero_T(self):
        engine = SimulationEngine()
        with pytest.raises(ValueError, match="T must be positive"):
            engine.simulate_gbm_time_varying(
                mu_times=[0, 1], mu_values=[0.05, 0.05],
                sigma_times=[0, 1], sigma_values=[0.2, 0.2],
                s0=100, n_paths=10, n_steps=10, T=0,
            )

    def test_config_validation_in_engine(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            SimulationEngine(config=SimulationConfig(batch_size=0))

    def test_engine_is_not_base_simulator(self):
        """SimulationEngine should NOT be a BaseSimulator subclass."""
        from simflux.core.base import BaseSimulator
        assert not isinstance(SimulationEngine(), BaseSimulator)
