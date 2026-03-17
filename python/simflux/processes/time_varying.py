"""
Time-varying parameter support for stochastic processes.

This module provides simple time-varying GBM by accepting time series inputs directly.
When the Rust backend is available the heavy lifting is delegated there; otherwise a
vectorised NumPy fallback is used transparently.
"""

import numpy as np
from typing import Any, Union, List, Optional, Tuple

from ..core.base import BaseSimulator, SimulationConfig
from ..core.engine import SimulationEngine


def _linear_interpolate(t: float, times: np.ndarray, values: np.ndarray) -> float:
    """Simple linear interpolation for a scalar time point."""
    return float(np.interp(t, times, values))


class TimeVaryingGBM(BaseSimulator):
    """
    Geometric Brownian Motion with time-varying drift (mu) and volatility (sigma).

    Simply provide time series arrays for mu and sigma values.
    """

    def __init__(self,
                 mu_times: Union[np.ndarray, List[float]],
                 mu_values: Union[np.ndarray, List[float]],
                 sigma_times: Union[np.ndarray, List[float]],
                 sigma_values: Union[np.ndarray, List[float]],
                 S0: float,
                 config: Optional[SimulationConfig] = None) -> None:
        super().__init__(config)
        self.mu_times = np.asarray(mu_times)
        self.mu_values = np.asarray(mu_values)
        self.sigma_times = np.asarray(sigma_times)
        self.sigma_values = np.asarray(sigma_values)
        self.S0 = S0
        self.engine = SimulationEngine(config)

        if len(self.mu_times) != len(self.mu_values):
            raise ValueError("mu_times and mu_values must have same length")
        if len(self.sigma_times) != len(self.sigma_values):
            raise ValueError("sigma_times and sigma_values must have same length")

    def validate_inputs(self, *args: Any, **kwargs: Any) -> None:
        """Construction-time validation only; simulation dims validated by engine."""
        pass

    def get_parameters_at_time(self, t: float) -> Tuple[float, float]:
        """Get mu and sigma values at time t."""
        mu = _linear_interpolate(t, self.mu_times, self.mu_values)
        sigma = _linear_interpolate(t, self.sigma_times, self.sigma_values)
        return mu, sigma

    def simulate(self,
                 n_paths: int,
                 n_steps: int,
                 T: float) -> np.ndarray:
        """
        Simulate time-varying GBM paths.

        Parameters
        ----------
        n_paths : int
            Number of simulation paths
        n_steps : int
            Number of time steps
        T : float
            Time horizon

        Returns
        -------
        np.ndarray
            Simulated paths of shape (n_paths, n_steps + 1)
        """
        return self.engine.simulate_gbm_time_varying(
            mu_times=self.mu_times.tolist(),
            mu_values=self.mu_values.tolist(),
            sigma_times=self.sigma_times.tolist(),
            sigma_values=self.sigma_values.tolist(),
            s0=self.S0,
            n_paths=n_paths,
            n_steps=n_steps,
            T=T,
        )


class TimeVaryingCorrelatedGBM(BaseSimulator):
    """
    Multi-asset time-varying GBM with correlation.

    Simply provide time series for each asset's parameters.
    """

    def __init__(self,
                 mu_times: List[Union[np.ndarray, List[float]]],
                 mu_values: List[Union[np.ndarray, List[float]]],
                 sigma_times: List[Union[np.ndarray, List[float]]],
                 sigma_values: List[Union[np.ndarray, List[float]]],
                 S0: Union[np.ndarray, List[float]],
                 correlation_matrix: np.ndarray,
                 config: Optional[SimulationConfig] = None) -> None:
        super().__init__(config)
        self.n_assets = len(S0)
        self.S0 = np.asarray(S0)
        self.correlation_matrix = correlation_matrix
        self.engine = SimulationEngine(config)

        self.mu_times = [np.asarray(times) for times in mu_times]
        self.mu_values = [np.asarray(values) for values in mu_values]
        self.sigma_times = [np.asarray(times) for times in sigma_times]
        self.sigma_values = [np.asarray(values) for values in sigma_values]

        if len(self.mu_times) != self.n_assets:
            raise ValueError("Must provide mu time series for each asset")
        if len(self.sigma_times) != self.n_assets:
            raise ValueError("Must provide sigma time series for each asset")

        # Validate Cholesky feasibility (actual decomposition done in engine)
        np.linalg.cholesky(correlation_matrix)

    def validate_inputs(self, *args: Any, **kwargs: Any) -> None:
        """Construction-time validation only; simulation dims validated by engine."""
        pass

    def get_parameters_at_time(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get mu and sigma vectors at time t."""
        mu = np.array([
            _linear_interpolate(t, self.mu_times[i], self.mu_values[i])
            for i in range(self.n_assets)
        ])
        sigma = np.array([
            _linear_interpolate(t, self.sigma_times[i], self.sigma_values[i])
            for i in range(self.n_assets)
        ])
        return mu, sigma

    def simulate(self,
                 n_paths: int,
                 n_steps: int,
                 T: float) -> np.ndarray:
        """
        Simulate multi-asset time-varying GBM paths.

        Returns
        -------
        np.ndarray
            Simulated paths of shape (n_paths, n_assets, n_steps + 1)
        """
        self.validate_inputs(n_paths, n_steps, T)

        return self.engine.simulate_gbm_time_varying_correlated(
            mu_times=[t.tolist() for t in self.mu_times],
            mu_values=[v.tolist() for v in self.mu_values],
            sigma_times=[t.tolist() for t in self.sigma_times],
            sigma_values=[v.tolist() for v in self.sigma_values],
            s0=self.S0.tolist(),
            correlation_matrix=self.correlation_matrix,
            n_paths=n_paths,
            n_steps=n_steps,
            T=T,
        )
