"""
Time-varying parameter support for stochastic processes.

This module provides simple time-varying GBM by accepting time series inputs directly.
"""

import numpy as np
from typing import Union, List, Optional, Tuple

from ..core.base import BaseSimulator, SimulationConfig


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
                 config: Optional[SimulationConfig] = None):
        """
        Initialize time-varying GBM.

        Parameters:
        -----------
        mu_times : array-like
            Time points for drift values
        mu_values : array-like
            Drift values at corresponding times
        sigma_times : array-like
            Time points for volatility values
        sigma_values : array-like
            Volatility values at corresponding times
        S0 : float
            Initial asset price
        config : SimulationConfig, optional
            Simulation configuration
        """
        super().__init__(config)
        self.mu_times = np.asarray(mu_times)
        self.mu_values = np.asarray(mu_values)
        self.sigma_times = np.asarray(sigma_times)
        self.sigma_values = np.asarray(sigma_values)
        self.S0 = S0

        # Validation
        if len(self.mu_times) != len(self.mu_values):
            raise ValueError("mu_times and mu_values must have same length")
        if len(self.sigma_times) != len(self.sigma_values):
            raise ValueError("sigma_times and sigma_values must have same length")

    def validate_inputs(self, n_paths: int, n_steps: int, T: float) -> None:
        """Validate simulation inputs."""
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if T <= 0:
            raise ValueError("T must be positive")

    def _interpolate(self, t: float, times: np.ndarray, values: np.ndarray) -> float:
        """Simple linear interpolation."""
        if t <= times[0]:
            return values[0]
        if t >= times[-1]:
            return values[-1]

        # Find surrounding points
        idx = np.searchsorted(times, t)
        if idx == 0:
            return values[0]
        if idx >= len(times):
            return values[-1]

        # Linear interpolation
        t0, t1 = times[idx-1], times[idx]
        v0, v1 = values[idx-1], values[idx]
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0)

    def get_parameters_at_time(self, t: float) -> Tuple[float, float]:
        """Get mu and sigma values at time t."""
        mu = self._interpolate(t, self.mu_times, self.mu_values)
        sigma = self._interpolate(t, self.sigma_times, self.sigma_values)
        return mu, sigma

    def simulate(self,
                 n_paths: int,
                 n_steps: int,
                 T: float) -> np.ndarray:
        """
        Simulate time-varying GBM paths.

        Parameters:
        -----------
        n_paths : int
            Number of simulation paths
        n_steps : int
            Number of time steps
        T : float
            Time horizon

        Returns:
        --------
        np.ndarray
            Simulated paths of shape (n_paths, n_steps + 1)
        """
        self.validate_inputs(n_paths, n_steps, T)

        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)

        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0

        # Generate random increments
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))

        # Simulate each time step
        for i in range(n_steps):
            t = times[i]
            mu, sigma = self.get_parameters_at_time(t)

            # GBM evolution: dS = mu*S*dt + sigma*S*dW
            paths[:, i+1] = paths[:, i] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * dW[:, i]
            )

        return paths


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
                 config: Optional[SimulationConfig] = None):
        """
        Initialize multi-asset time-varying GBM.

        Parameters:
        -----------
        mu_times : list of arrays
            Time points for drift values for each asset
        mu_values : list of arrays
            Drift values for each asset
        sigma_times : list of arrays
            Time points for volatility values for each asset
        sigma_values : list of arrays
            Volatility values for each asset
        S0 : array-like
            Initial prices for each asset
        correlation_matrix : np.ndarray
            Correlation matrix between assets
        """
        super().__init__(config)
        self.n_assets = len(S0)
        self.S0 = np.asarray(S0)
        self.correlation_matrix = correlation_matrix

        # Store time series for each asset
        self.mu_times = [np.asarray(times) for times in mu_times]
        self.mu_values = [np.asarray(values) for values in mu_values]
        self.sigma_times = [np.asarray(times) for times in sigma_times]
        self.sigma_values = [np.asarray(values) for values in sigma_values]

        # Validation
        if len(self.mu_times) != self.n_assets:
            raise ValueError("Must provide mu time series for each asset")
        if len(self.sigma_times) != self.n_assets:
            raise ValueError("Must provide sigma time series for each asset")

        # Cholesky decomposition for correlation
        self.chol = np.linalg.cholesky(correlation_matrix)

    def validate_inputs(self, n_paths: int, n_steps: int, T: float) -> None:
        """Validate simulation inputs."""
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if T <= 0:
            raise ValueError("T must be positive")

    def _interpolate(self, t: float, times: np.ndarray, values: np.ndarray) -> float:
        """Simple linear interpolation."""
        if t <= times[0]:
            return values[0]
        if t >= times[-1]:
            return values[-1]

        # Find surrounding points
        idx = np.searchsorted(times, t)
        if idx == 0:
            return values[0]
        if idx >= len(times):
            return values[-1]

        # Linear interpolation
        t0, t1 = times[idx-1], times[idx]
        v0, v1 = values[idx-1], values[idx]
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0)

    def get_parameters_at_time(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get mu and sigma vectors at time t."""
        mu = np.array([
            self._interpolate(t, self.mu_times[i], self.mu_values[i])
            for i in range(self.n_assets)
        ])
        sigma = np.array([
            self._interpolate(t, self.sigma_times[i], self.sigma_values[i])
            for i in range(self.n_assets)
        ])
        return mu, sigma

    def simulate(self,
                 n_paths: int,
                 n_steps: int,
                 T: float) -> np.ndarray:
        """
        Simulate multi-asset time-varying GBM paths.

        Returns:
        --------
        np.ndarray
            Simulated paths of shape (n_paths, n_assets, n_steps + 1)
        """
        self.validate_inputs(n_paths, n_steps, T)

        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)

        # Initialize paths
        paths = np.zeros((n_paths, self.n_assets, n_steps + 1))
        paths[:, :, 0] = self.S0

        # Simulate each time step
        for i in range(n_steps):
            t = times[i]
            mu, sigma = self.get_parameters_at_time(t)

            # Generate correlated random increments
            dW = np.random.normal(0, np.sqrt(dt), (n_paths, self.n_assets))
            dW_corr = dW @ self.chol.T

            # GBM evolution for each asset
            for j in range(self.n_assets):
                paths[:, j, i+1] = paths[:, j, i] * np.exp(
                    (mu[j] - 0.5 * sigma[j]**2) * dt + sigma[j] * dW_corr[:, j]
                )

        return paths