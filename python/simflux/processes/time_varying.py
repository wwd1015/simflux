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
from ..core.validation import (
    validate_time_varying_params,
    validate_time_varying_correlated_params,
)


def _linear_interpolate(t: float, times: np.ndarray, values: np.ndarray) -> float:
    """Simple linear interpolation for a scalar time point."""
    return float(np.interp(t, times, values))


class TimeVaryingGBM(BaseSimulator):
    """
    Geometric Brownian Motion with time-varying drift (mu) and volatility (sigma).

    Simply provide time series arrays for mu and sigma values.
    """

    def __init__(
        self,
        mu_times: Union[np.ndarray, List[float]],
        mu_values: Union[np.ndarray, List[float]],
        sigma_times: Union[np.ndarray, List[float]],
        sigma_values: Union[np.ndarray, List[float]],
        S0: float,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        super().__init__(config)
        self.mu_times = np.asarray(mu_times)
        self.mu_values = np.asarray(mu_values)
        self.sigma_times = np.asarray(sigma_times)
        self.sigma_values = np.asarray(sigma_values)
        self.S0 = S0
        self.engine = SimulationEngine(config)

        self.validate_inputs()

    def validate_inputs(self, *args: Any, **kwargs: Any) -> None:
        """Validate the time-series inputs (construction-time contract).

        Raises ``ValueError`` on mismatched series lengths.  Per-run dimension
        checks (n_paths/n_steps/T) are owned by the engine seam.
        """
        validate_time_varying_params(
            self.mu_times, self.mu_values, self.sigma_times, self.sigma_values
        )

    def get_parameters_at_time(self, t: float) -> Tuple[float, float]:
        """Interpolate mu and sigma at time ``t`` for inspection.

        Convenience/inspection helper.  The simulation does not call it: each
        backend interpolates the schedule internally over its own time grid.
        Use it to inspect the assumed parameters, not to predict the exact
        values a given run will use.
        """
        mu = _linear_interpolate(t, self.mu_times, self.mu_values)
        sigma = _linear_interpolate(t, self.sigma_times, self.sigma_values)
        return mu, sigma

    def simulate(self, n_paths: int, n_steps: int, T: float) -> np.ndarray:
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

    def __init__(
        self,
        mu_times: List[Union[np.ndarray, List[float]]],
        mu_values: List[Union[np.ndarray, List[float]]],
        sigma_times: List[Union[np.ndarray, List[float]]],
        sigma_values: List[Union[np.ndarray, List[float]]],
        S0: Union[np.ndarray, List[float]],
        correlation_matrix: np.ndarray,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        super().__init__(config)
        self.n_assets = len(S0)
        self.S0 = np.asarray(S0)
        self.correlation_matrix = np.asarray(correlation_matrix, dtype=float)
        self.engine = SimulationEngine(config)

        self.mu_times = [np.asarray(times) for times in mu_times]
        self.mu_values = [np.asarray(values) for values in mu_values]
        self.sigma_times = [np.asarray(times) for times in sigma_times]
        self.sigma_values = [np.asarray(values) for values in sigma_values]

        self.validate_inputs()

    def validate_inputs(self, *args: Any, **kwargs: Any) -> None:
        """Validate per-asset series and the correlation matrix (construction-time).

        Raises ``ValueError`` (never a raw ``LinAlgError``) on a missing series
        or an invalid correlation matrix, matching every other simulator's error
        mode.  Per-run dimension checks are owned by the engine seam.
        """
        validate_time_varying_correlated_params(
            self.mu_times, self.sigma_times, self.n_assets, self.correlation_matrix
        )

    def get_parameters_at_time(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate mu and sigma vectors at time ``t`` for inspection.

        Convenience/inspection helper; the simulation interpolates internally
        per backend and does not call this.
        """
        mu = np.array(
            [
                _linear_interpolate(t, self.mu_times[i], self.mu_values[i])
                for i in range(self.n_assets)
            ]
        )
        sigma = np.array(
            [
                _linear_interpolate(t, self.sigma_times[i], self.sigma_values[i])
                for i in range(self.n_assets)
            ]
        )
        return mu, sigma

    def simulate(self, n_paths: int, n_steps: int, T: float) -> np.ndarray:
        """
        Simulate multi-asset time-varying GBM paths.

        Returns
        -------
        np.ndarray
            Simulated paths of shape (n_paths, n_assets, n_steps + 1)
        """
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
