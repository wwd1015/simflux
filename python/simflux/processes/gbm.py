"""Geometric Brownian Motion simulation classes."""

import numpy as np
from typing import Optional, List, Union
from ..core.base import BaseSimulator, SimulationConfig
from ..core.engine import SimulationEngine
from ..core.validation import validate_gbm_params, validate_correlated_gbm_params


class GBM(BaseSimulator):
    """
    Single-asset Geometric Brownian Motion simulator.

    The GBM follows the stochastic differential equation:
    dS_t = μ S_t dt + σ S_t dW_t

    Where:
    - μ (mu) is the drift parameter
    - σ (sigma) is the volatility parameter
    - W_t is a Wiener process (Brownian motion)
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        S0: float,
        config: Optional[SimulationConfig] = None,
    ):
        """
        Initialize GBM simulator.

        Parameters
        ----------
        mu : float
            Drift parameter (annualized return)
        sigma : float
            Volatility parameter (annualized volatility)
        S0 : float
            Initial asset value
        config : SimulationConfig, optional
            Simulation configuration
        """
        super().__init__(config)
        self.mu = mu
        self.sigma = sigma
        self.S0 = S0
        self.engine = SimulationEngine(config)

        self.validate_inputs(mu, sigma, S0)

    def validate_inputs(self, mu, sigma, S0):
        """Validate GBM parameters (one validator, shared with the engine seam)."""
        validate_gbm_params(sigma, S0, s0_label="S0")

    def simulate(self, n_paths: int, n_steps: int, T: float = 1.0) -> np.ndarray:
        """
        Simulate GBM paths.

        Parameters
        ----------
        n_paths : int
            Number of simulation paths
        n_steps : int
            Number of time steps per path
        T : float, default=1.0
            Time horizon (years)

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, n_steps+1) containing simulated paths
        """
        return self.engine.simulate_gbm(
            mu=self.mu,
            sigma=self.sigma,
            s0=self.S0,
            n_paths=n_paths,
            n_steps=n_steps,
            T=T,
        )

    def simulate_single_path(self, n_steps: int, T: float = 1.0) -> np.ndarray:
        """
        Simulate a single GBM path.

        Parameters
        ----------
        n_steps : int
            Number of time steps
        T : float, default=1.0
            Time horizon (years)

        Returns
        -------
        np.ndarray
            Array of length n_steps+1 containing the simulated path
        """
        paths = self.simulate(n_paths=1, n_steps=n_steps, T=T)
        return paths[0]

    def get_time_grid(self, n_steps: int, T: float = 1.0) -> np.ndarray:
        """
        Get the time grid for simulation.

        Parameters
        ----------
        n_steps : int
            Number of time steps
        T : float, default=1.0
            Time horizon (years)

        Returns
        -------
        np.ndarray
            Time grid of length n_steps+1
        """
        return np.linspace(0, T, n_steps + 1)


class CorrelatedGBM(BaseSimulator):
    """
    Multi-asset correlated Geometric Brownian Motion simulator.

    Simulates multiple correlated assets following GBM dynamics:
    dS_i,t = μ_i S_i,t dt + σ_i S_i,t dW_i,t

    Where the Wiener processes W_i,t are correlated according to
    the specified correlation matrix.
    """

    def __init__(
        self,
        mu: List[float],
        sigma: List[float],
        S0: List[float],
        correlation_matrix: Union[np.ndarray, List[List[float]]],
        config: Optional[SimulationConfig] = None,
    ):
        """
        Initialize correlated GBM simulator.

        Parameters
        ----------
        mu : List[float]
            Drift parameters for each asset
        sigma : List[float]
            Volatility parameters for each asset
        S0 : List[float]
            Initial values for each asset
        correlation_matrix : np.ndarray or List[List[float]]
            Correlation matrix (n_assets x n_assets)
        config : SimulationConfig, optional
            Simulation configuration
        """
        super().__init__(config)
        self.mu = mu
        self.sigma = sigma
        self.S0 = S0
        self.correlation_matrix = np.array(correlation_matrix)
        self.n_assets = len(mu)
        self.engine = SimulationEngine(config)

        self.validate_inputs(mu, sigma, S0, self.correlation_matrix)

    def validate_inputs(self, mu, sigma, S0, correlation_matrix):
        """Validate correlated GBM parameters (shared with the engine seam)."""
        validate_correlated_gbm_params(mu, sigma, S0, correlation_matrix, s0_label="S0")

    def simulate(self, n_paths: int, n_steps: int, T: float = 1.0) -> np.ndarray:
        """
        Simulate correlated GBM paths.

        Parameters
        ----------
        n_paths : int
            Number of simulation paths
        n_steps : int
            Number of time steps per path
        T : float, default=1.0
            Time horizon (years)

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, n_assets, n_steps+1) containing simulated paths
        """
        return self.engine.simulate_gbm_correlated(
            mu=self.mu,
            sigma=self.sigma,
            s0=self.S0,
            correlation_matrix=self.correlation_matrix,
            n_paths=n_paths,
            n_steps=n_steps,
            T=T,
        )

    def simulate_single_path(self, n_steps: int, T: float = 1.0) -> np.ndarray:
        """
        Simulate a single set of correlated paths.

        Parameters
        ----------
        n_steps : int
            Number of time steps
        T : float, default=1.0
            Time horizon (years)

        Returns
        -------
        np.ndarray
            Array of shape (n_assets, n_steps+1) containing the simulated paths
        """
        paths = self.simulate(n_paths=1, n_steps=n_steps, T=T)
        return paths[0]

    def get_asset_names(self) -> List[str]:
        """Get default asset names."""
        return [f"Asset_{i}" for i in range(self.n_assets)]

    def get_correlation_matrix(self) -> np.ndarray:
        """Get the correlation matrix."""
        return self.correlation_matrix.copy()

    @classmethod
    def from_single_gbm(
        cls,
        gbm: GBM,
        n_assets: int,
        correlation_matrix: Union[np.ndarray, List[List[float]]],
        config: Optional[SimulationConfig] = None,
    ) -> "CorrelatedGBM":
        """
        Create a CorrelatedGBM from a single GBM with identical parameters.

        Parameters
        ----------
        gbm : GBM
            Single GBM to replicate
        n_assets : int
            Number of assets
        correlation_matrix : np.ndarray or List[List[float]]
            Correlation matrix
        config : SimulationConfig, optional
            Simulation configuration

        Returns
        -------
        CorrelatedGBM
            New CorrelatedGBM instance
        """
        mu = [gbm.mu] * n_assets
        sigma = [gbm.sigma] * n_assets
        S0 = [gbm.S0] * n_assets

        return cls(mu, sigma, S0, correlation_matrix, config)
