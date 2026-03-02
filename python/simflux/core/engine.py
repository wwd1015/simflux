"""Main simulation engine orchestrating Rust backend calls."""

import numpy as np
from typing import Optional, Dict, Any, List, NoReturn
from .base import BaseSimulator, SimulationConfig, SimulationResults
from ..utils.random_utils import validate_correlation_matrix_strict

try:
    from simflux import _rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    _rust = None


class SimulationEngine(BaseSimulator):
    """Main simulation engine that coordinates between Python and Rust backends."""

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        super().__init__(config)

        if not RUST_AVAILABLE:
            import warnings
            warnings.warn(
                "Rust backend not available. Using slower NumPy fallback. "
                "For best performance, install from binary wheel.",
                RuntimeWarning
            )

    def simulate_gbm(self,
                     mu: float,
                     sigma: float,
                     s0: float,
                     n_paths: int,
                     n_steps: int,
                     T: float = 1.0,
                     **kwargs: Any) -> np.ndarray:
        """
        Simulate single-asset Geometric Brownian Motion.

        Parameters:
        -----------
        mu : float
            Drift parameter
        sigma : float
            Volatility parameter
        s0 : float
            Initial asset value
        n_paths : int
            Number of simulation paths
        n_steps : int
            Number of time steps per path
        T : float, default=1.0
            Time horizon

        Returns:
        --------
        np.ndarray
            Array of shape (n_paths, n_steps+1) containing simulated paths
        """
        self.validate_gbm_inputs(mu, sigma, s0, n_paths, n_steps, T)

        if RUST_AVAILABLE:
            return self._rust_simulate_gbm(mu, sigma, s0, n_paths, n_steps, T)
        else:
            return self._numpy_simulate_gbm(mu, sigma, s0, n_paths, n_steps, T)

    def simulate_gbm_correlated(self,
                               mu: List[float],
                               sigma: List[float],
                               s0: List[float],
                               correlation_matrix: np.ndarray,
                               n_paths: int,
                               n_steps: int,
                               T: float = 1.0,
                               **kwargs: Any) -> np.ndarray:
        """
        Simulate correlated multi-asset Geometric Brownian Motion.

        Parameters:
        -----------
        mu : List[float]
            Drift parameters for each asset
        sigma : List[float]
            Volatility parameters for each asset
        s0 : List[float]
            Initial asset values
        correlation_matrix : np.ndarray
            Correlation matrix (n_assets x n_assets)
        n_paths : int
            Number of simulation paths
        n_steps : int
            Number of time steps per path
        T : float, default=1.0
            Time horizon

        Returns:
        --------
        np.ndarray
            Array of shape (n_paths, n_assets, n_steps+1) containing simulated paths
        """
        self.validate_gbm_correlated_inputs(
            mu, sigma, s0, correlation_matrix, n_paths, n_steps, T
        )

        if RUST_AVAILABLE:
            return self._rust_simulate_gbm_correlated(
                mu, sigma, s0, correlation_matrix, n_paths, n_steps, T
            )
        else:
            return self._numpy_simulate_gbm_correlated(
                mu, sigma, s0, correlation_matrix, n_paths, n_steps, T
            )

    def validate_gbm_inputs(self, mu: float, sigma: float, s0: float,
                            n_paths: int, n_steps: int, T: float) -> None:
        """Validate GBM simulation inputs."""
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if s0 <= 0:
            raise ValueError("s0 must be positive")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if T <= 0:
            raise ValueError("T must be positive")

    def validate_gbm_correlated_inputs(self, mu: List[float], sigma: List[float],
                                       s0: List[float], correlation_matrix: np.ndarray,
                                       n_paths: int, n_steps: int, T: float) -> None:
        """Validate correlated GBM simulation inputs."""
        n_assets = len(mu)

        if len(sigma) != n_assets:
            raise ValueError("sigma must have same length as mu")
        if len(s0) != n_assets:
            raise ValueError("s0 must have same length as mu")

        if correlation_matrix.shape != (n_assets, n_assets):
            raise ValueError(f"correlation_matrix must be {n_assets}x{n_assets}")

        validate_correlation_matrix_strict(correlation_matrix)

        for s in sigma:
            if s <= 0:
                raise ValueError("all sigma values must be positive")

        for s in s0:
            if s <= 0:
                raise ValueError("all s0 values must be positive")

        self.validate_gbm_inputs(0.0, 0.1, 1.0, n_paths, n_steps, T)  # Validate common params

    # BaseSimulator compatibility -------------------------------------------------
    def validate_inputs(self, *_: Any, **__: Any) -> None:
        """BaseSimulator hook; validation is handled by specific helpers."""
        return None

    def _rust_simulate_gbm(self, mu: float, sigma: float, s0: float,
                           n_paths: int, n_steps: int, T: float) -> np.ndarray:
        """Call Rust backend for GBM simulation."""
        dt = T / n_steps
        paths = _rust.simulate_gbm(
            mu=mu, sigma=sigma, s0=s0,
            n_paths=n_paths, n_steps=n_steps, dt=dt,
            seed=self.config.seed
        )
        return np.array(paths)

    def _numpy_simulate_gbm(self, mu: float, sigma: float, s0: float,
                            n_paths: int, n_steps: int, T: float) -> np.ndarray:
        """NumPy fallback for GBM simulation."""
        self._check_memory(n_paths * (n_steps + 1))

        rng = np.random.default_rng(self.config.seed)

        dt = T / n_steps
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        # Generate random increments
        dW = rng.normal(0, 1, (n_paths, n_steps))

        # Calculate price paths
        log_returns = drift + diffusion * dW
        log_prices = np.cumsum(log_returns, axis=1)
        log_prices = np.column_stack([np.zeros(n_paths), log_prices])

        # Convert to prices
        prices = s0 * np.exp(log_prices)
        return prices

    def _rust_simulate_gbm_correlated(self, mu: List[float], sigma: List[float],
                                      s0: List[float], correlation_matrix: np.ndarray,
                                      n_paths: int, n_steps: int, T: float) -> np.ndarray:
        """Call Rust backend for correlated GBM simulation."""
        dt = T / n_steps
        correlation_list = correlation_matrix.tolist()

        paths = _rust.simulate_gbm_multi(
            mu=mu, sigma=sigma, s0=s0,
            correlation_matrix=correlation_list,
            n_paths=n_paths, n_steps=n_steps, dt=dt,
            seed=self.config.seed
        )
        return np.array(paths)

    def _numpy_simulate_gbm_correlated(self, mu: List[float], sigma: List[float],
                                       s0: List[float], correlation_matrix: np.ndarray,
                                       n_paths: int, n_steps: int, T: float) -> np.ndarray:
        """NumPy fallback for correlated GBM simulation."""
        n_assets = len(mu)
        self._check_memory(n_paths * n_assets * (n_steps + 1))

        rng = np.random.default_rng(self.config.seed)

        dt = T / n_steps

        # Precompute constants
        drift = np.array([(m - 0.5 * s**2) * dt for m, s in zip(mu, sigma)])
        vol_sqrt_dt = np.array([s * np.sqrt(dt) for s in sigma])

        # Cholesky decomposition for correlation
        try:
            L = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            # If not positive definite, use eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive
            L = eigenvecs @ np.diag(np.sqrt(eigenvals))

        # Initialize paths
        paths = np.zeros((n_paths, n_assets, n_steps + 1))

        # Set initial values
        for i in range(n_assets):
            paths[:, i, 0] = s0[i]

        # Generate correlated paths
        for step in range(n_steps):
            # Generate independent random numbers
            independent_randoms = rng.normal(0, 1, (n_paths, n_assets))

            # Apply correlation structure
            correlated_randoms = independent_randoms @ L.T

            # Update paths for each asset
            for i in range(n_assets):
                dW = correlated_randoms[:, i]
                log_return = drift[i] + vol_sqrt_dt[i] * dW
                paths[:, i, step + 1] = paths[:, i, step] * np.exp(log_return)

        return paths

    def simulate(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Generic simulate method - delegates to specific simulators."""
        raise NotImplementedError("Use specific simulation methods like simulate_gbm()")
