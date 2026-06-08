"""Simulation backend dispatching Rust and NumPy implementations."""

import numpy as np
from typing import Optional, List, Any
from .base import SimulationConfig, check_memory
from .backend import Backend
from .validation import (
    validate_gbm_params,
    validate_correlated_gbm_params,
    validate_time_varying_params,
    validate_time_varying_correlated_params,
)
from ..utils.random_utils import safe_cholesky


def _validate_sim_dims(n_paths: int, n_steps: int, T: float) -> None:
    """Shared validation for simulation dimension parameters."""
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if T <= 0:
        raise ValueError("T must be positive")


class SimulationEngine:
    """Backend dispatcher that delegates simulation work to Rust or NumPy.

    Unlike the simulator classes (GBM, CorrelatedGBM, …) this is **not** a
    ``BaseSimulator``.  It owns no model parameters and exposes only
    low-level simulation entry-points.
    """

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig()
        self.config.validate()

        if not Backend.is_available():
            import warnings

            warnings.warn(
                "Rust backend not available. Using slower NumPy fallback. "
                "For best performance, install from binary wheel.",
                RuntimeWarning,
            )

    def _check_memory(self, n_elements: int, element_bytes: int = 8) -> None:
        check_memory(self.config, n_elements, element_bytes)

    # ------------------------------------------------------------------
    # Standard GBM
    # ------------------------------------------------------------------
    def simulate_gbm(
        self,
        mu: float,
        sigma: float,
        s0: float,
        n_paths: int,
        n_steps: int,
        T: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Simulate single-asset GBM paths.

        Returns shape ``(n_paths, n_steps + 1)``.
        """
        validate_gbm_params(sigma, s0)
        _validate_sim_dims(n_paths, n_steps, T)
        self._check_memory(n_paths * (n_steps + 1))

        run = Backend.choose(self._rust_simulate_gbm, self._numpy_simulate_gbm)
        return run(mu, sigma, s0, n_paths, n_steps, T)

    def simulate_gbm_correlated(
        self,
        mu: List[float],
        sigma: List[float],
        s0: List[float],
        correlation_matrix: np.ndarray,
        n_paths: int,
        n_steps: int,
        T: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Simulate correlated multi-asset GBM paths.

        Returns shape ``(n_paths, n_assets, n_steps + 1)``.
        """
        n_assets = len(mu)
        validate_correlated_gbm_params(mu, sigma, s0, correlation_matrix)
        _validate_sim_dims(n_paths, n_steps, T)
        self._check_memory(n_paths * n_assets * (n_steps + 1))

        run = Backend.choose(
            self._rust_simulate_gbm_correlated, self._numpy_simulate_gbm_correlated
        )
        return run(mu, sigma, s0, correlation_matrix, n_paths, n_steps, T)

    # ------------------------------------------------------------------
    # Time-varying GBM
    # ------------------------------------------------------------------
    def simulate_gbm_time_varying(
        self,
        mu_times: List[float],
        mu_values: List[float],
        sigma_times: List[float],
        sigma_values: List[float],
        s0: float,
        n_paths: int,
        n_steps: int,
        T: float,
    ) -> np.ndarray:
        """Simulate single-asset time-varying GBM.

        Returns shape ``(n_paths, n_steps + 1)``.
        """
        validate_time_varying_params(mu_times, mu_values, sigma_times, sigma_values)
        _validate_sim_dims(n_paths, n_steps, T)
        self._check_memory(n_paths * (n_steps + 1))

        run = Backend.choose(
            self._rust_simulate_gbm_time_varying,
            self._numpy_simulate_gbm_time_varying,
        )
        return run(
            mu_times, mu_values, sigma_times, sigma_values, s0, n_paths, n_steps, T
        )

    def simulate_gbm_time_varying_correlated(
        self,
        mu_times: List[List[float]],
        mu_values: List[List[float]],
        sigma_times: List[List[float]],
        sigma_values: List[List[float]],
        s0: List[float],
        correlation_matrix: np.ndarray,
        n_paths: int,
        n_steps: int,
        T: float,
    ) -> np.ndarray:
        """Simulate correlated multi-asset time-varying GBM.

        Returns shape ``(n_paths, n_assets, n_steps + 1)``.
        """
        n_assets = len(s0)
        validate_time_varying_correlated_params(
            mu_times, sigma_times, n_assets, correlation_matrix
        )
        _validate_sim_dims(n_paths, n_steps, T)
        self._check_memory(n_paths * n_assets * (n_steps + 1))

        run = Backend.choose(
            self._rust_simulate_gbm_time_varying_correlated,
            self._numpy_simulate_gbm_time_varying_correlated,
        )
        return run(
            mu_times,
            mu_values,
            sigma_times,
            sigma_values,
            s0,
            correlation_matrix,
            n_paths,
            n_steps,
            T,
        )

    # ==================================================================
    # Rust implementations
    # ==================================================================
    def _rust_call(
        self, fn_name: str, expected_shape: tuple, **kwargs: Any
    ) -> np.ndarray:
        """Marshal one Rust simulation call and guard its result shape.

        Concentrates the three steps every Rust path shares — fetch the backend,
        rebuild the ndarray, and assert the documented axis convention — so an
        axis/ordering change in the Rust layer fails loudly here rather than
        surfacing as silently wrong numbers downstream.
        """
        _rust = Backend.get_rust()
        # The Rust backend returns a contiguous NumPy array (not a list-of-lists),
        # so asarray is zero-copy rather than re-parsing/boxing every element.
        paths = np.asarray(getattr(_rust, fn_name)(**kwargs))
        if paths.shape != expected_shape:
            raise RuntimeError(
                f"Rust backend '{fn_name}' returned shape {paths.shape}, "
                f"expected {expected_shape}; Python/Rust marshalling is out of sync."
            )
        return paths

    def _rust_simulate_gbm(self, mu, sigma, s0, n_paths, n_steps, T):
        return self._rust_call(
            "simulate_gbm",
            (n_paths, n_steps + 1),
            mu=mu,
            sigma=sigma,
            s0=s0,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=T / n_steps,
            seed=self.config.seed,
        )

    def _rust_simulate_gbm_correlated(
        self, mu, sigma, s0, correlation_matrix, n_paths, n_steps, T
    ):
        return self._rust_call(
            "simulate_gbm_multi",
            (n_paths, len(mu), n_steps + 1),
            mu=mu,
            sigma=sigma,
            s0=s0,
            correlation_matrix=correlation_matrix.tolist(),
            n_paths=n_paths,
            n_steps=n_steps,
            dt=T / n_steps,
            seed=self.config.seed,
        )

    def _rust_simulate_gbm_time_varying(
        self, mu_times, mu_values, sigma_times, sigma_values, s0, n_paths, n_steps, T
    ):
        return self._rust_call(
            "simulate_gbm_tv",
            (n_paths, n_steps + 1),
            mu_times=list(mu_times),
            mu_values=list(mu_values),
            sigma_times=list(sigma_times),
            sigma_values=list(sigma_values),
            s0=s0,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=T / n_steps,
            t_start=0.0,
            seed=self.config.seed,
        )

    def _rust_simulate_gbm_time_varying_correlated(
        self,
        mu_times,
        mu_values,
        sigma_times,
        sigma_values,
        s0,
        correlation_matrix,
        n_paths,
        n_steps,
        T,
    ):
        return self._rust_call(
            "simulate_gbm_tv_multi",
            (n_paths, len(s0), n_steps + 1),
            mu_times=[list(t) for t in mu_times],
            mu_values=[list(v) for v in mu_values],
            sigma_times=[list(t) for t in sigma_times],
            sigma_values=[list(v) for v in sigma_values],
            s0=list(s0),
            correlation_matrix=correlation_matrix.tolist(),
            n_paths=n_paths,
            n_steps=n_steps,
            dt=T / n_steps,
            t_start=0.0,
            seed=self.config.seed,
        )

    # ==================================================================
    # NumPy fallback implementations
    # ==================================================================
    def _numpy_simulate_gbm(self, mu, sigma, s0, n_paths, n_steps, T):
        rng = np.random.default_rng(self.config.seed)
        dt = T / n_steps
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        dW = rng.normal(0, 1, (n_paths, n_steps))
        log_returns = drift + diffusion * dW
        log_prices = np.cumsum(log_returns, axis=1)
        log_prices = np.column_stack([np.zeros(n_paths), log_prices])

        return s0 * np.exp(log_prices)

    def _numpy_simulate_gbm_correlated(
        self, mu, sigma, s0, correlation_matrix, n_paths, n_steps, T
    ):
        n_assets = len(mu)
        rng = np.random.default_rng(self.config.seed)
        dt = T / n_steps

        sigma_arr = np.asarray(sigma)
        drift = (np.asarray(mu) - 0.5 * sigma_arr**2) * dt
        vol_sqrt_dt = sigma_arr * np.sqrt(dt)

        L = safe_cholesky(correlation_matrix)

        paths = np.zeros((n_paths, n_assets, n_steps + 1))
        paths[:, :, 0] = np.asarray(s0)

        for step in range(n_steps):
            independent_randoms = rng.normal(0, 1, (n_paths, n_assets))
            correlated_randoms = independent_randoms @ L.T
            log_returns = drift + vol_sqrt_dt * correlated_randoms
            paths[:, :, step + 1] = paths[:, :, step] * np.exp(log_returns)

        return paths

    def _numpy_simulate_gbm_time_varying(
        self, mu_times, mu_values, sigma_times, sigma_values, s0, n_paths, n_steps, T
    ):
        mu_t = np.asarray(mu_times)
        mu_v = np.asarray(mu_values)
        sigma_t = np.asarray(sigma_times)
        sigma_v = np.asarray(sigma_values)

        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = s0

        rng = np.random.default_rng(self.config.seed)
        dW = rng.normal(0, np.sqrt(dt), (n_paths, n_steps))

        mu_all = np.interp(times[:-1], mu_t, mu_v)
        sigma_all = np.interp(times[:-1], sigma_t, sigma_v)

        for i in range(n_steps):
            mu, sigma = mu_all[i], sigma_all[i]
            paths[:, i + 1] = paths[:, i] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * dW[:, i]
            )

        return paths

    def _numpy_simulate_gbm_time_varying_correlated(
        self,
        mu_times,
        mu_values,
        sigma_times,
        sigma_values,
        s0,
        correlation_matrix,
        n_paths,
        n_steps,
        T,
    ):
        n_assets = len(s0)
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)

        chol = safe_cholesky(correlation_matrix)

        mu_all = np.array(
            [
                np.interp(times[:-1], np.asarray(mu_times[j]), np.asarray(mu_values[j]))
                for j in range(n_assets)
            ]
        ).T
        sigma_all = np.array(
            [
                np.interp(
                    times[:-1], np.asarray(sigma_times[j]), np.asarray(sigma_values[j])
                )
                for j in range(n_assets)
            ]
        ).T

        paths = np.zeros((n_paths, n_assets, n_steps + 1))
        paths[:, :, 0] = np.asarray(s0)

        rng = np.random.default_rng(self.config.seed)

        for i in range(n_steps):
            mu, sigma = mu_all[i], sigma_all[i]
            dW = rng.normal(0, np.sqrt(dt), (n_paths, n_assets))
            dW_corr = dW @ chol.T
            paths[:, :, i + 1] = paths[:, :, i] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * dW_corr
            )

        return paths
