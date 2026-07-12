"""Shared parameter-invariant validators for the GBM-family simulators.

Each parameter invariant (``sigma > 0``, ``s0 > 0``, matching series lengths,
positive-definite correlation) is written **once** here and called from both
seams that need it:

* the simulator constructor (``processes/gbm.py``, ``processes/time_varying.py``)
  — so a misconfigured model fails early, at construction, with a clear error;
* :class:`~simflux.core.engine.SimulationEngine` — which is independently
  callable, so it self-validates rather than trusting its caller.

Both CLAUDE.md principles are satisfied without copy-pasting the check at both
seams.  ``s0_label`` lets each seam name the parameter as its own caller passed
it (the simulator's public parameter is ``S0``; the engine's is ``s0``).

Per-run *dimension* invariants (``n_paths``/``n_steps``/``T``) are not here — they
remain the engine's job (see ``SimulationEngine._validate_sim_dims``).
"""

from typing import Sequence, Sized

import numpy as np

from ..exceptions import ValidationError

from ..utils.random_utils import validate_correlation_matrix_strict


def validate_gbm_params(sigma: float, s0: float, *, s0_label: str = "s0") -> None:
    """Single-asset GBM parameter invariants: ``sigma > 0`` and ``s0 > 0``."""
    if sigma <= 0:
        raise ValidationError("sigma must be positive")
    if s0 <= 0:
        raise ValidationError(f"{s0_label} must be positive")


def validate_correlated_gbm_params(
    mu: Sequence[float],
    sigma: Sequence[float],
    s0: Sequence[float],
    correlation_matrix: np.ndarray,
    *,
    s0_label: str = "s0",
) -> None:
    """Multi-asset GBM parameter invariants: matching lengths, square correlation
    matrix of the right size, each ``sigma``/``s0`` positive, correlation positive
    definite."""
    n_assets = len(mu)
    if len(sigma) != n_assets:
        raise ValidationError("sigma must have same length as mu")
    if len(s0) != n_assets:
        raise ValidationError(f"{s0_label} must have same length as mu")

    cm = np.asarray(correlation_matrix)
    if cm.shape != (n_assets, n_assets):
        raise ValidationError(f"correlation_matrix must be {n_assets}x{n_assets}")

    for i in range(n_assets):
        if sigma[i] <= 0:
            raise ValidationError(f"sigma[{i}] must be positive")
        if s0[i] <= 0:
            raise ValidationError(f"{s0_label}[{i}] must be positive")

    validate_correlation_matrix_strict(cm)


def validate_time_varying_params(
    mu_times: Sized,
    mu_values: Sized,
    sigma_times: Sized,
    sigma_values: Sized,
) -> None:
    """Single-asset time-varying GBM invariants: each schedule's times and values
    have matching lengths.  Accepts lists or ndarrays (only lengths are read)."""
    if len(mu_times) != len(mu_values):
        raise ValidationError("mu_times and mu_values must have same length")
    if len(sigma_times) != len(sigma_values):
        raise ValidationError("sigma_times and sigma_values must have same length")


def validate_time_varying_correlated_params(
    mu_times: Sized,
    sigma_times: Sized,
    n_assets: int,
    correlation_matrix: np.ndarray,
) -> None:
    """Multi-asset time-varying GBM invariants: one schedule per asset, and a
    positive-definite correlation matrix.  Accepts lists or ndarrays."""
    if len(mu_times) != n_assets:
        raise ValidationError("Must provide mu time series for each asset")
    if len(sigma_times) != n_assets:
        raise ValidationError("Must provide sigma time series for each asset")
    validate_correlation_matrix_strict(
        np.asarray(correlation_matrix), name="correlation_matrix"
    )
