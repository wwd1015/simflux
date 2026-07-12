"""The exception hierarchy contract.

Every deliberate SimFlux error derives from SimfluxError so callers can catch
"anything SimFlux" with one clause, while each concrete type also subclasses
the builtin it historically was — so pre-0.7.0 ``except ValueError`` /
``except RuntimeError`` / ``except MemoryError`` code keeps working.
"""

import numpy as np
import pytest

import simflux as sf
from simflux import (
    BackendError,
    MemoryLimitError,
    SimfluxError,
    ValidationError,
)
from simflux.core import SimulationConfig


class TestHierarchy:
    def test_all_types_derive_from_simflux_error(self):
        for exc_type in (
            ValidationError,
            BackendError,
            MemoryLimitError,
        ):
            assert issubclass(exc_type, SimfluxError)

    def test_builtin_compatibility(self):
        # The builtin aliases the pre-0.7.0 API raised must keep catching.
        assert issubclass(ValidationError, ValueError)
        assert issubclass(BackendError, RuntimeError)
        assert issubclass(MemoryLimitError, MemoryError)


class TestRaisedTypes:
    def test_invalid_gbm_params_raise_validation_error(self):
        with pytest.raises(ValidationError, match="sigma"):
            sf.GBM(mu=0.05, sigma=-0.2, S0=100.0)

    def test_invalid_correlation_matrix_raises_validation_error(self):
        bad = np.array([[1.0, 0.9], [0.2, 1.0]])  # asymmetric
        with pytest.raises(ValidationError):
            sf.CorrelatedGBM(
                mu=[0.05, 0.05],
                sigma=[0.2, 0.2],
                S0=[100.0, 100.0],
                correlation_matrix=bad,
            )

    def test_invalid_timing_raises_validation_error(self):
        with pytest.raises(ValidationError):
            sf.Frailty(persistence=1.5)

    def test_memory_guard_raises_memory_limit_error(self):
        gbm = sf.GBM(
            mu=0.05,
            sigma=0.2,
            S0=100.0,
            config=SimulationConfig(memory_limit_gb=0.0001),
        )
        with pytest.raises(MemoryLimitError, match="exceeds limit"):
            gbm.simulate(n_paths=100_000, n_steps=252, T=1.0)

    def test_one_clause_catches_everything(self):
        # The point of the hierarchy: one except clause for any SimFlux error.
        try:
            sf.GBM(mu=0.05, sigma=-1.0, S0=100.0)
        except SimfluxError as exc:
            assert isinstance(exc, ValueError)  # and still the builtin
        else:
            pytest.fail("expected a SimfluxError")
