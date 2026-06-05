"""Base classes and configurations for simulation components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, Callable


@dataclass
class SimulationConfig:
    """Configuration for simulation runs.

    Attributes
    ----------
    seed : int, optional
        Seed for the random number generator.

        Reproducibility is **within-backend**, not across backends.  Fixing a
        seed makes a run repeatable on whichever backend produced it, but the
        Rust and NumPy backends consume the seed through different RNG schemes
        (Rust draws per-stream ``StdRng`` in parallel; NumPy draws a single
        ``default_rng`` sequence).  The same seed therefore yields *different*
        individual paths depending on which backend is installed — the two
        agree only in distribution, which is exactly what the cross-validation
        tests assert.
    batch_size : int
        Number of records buffered before flushing during interim storage.
    memory_limit_gb : float, optional
        Soft cap on estimated array allocation; ``None`` disables the check.
    progress_callback : callable, optional
        Invoked with a completion fraction in ``[0, 1]`` during long runs.
    """

    seed: Optional[int] = None
    batch_size: int = 10000
    memory_limit_gb: Optional[float] = None
    progress_callback: Optional[Callable[[float], None]] = None

    def validate(self) -> None:
        """Raise ``ValueError`` if any field is invalid."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")


def check_memory(
    config: SimulationConfig, n_elements: int, element_bytes: int = 8
) -> None:
    """Check if estimated memory usage exceeds configured limit.

    Parameters
    ----------
    config : SimulationConfig
        Configuration with optional ``memory_limit_gb``.
    n_elements : int
        Total number of array elements to allocate.
    element_bytes : int
        Bytes per element (default 8 for float64).

    Raises
    ------
    MemoryError
        If estimated memory exceeds ``memory_limit_gb``.
    """
    if config.memory_limit_gb is None:
        return
    estimated_gb = (n_elements * element_bytes) / (1024**3)
    if estimated_gb > config.memory_limit_gb:
        raise MemoryError(
            f"Estimated memory usage ({estimated_gb:.2f} GB) exceeds limit "
            f"({config.memory_limit_gb:.2f} GB). Reduce n_paths/n_steps or "
            f"increase memory_limit_gb."
        )


class BaseSimulator(ABC):
    """Abstract base class for all simulators."""

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig()
        self.config.validate()

    def _check_memory(self, n_elements: int, element_bytes: int = 8) -> None:
        check_memory(self.config, n_elements, element_bytes)

    def _report_progress(self, fraction: float) -> None:
        """Invoke the progress callback if one was configured."""
        if self.config.progress_callback is not None:
            self.config.progress_callback(fraction)

    @abstractmethod
    def simulate(self, *args: Any, **kwargs: Any) -> Any:
        """Run simulation. To be implemented by subclasses."""
        pass

    @abstractmethod
    def validate_inputs(self, *args: Any, **kwargs: Any) -> None:
        """Validate simulation inputs. To be implemented by subclasses."""
        pass
