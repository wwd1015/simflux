"""Base classes and configurations for simulation components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""

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


def check_memory(config: SimulationConfig, n_elements: int, element_bytes: int = 8) -> None:
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
    estimated_gb = (n_elements * element_bytes) / (1024 ** 3)
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


class SimulationResults:
    """Container for simulation results with lazy loading capabilities."""

    def __init__(self,
                 summary_stats: Dict[str, Any],
                 interim_data_path: Optional[str] = None) -> None:
        self.summary_stats = summary_stats
        self.interim_data_path = interim_data_path
        self._interim_data: Optional[Any] = None

    @property
    def has_interim_data(self) -> bool:
        """Check if interim data is available."""
        return self.interim_data_path is not None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return self.summary_stats

    def load_interim_data(self) -> Any:
        """Load interim data on demand via ParquetResultsAnalyzer."""
        if not self.has_interim_data:
            raise ValueError("No interim data available")

        if self._interim_data is None:
            from ..utils.storage import ParquetResultsAnalyzer
            self._interim_data = ParquetResultsAnalyzer(self.interim_data_path)

        return self._interim_data
