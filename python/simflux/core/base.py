"""Base classes and configurations for simulation components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import numpy as np


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""
    
    seed: Optional[int] = None
    n_threads: Optional[int] = None
    batch_size: int = 10000
    memory_limit_gb: Optional[float] = None
    progress_callback: Optional[callable] = None


@dataclass 
class StorageConfig:
    """Configuration for interim results storage."""
    
    store_interim: bool = False
    store_defaults: bool = True
    store_losses: bool = True
    store_systematic_factors: bool = False
    format: str = "parquet"  # "parquet" or "hdf5"
    output_path: Optional[str] = None
    partition_by: Optional[list] = field(default_factory=lambda: ["sector"])
    compression: str = "snappy"  # "snappy", "gzip", "lz4", "zstd"
    batch_size: int = 10000


class BaseSimulator(ABC):
    """Abstract base class for all simulators."""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self._validate_config()
    
    def _validate_config(self):
        """Validate simulation configuration."""
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.config.memory_limit_gb is not None and self.config.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")
    
    @abstractmethod
    def simulate(self, *args, **kwargs):
        """Run simulation. To be implemented by subclasses."""
        pass
    
    @abstractmethod
    def validate_inputs(self, *args, **kwargs):
        """Validate simulation inputs. To be implemented by subclasses."""
        pass


class SimulationResults:
    """Container for simulation results with lazy loading capabilities."""
    
    def __init__(self, 
                 summary_stats: Dict[str, Any],
                 interim_data_path: Optional[str] = None):
        self.summary_stats = summary_stats
        self.interim_data_path = interim_data_path
        self._interim_data = None
    
    @property
    def has_interim_data(self) -> bool:
        """Check if interim data is available."""
        return self.interim_data_path is not None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return self.summary_stats
    
    def load_interim_data(self):
        """Load interim data on demand."""
        if not self.has_interim_data:
            raise ValueError("No interim data available")
        
        if self._interim_data is None:
            # Lazy loading implementation would go here
            # For now, return placeholder
            self._interim_data = {}
        
        return self._interim_data