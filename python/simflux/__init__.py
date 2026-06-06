"""
SimFlux: High-performance financial simulation library with Rust backend

A comprehensive library for stochastic process simulation and portfolio risk modeling,
featuring:
- Correlated Geometric Brownian Motion simulation
- Two-factor portfolio loss modeling (Merton framework)
- Efficient Rust backend for performance-critical operations
- Flexible Parquet-based storage for interim results analysis
"""

from .processes.gbm import GBM, CorrelatedGBM
from .processes.time_varying import TimeVaryingGBM, TimeVaryingCorrelatedGBM
from .portfolio.two_factor_model import TwoFactorPortfolio, AssetData
from .portfolio.correlation import TwoFactorCorrelationStructure
from .utils.storage import StorageConfig, ParquetResultsAnalyzer
from .utils.random_utils import set_seed
from .core.engine import SimulationEngine
from .core.backend import Backend

__version__ = "0.4.1"
__author__ = "SimFlux Contributors"

__all__ = [
    # Standard GBM
    "GBM",
    "CorrelatedGBM",
    # Time-varying GBM
    "TimeVaryingGBM",
    "TimeVaryingCorrelatedGBM",
    # Portfolio modeling
    "TwoFactorPortfolio",
    "AssetData",
    "TwoFactorCorrelationStructure",
    # Storage and utilities
    "StorageConfig",
    "ParquetResultsAnalyzer",
    "SimulationEngine",
    "Backend",
    "set_seed",
]
