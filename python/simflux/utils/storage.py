"""Storage utilities for simulation results using Parquet format."""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
from dataclasses import dataclass


@dataclass
class StorageConfig:
    """Configuration for persisting interim simulation results.

    Only the fields the Parquet writer actually honors are exposed:

    - ``store_interim`` — whether to persist per-asset interim records.
    - ``output_path`` — destination Parquet file (required when storing).
    - ``batch_size`` — rows buffered before each write; threaded into the Rust
      writer.

    The interim writer always emits a single Snappy-compressed Parquet file.
    Format, compression, and partitioning are not configurable.
    """

    store_interim: bool = False
    output_path: Optional[str] = None
    batch_size: int = 10000

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


class Columns:
    """Physical column names of the interim Parquet schema.

    The single Python-side source of truth for the columns the Rust writer
    produces (see ``src/storage.rs::create_parquet_schema``).  Every analyzer
    query references these constants, so a schema rename touches one place
    instead of eight methods.
    """

    TRIAL_ID = "trial_id"
    ASSET_ID = "asset_id"
    SECTOR_ID = "sector_id"
    SECTOR = "sector"
    DEFAULTED = "defaulted"
    TIME_TO_DEFAULT = "time_to_default"
    LOSS_AMOUNT = "loss_amount"
    RECOVERY_RATE = "recovery_rate"
    ASSET_VALUE = "asset_value"
    SYSTEMATIC_FACTOR_PREFIX = "systematic_factor"
    PD = "pd"
    LGD_MEAN = "lgd_mean"
    EXPOSURE = "exposure"


class ParquetResultsAnalyzer:
    """Analyzer for Parquet-stored simulation results with efficient querying."""
    
    def __init__(self, parquet_path: Union[str, Path]):
        """
        Initialize analyzer with Parquet file/directory path.
        
        Parameters:
        -----------
        parquet_path : str or Path
            Path to Parquet file or partitioned directory
        """
        self.path = str(parquet_path)
        self._validate_path()
        self._lazy_frame = None
    
    def _validate_path(self) -> None:
        """Validate that the Parquet path exists and is readable."""
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet path not found: {self.path}")
    
    @property
    def lazy_frame(self) -> pl.LazyFrame:
        """Get lazy frame for efficient querying."""
        if self._lazy_frame is None:
            self._lazy_frame = pl.scan_parquet(self.path)
        return self._lazy_frame
    
    def get_schema(self) -> Dict[str, str]:
        """Get the schema of the stored data."""
        return dict(self.lazy_frame.schema)
    
    def count_simulations(self) -> int:
        """Count total number of simulation trials."""
        return (self.lazy_frame
                .select(pl.col(Columns.TRIAL_ID).n_unique())
                .collect()
                .item())

    def count_assets(self) -> int:
        """Count total number of assets."""
        return (self.lazy_frame
                .select(pl.col(Columns.ASSET_ID).n_unique())
                .collect()
                .item())
    
    def get_defaults(self,
                     trial_range: Optional[tuple[int, int]] = None,
                     sectors: Optional[List[str]] = None,
                     asset_ids: Optional[List[int]] = None) -> pl.DataFrame:
        """
        Get defaulted assets with efficient filtering.
        
        Parameters:
        -----------
        trial_range : tuple, optional
            (start, end) trial range to filter
        sectors : List[str], optional
            Sectors to include
        asset_ids : List[int], optional
            Specific asset IDs to include
            
        Returns:
        --------
        pl.DataFrame
            DataFrame of defaulted assets
        """
        query = self._apply_filters(
            self.lazy_frame.filter(pl.col(Columns.DEFAULTED) == True),
            trial_range=trial_range, sectors=sectors, asset_ids=asset_ids,
        )
        return query.collect()

    def _apply_filters(self,
                       query: pl.LazyFrame,
                       trial_range: Optional[tuple[int, int]] = None,
                       sectors: Optional[List[str]] = None,
                       asset_ids: Optional[List[int]] = None) -> pl.LazyFrame:
        """Apply the shared domain filters (trial range, sectors, asset ids)."""
        if trial_range:
            start, end = trial_range
            query = query.filter(pl.col(Columns.TRIAL_ID).is_between(start, end))
        if sectors:
            query = query.filter(pl.col(Columns.SECTOR).is_in(sectors))
        if asset_ids:
            query = query.filter(pl.col(Columns.ASSET_ID).is_in(asset_ids))
        return query
    
    def get_trial_losses(self, trial_ids: Optional[List[int]] = None) -> pl.DataFrame:
        """
        Get total losses per trial.
        
        Parameters:
        -----------
        trial_ids : List[int], optional
            Specific trial IDs to include
            
        Returns:
        --------
        pl.DataFrame
            DataFrame with trial_id and total_loss columns
        """
        query = self.lazy_frame.group_by(Columns.TRIAL_ID).agg(
            pl.col(Columns.LOSS_AMOUNT).sum().alias("total_loss"),
            pl.col(Columns.DEFAULTED).sum().alias("total_defaults")
        )

        if trial_ids:
            query = query.filter(pl.col(Columns.TRIAL_ID).is_in(trial_ids))

        return query.collect().sort(Columns.TRIAL_ID)
    
    def analyze_by_sector(self, metrics: List[str] = None) -> pl.DataFrame:
        """
        Analyze results by sector across all trials.
        
        Parameters:
        -----------
        metrics : List[str], optional
            Metrics to calculate. Defaults to ['default_rate', 'avg_loss']
            
        Returns:
        --------
        pl.DataFrame
            DataFrame with sector-level analysis
        """
        if metrics is None:
            metrics = ['default_rate', 'avg_loss', 'total_exposure']
        
        agg_exprs = []

        if 'default_rate' in metrics:
            agg_exprs.append(pl.col(Columns.DEFAULTED).mean().alias("default_rate"))

        if 'avg_loss' in metrics:
            agg_exprs.append(pl.col(Columns.LOSS_AMOUNT).mean().alias("avg_loss"))

        if 'total_loss' in metrics:
            agg_exprs.append(pl.col(Columns.LOSS_AMOUNT).sum().alias("total_loss"))

        if 'total_defaults' in metrics:
            agg_exprs.append(pl.col(Columns.DEFAULTED).sum().alias("total_defaults"))

        if 'total_exposure' in metrics:
            agg_exprs.append(pl.col(Columns.ASSET_ID).n_unique().alias("total_assets"))

        return (self.lazy_frame
                .group_by([Columns.TRIAL_ID, Columns.SECTOR])
                .agg(agg_exprs)
                .group_by(Columns.SECTOR)
                .agg([
                    pl.col(col).mean().name.suffix("_mean") for col in [expr.meta.output_name() for expr in agg_exprs]
                ] + [
                    pl.col(col).std().name.suffix("_std") for col in [expr.meta.output_name() for expr in agg_exprs]
                ])
                .collect())
    
    def query_high_loss_trials(self, percentile: float = 95) -> List[int]:
        """
        Find trial IDs with losses above specified percentile.
        
        Parameters:
        -----------
        percentile : float, default=95
            Percentile threshold (0-100)
            
        Returns:
        --------
        List[int]
            List of trial IDs with high losses
        """
        trial_losses = self.get_trial_losses()
        threshold = np.percentile(trial_losses["total_loss"], percentile)
        
        high_loss_trials = trial_losses.filter(
            pl.col("total_loss") >= threshold
        )["trial_id"].to_list()
        
        return sorted(high_loss_trials)
    
    def get_systematic_factors(self, 
                              trial_ids: Optional[List[int]] = None) -> Optional[pl.DataFrame]:
        """
        Get systematic factors if stored.
        
        Parameters:
        -----------
        trial_ids : List[int], optional
            Specific trial IDs to include
            
        Returns:
        --------
        pl.DataFrame or None
            DataFrame of systematic factors, or None if not stored
        """
        schema = self.get_schema()
        factor_columns = [col for col in schema.keys()
                          if col.startswith(Columns.SYSTEMATIC_FACTOR_PREFIX)]

        if not factor_columns:
            return None

        query = self.lazy_frame.select([Columns.TRIAL_ID] + factor_columns).unique()

        if trial_ids:
            query = query.filter(pl.col(Columns.TRIAL_ID).is_in(trial_ids))

        return query.collect().sort(Columns.TRIAL_ID)
    
    def calculate_portfolio_statistics(self) -> Dict[str, float]:
        """
        Calculate portfolio-level risk statistics.
        
        Returns:
        --------
        Dict[str, float]
            Dictionary of portfolio statistics
        """
        trial_losses = self.get_trial_losses()
        losses = trial_losses["total_loss"].to_numpy()
        
        return {
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
            "var_95": float(np.percentile(losses, 95)),
            "var_99": float(np.percentile(losses, 99)),
            "var_999": float(np.percentile(losses, 99.9)),
            "expected_shortfall_95": float(np.mean(losses[losses >= np.percentile(losses, 95)])),
            "expected_shortfall_99": float(np.mean(losses[losses >= np.percentile(losses, 99)])),
            "max_loss": float(np.max(losses)),
            "min_loss": float(np.min(losses)),
        }
    
    def export_to_pandas(self,
                        trial_range: Optional[tuple[int, int]] = None,
                        sectors: Optional[List[str]] = None,
                        asset_ids: Optional[List[int]] = None,
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Export filtered data to a pandas DataFrame using domain filters.

        Callers express filters in domain terms (trial range, sectors, asset
        ids) rather than authoring Polars expressions against the physical
        schema, mirroring :meth:`get_defaults`.

        Parameters:
        -----------
        trial_range : tuple, optional
            (start, end) trial range to filter.
        sectors : List[str], optional
            Sectors to include.
        asset_ids : List[int], optional
            Specific asset IDs to include.
        columns : List[str], optional
            Columns to include (defaults to all).

        Returns:
        --------
        pd.DataFrame
            Pandas DataFrame.
        """
        query = self._apply_filters(
            self.lazy_frame, trial_range=trial_range,
            sectors=sectors, asset_ids=asset_ids,
        )

        if columns:
            query = query.select(columns)

        return query.collect().to_pandas()

    def close(self) -> None:
        """Clean up resources."""
        self._lazy_frame = None