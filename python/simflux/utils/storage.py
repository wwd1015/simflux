"""Storage utilities for simulation results using Parquet format."""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import pyarrow as pa


@dataclass
class StorageConfig:
    """Configuration for storing interim simulation results."""
    
    store_interim: bool = False
    store_defaults: bool = True
    store_losses: bool = True
    store_systematic_factors: bool = False
    format: str = "parquet"
    output_path: Optional[str] = None
    partition_by: Optional[List[str]] = None
    compression: str = "snappy"
    batch_size: int = 10000
    
    def __post_init__(self) -> None:
        if self.partition_by is None:
            self.partition_by = ["sector"] if self.store_interim else []
        
        valid_formats = ["parquet", "hdf5"]
        if self.format not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}")
        
        valid_compressions = ["snappy", "gzip", "lz4", "zstd", "brotli"]
        if self.compression not in valid_compressions:
            raise ValueError(f"compression must be one of {valid_compressions}")


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
                .select(pl.col("trial_id").n_unique())
                .collect()
                .item())
    
    def count_assets(self) -> int:
        """Count total number of assets."""
        return (self.lazy_frame
                .select(pl.col("asset_id").n_unique())
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
        query = self.lazy_frame.filter(pl.col("defaulted") == True)
        
        if trial_range:
            start, end = trial_range
            query = query.filter(pl.col("trial_id").is_between(start, end))
        
        if sectors:
            query = query.filter(pl.col("sector").is_in(sectors))
        
        if asset_ids:
            query = query.filter(pl.col("asset_id").is_in(asset_ids))
        
        return query.collect()
    
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
        query = self.lazy_frame.group_by("trial_id").agg(
            pl.col("loss_amount").sum().alias("total_loss"),
            pl.col("defaulted").sum().alias("total_defaults")
        )
        
        if trial_ids:
            query = query.filter(pl.col("trial_id").is_in(trial_ids))
        
        return query.collect().sort("trial_id")
    
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
            agg_exprs.append(pl.col("defaulted").mean().alias("default_rate"))
        
        if 'avg_loss' in metrics:
            agg_exprs.append(pl.col("loss_amount").mean().alias("avg_loss"))
        
        if 'total_loss' in metrics:
            agg_exprs.append(pl.col("loss_amount").sum().alias("total_loss"))
        
        if 'total_defaults' in metrics:
            agg_exprs.append(pl.col("defaulted").sum().alias("total_defaults"))
        
        if 'total_exposure' in metrics:
            agg_exprs.append(pl.col("asset_id").n_unique().alias("total_assets"))
        
        return (self.lazy_frame
                .group_by(["trial_id", "sector"])
                .agg(agg_exprs)
                .group_by("sector")
                .agg([
                    pl.col(col).mean().suffix("_mean") for col in [expr.meta.output_name() for expr in agg_exprs]
                ] + [
                    pl.col(col).std().suffix("_std") for col in [expr.meta.output_name() for expr in agg_exprs]
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
        factor_columns = [col for col in schema.keys() if col.startswith("systematic_factor")]
        
        if not factor_columns:
            return None
        
        query = self.lazy_frame.select(["trial_id"] + factor_columns).unique()
        
        if trial_ids:
            query = query.filter(pl.col("trial_id").is_in(trial_ids))
        
        return query.collect().sort("trial_id")
    
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
                        query_filter: Optional[pl.Expr] = None,
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Export filtered data to pandas DataFrame.
        
        Parameters:
        -----------
        query_filter : pl.Expr, optional
            Polars expression for filtering
        columns : List[str], optional
            Columns to include
            
        Returns:
        --------
        pd.DataFrame
            Pandas DataFrame
        """
        query = self.lazy_frame
        
        if query_filter is not None:
            query = query.filter(query_filter)
        
        if columns:
            query = query.select(columns)
        
        return query.collect().to_pandas()
    
    def close(self) -> None:
        """Clean up resources."""
        self._lazy_frame = None


def create_parquet_schema() -> pa.Schema:
    """Create the standard Parquet schema for simulation results."""
    return pa.schema([
        pa.field("trial_id", pa.int64()),
        pa.field("asset_id", pa.int32()),
        pa.field("sector", pa.string()),
        pa.field("defaulted", pa.bool_()),
        pa.field("time_to_default", pa.float32()),
        pa.field("loss_amount", pa.float32()),
        pa.field("recovery_rate", pa.float32()),
        pa.field("systematic_factor_global", pa.float32()),
        pa.field("systematic_factor_sector", pa.float32()),
        pa.field("asset_value", pa.float32()),
        pa.field("pd", pa.float32()),
        pa.field("lgd_mean", pa.float32()),
        pa.field("exposure", pa.float32())
    ])