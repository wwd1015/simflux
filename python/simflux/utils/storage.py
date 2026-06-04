"""Storage utilities for simulation results using Parquet format."""

import json
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Optional, List, Dict, Union
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

    The store is **defaults-only**: one row per defaulted obligor per trial
    (non-defaulted asset-trials are not written). Portfolio totals the rows no
    longer carry — trial/asset counts, default rates — are read from the
    file-level key-value metadata (see :class:`MetadataKeys`).
    """

    TRIAL_ID = "trial_id"
    ASSET_ID = "asset_id"
    SECTOR_ID = "sector_id"
    SECTOR = "sector"
    DEFAULT_PERIOD = "default_period"
    TIME_TO_DEFAULT = "time_to_default"
    LOSS_AMOUNT = "loss_amount"
    RECOVERY_RATE = "recovery_rate"
    ASSET_VALUE = "asset_value"
    SYSTEMATIC_FACTOR = "systematic_factor"
    IDIOSYNCRATIC_FACTOR = "idiosyncratic_factor"
    PD = "pd"
    LGD_MEAN = "lgd_mean"
    EXPOSURE = "exposure"


class MetadataKeys:
    """File-level Parquet key-value metadata keys written by the Rust writer
    (see ``src/storage.rs::set_metadata``). Mirror of that source."""

    N_TRIALS = "simflux.n_trials"
    N_ASSETS = "simflux.n_assets"
    SECTOR_ASSET_COUNTS = "simflux.sector_asset_counts"


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
        self._lazy_frame: Optional[pl.LazyFrame] = None
        self._meta: Optional[Dict[str, Any]] = None

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

    @property
    def metadata(self) -> Dict[str, Any]:
        """Portfolio-shape metadata stored in the Parquet file footer.

        The defaults-only store omits non-defaulted asset-trials, so totals
        (``n_trials``, ``n_assets``, per-sector asset counts) are recovered from
        the file's key-value metadata. Returns ``{}`` for files written without
        it (e.g. a hand-built Parquet), and callers fall back accordingly.
        """
        if self._meta is None:
            self._meta = self._read_metadata()
        return self._meta

    def _read_metadata(self) -> Dict[str, Any]:
        try:
            import pyarrow.parquet as pq

            src = self.path
            if Path(src).is_dir():
                files = sorted(Path(src).glob("*.parquet"))
                if not files:
                    return {}
                src = str(files[0])
            raw = pq.read_metadata(src).metadata or {}
        except Exception:
            return {}

        def _get(key: str) -> Optional[str]:
            v = raw.get(key.encode()) if raw else None
            return v.decode() if isinstance(v, (bytes, bytearray)) else v

        meta: Dict[str, Any] = {}
        n_trials = _get(MetadataKeys.N_TRIALS)
        n_assets = _get(MetadataKeys.N_ASSETS)
        counts = _get(MetadataKeys.SECTOR_ASSET_COUNTS)
        if n_trials is not None:
            meta["n_trials"] = int(n_trials)
        if n_assets is not None:
            meta["n_assets"] = int(n_assets)
        if counts is not None:
            try:
                meta["sector_asset_counts"] = json.loads(counts)
            except (ValueError, TypeError):
                pass
        return meta

    def get_schema(self) -> Dict[str, str]:
        """Get the schema of the stored data."""
        return {
            name: str(dtype) for name, dtype in self.lazy_frame.collect_schema().items()
        }

    def count_simulations(self) -> int:
        """Total number of simulation trials.

        Read from file metadata — a trial with zero defaults has no rows, so a
        ``trial_id`` distinct-count would undercount. Falls back to the distinct
        count only for files written without metadata.
        """
        n = self.metadata.get("n_trials")
        if n is not None:
            return int(n)
        return (
            self.lazy_frame.select(pl.col(Columns.TRIAL_ID).n_unique()).collect().item()
        )

    def count_assets(self) -> int:
        """Total number of assets in the portfolio.

        From file metadata (an asset that never defaults has no rows). Falls
        back to the distinct ``asset_id`` count for metadata-less files.
        """
        n = self.metadata.get("n_assets")
        if n is not None:
            return int(n)
        return (
            self.lazy_frame.select(pl.col(Columns.ASSET_ID).n_unique()).collect().item()
        )

    def get_defaults(
        self,
        trial_range: Optional[tuple[int, int]] = None,
        sectors: Optional[List[str]] = None,
        asset_ids: Optional[List[int]] = None,
    ) -> pl.DataFrame:
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
        # Every stored row is already a default event, so no defaulted filter.
        query = self._apply_filters(
            self.lazy_frame,
            trial_range=trial_range,
            sectors=sectors,
            asset_ids=asset_ids,
        )
        return query.collect()

    def _apply_filters(
        self,
        query: pl.LazyFrame,
        trial_range: Optional[tuple[int, int]] = None,
        sectors: Optional[List[str]] = None,
        asset_ids: Optional[List[int]] = None,
    ) -> pl.LazyFrame:
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
        present = (
            self.lazy_frame.group_by(Columns.TRIAL_ID)
            .agg(
                pl.col(Columns.LOSS_AMOUNT).sum().alias("total_loss"),
                pl.len().alias("total_defaults"),  # every stored row is a default
            )
            .collect()
            .with_columns(pl.col(Columns.TRIAL_ID).cast(pl.Int64))
        )

        n_trials = self.metadata.get("n_trials")
        if n_trials is not None:
            # Reindex against every trial: trials with zero defaults are absent
            # from the sparse store but contribute zero loss / zero defaults.
            full = pl.DataFrame(
                {Columns.TRIAL_ID: np.arange(int(n_trials), dtype=np.int64)}
            )
            df = full.join(present, on=Columns.TRIAL_ID, how="left").with_columns(
                pl.col("total_loss").fill_null(0.0),
                pl.col("total_defaults").fill_null(0).cast(pl.Int64),
            )
        else:
            df = present

        if trial_ids:
            df = df.filter(pl.col(Columns.TRIAL_ID).is_in(trial_ids))

        return df.sort(Columns.TRIAL_ID)

    def analyze_by_sector(self, metrics: Optional[List[str]] = None) -> pl.DataFrame:
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
            metrics = ["default_rate", "avg_loss", "total_loss"]

        # Aggregate the stored default events per sector. avg_loss is the mean
        # loss per default event; total_defaults is the event count.
        agg = (
            self.lazy_frame.group_by(Columns.SECTOR)
            .agg(
                pl.col(Columns.LOSS_AMOUNT).sum().alias("total_loss"),
                pl.col(Columns.LOSS_AMOUNT).mean().alias("avg_loss"),
                pl.len().alias("total_defaults"),
            )
            .collect()
        )

        if "default_rate" in metrics:
            # rate = defaults / (n_trials x assets in sector), recovered from
            # metadata since non-defaulted asset-trials are not stored.
            n_trials = self.metadata.get("n_trials")
            sector_counts = self.metadata.get("sector_asset_counts") or {}
            if n_trials and sector_counts:
                rates = [
                    (
                        defaults / denom
                        if (denom := float(sector_counts.get(sec, 0)) * float(n_trials))
                        else None
                    )
                    for sec, defaults in zip(
                        agg[Columns.SECTOR].to_list(), agg["total_defaults"].to_list()
                    )
                ]
                agg = agg.with_columns(
                    pl.Series("default_rate", rates, dtype=pl.Float64)
                )
            else:
                agg = agg.with_columns(
                    pl.lit(None, dtype=pl.Float64).alias("default_rate")
                )

        keep = [Columns.SECTOR] + [
            m
            for m in ("default_rate", "avg_loss", "total_loss", "total_defaults")
            if m in metrics and m in agg.columns
        ]
        return agg.select(keep)

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

        high_loss_trials = trial_losses.filter(pl.col("total_loss") >= threshold)[
            "trial_id"
        ].to_list()

        return sorted(high_loss_trials)

    def get_systematic_factors(
        self, trial_ids: Optional[List[int]] = None
    ) -> Optional[pl.DataFrame]:
        """The systematic and idiosyncratic factors recorded at each stored
        default event.

        Returns ``trial_id``, ``asset_id`` and the factor columns (one row per
        default), or ``None`` if the factor columns are absent. Useful for
        debugging *why* an obligor defaulted.

        Parameters:
        -----------
        trial_ids : List[int], optional
            Specific trial IDs to include
        """
        schema = self.get_schema()
        factor_columns = [
            c
            for c in (Columns.SYSTEMATIC_FACTOR, Columns.IDIOSYNCRATIC_FACTOR)
            if c in schema
        ]
        if not factor_columns:
            return None

        select = [Columns.TRIAL_ID]
        if Columns.ASSET_ID in schema:
            select.append(Columns.ASSET_ID)

        query = self.lazy_frame.select(select + factor_columns)
        if trial_ids:
            query = query.filter(pl.col(Columns.TRIAL_ID).is_in(trial_ids))

        return query.collect().sort(select)

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
            "expected_shortfall_95": float(
                np.mean(losses[losses >= np.percentile(losses, 95)])
            ),
            "expected_shortfall_99": float(
                np.mean(losses[losses >= np.percentile(losses, 99)])
            ),
            "max_loss": float(np.max(losses)),
            "min_loss": float(np.min(losses)),
        }

    def export_to_pandas(
        self,
        trial_range: Optional[tuple[int, int]] = None,
        sectors: Optional[List[str]] = None,
        asset_ids: Optional[List[int]] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
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
            self.lazy_frame,
            trial_range=trial_range,
            sectors=sectors,
            asset_ids=asset_ids,
        )

        if columns:
            query = query.select(columns)

        return query.collect().to_pandas()

    def close(self) -> None:
        """Clean up resources."""
        self._lazy_frame = None
