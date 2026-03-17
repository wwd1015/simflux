"""End-to-end storage roundtrip tests using real Rust backend + ParquetResultsAnalyzer."""

import pytest
import numpy as np
import tempfile
import os

import simflux as sf
from simflux.core.backend import Backend


@pytest.fixture
def sample_portfolio():
    """Small portfolio for fast storage tests."""
    return sf.TwoFactorPortfolio.create_sample_portfolio(
        n_assets_per_sector=[5, 5],
        sectors=["Tech", "Finance"],
        inter_sector_correlation=0.2,
        intra_sector_correlations=0.4,
    )


class TestStorageRoundtrip:
    """Write interim data via Rust, read back via ParquetResultsAnalyzer."""

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_write_and_read_parquet(self, sample_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)

            results = sample_portfolio.simulate(
                n_simulations=50, storage_config=config
            )

            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            assert "portfolio_statistics" in results
            assert "analyzer" in results

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_analyzer_counts(self, sample_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            sample_portfolio.simulate(n_simulations=30, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            assert analyzer.count_simulations() == 30
            assert analyzer.count_assets() == 10  # 5 + 5

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_analyzer_schema(self, sample_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            sample_portfolio.simulate(n_simulations=20, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            schema = analyzer.get_schema()
            for col in ["trial_id", "asset_id", "sector", "defaulted", "loss_amount"]:
                assert col in schema

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_get_trial_losses(self, sample_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            sample_portfolio.simulate(n_simulations=40, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            trial_losses = analyzer.get_trial_losses()
            assert len(trial_losses) == 40
            assert "total_loss" in trial_losses.columns
            assert "total_defaults" in trial_losses.columns

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_get_defaults(self, sample_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            sample_portfolio.simulate(n_simulations=100, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            defaults = analyzer.get_defaults()
            # All returned rows should be defaulted
            assert all(defaults["defaulted"])

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_get_defaults_with_sector_filter(self, sample_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            sample_portfolio.simulate(n_simulations=100, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            defaults = analyzer.get_defaults(sectors=["Tech"])
            if len(defaults) > 0:
                assert all(defaults["sector"] == "Tech")

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_query_high_loss_trials(self, sample_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            sample_portfolio.simulate(n_simulations=200, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            high = analyzer.query_high_loss_trials(percentile=90)
            assert isinstance(high, list)
            # At most 10% should be above 90th percentile (allowing for ties)
            assert len(high) <= 200 * 0.12

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_calculate_portfolio_statistics(self, sample_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            sample_portfolio.simulate(n_simulations=100, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            stats = analyzer.calculate_portfolio_statistics()
            for key in ["mean_loss", "std_loss", "var_95", "var_99", "max_loss"]:
                assert key in stats
                assert isinstance(stats[key], float)

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_export_to_pandas(self, sample_portfolio):
        import pandas as pd

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            sample_portfolio.simulate(n_simulations=20, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            df = analyzer.export_to_pandas()
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_close(self, sample_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            sample_portfolio.simulate(n_simulations=10, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            analyzer.close()
            assert analyzer._lazy_frame is None
            # Re-access should recreate it
            _ = analyzer.lazy_frame
            assert analyzer._lazy_frame is not None


class TestStorageConfigDefaults:
    """Test StorageConfig partition_by auto-population."""

    def test_partition_by_default_when_interim(self):
        config = sf.StorageConfig(store_interim=True)
        assert config.partition_by == ["sector"]

    def test_partition_by_default_when_no_interim(self):
        config = sf.StorageConfig(store_interim=False)
        assert config.partition_by == []

    def test_partition_by_custom_preserved(self):
        config = sf.StorageConfig(store_interim=True, partition_by=["trial_id"])
        assert config.partition_by == ["trial_id"]
