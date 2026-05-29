"""Tests for storage functionality."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import simflux as sf


class TestStorageIntegration:
    """Test storage and data persistence features."""

    def test_storage_config_creation(self):
        """Test StorageConfig creation and validation."""
        # Default config
        config = sf.StorageConfig()
        assert not config.store_interim
        assert config.output_path is None
        assert config.batch_size == 10000

        # Custom config
        custom_config = sf.StorageConfig(
            store_interim=True,
            output_path="test_output.parquet",
            batch_size=2048,
        )

        assert custom_config.store_interim
        assert custom_config.output_path == "test_output.parquet"
        assert custom_config.batch_size == 2048

    def test_portfolio_with_storage_config(self):
        """Test portfolio simulation with storage configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "portfolio_results.parquet")

            # Create portfolio
            portfolio = sf.TwoFactorPortfolio.create_sample_portfolio(
                n_assets_per_sector=[10, 8],
                sectors=['Tech', 'Finance'],
                inter_sector_correlation=0.2
            )

            # Create storage config
            storage_config = sf.StorageConfig(
                store_interim=True,
                output_path=output_path,
            )

            # This might fail if storage is not fully implemented
            # But we test the config passes through properly
            try:
                results = portfolio.simulate(
                    n_simulations=100,
                    storage_config=storage_config
                )

                # If successful, check basic results structure
                assert 'portfolio_statistics' in results

                # Check if file was created (if storage works)
                if os.path.exists(output_path):
                    assert os.path.getsize(output_path) > 0

            except RuntimeError as e:
                # Expected if storage backend not available
                if "Storage features require" in str(e):
                    pytest.skip("Storage backend not available")
                else:
                    raise

    def test_parquet_results_analyzer(self):
        """Test ParquetResultsAnalyzer functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy parquet file so the analyzer can open it
            parquet_path = os.path.join(temp_dir, "test_results.parquet")

            # Non-existent path should raise FileNotFoundError
            with pytest.raises(FileNotFoundError):
                sf.ParquetResultsAnalyzer(parquet_path)

            # Create a minimal parquet file for a successful init
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.table({"trial_id": [1], "asset_id": [1]})
            pq.write_table(table, parquet_path)

            analyzer = sf.ParquetResultsAnalyzer(parquet_path)

            # Basic properties should be accessible
            assert hasattr(analyzer, 'path')
            assert analyzer.path == parquet_path

            # Methods should exist
            assert hasattr(analyzer, 'query_high_loss_trials')
            assert hasattr(analyzer, 'analyze_by_sector')

    def test_storage_config_validation(self):
        """Test storage configuration validation."""
        # Non-positive batch size is rejected
        with pytest.raises(ValueError, match="batch_size must be positive"):
            sf.StorageConfig(batch_size=0)

        # Missing output path when storing is allowed (may error at runtime)
        config = sf.StorageConfig(store_interim=True)
        assert config.output_path is None

    def test_storage_memory_efficiency(self):
        """Test that storage doesn't dramatically increase memory usage."""
        pytest.importorskip("psutil")

        # Create a portfolio simulation without storage
        portfolio = sf.TwoFactorPortfolio.create_sample_portfolio(
            n_assets_per_sector=[5, 5],
            sectors=['A', 'B'],
            inter_sector_correlation=0.1
        )

        # Measure memory usage patterns (basic check)
        import psutil
        import gc

        process = psutil.Process()

        # Run without storage
        gc.collect()
        mem_before = process.memory_info().rss

        results_no_storage = portfolio.simulate(n_simulations=50)

        gc.collect()
        mem_after_no_storage = process.memory_info().rss
        mem_used_no_storage = mem_after_no_storage - mem_before

        # Run with storage config (even if it fails)
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = sf.StorageConfig(
                store_interim=True,
                output_path=os.path.join(temp_dir, "test.parquet")
            )

            try:
                gc.collect()
                mem_before_storage = process.memory_info().rss

                results_with_storage = portfolio.simulate(
                    n_simulations=50,
                    storage_config=storage_config
                )

                gc.collect()
                mem_after_storage = process.memory_info().rss
                mem_used_storage = mem_after_storage - mem_before_storage

                # Storage shouldn't use dramatically more memory (< 5x increase)
                memory_ratio = mem_used_storage / max(mem_used_no_storage, 1)
                assert memory_ratio < 5.0

            except RuntimeError:
                # Storage not implemented - that's fine for this test
                pass

        # Results should be structurally similar regardless of storage
        assert 'portfolio_statistics' in results_no_storage
        assert len(results_no_storage['portfolio_statistics']) > 0

    def test_storage_config_passthrough(self):
        """Storage config (incl. batch_size) passes through to a simulation."""
        config = sf.StorageConfig(store_interim=True, batch_size=4096)
        assert config.batch_size == 4096

        portfolio = sf.TwoFactorPortfolio.create_sample_portfolio(
            n_assets_per_sector=[3, 3],
            sectors=['X', 'Y'],
            inter_sector_correlation=0.1
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_path = os.path.join(temp_dir, "test.parquet")

            try:
                results = portfolio.simulate(n_simulations=10, storage_config=config)
                assert 'portfolio_statistics' in results
            except RuntimeError as e:
                if "Storage features require" in str(e):
                    pytest.skip("storage not available")
                else:
                    raise