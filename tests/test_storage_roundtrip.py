"""End-to-end storage roundtrip tests using real Rust backend + ParquetResultsAnalyzer."""

import pytest
import tempfile
import os

import simflux as sf
from simflux.core.backend import Backend


@pytest.fixture
def sample_portfolio():
    """Small portfolio for fast storage tests."""
    return sf.CreditPortfolio.create_sample_portfolio(
        n_assets_per_sector=[5, 5],
        sectors=["Tech", "Finance"],
        inter_sector_correlation=0.2,
        intra_sector_correlations=0.4,
    )


@pytest.fixture
def default_heavy_portfolio():
    """High-PD portfolio so the defaults-only store has plenty of rows for the
    filter/round-trip tests (the sample portfolio's PD is ~2-5%, too sparse)."""
    assets = [
        sf.AssetData(
            i, i % 2, 0.30, 0.5, 0.1, 1_000_000.0, "Tech" if i % 2 == 0 else "Finance"
        )
        for i in range(10)
    ]
    return sf.CreditPortfolio(
        assets=assets,
        intra_sector_correlations=0.3,
        sector_correlation_matrix=[[1.0, 0.2], [0.2, 1.0]],
    )


class TestStorageRoundtrip:
    """Write interim data via Rust, read back via ParquetResultsAnalyzer."""

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_write_and_read_parquet(self, sample_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)

            results = sample_portfolio.simulate(n_simulations=50, storage_config=config)

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
            for col in [
                "trial_id",
                "asset_id",
                "sector",
                "default_period",
                "loss_amount",
                "systematic_factor",
                "idiosyncratic_factor",
            ]:
                assert col in schema
            # The sparse store no longer carries a 'defaulted' flag (all rows are
            # defaults).
            assert "defaulted" not in schema

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
    def test_get_defaults(self, default_heavy_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            default_heavy_portfolio.simulate(n_simulations=100, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            defaults = analyzer.get_defaults()
            # Every stored row is a default event; with PD=0.3 there are many.
            assert len(defaults) > 0
            assert (defaults["loss_amount"] >= 0).all()
            assert "defaulted" not in defaults.columns

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_get_defaults_with_sector_filter(self, default_heavy_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            default_heavy_portfolio.simulate(n_simulations=100, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            defaults = analyzer.get_defaults(sectors=["Tech"])
            assert len(defaults) > 0
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
    def test_export_to_pandas(self, default_heavy_portfolio):
        import pandas as pd

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            default_heavy_portfolio.simulate(n_simulations=100, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            df = analyzer.export_to_pandas()
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0

            # The domain filters must actually wire through to the query.
            cols = analyzer.export_to_pandas(columns=["trial_id", "loss_amount"])
            assert list(cols.columns) == ["trial_id", "loss_amount"]

            tr = analyzer.export_to_pandas(trial_range=(0, 5))
            assert len(tr) > 0
            assert tr["trial_id"].min() >= 0 and tr["trial_id"].max() <= 5

            tech = analyzer.export_to_pandas(sectors=["Tech"])
            assert len(tech) > 0
            assert set(tech["sector"].unique()) == {"Tech"}

            a01 = analyzer.export_to_pandas(asset_ids=[0, 1])
            assert len(a01) > 0
            assert set(a01["asset_id"].unique()) <= {0, 1}

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


class TestSparseInterimStore:
    """The interim store is defaults-only with portfolio metadata in the footer."""

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_row_count_equals_total_defaults(self, default_heavy_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            default_heavy_portfolio.simulate(n_simulations=80, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            rows = len(analyzer.export_to_pandas())
            tl = analyzer.get_trial_losses()
            # One row per default event == sum of per-trial default counts.
            assert rows == int(tl["total_defaults"].sum())
            # Sparse: far fewer rows than the full asset x trial grid (10 x 80).
            assert rows < 80 * 10

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_counts_and_reindex_from_metadata(self, default_heavy_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            default_heavy_portfolio.simulate(n_simulations=55, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            assert (
                analyzer.count_simulations() == 55
            )  # exact even if some trials had 0 defaults
            assert analyzer.count_assets() == 10
            assert len(analyzer.get_trial_losses()) == 55  # reindexed to all trials

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_factors_recorded_per_default(self, default_heavy_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            default_heavy_portfolio.simulate(n_simulations=50, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            factors = analyzer.get_systematic_factors()
            assert factors is not None
            assert "systematic_factor" in factors.columns
            assert "idiosyncratic_factor" in factors.columns

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_default_rate_recovered_from_metadata(self, default_heavy_portfolio):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            default_heavy_portfolio.simulate(n_simulations=200, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            by_sector = analyzer.analyze_by_sector(metrics=["default_rate"])
            rates = by_sector["default_rate"].to_list()
            # Copula reproduces the marginal PD (0.30) over the 1-period horizon.
            assert all(0.15 < r < 0.45 for r in rates)

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_zero_default_sector_still_appears(self):
        # A sector with negligible PD has no rows in the defaults-only store, but
        # must still appear in analyze_by_sector (from metadata) with rate/total 0
        # — dropping it would be a silently-missing risk number.
        assets = [sf.AssetData(i, 0, 0.5, 0.5, 0.1, 1e6, "Loud") for i in range(5)] + [
            sf.AssetData(5 + i, 1, 1e-9, 0.5, 0.1, 1e6, "Quiet") for i in range(5)
        ]
        port = sf.CreditPortfolio(
            assets=assets,
            intra_sector_correlations=0.2,
            sector_correlation_matrix=[[1.0, 0.1], [0.1, 1.0]],
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            port.simulate(n_simulations=100, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            rows = {
                r["sector"]: r
                for r in analyzer.analyze_by_sector(
                    metrics=["default_rate", "total_loss"]
                ).to_dicts()
            }
            assert set(rows) == {"Loud", "Quiet"}  # zero-default sector not dropped
            assert rows["Quiet"]["total_loss"] == 0.0
            assert rows["Quiet"]["default_rate"] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_zero_default_file_is_readable(self):
        # Negligible PD: almost surely zero defaults, but the file must still be a
        # valid, metadata-bearing Parquet the analyzer can read and reindex.
        assets = [sf.AssetData(i, 0, 0.0001, 0.5, 0.1, 1e6, "S") for i in range(4)]
        port = sf.CreditPortfolio(
            assets=assets,
            intra_sector_correlations=0.2,
            sector_correlation_matrix=[[1.0]],
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "results.parquet")
            config = sf.StorageConfig(store_interim=True, output_path=path)
            port.simulate(n_simulations=10, storage_config=config)

            analyzer = sf.ParquetResultsAnalyzer(path)
            assert analyzer.count_simulations() == 10
            tl = analyzer.get_trial_losses()
            assert len(tl) == 10
            assert float(tl["total_loss"].sum()) >= 0.0
