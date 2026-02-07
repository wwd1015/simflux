"""Tests for portfolio simulation functionality."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import simflux as sf


class TestAssetData:
    """Test AssetData functionality."""
    
    def test_asset_data_creation(self):
        """Test AssetData creation with valid parameters."""
        asset = sf.AssetData(
            asset_id=1,
            sector_id=0,
            pd=0.05,
            lgd_mean=0.6,
            lgd_std=0.2,
            exposure=1000000.0,
            sector_name="Technology"
        )
        
        assert asset.asset_id == 1
        assert asset.sector_id == 0
        assert asset.pd == 0.05
        assert asset.lgd_mean == 0.6
        assert asset.lgd_std == 0.2
        assert asset.exposure == 1000000.0
        assert asset.sector_name == "Technology"
    
    def test_asset_data_invalid_parameters(self):
        """Test AssetData with invalid parameters."""
        # Invalid PD
        with pytest.raises(ValueError, match="PD must be between 0 and 1"):
            sf.AssetData(1, 0, -0.1, 0.6, 0.2, 1000000.0, "Tech")
        
        with pytest.raises(ValueError, match="PD must be between 0 and 1"):
            sf.AssetData(1, 0, 1.5, 0.6, 0.2, 1000000.0, "Tech")
        
        # Invalid LGD mean
        with pytest.raises(ValueError, match="LGD mean must be between 0 and 1"):
            sf.AssetData(1, 0, 0.05, -0.1, 0.2, 1000000.0, "Tech")
        
        with pytest.raises(ValueError, match="LGD mean must be between 0 and 1"):
            sf.AssetData(1, 0, 0.05, 1.5, 0.2, 1000000.0, "Tech")
        
        # Invalid LGD std
        with pytest.raises(ValueError, match="LGD std must be positive"):
            sf.AssetData(1, 0, 0.05, 0.6, -0.1, 1000000.0, "Tech")
        
        # Invalid exposure
        with pytest.raises(ValueError, match="Exposure must be non-negative"):
            sf.AssetData(1, 0, 0.05, 0.6, 0.2, -1000.0, "Tech")
    
    def test_asset_data_from_dataframe(self):
        """Test creating AssetData list from DataFrame."""
        df = pd.DataFrame({
            'asset_id': [1, 2, 3],
            'sector': ['Tech', 'Finance', 'Tech'],
            'pd': [0.02, 0.05, 0.03],
            'lgd_mean': [0.6, 0.45, 0.65],
            'lgd_std': [0.2, 0.15, 0.25],
            'exposure': [1000000, 500000, 750000]
        })
        
        assets = sf.AssetData.from_dataframe(df)
        
        assert len(assets) == 3
        assert assets[0].asset_id == 1
        assert assets[0].sector_name == 'Tech'
        assert assets[0].sector_id == 0  # First unique sector gets ID 0
        assert assets[1].sector_name == 'Finance' 
        assert assets[1].sector_id == 1  # Second unique sector gets ID 1
        assert assets[2].sector_id == 0  # Same as first asset (Tech)
    
    def test_asset_data_from_dataframe_missing_columns(self):
        """Test AssetData creation with missing required columns."""
        df = pd.DataFrame({
            'asset_id': [1, 2],
            'pd': [0.02, 0.05],
            # Missing other required columns
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            sf.AssetData.from_dataframe(df)
    
    def test_asset_data_from_dataframe_custom_mapping(self):
        """Test AssetData creation with custom sector mapping."""
        df = pd.DataFrame({
            'asset_id': [1, 2],
            'sector': ['Tech', 'Finance'],
            'pd': [0.02, 0.05],
            'lgd_mean': [0.6, 0.45],
            'lgd_std': [0.2, 0.15],
            'exposure': [1000000, 500000]
        })
        
        sector_mapping = {'Finance': 0, 'Tech': 1}
        assets = sf.AssetData.from_dataframe(df, sector_mapping)
        
        assert assets[0].sector_id == 1  # Tech mapped to 1
        assert assets[1].sector_id == 0  # Finance mapped to 0


class TestTwoFactorPortfolio:
    """Test TwoFactorPortfolio functionality."""
    
    def create_sample_assets(self):
        """Helper to create sample assets."""
        return [
            sf.AssetData(1, 0, 0.02, 0.6, 0.2, 1000000, "Tech"),
            sf.AssetData(2, 0, 0.03, 0.65, 0.25, 800000, "Tech"),
            sf.AssetData(3, 1, 0.05, 0.45, 0.15, 1200000, "Finance"),
            sf.AssetData(4, 1, 0.04, 0.50, 0.18, 900000, "Finance"),
        ]
    
    @patch('simflux.portfolio.two_factor_model.RUST_AVAILABLE', True)
    def test_portfolio_initialization(self):
        """Test TwoFactorPortfolio initialization."""
        assets = self.create_sample_assets()
        
        matrix = np.array([[1.0, 0.2], [0.2, 1.0]])
        portfolio = sf.TwoFactorPortfolio(
            assets=assets,
            intra_sector_correlations=0.4,
            sector_correlation_matrix=matrix,
            systematic_lgd_correlation=0.3
        )

        assert len(portfolio.assets) == 4
        assert portfolio.intra_sector_correlations == [0.4, 0.4]
        assert portfolio.systematic_lgd_correlation == 0.3
        assert portfolio.sector_names == ['Finance', 'Tech']  # Sorted alphabetically
        np.testing.assert_allclose(portfolio.sector_correlation_matrix, matrix)
    
    @patch('simflux.portfolio.two_factor_model.RUST_AVAILABLE', True)
    def test_portfolio_with_dict_correlations(self):
        """Test portfolio with dictionary of intra-sector correlations."""
        assets = self.create_sample_assets()
        
        portfolio = sf.TwoFactorPortfolio(
            assets=assets,
            intra_sector_correlations={'Tech': 0.35, 'Finance': 0.45}
        )
        
        # Should be ordered by sorted sector names: Finance, Tech
        assert portfolio.intra_sector_correlations == [0.45, 0.35]
    
    @patch('simflux.portfolio.two_factor_model.RUST_AVAILABLE', True)
    def test_portfolio_from_dataframe(self):
        """Test creating portfolio from DataFrame."""
        df = pd.DataFrame({
            'asset_id': [1, 2, 3],
            'sector': ['Tech', 'Finance', 'Tech'],
            'pd': [0.02, 0.05, 0.03],
            'lgd_mean': [0.6, 0.45, 0.65],
            'lgd_std': [0.2, 0.15, 0.25],
            'exposure': [1000000, 500000, 750000]
        })
        
        portfolio = sf.TwoFactorPortfolio(assets=df)
        
        assert len(portfolio.assets) == 3
        assert len(portfolio.sector_names) == 2
        assert 'Tech' in portfolio.sector_names
        assert 'Finance' in portfolio.sector_names

    def test_portfolio_dataframe_inferred_correlations(self):
        """Test that correlations can be inferred from DataFrame metadata."""
        df = pd.DataFrame({
            'asset_id': [1, 2, 3, 4],
            'sector': ['Tech', 'Tech', 'Finance', 'Healthcare'],
            'pd': [0.02, 0.03, 0.05, 0.04],
            'lgd_mean': [0.6, 0.55, 0.45, 0.5],
            'lgd_std': [0.2, 0.18, 0.15, 0.17],
            'exposure': [1_000_000, 750_000, 1_200_000, 900_000],
            'intra_sector_correlation': [0.35, 0.35, 0.45, 0.30],
        })

        sector_matrix = np.array([
            [1.0, 0.15, 0.20],
            [0.15, 1.0, 0.10],
            [0.20, 0.10, 1.0],
        ])

        portfolio = sf.TwoFactorPortfolio(
            assets=df,
            sector_correlation_matrix=sector_matrix,
        )

        # Intra correlations inferred from metadata (order: Finance, Healthcare, Tech)
        expected_intra = {
            'Finance': 0.45,
            'Healthcare': 0.30,
            'Tech': 0.35,
        }
        for name, value in zip(portfolio.sector_names, portfolio.intra_sector_correlations):
            assert value == pytest.approx(expected_intra[name], rel=1e-12)

        np.testing.assert_allclose(portfolio.sector_correlation_matrix, sector_matrix)
    
    def test_portfolio_invalid_correlations(self):
        """Test portfolio with invalid correlation parameters."""
        assets = self.create_sample_assets()
        
        # Invalid sector correlation matrix (not PSD)
        with pytest.raises(ValueError, match="sector_correlation_matrix must be positive semi-definite"):
            sf.TwoFactorPortfolio(assets=assets, sector_correlation_matrix=[[1.0, 1.5], [1.5, 1.0]])

        # Invalid systematic LGD correlation
        with pytest.raises(ValueError, match="systematic_lgd_correlation must be between -1 and 1"):
            sf.TwoFactorPortfolio(assets=assets, systematic_lgd_correlation=-1.5)
        
        # Invalid intra-sector correlation
        with pytest.raises(ValueError, match="All intra_sector_correlations must be between 0 and 1"):
            sf.TwoFactorPortfolio(assets=assets, intra_sector_correlations=1.2)
    
    def test_portfolio_empty_assets(self):
        """Test portfolio with no assets."""
        with pytest.raises(ValueError, match="No assets provided"):
            sf.TwoFactorPortfolio(assets=[])
    
    @patch('simflux.portfolio.two_factor_model.RUST_AVAILABLE', True)
    @patch('simflux.portfolio.two_factor_model._rust')
    def test_portfolio_simulate(self, mock_rust):
        """Test portfolio simulation."""
        # Mock the Rust simulation results
        mock_results = {
            'portfolio_statistics': {
                'mean': 50000.0,
                'std_dev': 75000.0,
                'var_95': 200000.0,
                'var_99': 350000.0,
                'var_999': 500000.0,
                'expected_shortfall_95': 250000.0,
                'expected_shortfall_99': 400000.0,
                'max_loss': 800000.0
            },
            'sector_statistics': {
                'Tech': {'mean': 30000.0, 'var_95': 120000.0, 'var_99': 200000.0},
                'Finance': {'mean': 20000.0, 'var_95': 80000.0, 'var_99': 150000.0}
            }
        }
        
        # Mock Rust classes
        mock_rust.PortfolioConfig = MagicMock()
        mock_rust.AssetData = MagicMock()
        mock_rust.simulate_portfolio.return_value = mock_results
        
        assets = self.create_sample_assets()
        portfolio = sf.TwoFactorPortfolio(assets=assets)
        
        results = portfolio.simulate(n_simulations=10000)
        
        # Check that Rust functions were called
        mock_rust.simulate_portfolio.assert_called_once()
        
        # Check results structure
        assert 'portfolio_statistics' in results
        assert 'sector_statistics' in results
        assert 'n_assets' in results
        assert 'n_sectors' in results
        assert results['n_assets'] == 4
        assert results['n_sectors'] == 2
    
    def test_portfolio_simulate_invalid_inputs(self):
        """Test portfolio simulation with invalid inputs."""
        assets = self.create_sample_assets()
        portfolio = sf.TwoFactorPortfolio(assets=assets)
        
        # Zero simulations
        with pytest.raises(ValueError, match="n_simulations must be positive"):
            portfolio.simulate(n_simulations=0)
        
        # Negative simulations
        with pytest.raises(ValueError, match="n_simulations must be positive"):
            portfolio.simulate(n_simulations=-100)
    
    @patch('simflux.portfolio.two_factor_model.RUST_AVAILABLE', True)
    def test_portfolio_summary(self):
        """Test portfolio summary generation."""
        assets = self.create_sample_assets()
        portfolio = sf.TwoFactorPortfolio(assets=assets)
        
        summary = portfolio.get_portfolio_summary()
        
        assert summary['n_assets'] == 4
        assert summary['n_sectors'] == 2
        assert summary['total_exposure'] == 3900000  # Sum of all exposures
        assert 'avg_pd' in summary
        assert 'avg_lgd' in summary
        assert 'sectors' in summary
        assert 'correlation_structure' in summary
        
        # Check sector breakdown
        assert len(summary['sectors']) == 2
        for sector_info in summary['sectors'].values():
            assert 'n_assets' in sector_info
            assert 'total_exposure' in sector_info
            assert 'exposure_pct' in sector_info
            assert 'avg_pd' in sector_info
            assert 'avg_lgd' in sector_info
    
    def test_create_sample_portfolio(self):
        """Test sample portfolio creation."""
        matrix = np.array([[1.0, 0.15], [0.15, 1.0]])
        portfolio = sf.TwoFactorPortfolio.create_sample_portfolio(
            n_assets_per_sector=10,
            sectors=['Tech', 'Finance'],
            sector_correlation_matrix=matrix
        )
        
        assert len(portfolio.assets) == 20
        assert len(portfolio.sector_names) == 2
        np.testing.assert_allclose(portfolio.sector_correlation_matrix, matrix)
        
        # Check that assets are distributed across sectors
        tech_count = sum(1 for asset in portfolio.assets if asset.sector_name == 'Tech')
        finance_count = sum(1 for asset in portfolio.assets if asset.sector_name == 'Finance')
        assert tech_count == 10
        assert finance_count == 10
    
    def test_create_sample_portfolio_different_sizes(self):
        """Test sample portfolio with different sector sizes."""
        portfolio = sf.TwoFactorPortfolio.create_sample_portfolio(
            n_assets_per_sector=[15, 5, 20],
            sectors=['Tech', 'Finance', 'Healthcare']
        )
        
        assert len(portfolio.assets) == 40
        assert len(portfolio.sector_names) == 3
        
        # Check sector distributions
        sector_counts = {}
        for asset in portfolio.assets:
            sector_counts[asset.sector_name] = sector_counts.get(asset.sector_name, 0) + 1
        
        assert sector_counts['Tech'] == 15
        assert sector_counts['Finance'] == 5
        assert sector_counts['Healthcare'] == 20
    
    @patch('simflux.portfolio.two_factor_model.RUST_AVAILABLE', False)
    def test_portfolio_without_rust(self):
        """Test portfolio falls back to NumPy implementation when Rust missing."""
        assets = self.create_sample_assets()

        with pytest.warns(RuntimeWarning):
            portfolio = sf.TwoFactorPortfolio(assets=assets)

        results = portfolio.simulate(n_simulations=10)
        assert 'portfolio_statistics' in results
        assert 'sector_statistics' in results


class TestStorageConfig:
    """Test StorageConfig functionality."""
    
    def test_storage_config_defaults(self):
        """Test default StorageConfig values."""
        config = sf.StorageConfig()
        
        assert config.store_interim is False
        assert config.store_defaults is True
        assert config.store_losses is True
        assert config.store_systematic_factors is False
        assert config.format == "parquet"
        assert config.output_path is None
        assert config.partition_by == ["sector"]
        assert config.compression == "snappy"
        assert config.batch_size == 10000
    
    def test_storage_config_custom_values(self):
        """Test StorageConfig with custom values."""
        config = sf.StorageConfig(
            store_interim=True,
            output_path="test_results.parquet",
            compression="zstd",
            partition_by=["sector", "trial_id"]
        )
        
        assert config.store_interim is True
        assert config.output_path == "test_results.parquet"
        assert config.compression == "zstd"
        assert config.partition_by == ["sector", "trial_id"]
    
    def test_storage_config_invalid_format(self):
        """Test StorageConfig with invalid format."""
        with pytest.raises(ValueError, match="format must be one of"):
            sf.StorageConfig(format="invalid_format")
    
    def test_storage_config_invalid_compression(self):
        """Test StorageConfig with invalid compression."""
        with pytest.raises(ValueError, match="compression must be one of"):
            sf.StorageConfig(compression="invalid_compression")


class TestInterimStorage:
    """Tests for interim storage behaviour."""

    @patch('simflux.portfolio.two_factor_model.RUST_AVAILABLE', True)
    @patch('simflux.portfolio.two_factor_model._rust')
    def test_store_interim_not_supported(self, mock_rust, *_):
        """Ensure simulate raises when interim storage requested with Rust backend."""
        assets = [
            sf.AssetData(1, 0, 0.02, 0.6, 0.2, 1000000, "Tech"),
            sf.AssetData(2, 0, 0.03, 0.65, 0.25, 800000, "Tech"),
        ]

        # Configure mock Rust module
        mock_rust.PortfolioConfig.return_value = MagicMock()
        mock_rust.AssetData.side_effect = lambda **kwargs: MagicMock()
        mock_rust.simulate_portfolio.side_effect = RuntimeError(
            "Interim storage via the Rust backend is temporarily unavailable"
        )

        portfolio = sf.TwoFactorPortfolio(assets=assets)
        storage_config = sf.StorageConfig(store_interim=True, output_path="results.parquet")

        with pytest.raises(RuntimeError, match="Interim storage via the Rust backend is not available"):
            portfolio.simulate(n_simulations=10, storage_config=storage_config)
