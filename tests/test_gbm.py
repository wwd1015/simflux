"""Tests for GBM simulation functionality."""

import pytest
import numpy as np
import warnings
from unittest.mock import patch, MagicMock
import simflux as sf


class TestGBM:
    """Test single-asset GBM simulation."""
    
    def test_gbm_initialization(self):
        """Test GBM object creation."""
        gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
        
        assert gbm.mu == 0.05
        assert gbm.sigma == 0.2
        assert gbm.S0 == 100
    
    def test_gbm_invalid_parameters(self):
        """Test GBM with invalid parameters."""
        # Negative sigma
        with pytest.raises(ValueError, match="sigma must be positive"):
            sf.GBM(mu=0.05, sigma=-0.2, S0=100)
        
        # Zero S0
        with pytest.raises(ValueError, match="S0 must be positive"):
            sf.GBM(mu=0.05, sigma=0.2, S0=0)
        
        # Negative S0
        with pytest.raises(ValueError, match="S0 must be positive"):
            sf.GBM(mu=0.05, sigma=0.2, S0=-100)
    
    @patch('simflux.processes.gbm.RUST_AVAILABLE', True)
    @patch('simflux.processes.gbm._rust')
    def test_gbm_simulate(self, mock_rust):
        """Test GBM simulation with mocked Rust backend."""
        # Mock the Rust simulate_gbm function
        mock_paths = [[100.0, 105.0, 110.0], [100.0, 95.0, 98.0]]
        mock_rust.simulate_gbm.return_value = mock_paths
        
        gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
        paths = gbm.simulate(n_paths=2, n_steps=2, T=1.0)
        
        # Check that Rust function was called with correct parameters
        mock_rust.simulate_gbm.assert_called_once()
        call_args = mock_rust.simulate_gbm.call_args[1]
        assert call_args['mu'] == 0.05
        assert call_args['sigma'] == 0.2
        assert call_args['s0'] == 100.0
        assert call_args['n_paths'] == 2
        assert call_args['n_steps'] == 2
        
        # Check output shape and values
        assert paths.shape == (2, 3)
        assert paths[0, 0] == 100.0
        np.testing.assert_array_equal(paths, mock_paths)
    
    def test_gbm_simulate_invalid_inputs(self):
        """Test GBM simulation with invalid inputs."""
        gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
        
        # Zero paths
        with pytest.raises(ValueError, match="n_paths must be positive"):
            gbm.simulate(n_paths=0, n_steps=252)
        
        # Negative steps
        with pytest.raises(ValueError, match="n_steps must be positive"):
            gbm.simulate(n_paths=100, n_steps=-1)
        
        # Zero time horizon
        with pytest.raises(ValueError, match="T must be positive"):
            gbm.simulate(n_paths=100, n_steps=252, T=0)
    
    @patch('simflux.processes.gbm.RUST_AVAILABLE', True)
    @patch('simflux.processes.gbm._rust')
    def test_gbm_single_path(self, mock_rust):
        """Test single path simulation."""
        mock_paths = [[100.0, 105.0, 110.0]]
        mock_rust.simulate_gbm.return_value = mock_paths
        
        gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
        path = gbm.simulate_single_path(n_steps=2, T=1.0)
        
        assert len(path) == 3
        assert path[0] == 100.0
    
    def test_gbm_time_grid(self):
        """Test time grid generation."""
        gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
        time_grid = gbm.get_time_grid(n_steps=252, T=1.0)
        
        assert len(time_grid) == 253
        assert time_grid[0] == 0.0
        assert time_grid[-1] == 1.0
        np.testing.assert_allclose(np.diff(time_grid), 1.0/252, rtol=1e-10)


class TestCorrelatedGBM:
    """Test multi-asset correlated GBM simulation."""
    
    def test_correlated_gbm_initialization(self):
        """Test CorrelatedGBM object creation."""
        correlation_matrix = [[1.0, 0.3], [0.3, 1.0]]
        gbm = sf.CorrelatedGBM(
            mu=[0.05, 0.03],
            sigma=[0.2, 0.15],
            S0=[100, 50],
            correlation_matrix=correlation_matrix
        )
        
        assert gbm.mu == [0.05, 0.03]
        assert gbm.sigma == [0.2, 0.15]
        assert gbm.S0 == [100, 50]
        assert gbm.n_assets == 2
        np.testing.assert_array_equal(gbm.correlation_matrix, correlation_matrix)
    
    def test_correlated_gbm_invalid_dimensions(self):
        """Test CorrelatedGBM with mismatched dimensions."""
        correlation_matrix = [[1.0, 0.3], [0.3, 1.0]]
        
        # Mismatched sigma length
        with pytest.raises(ValueError, match="sigma must have same length as mu"):
            sf.CorrelatedGBM(
                mu=[0.05, 0.03],
                sigma=[0.2],  # Wrong length
                S0=[100, 50],
                correlation_matrix=correlation_matrix
            )
        
        # Wrong correlation matrix size
        with pytest.raises(ValueError, match="correlation_matrix must be 2x2"):
            sf.CorrelatedGBM(
                mu=[0.05, 0.03],
                sigma=[0.2, 0.15],
                S0=[100, 50],
                correlation_matrix=[[1.0]]  # Wrong size
            )
    
    def test_correlated_gbm_invalid_correlation_matrix(self):
        """Test CorrelatedGBM with invalid correlation matrix."""
        # Non-symmetric matrix
        with pytest.raises(ValueError, match="correlation_matrix must be symmetric"):
            sf.CorrelatedGBM(
                mu=[0.05, 0.03],
                sigma=[0.2, 0.15],
                S0=[100, 50],
                correlation_matrix=[[1.0, 0.3], [0.5, 1.0]]  # Not symmetric
            )
        
        # Diagonal not 1
        with pytest.raises(ValueError, match="correlation_matrix diagonal must be 1.0"):
            sf.CorrelatedGBM(
                mu=[0.05, 0.03],
                sigma=[0.2, 0.15],
                S0=[100, 50],
                correlation_matrix=[[0.8, 0.3], [0.3, 1.0]]  # Diagonal not 1
            )
        
        # Values outside [-1, 1]
        with pytest.raises(ValueError, match="correlation_matrix values must be between -1 and 1"):
            sf.CorrelatedGBM(
                mu=[0.05, 0.03],
                sigma=[0.2, 0.15],
                S0=[100, 50],
                correlation_matrix=[[1.0, 1.5], [1.5, 1.0]]  # Value > 1
            )
    
    @patch('simflux.processes.gbm.RUST_AVAILABLE', True)
    @patch('simflux.processes.gbm._rust')
    def test_correlated_gbm_simulate(self, mock_rust):
        """Test correlated GBM simulation."""
        # Mock return: paths[trial][asset][time_step]
        mock_paths = [
            [[100.0, 105.0], [50.0, 48.0]],  # Trial 1
            [[100.0, 98.0], [50.0, 52.0]]   # Trial 2
        ]
        mock_rust.simulate_gbm_multi.return_value = mock_paths
        
        correlation_matrix = [[1.0, 0.3], [0.3, 1.0]]
        gbm = sf.CorrelatedGBM(
            mu=[0.05, 0.03],
            sigma=[0.2, 0.15],
            S0=[100, 50],
            correlation_matrix=correlation_matrix
        )
        
        paths = gbm.simulate(n_paths=2, n_steps=1, T=1.0)
        
        # Check function call
        mock_rust.simulate_gbm_multi.assert_called_once()
        call_args = mock_rust.simulate_gbm_multi.call_args[1]
        assert call_args['mu'] == [0.05, 0.03]
        assert call_args['sigma'] == [0.2, 0.15]
        assert call_args['s0'] == [100, 50]
        
        # Check output shape
        assert paths.shape == (2, 2, 2)  # (n_paths, n_assets, n_steps+1)
        np.testing.assert_array_equal(paths, mock_paths)
    
    def test_correlated_gbm_from_single_gbm(self):
        """Test creating CorrelatedGBM from single GBM."""
        single_gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
        correlation_matrix = [[1.0, 0.3], [0.3, 1.0]]
        
        multi_gbm = sf.CorrelatedGBM.from_single_gbm(
            single_gbm, 
            n_assets=2, 
            correlation_matrix=correlation_matrix
        )
        
        assert multi_gbm.mu == [0.05, 0.05]
        assert multi_gbm.sigma == [0.2, 0.2]
        assert multi_gbm.S0 == [100, 100]
        assert multi_gbm.n_assets == 2
    
    def test_correlated_gbm_asset_names(self):
        """Test asset name generation."""
        correlation_matrix = [[1.0, 0.3], [0.3, 1.0]]
        gbm = sf.CorrelatedGBM(
            mu=[0.05, 0.03],
            sigma=[0.2, 0.15],
            S0=[100, 50],
            correlation_matrix=correlation_matrix
        )
        
        names = gbm.get_asset_names()
        assert names == ["Asset_0", "Asset_1"]
    
    def test_get_correlation_matrix(self):
        """Test correlation matrix getter."""
        correlation_matrix = np.array([[1.0, 0.3], [0.3, 1.0]])
        gbm = sf.CorrelatedGBM(
            mu=[0.05, 0.03],
            sigma=[0.2, 0.15],
            S0=[100, 50],
            correlation_matrix=correlation_matrix
        )
        
        retrieved_matrix = gbm.get_correlation_matrix()
        np.testing.assert_array_equal(retrieved_matrix, correlation_matrix)
        
        # Should be a copy, not the same object
        retrieved_matrix[0, 1] = 0.5
        np.testing.assert_array_equal(gbm.correlation_matrix, correlation_matrix)


class TestRustUnavailable:
    """Test behavior when Rust backend is unavailable."""

    @patch('simflux.core.engine.RUST_AVAILABLE', False)
    def test_gbm_without_rust(self):
        """Test GBM works with fallback when Rust backend unavailable."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
            paths = gbm.simulate(n_paths=10, n_steps=5, T=1.0)

            # Check warning was issued
            assert len(w) == 1
            assert "Rust backend not available" in str(w[0].message)
            assert issubclass(w[0].category, RuntimeWarning)

            # Check fallback works
            assert paths.shape == (10, 6)
            assert np.all(paths[:, 0] == 100.0)  # Initial values correct

    @patch('simflux.core.engine.RUST_AVAILABLE', False)
    def test_correlated_gbm_without_rust(self):
        """Test CorrelatedGBM works with fallback when Rust backend unavailable."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            corr_gbm = sf.CorrelatedGBM(
                mu=[0.05, 0.03],
                sigma=[0.2, 0.15],
                S0=[100, 50],
                correlation_matrix=[[1.0, 0.3], [0.3, 1.0]]
            )
            paths = corr_gbm.simulate(n_paths=10, n_steps=5, T=1.0)

            # Check warning was issued
            assert len(w) == 1
            assert "Rust backend not available" in str(w[0].message)

            # Check fallback works
            assert paths.shape == (10, 2, 6)
            assert np.all(paths[:, 0, 0] == 100.0)  # Asset 1 initial values
            assert np.all(paths[:, 1, 0] == 50.0)   # Asset 2 initial values