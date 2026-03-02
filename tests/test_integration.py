"""Integration tests for end-to-end workflows."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import simflux as sf


class TestIntegrationWorkflows:
    """Test complete workflows from start to finish."""

    def test_single_asset_workflow(self):
        """Test complete single-asset GBM workflow."""
        # Create GBM
        gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)

        # Simulate paths
        n_paths, n_steps = 1000, 252
        paths = gbm.simulate(n_paths=n_paths, n_steps=n_steps, T=1.0)

        # Verify basic properties
        assert paths.shape == (n_paths, n_steps + 1)
        assert np.all(paths[:, 0] == 100.0)  # Initial values
        assert np.all(paths > 0)  # Prices stay positive

        # Check statistical properties (Monte Carlo)
        final_prices = paths[:, -1]
        log_returns = np.log(final_prices / 100.0)

        # Expected log return: (mu - 0.5*sigma^2)*T
        expected_log_return = (0.05 - 0.5 * 0.2**2) * 1.0
        actual_mean = np.mean(log_returns)

        # Allow 5% tolerance for Monte Carlo error
        assert abs(actual_mean - expected_log_return) < 0.02

        # Check volatility
        expected_volatility = 0.2 * np.sqrt(1.0)  # sigma * sqrt(T)
        actual_volatility = np.std(log_returns)
        assert abs(actual_volatility - expected_volatility) < 0.02

    def test_multi_asset_correlation_workflow(self):
        """Test correlated multi-asset workflow."""
        # Create correlation matrix
        correlation_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])

        corr_gbm = sf.CorrelatedGBM(
            mu=[0.06, 0.04, 0.08],
            sigma=[0.2, 0.15, 0.25],
            S0=[100, 50, 200],
            correlation_matrix=correlation_matrix
        )

        # Simulate
        n_paths, n_steps = 2000, 100
        paths = corr_gbm.simulate(n_paths=n_paths, n_steps=n_steps, T=0.5)

        # Verify shape
        assert paths.shape == (n_paths, 3, n_steps + 1)

        # Check initial values
        assert np.all(paths[:, 0, 0] == 100)
        assert np.all(paths[:, 1, 0] == 50)
        assert np.all(paths[:, 2, 0] == 200)

        # Check correlation in final returns
        final_returns = []
        for i in range(3):
            returns = (paths[:, i, -1] / paths[:, i, 0]) - 1
            final_returns.append(returns)

        realized_corr = np.corrcoef(final_returns)

        # Check correlation is roughly preserved (looser tolerance for MC)
        for i in range(3):
            for j in range(3):
                expected = correlation_matrix[i, j]
                actual = realized_corr[i, j]
                tolerance = 0.15 if i != j else 0.05  # Diagonal should be close to 1
                assert abs(actual - expected) < tolerance

    def test_portfolio_risk_workflow(self):
        """Test complete portfolio risk analysis workflow."""
        # Create sample portfolio
        n_assets_per_sector = [25, 20, 15]
        sectors = ['Technology', 'Finance', 'Healthcare']

        sector_corr = np.array([
            [1.0, 0.2, 0.1],
            [0.2, 1.0, 0.15],
            [0.1, 0.15, 1.0]
        ])

        portfolio = sf.TwoFactorPortfolio.create_sample_portfolio(
            n_assets_per_sector=n_assets_per_sector,
            sectors=sectors,
            sector_correlation_matrix=sector_corr,
            intra_sector_correlations={
                'Technology': 0.4,
                'Finance': 0.5,
                'Healthcare': 0.3
            }
        )

        # Run simulation
        results = portfolio.simulate(n_simulations=5000)

        # Check results structure
        assert 'portfolio_statistics' in results
        assert 'sector_statistics' in results

        portfolio_stats = results['portfolio_statistics']
        required_stats = ['mean', 'std_dev', 'var_95', 'var_99', 'var_999',
                         'expected_shortfall_99', 'max_loss']

        for stat in required_stats:
            assert stat in portfolio_stats
            assert isinstance(portfolio_stats[stat], (int, float))
            assert portfolio_stats[stat] >= 0  # Losses should be positive

        # Check sector statistics
        sector_stats = results['sector_statistics']
        assert len(sector_stats) == 3

        for sector in sectors:
            assert sector in sector_stats
            sector_data = sector_stats[sector]
            for stat in required_stats:
                assert stat in sector_data

    def test_performance_comparison_workflow(self):
        """Test that we can run basic performance comparisons."""
        # Small test to ensure benchmark infrastructure works
        gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)

        # Time a small simulation
        import time
        start_time = time.time()
        paths = gbm.simulate(n_paths=100, n_steps=50, T=1.0)
        execution_time = time.time() - start_time

        # Should complete in reasonable time (< 1 second for small problem)
        assert execution_time < 1.0
        assert paths.shape == (100, 51)

        # Test fallback comparison
        engine = gbm.engine
        numpy_paths = engine._numpy_simulate_gbm(
            mu=0.05, sigma=0.2, s0=100,
            n_paths=100, n_steps=50, T=1.0
        )

        # Both should produce valid results
        assert numpy_paths.shape == paths.shape
        assert np.all(numpy_paths > 0)
        assert np.all(paths > 0)

    def test_error_handling_workflow(self):
        """Test error handling in realistic scenarios."""
        # Test invalid correlation matrix
        with pytest.raises(ValueError, match="correlation_matrix must be symmetric"):
            sf.CorrelatedGBM(
                mu=[0.05, 0.03],
                sigma=[0.2, 0.15],
                S0=[100, 50],
                correlation_matrix=[[1.0, 0.3], [0.5, 1.0]]  # Not symmetric
            )

        # Test dimension mismatches
        with pytest.raises(ValueError, match="sigma must have same length as mu"):
            sf.CorrelatedGBM(
                mu=[0.05, 0.03, 0.04],  # 3 elements
                sigma=[0.2, 0.15],      # 2 elements
                S0=[100, 50, 75],
                correlation_matrix=np.eye(3)
            )

        # Test invalid simulation parameters
        gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)

        with pytest.raises(ValueError, match="n_paths must be positive"):
            gbm.simulate(n_paths=0, n_steps=10)

        with pytest.raises(ValueError, match="T must be positive"):
            gbm.simulate(n_paths=10, n_steps=10, T=0)

    def test_edge_cases_workflow(self):
        """Test edge cases and boundary conditions."""
        # Very high volatility
        gbm_high_vol = sf.GBM(mu=0.0, sigma=2.0, S0=100)
        paths_high_vol = gbm_high_vol.simulate(n_paths=100, n_steps=10, T=0.1)

        # Should still be positive and finite
        assert np.all(paths_high_vol > 0)
        assert np.all(np.isfinite(paths_high_vol))

        # Very low volatility
        gbm_low_vol = sf.GBM(mu=0.05, sigma=0.001, S0=100)
        paths_low_vol = gbm_low_vol.simulate(n_paths=100, n_steps=252, T=1.0)

        # Should be close to deterministic growth
        expected_final = 100 * np.exp(0.05 * 1.0)
        actual_finals = paths_low_vol[:, -1]
        assert np.abs(np.mean(actual_finals) - expected_final) < 1.0

        # Near-perfect correlation (1.0 exactly is singular, not positive definite)
        near_perfect_corr = sf.CorrelatedGBM(
            mu=[0.05, 0.03],
            sigma=[0.2, 0.15],
            S0=[100, 50],
            correlation_matrix=[[1.0, 0.999], [0.999, 1.0]]
        )

        paths_perfect = near_perfect_corr.simulate(n_paths=500, n_steps=50, T=0.25)

        # Returns should be highly correlated
        returns_1 = (paths_perfect[:, 0, -1] / paths_perfect[:, 0, 0]) - 1
        returns_2 = (paths_perfect[:, 1, -1] / paths_perfect[:, 1, 0]) - 1
        actual_corr = np.corrcoef(returns_1, returns_2)[0, 1]

        assert actual_corr > 0.9  # Should be very highly correlated