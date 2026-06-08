"""Tests for correlation validation edge cases, AssetData boundaries, and misc gaps."""

import pytest
import numpy as np

from simflux.utils.random_utils import (
    validate_correlation_matrix_strict,
    validate_correlation_matrix,
    adjust_correlation_strength,
)
from simflux.core.backend import CORRELATION_TOLERANCE
import simflux as sf


# ---------------------------------------------------------------------------
# validate_correlation_matrix_strict — error messages and modes
# ---------------------------------------------------------------------------

class TestStrictValidationMessages:
    """Verify exact error messages and parameter modes."""

    def test_non_square(self):
        m = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.1]])
        with pytest.raises(ValueError, match="must be square"):
            validate_correlation_matrix_strict(m)

    def test_not_symmetric(self):
        m = np.array([[1.0, 0.3], [0.5, 1.0]])
        with pytest.raises(ValueError, match="must be symmetric"):
            validate_correlation_matrix_strict(m)

    def test_diagonal_not_one(self):
        m = np.array([[0.9, 0.3], [0.3, 1.0]])
        with pytest.raises(ValueError, match="diagonal must be 1.0"):
            validate_correlation_matrix_strict(m)

    def test_values_out_of_range(self):
        m = np.array([[1.0, 1.5], [1.5, 1.0]])
        with pytest.raises(ValueError, match="values must be between -1 and 1"):
            validate_correlation_matrix_strict(m)

    def test_not_positive_definite(self):
        # Singular matrix (eigenvalue = 0)
        m = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="must be positive definite"):
            validate_correlation_matrix_strict(m, check_positive_definite=True)

    def test_positive_semi_definite_mode(self):
        """Semi-definite check should accept a singular but PSD matrix."""
        # Rank-deficient but PSD 3x3
        m = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        # This is PSD (eigenvalues: 3, 0, 0) but NOT PD
        with pytest.raises(ValueError, match="must be positive definite"):
            validate_correlation_matrix_strict(m, check_positive_definite=True)
        # Semi-definite should pass (eigenvalues >= 0)
        validate_correlation_matrix_strict(m, check_positive_definite=False)

    def test_custom_name(self):
        m = np.array([[1.0, 0.3], [0.5, 1.0]])
        with pytest.raises(ValueError, match="my_matrix must be symmetric"):
            validate_correlation_matrix_strict(m, name="my_matrix")

    def test_custom_pd_tolerance(self):
        """Near-singular matrix passes with loose tolerance."""
        m = np.array([[1.0, 0.9999], [0.9999, 1.0]])
        # With tight tolerance this fails (smallest eigenvalue ~ 1e-4)
        validate_correlation_matrix_strict(m, pd_tolerance=1e-6)


# ---------------------------------------------------------------------------
# adjust_correlation_strength (previously only tested indirectly)
# ---------------------------------------------------------------------------

class TestAdjustCorrelationStrength:
    """Direct tests for the strength-scaling utility."""

    def test_identity_returns_identity(self):
        m = np.eye(3)
        result = adjust_correlation_strength(m, target_strength=0.5)
        # Identity has zero off-diagonal → should remain identity
        np.testing.assert_array_equal(result, np.eye(3))

    def test_scales_off_diagonal(self):
        m = np.array([[1.0, 0.4], [0.4, 1.0]])
        result = adjust_correlation_strength(m, target_strength=0.2)
        np.testing.assert_allclose(np.diag(result), 1.0)
        assert abs(result[0, 1]) == pytest.approx(0.2, abs=0.01)


# ---------------------------------------------------------------------------
# AssetData boundary conditions
# ---------------------------------------------------------------------------

class TestAssetDataBoundaries:
    """Edge-case validation for AssetData."""

    def test_pd_at_boundaries(self):
        # PD = 0 and PD = 1 should be valid
        a0 = sf.AssetData(0, 0, 0.0, 0.5, 0.1, 1000, "S")
        assert a0.pd == 0.0
        a1 = sf.AssetData(1, 0, 1.0, 0.5, 0.1, 1000, "S")
        assert a1.pd == 1.0

    def test_intra_corr_at_boundaries(self):
        a0 = sf.AssetData(0, 0, 0.05, 0.5, 0.1, 1000, "S", intra_sector_correlation=0.0)
        assert a0.intra_sector_correlation == 0.0
        a1 = sf.AssetData(1, 0, 0.05, 0.5, 0.1, 1000, "S", intra_sector_correlation=1.0)
        assert a1.intra_sector_correlation == 1.0

    def test_zero_exposure_allowed(self):
        a = sf.AssetData(0, 0, 0.05, 0.5, 0.1, 0.0, "S")
        assert a.exposure == 0.0

    def test_lgd_std_at_limit_rejected(self):
        """lgd_std must be strictly less than sqrt(lgd_mean*(1-lgd_mean))."""
        import math
        lgd_mean = 0.5
        max_std = math.sqrt(lgd_mean * (1 - lgd_mean))
        with pytest.raises(ValueError, match="lgd_std"):
            sf.AssetData(0, 0, 0.05, lgd_mean, max_std, 1000, "S")


# ---------------------------------------------------------------------------
# CreditPortfolio edge cases
# ---------------------------------------------------------------------------

class TestPortfolioEdgeCases:
    """Portfolio with single asset and single sector."""

    def test_single_asset_portfolio(self):
        assets = [sf.AssetData(0, 0, 0.05, 0.5, 0.1, 1_000_000, "A")]
        p = sf.CreditPortfolio(assets=assets, intra_sector_correlations=0.3)
        results = p.simulate(n_simulations=100)
        assert results["n_assets"] == 1
        assert results["n_sectors"] == 1
        assert results["portfolio_statistics"]["mean"] >= 0

    def test_single_sector_portfolio(self):
        assets = [
            sf.AssetData(i, 0, 0.05, 0.5, 0.1, 500_000, "Only")
            for i in range(5)
        ]
        p = sf.CreditPortfolio(assets=assets, intra_sector_correlations=0.4)
        results = p.simulate(n_simulations=50)
        assert len(results["sector_statistics"]) == 1
        assert "Only" in results["sector_statistics"]


# ---------------------------------------------------------------------------
# TwoFactorCorrelationStructure generate_factors
# ---------------------------------------------------------------------------

class TestCorrelationStructureFactors:
    """Test generate_factors and negative sector size rejection."""

    def test_generate_factors_shape(self):
        s = sf.TwoFactorCorrelationStructure(
            inter_sector_correlation=0.2,
            intra_sector_correlations=[0.3, 0.4],
            sector_sizes=[5, 3],
        )
        factors = s.generate_factors(n_simulations=100, seed=42)
        assert len(factors) == 100
        assert factors[0].sector_factors.shape == (2,)

    def test_negative_sector_size_rejected(self):
        with pytest.raises(ValueError, match="All sector sizes must be positive"):
            sf.TwoFactorCorrelationStructure(0.2, [0.3], [-1])

    def test_invalid_sector_correlation_matrix(self):
        """Non-PSD sector correlation matrix should fail."""
        bad_matrix = [[1.0, 1.5], [1.5, 1.0]]
        with pytest.raises(ValueError):
            sf.TwoFactorCorrelationStructure(
                0.2, [0.3, 0.4], [5, 3],
                sector_correlation_matrix=bad_matrix,
            )
