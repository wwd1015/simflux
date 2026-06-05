"""Tests for the scipy-free special functions (utils/special).

The Beta inverse-CDF is validated three ways: against closed forms that need no
scipy (Beta(1,1), Beta(2,1), Beta(1,2)), by a round-trip through the regularized
incomplete beta, and — when scipy is installed — against scipy.stats.beta.ppf.
A final end-to-end test pins the regression this fixes: the NumPy fallback must
realize LGD at its mean, not a silent ~0.5 uniform.
"""

import numpy as np
import pytest
from unittest.mock import patch

import simflux as sf
from simflux.core.base import SimulationConfig
from simflux.utils.special import beta_ppf, reg_incomplete_beta

try:
    from scipy.stats import beta as _scipy_beta  # noqa: F401
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


U = np.linspace(1e-4, 1 - 1e-4, 999)


class TestBetaPpfClosedForms:
    """Closed-form quantiles that require no scipy."""

    def test_uniform_beta_1_1(self):
        # Beta(1,1) is Uniform(0,1): ppf(u) = u.
        assert np.allclose(beta_ppf(U, 1.0, 1.0), U, atol=2e-3)

    def test_beta_2_1_is_sqrt(self):
        # Beta(2,1): cdf = x^2, so ppf(u) = sqrt(u).
        assert np.allclose(beta_ppf(U, 2.0, 1.0), np.sqrt(U), atol=2e-3)

    def test_beta_1_2_is_one_minus_sqrt(self):
        # Beta(1,2): cdf = 1-(1-x)^2, so ppf(u) = 1 - sqrt(1-u).
        assert np.allclose(beta_ppf(U, 1.0, 2.0), 1.0 - np.sqrt(1.0 - U), atol=2e-3)


class TestBetaPpfRoundTrip:
    """I_x(a,b) evaluated at the quantile must recover the probability."""

    @pytest.mark.parametrize("a,b", [(12.0, 12.0), (3.0, 12.0), (2.0, 5.0), (0.8, 2.0), (5.0, 1.5)])
    def test_cdf_of_ppf_is_identity(self, a, b):
        x = beta_ppf(U, a, b)
        assert np.all(np.diff(x) >= -1e-9)  # monotone in u
        recovered = reg_incomplete_beta(a, b, x)
        assert np.allclose(recovered, U, atol=3e-3)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
class TestBetaPpfVsScipy:
    @pytest.mark.parametrize("a,b", [(12.0, 12.0), (3.0, 12.0), (2.0, 5.0), (0.8, 2.0)])
    def test_matches_scipy(self, a, b):
        ref = _scipy_beta.ppf(U, a, b)
        assert np.max(np.abs(beta_ppf(U, a, b) - ref)) < 2e-3


class TestBetaPpfPerColumn:
    """Per-column params (one (alpha,beta) per obligor) invert independently."""

    def test_distinct_columns(self):
        u = np.column_stack([U, U])                     # (999, 2)
        a = np.array([2.0, 1.0])                          # col0: sqrt, col1: uniform
        b = np.array([1.0, 1.0])
        out = beta_ppf(u, a[np.newaxis, :], b[np.newaxis, :])
        assert np.allclose(out[:, 0], np.sqrt(U), atol=2e-3)
        assert np.allclose(out[:, 1], U, atol=2e-3)


class TestNumpyFallbackLgdRegression:
    """The no-Rust fallback must realize LGD at its mean, not a silent ~0.5."""

    @patch.object(sf.core.backend.Backend, 'is_available', return_value=False)
    @patch.object(sf.core.backend.Backend, 'get_rust', return_value=None)
    def test_fallback_lgd_respects_mean(self, *_):
        lgd_mean = 0.2
        n, pd, exposure = 50, 0.10, 1_000_000.0
        assets = [sf.AssetData(i, 0, pd, lgd_mean, 0.08, exposure, "A") for i in range(n)]
        port = sf.TwoFactorPortfolio(
            assets=assets, intra_sector_correlations=0.1,
            systematic_lgd_correlation=0.0,  # E[LGD|default] == lgd_mean
            config=SimulationConfig(seed=11),
        )
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")  # scipy-absent fallback notice
            mean = port.simulate(n_simulations=40_000)["portfolio_statistics"]["mean"]

        expected = n * pd * lgd_mean * exposure        # ~1.0e6
        buggy = n * pd * 0.5 * exposure                # the old uniform-mean value ~2.5e6
        assert abs(mean - expected) / expected < 0.1   # close to the true mean
        assert abs(mean - buggy) / buggy > 0.3         # clearly NOT the buggy ~0.5 LGD
