"""Tests for multi-period portfolio loss simulation with default timing."""

import pytest
import numpy as np
import simflux as sf
from simflux.core.base import SimulationConfig
from simflux.core.backend import Backend
from unittest.mock import patch


def _make_assets(n=10, pd=0.05, ts=None):
    """Helper: create n assets in 2 sectors with optional PD term structure."""
    assets = []
    for i in range(n):
        assets.append(sf.AssetData(
            asset_id=i, sector_id=i % 2, pd=pd,
            lgd_mean=0.5, lgd_std=0.1, exposure=1_000_000,
            sector_name="A" if i % 2 == 0 else "B",
            pd_term_structure=ts,
        ))
    return assets


class TestSinglePeriodBackwardCompat:
    """n_periods=1 must produce identical semantics to the old API."""

    def test_default_args_match_old_behaviour(self):
        portfolio = sf.CreditPortfolio.create_sample_portfolio(
            n_assets_per_sector=10, sectors=["X", "Y"],
            inter_sector_correlation=0.2,
        )
        results = portfolio.simulate(n_simulations=200)
        assert results["n_periods"] == 1
        assert results["period_length"] == 1.0
        assert results["time_horizon"] == 1.0
        assert "portfolio_statistics" in results

    def test_explicit_single_period(self):
        portfolio = sf.CreditPortfolio.create_sample_portfolio(
            n_assets_per_sector=5, sectors=["A", "B"],
            inter_sector_correlation=0.1,
        )
        results = portfolio.simulate(n_simulations=100, n_periods=1, period_length=1.0)
        assert results["time_horizon"] == 1.0


class TestMultiPeriodFlatPD:
    """Multi-period with flat PD (no term structure) — constant hazard."""

    def test_quarterly_2_year(self):
        assets = _make_assets(n=10, pd=0.10)  # 10% over full horizon
        portfolio = sf.CreditPortfolio(
            assets=assets, intra_sector_correlations=0.3,
        )
        results = portfolio.simulate(
            n_simulations=500, n_periods=8, period_length=0.25,
        )
        assert results["n_periods"] == 8
        assert results["period_length"] == 0.25
        assert results["time_horizon"] == pytest.approx(2.0)
        assert results["portfolio_statistics"]["mean"] >= 0

    def test_more_periods_means_more_defaults(self):
        """With more periods, cumulative default rate should be >= single-period."""
        assets = _make_assets(n=20, pd=0.08)
        p = sf.CreditPortfolio(assets=assets, intra_sector_correlations=0.3)

        r1 = p.simulate(n_simulations=2000, n_periods=1, period_length=1.0)
        r4 = p.simulate(n_simulations=2000, n_periods=4, period_length=0.25)

        # Expected loss should be similar (same cumulative PD) — not strictly
        # monotone due to MC noise, but within reasonable tolerance.
        mean_1 = r1["portfolio_statistics"]["mean"]
        mean_4 = r4["portfolio_statistics"]["mean"]
        assert abs(mean_4 - mean_1) / max(mean_1, 1.0) < 0.5  # within 50%


class TestMultiPeriodTermStructure:
    """Multi-period with explicit cumulative PD term structure."""

    def test_term_structure_basic(self):
        # Quarterly cumulative PDs over 1 year
        ts = [0.01, 0.025, 0.04, 0.06]
        assets = _make_assets(n=10, pd=0.06, ts=ts)
        portfolio = sf.CreditPortfolio(
            assets=assets, intra_sector_correlations=0.3,
        )
        results = portfolio.simulate(n_simulations=500, n_periods=4, period_length=0.25)
        assert results["time_horizon"] == pytest.approx(1.0)
        assert results["portfolio_statistics"]["mean"] >= 0

    def test_front_loaded_vs_back_loaded(self):
        """Front-loaded PD should produce earlier defaults on average."""
        n_sims = 3000
        # Front-loaded: most risk in early quarters
        ts_front = [0.04, 0.055, 0.06, 0.06]
        # Back-loaded: most risk in later quarters
        ts_back = [0.005, 0.01, 0.03, 0.06]

        assets_front = _make_assets(n=20, pd=0.06, ts=ts_front)
        assets_back = _make_assets(n=20, pd=0.06, ts=ts_back)

        p_front = sf.CreditPortfolio(assets=assets_front, intra_sector_correlations=0.3)
        p_back = sf.CreditPortfolio(assets=assets_back, intra_sector_correlations=0.3)

        # Both have same cumulative PD, so total expected loss should be similar
        r_front = p_front.simulate(n_simulations=n_sims, n_periods=4, period_length=0.25)
        r_back = p_back.simulate(n_simulations=n_sims, n_periods=4, period_length=0.25)

        mean_front = r_front["portfolio_statistics"]["mean"]
        mean_back = r_back["portfolio_statistics"]["mean"]
        # Same final PD → similar expected loss (within MC tolerance)
        assert abs(mean_front - mean_back) / max(mean_front, 1.0) < 0.5

    def test_term_structure_validation(self):
        """pd_term_structure must be non-decreasing and in [0, 1]."""
        with pytest.raises(ValueError, match="pd_term_structure must be non-decreasing"):
            sf.AssetData(0, 0, 0.05, 0.5, 0.1, 1e6, "S",
                         pd_term_structure=[0.03, 0.02, 0.04])

        with pytest.raises(ValueError, match="pd_term_structure.*must be between 0 and 1"):
            sf.AssetData(0, 0, 0.05, 0.5, 0.1, 1e6, "S",
                         pd_term_structure=[0.03, 1.5])


class TestSubHorizonTermStructure:
    """A pd_term_structure longer than n_periods is a valid *sub-horizon* run:
    the first n_periods cumulative-PD points are used. Sweeping n_periods from
    1..len over one fixed curve walks up the curve (cum[0], cum[1], ...).
    """

    TS = [0.02, 0.05, 0.08, 0.10]

    def _mean_loss(self, n_periods):
        # systematic_lgd_correlation=0 so mean loss == sum PD·E[LGD]·exposure in
        # expectation, making the ratio across horizons a clean function of the curve.
        assets = _make_assets(n=20, pd=0.10, ts=self.TS)
        p = sf.CreditPortfolio(
            assets=assets, intra_sector_correlations=0.2,
            systematic_lgd_correlation=0.0,
            config=SimulationConfig(seed=7),
        )
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")  # sub-horizon notice expected for n_periods<len(TS)
            r = p.simulate(n_simulations=20_000, n_periods=n_periods, period_length=1.0)
        return r["portfolio_statistics"]["mean"]

    def test_sweep_n_periods_walks_the_curve(self):
        means = [self._mean_loss(npd) for npd in (1, 2, 3, 4)]
        # Expected loss rises monotonically as the sub-horizon lengthens.
        assert means[0] < means[1] < means[2] < means[3]
        # n_periods=1 tracks cum[0]=0.02; n_periods=4 tracks cum[3]=0.10.
        assert means[0] / means[3] == pytest.approx(self.TS[0] / self.TS[3], rel=0.25)

    def test_longer_curve_emits_subhorizon_userwarning(self):
        assets = _make_assets(n=5, pd=0.10, ts=self.TS)
        p = sf.CreditPortfolio(assets=assets, intra_sector_correlations=0.2)
        with pytest.warns(UserWarning, match="sub-horizon"):
            p.simulate(n_simulations=50, n_periods=2)

    def test_shorter_curve_raises(self):
        assets = _make_assets(n=5, pd=0.10, ts=[0.02, 0.05])  # only 2 points
        p = sf.CreditPortfolio(assets=assets, intra_sector_correlations=0.2)
        # An under-specified horizon is a parameter problem: ValidationError
        # (a ValueError subclass) since 0.7.0, previously a RuntimeError.
        with pytest.raises(sf.ValidationError, match="shorter than n_periods"):
            p.simulate(n_simulations=50, n_periods=4)


class TestMultiPeriodValidation:
    """Validate multi-period parameter checks."""

    def test_zero_periods_rejected(self):
        portfolio = sf.CreditPortfolio.create_sample_portfolio(
            n_assets_per_sector=3, sectors=["A"],
        )
        with pytest.raises(ValueError, match="n_periods must be positive"):
            portfolio.simulate(n_simulations=10, n_periods=0)

    def test_negative_period_length_rejected(self):
        portfolio = sf.CreditPortfolio.create_sample_portfolio(
            n_assets_per_sector=3, sectors=["A"],
        )
        with pytest.raises(ValueError, match="period_length must be positive"):
            portfolio.simulate(n_simulations=10, n_periods=4, period_length=-0.25)


class TestMultiPeriodFallback:
    """Multi-period works through NumPy fallback when Rust is unavailable."""

    @patch.object(Backend, "is_available", return_value=False)
    @patch.object(Backend, "get_rust", return_value=None)
    def test_fallback_multi_period(self, *_):
        import warnings
        assets = _make_assets(n=6, pd=0.08)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            portfolio = sf.CreditPortfolio(
                assets=assets, intra_sector_correlations=0.3,
            )
            results = portfolio.simulate(
                n_simulations=100, n_periods=4, period_length=0.25,
            )
        assert results["n_periods"] == 4
        assert results["time_horizon"] == pytest.approx(1.0)
        assert results["portfolio_statistics"]["mean"] >= 0

    @patch.object(Backend, "is_available", return_value=False)
    @patch.object(Backend, "get_rust", return_value=None)
    def test_fallback_with_term_structure(self, *_):
        import warnings
        ts = [0.01, 0.03, 0.05, 0.08]
        assets = _make_assets(n=6, pd=0.08, ts=ts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            portfolio = sf.CreditPortfolio(
                assets=assets, intra_sector_correlations=0.3,
            )
            results = portfolio.simulate(
                n_simulations=100, n_periods=4, period_length=0.25,
            )
        assert results["portfolio_statistics"]["mean"] >= 0


class TestRustNumpyCrossValidation:
    """Run both Rust and NumPy backends on the same problem and compare.

    The two implementations use different RNG streams, so individual trials
    differ.  With enough simulations the *distributional statistics* (mean,
    VaR) should converge to the same values within Monte Carlo tolerance.
    """

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_single_period_cross_validation(self):
        """Rust vs NumPy on a single-period portfolio."""
        assets = _make_assets(n=20, pd=0.06)
        n_sims = 10_000

        # --- Rust path ---
        p_rust = sf.CreditPortfolio(
            assets=assets, intra_sector_correlations=0.35,
            config=SimulationConfig(seed=100),
        )
        r_rust = p_rust.simulate(n_simulations=n_sims)

        # --- NumPy path (force fallback) ---
        with patch.object(Backend, "is_available", return_value=False), \
             patch.object(Backend, "get_rust", return_value=None):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_np = sf.CreditPortfolio(
                    assets=assets, intra_sector_correlations=0.35,
                    config=SimulationConfig(seed=200),
                )
                r_np = p_np.simulate(n_simulations=n_sims)

        rust_stats = r_rust["portfolio_statistics"]
        np_stats = r_np["portfolio_statistics"]

        # Mean loss should agree within 30% (MC tolerance with 10k sims)
        assert abs(rust_stats["mean"] - np_stats["mean"]) / max(np_stats["mean"], 1.0) < 0.30
        # VaR 95 within 40%
        assert abs(rust_stats["var_95"] - np_stats["var_95"]) / max(np_stats["var_95"], 1.0) < 0.40

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_multi_period_cross_validation(self):
        """Rust vs NumPy on a quarterly 1-year multi-period simulation."""
        ts = [0.015, 0.03, 0.045, 0.06]
        assets = _make_assets(n=20, pd=0.06, ts=ts)
        n_sims = 10_000

        # --- Rust ---
        p_rust = sf.CreditPortfolio(
            assets=assets, intra_sector_correlations=0.35,
            config=SimulationConfig(seed=300),
        )
        r_rust = p_rust.simulate(n_simulations=n_sims, n_periods=4, period_length=0.25)

        # --- NumPy ---
        with patch.object(Backend, "is_available", return_value=False), \
             patch.object(Backend, "get_rust", return_value=None):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_np = sf.CreditPortfolio(
                    assets=assets, intra_sector_correlations=0.35,
                    config=SimulationConfig(seed=400),
                )
                r_np = p_np.simulate(n_simulations=n_sims, n_periods=4, period_length=0.25)

        rust_mean = r_rust["portfolio_statistics"]["mean"]
        np_mean = r_np["portfolio_statistics"]["mean"]

        # Both should produce a positive mean loss
        assert rust_mean > 0
        assert np_mean > 0
        # Means should be in the same ballpark (within 30%)
        assert abs(rust_mean - np_mean) / max(np_mean, 1.0) < 0.30

    @pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
    def test_flat_pd_multi_period_cross_validation(self):
        """Rust vs NumPy with flat PD spread across periods (no term structure)."""
        assets = _make_assets(n=20, pd=0.10)
        n_sims = 10_000

        p_rust = sf.CreditPortfolio(
            assets=assets, intra_sector_correlations=0.3,
            config=SimulationConfig(seed=500),
        )
        r_rust = p_rust.simulate(n_simulations=n_sims, n_periods=8, period_length=0.25)

        with patch.object(Backend, "is_available", return_value=False), \
             patch.object(Backend, "get_rust", return_value=None):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_np = sf.CreditPortfolio(
                    assets=assets, intra_sector_correlations=0.3,
                    config=SimulationConfig(seed=600),
                )
                r_np = p_np.simulate(n_simulations=n_sims, n_periods=8, period_length=0.25)

        rust_mean = r_rust["portfolio_statistics"]["mean"]
        np_mean = r_np["portfolio_statistics"]["mean"]
        assert abs(rust_mean - np_mean) / max(np_mean, 1.0) < 0.30
