"""Time-varying LGD: a per-period LGD mean term structure on AssetData.

An obligor that defaults in period k draws LGD from a Beta with mean
``lgd_term_structure[k]`` (lgd_std constant). A constant term structure reduces
exactly to the flat ``lgd_mean``; a rising one raises loss; both backends agree.
"""

from unittest.mock import patch

import pytest

import simflux as sf
from simflux.core.backend import Backend
from simflux.core.base import SimulationConfig


def _portfolio(lgd_mean=0.5, lgd_std=0.1, lgd_term=None, n=120, seed=11):
    assets = [
        sf.AssetData(
            asset_id=i,
            sector_id=0,
            pd=0.08,
            lgd_mean=lgd_mean,
            lgd_std=lgd_std,
            exposure=1_000_000,
            sector_name="S",
            lgd_term_structure=lgd_term,
        )
        for i in range(n)
    ]
    return sf.CreditPortfolio(
        assets, intra_sector_correlations=0.3, config=SimulationConfig(seed=seed)
    )


def test_lgd_term_structure_validated():
    """Out-of-range or Beta-infeasible per-period means are rejected."""
    with pytest.raises(ValueError, match="lgd_term_structure"):
        sf.AssetData(0, 0, 0.05, 0.5, 0.1, 1e6, "S", lgd_term_structure=[0.5, 1.5])
    with pytest.raises(ValueError, match="infeasible"):
        # std 0.3 is feasible for the flat mean 0.5 but not for a period mean 0.05
        sf.AssetData(0, 0, 0.05, 0.5, 0.3, 1e6, "S", lgd_term_structure=[0.5, 0.05])


def test_constant_term_structure_equals_flat_lgd():
    """A constant term structure reproduces the flat-LGD result exactly."""
    flat = _portfolio(lgd_mean=0.5).simulate(n_simulations=20_000, n_periods=6)
    tv = _portfolio(lgd_mean=0.5, lgd_term=[0.5] * 6).simulate(
        n_simulations=20_000, n_periods=6
    )
    assert flat["portfolio_statistics"]["var_99"] == pytest.approx(
        tv["portfolio_statistics"]["var_99"], rel=1e-9
    )


def test_rising_lgd_raises_loss_above_flat_low():
    """A term structure rising above the flat mean produces larger mean loss."""
    flat_low = _portfolio(lgd_mean=0.3).simulate(n_simulations=30_000, n_periods=8)
    rising = _portfolio(
        lgd_mean=0.3, lgd_term=[0.3, 0.37, 0.44, 0.51, 0.58, 0.65, 0.72, 0.8]
    ).simulate(n_simulations=30_000, n_periods=8)
    assert (
        rising["portfolio_statistics"]["mean"]
        > flat_low["portfolio_statistics"]["mean"]
    )


@pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
def test_backends_agree_time_varying_lgd():
    """Rust and the NumPy fallback agree in distribution with a rising LGD curve."""
    curve = [0.3, 0.37, 0.44, 0.51, 0.58, 0.65, 0.72, 0.8]
    pf = _portfolio(lgd_mean=0.3, lgd_term=curve, n=160, seed=99)

    rust = pf.simulate(n_simulations=60_000, n_periods=8)
    with patch.object(Backend, "is_available", return_value=False):
        npy = pf.simulate(n_simulations=60_000, n_periods=8)

    r = rust["portfolio_statistics"]
    n = npy["portfolio_statistics"]
    assert abs(r["mean"] - n["mean"]) / max(n["mean"], 1.0) < 0.05
    assert abs(r["var_99"] - n["var_99"]) / max(n["var_99"], 1.0) < 0.15
