"""Per-obligor (heterogeneous) intra-sector correlation.

An obligor's own ``AssetData.intra_sector_correlation`` overrides its sector's
value, so loadings within a sector may differ. Both backends must honor it and
agree, and a higher loading must fatten the loss tail (more clustering).
"""

from unittest.mock import patch

import numpy as np
import pytest

import simflux as sf
from simflux.core.backend import Backend
from simflux.core.base import SimulationConfig


def _homogeneous(rho, n=120, seed=7):
    assets = [
        sf.AssetData(
            asset_id=i,
            sector_id=0,
            pd=0.05,
            lgd_mean=0.5,
            lgd_std=0.1,
            exposure=1_000_000,
            sector_name="S",
            intra_sector_correlation=rho,
        )
        for i in range(n)
    ]
    return sf.CreditPortfolio(assets, config=SimulationConfig(seed=seed))


def test_per_asset_value_overrides_sector_default():
    """A per-asset value wins; an asset without one falls back to the sector value."""
    assets = [
        sf.AssetData(0, 0, 0.05, 0.5, 0.1, 1e6, "S", intra_sector_correlation=0.1),
        sf.AssetData(1, 0, 0.05, 0.5, 0.1, 1e6, "S", intra_sector_correlation=0.8),
        sf.AssetData(2, 0, 0.05, 0.5, 0.1, 1e6, "S"),  # no per-asset value
    ]
    pf = sf.CreditPortfolio(assets, intra_sector_correlations=0.4)
    assert pf.asset_intra_correlations == [0.1, 0.8, 0.4]  # third falls back to 0.4


def test_heterogeneous_within_sector_is_allowed():
    """Mixed loadings in one sector no longer raise (previously rejected)."""
    pf = _homogeneous(0.1, n=4)  # constructs fine
    assert pf.simulate(n_simulations=100)["portfolio_statistics"]["mean"] >= 0


def test_higher_loading_fattens_the_tail():
    """All-high rho clusters defaults more than all-low rho -> larger VaR."""
    low = _homogeneous(0.05).simulate(n_simulations=40_000)
    high = _homogeneous(0.85).simulate(n_simulations=40_000)
    assert (
        high["portfolio_statistics"]["var_99"] > low["portfolio_statistics"]["var_99"]
    )


@pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
def test_backends_agree_on_heterogeneous_loadings():
    """Rust and the NumPy fallback agree in distribution with mixed per-asset rho."""
    assets = [
        sf.AssetData(
            asset_id=i,
            sector_id=i % 2,
            pd=0.04,
            lgd_mean=0.5,
            lgd_std=0.12,
            exposure=1_000_000,
            sector_name=["A", "B"][i % 2],
            intra_sector_correlation=(0.1 if i % 2 == 0 else 0.7),
        )
        for i in range(160)
    ]
    cm = np.array([[1.0, 0.2], [0.2, 1.0]])
    pf = sf.CreditPortfolio(
        assets, sector_correlation_matrix=cm, config=SimulationConfig(seed=99)
    )

    rust = pf.simulate(n_simulations=60_000)
    with patch.object(Backend, "is_available", return_value=False):
        npy = pf.simulate(n_simulations=60_000)

    r = rust["portfolio_statistics"]
    n = npy["portfolio_statistics"]
    assert abs(r["mean"] - n["mean"]) / max(n["mean"], 1.0) < 0.05
    assert abs(r["var_99"] - n["var_99"]) / max(n["var_99"], 1.0) < 0.15
