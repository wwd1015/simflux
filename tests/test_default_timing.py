"""Multi-period default-timing modes: "copula" (Li 2000) vs "frailty" (Duffie 2009).

Both reproduce the marginal cumulative PD exactly; they differ in the
cross-period dependence of the systematic factor (and therefore the tail):

* copula  — single frozen latent vs the cumulative-PD staircase. Grid-invariant
  loss distribution; all uncertainty resolves at t=0.
* frailty — persistent AR(1) factor + fresh idiosyncratic each period, with
  per-period barriers calibrated to preserve the marginal for ANY persistence.
  factor_persistence=0 => independent periods; =1 => frozen factor.

At n_periods=1 the two coincide.
"""

import warnings
from unittest.mock import patch

import pytest

import simflux as sf
from simflux.core.backend import Backend
from simflux.core.base import SimulationConfig


def _run(n_periods, timing, *, fp=0.5, rho_lgd=0.0, intra=0.4, force_numpy=False, n_sims=30_000):
    assets = [sf.AssetData(i, 0, 0.08, 0.5, 0.1, 1.0, "A") for i in range(200)]
    port = sf.TwoFactorPortfolio(
        assets=assets, intra_sector_correlations=intra,
        systematic_lgd_correlation=rho_lgd, sector_correlation_matrix=[[1.0]],
        config=SimulationConfig(seed=2024),
    )
    pl = 1.0 / n_periods
    kw = dict(n_simulations=n_sims, n_periods=n_periods, period_length=pl,
              default_timing=timing, factor_persistence=fp)
    if force_numpy:
        with patch.object(Backend, "is_available", return_value=False), \
             patch.object(Backend, "get_rust", return_value=None), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return port.simulate(**kw)
    return port.simulate(**kw)


@pytest.mark.parametrize("force_numpy", [True, False])
@pytest.mark.parametrize("timing,fp", [("copula", 0.5), ("frailty", 0.0), ("frailty", 0.6), ("frailty", 1.0)])
def test_all_modes_reproduce_cumulative_pd(timing, fp, force_numpy):
    """mean / (lgd * N) == horizon PD (0.08), every mode/persistence/backend."""
    if not force_numpy and not Backend.is_available():
        pytest.skip("Rust backend required")
    stats = _run(4, timing, fp=fp, rho_lgd=0.0, force_numpy=force_numpy)["portfolio_statistics"]
    implied_pd = stats["mean"] / (0.5 * 200)
    assert implied_pd == pytest.approx(0.08, rel=0.08)


@pytest.mark.parametrize("force_numpy", [True, False])
def test_copula_is_grid_invariant(force_numpy):
    """copula loss distribution does not depend on n_periods (single draw)."""
    if not force_numpy and not Backend.is_available():
        pytest.skip("Rust backend required")
    base = _run(1, "copula", force_numpy=force_numpy)["portfolio_statistics"]
    for k in (2, 4, 8):
        s = _run(k, "copula", force_numpy=force_numpy)["portfolio_statistics"]
        assert s["var_99"] == pytest.approx(base["var_99"], rel=1e-9)
        assert s["mean"] == pytest.approx(base["mean"], rel=1e-9)


def test_frailty_tail_monotone_in_persistence():
    """Tail risk rises with persistence; fp=0 is the (thin) independent limit."""
    v0 = _run(8, "frailty", fp=0.0)["portfolio_statistics"]["var_99"]
    v5 = _run(8, "frailty", fp=0.5)["portfolio_statistics"]["var_99"]
    v9 = _run(8, "frailty", fp=0.9)["portfolio_statistics"]["var_99"]
    assert v0 < v5 < v9


def test_modes_coincide_at_one_period():
    c = _run(1, "copula")["portfolio_statistics"]["var_99"]
    f = _run(1, "frailty", fp=0.7)["portfolio_statistics"]["var_99"]
    assert c == pytest.approx(f, rel=1e-9)


def test_result_stamps_mode_and_persistence():
    cop = _run(4, "copula")
    assert cop["default_timing"] == "copula" and cop["factor_persistence"] is None
    fr = _run(4, "frailty", fp=0.6)
    assert fr["default_timing"] == "frailty" and fr["factor_persistence"] == 0.6


def test_invalid_inputs_rejected():
    assets = [sf.AssetData(0, 0, 0.05, 0.5, 0.1, 1.0, "A")]
    port = sf.TwoFactorPortfolio(assets=assets, sector_correlation_matrix=[[1.0]])
    # "hazard" was removed as a named mode.
    with pytest.raises(ValueError, match="default_timing must be"):
        port.simulate(n_simulations=10, default_timing="hazard")
    with pytest.raises(ValueError, match="factor_persistence must be"):
        port.simulate(n_simulations=10, default_timing="frailty", factor_persistence=1.5)


@pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
@pytest.mark.parametrize("timing,fp", [("copula", 0.5), ("frailty", 0.6)])
def test_rust_numpy_parity(timing, fp):
    """Rust and NumPy agree distributionally (different RNG streams)."""
    r = _run(4, timing, fp=fp, force_numpy=False)["portfolio_statistics"]
    n = _run(4, timing, fp=fp, force_numpy=True)["portfolio_statistics"]
    assert abs(r["mean"] - n["mean"]) / max(n["mean"], 1.0) < 0.30
    assert abs(r["var_99"] - n["var_99"]) / max(n["var_99"], 1.0) < 0.30
