"""Behavioral invariants — model correctness against theory, not against the
other backend.

Cross-validation compares Rust vs. NumPy and so is blind to errors both share.
These tests instead assert economic properties that must hold on *whichever*
backend runs: marginal loss = PD·LGD, tail risk rising in correlation, and the
coherence ordering of the risk measures. The wrong-way-LGD guard
(`test_lgd_wrong_way.py`) and the Vasicek golden master
(`test_analytic_vasicek.py`) are the other two members of this family.

`lgd_mean = 0.5` is used for the marginal-loss identities on purpose: the
no-scipy NumPy fallback approximates LGD as ~Uniform[0,1] (mean 0.5), so the
identity `E[loss] = P(default)·0.5·exposure` holds across the Rust, scipy-Beta,
and no-scipy paths alike.
"""

import warnings
from unittest.mock import patch

import pytest

import simflux as sf
from simflux.core.backend import Backend
from simflux.core.base import SimulationConfig

EXPOSURE = 1.0


def _simulate(port, n_sims, force_numpy, **sim_kwargs):
    if force_numpy:
        with patch.object(Backend, "is_available", return_value=False), \
             patch.object(Backend, "get_rust", return_value=None), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return port.simulate(n_simulations=n_sims, **sim_kwargs)
    return port.simulate(n_simulations=n_sims, **sim_kwargs)


def _homogeneous(pd, n=200, intra=0.2, rho_lgd=0.0, lgd_mean=0.5, seed=1, ts=None):
    assets = [
        sf.AssetData(i, 0, pd, lgd_mean, 0.1, EXPOSURE, "A", pd_term_structure=ts)
        for i in range(n)
    ]
    return sf.TwoFactorPortfolio(
        assets=assets, intra_sector_correlations=intra,
        systematic_lgd_correlation=rho_lgd, sector_correlation_matrix=[[1.0]],
        config=SimulationConfig(seed=seed),
    )


@pytest.mark.parametrize("force_numpy", [True, False])
def test_marginal_loss_equals_pd_times_lgd(force_numpy):
    """Single period: E[loss] / (lgd·exposure·n) == PD (pins the threshold)."""
    if not force_numpy and not Backend.is_available():
        pytest.skip("Rust backend required")
    pd, n = 0.05, 200
    port = _homogeneous(pd, n=n, rho_lgd=0.0, lgd_mean=0.5, seed=3)
    mean = _simulate(port, 30_000, force_numpy)["portfolio_statistics"]["mean"]
    implied_pd = mean / (0.5 * n * EXPOSURE)
    assert implied_pd == pytest.approx(pd, rel=0.08)


@pytest.mark.parametrize("force_numpy", [True, False])
def test_marginal_loss_multi_period_uses_cumulative_pd(force_numpy):
    """Over a multi-period horizon, marginal loss reflects the *cumulative* PD."""
    if not force_numpy and not Backend.is_available():
        pytest.skip("Rust backend required")
    ts = [0.02, 0.05, 0.08, 0.10]   # cumulative PD; P(ever default) = 0.10
    n = 200
    port = _homogeneous(0.10, n=n, rho_lgd=0.0, lgd_mean=0.5, seed=4, ts=ts)
    # Must run with one period per term-structure point, else only ts[0] is used.
    res = _simulate(port, 30_000, force_numpy, n_periods=4, period_length=0.25)
    mean = res["portfolio_statistics"]["mean"]
    implied_cum_pd = mean / (0.5 * n * EXPOSURE)
    assert implied_cum_pd == pytest.approx(ts[-1], rel=0.08)


def test_var_increases_with_intra_correlation():
    """Higher intra-sector correlation clusters defaults => fatter tail."""
    low = _homogeneous(0.05, intra=0.1, seed=5).simulate(n_simulations=30_000)
    high = _homogeneous(0.05, intra=0.5, seed=5).simulate(n_simulations=30_000)
    assert high["portfolio_statistics"]["var_99"] > 1.5 * low["portfolio_statistics"]["var_99"]


def test_var_increases_with_cross_sector_correlation():
    """Stronger cross-sector coupling makes sectors crash together => fatter
    portfolio tail, even though per-sector marginals are unchanged."""
    assets = [
        sf.AssetData(i, i % 2, 0.05, 0.5, 0.1, EXPOSURE, "A" if i % 2 == 0 else "B")
        for i in range(200)
    ]

    def run(off_diag):
        port = sf.TwoFactorPortfolio(
            assets=assets, intra_sector_correlations=0.3,
            sector_correlation_matrix=[[1.0, off_diag], [off_diag, 1.0]],
            config=SimulationConfig(seed=6),
        )
        return port.simulate(n_simulations=30_000)["portfolio_statistics"]["var_99"]

    assert run(0.8) > run(0.0)


def test_risk_measures_are_coherent():
    """var_999 >= var_99 >= var_95 >= 0 and ES_q >= VaR_q, on one run."""
    stats = _homogeneous(0.05, intra=0.3, seed=7).simulate(n_simulations=20_000)["portfolio_statistics"]
    assert 0 <= stats["var_95"] <= stats["var_99"] <= stats["var_999"]
    assert stats["expected_shortfall_95"] >= stats["var_95"]
    assert stats["expected_shortfall_99"] >= stats["var_99"]
    assert stats["max_loss"] >= stats["var_999"]
