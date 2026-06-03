"""Analytic golden master: SimFlux vs. the Vasicek / Basel-ASRF closed form.

This is the test the cross-validation suite cannot be: it pins the model against
*external truth*, not against the other backend. A single homogeneous sector,
many obligors, near-constant LGD, and ρ_lgd = 0 reduces the two-factor model to
the single-factor asymptotic-single-risk-factor (ASRF) limit, whose loss-rate
VaR has a closed form:

    VaR_q(loss rate) = lgd · Φ( (Φ⁻¹(PD) + √ρ · Φ⁻¹(q)) / √(1 − ρ) )

A misplaced √ρ loading, a wrong default threshold, or a broken factor structure
would all break this equality — none of which a Rust-vs-NumPy parity check sees.
"""

import math
from statistics import NormalDist

import pytest

import simflux as sf
from simflux.core.backend import Backend
from simflux.core.base import SimulationConfig

# Keep n_assets * n_simulations bounded: the Rust backend materialises a
# per-asset result for every trial, so this product drives memory.
N_ASSETS = 300
N_SIMS = 30_000
PD = 0.05
RHO = 0.20          # intra-sector (asset) correlation
LGD = 0.45
EXPOSURE = 1_000_000.0


@pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required (large N granularity)")
def test_vasicek_asrf_golden_master():
    nd = NormalDist()  # stdlib normal — independent of scipy and of the library

    assets = [sf.AssetData(i, 0, PD, LGD, 0.02, EXPOSURE, "A") for i in range(N_ASSETS)]
    port = sf.TwoFactorPortfolio(
        assets=assets,
        intra_sector_correlations=RHO,
        systematic_lgd_correlation=0.0,          # decouple LGD from the cycle
        sector_correlation_matrix=[[1.0]],       # single sector => F is THE factor
        config=SimulationConfig(seed=20260601),
    )
    stats = port.simulate(n_simulations=N_SIMS)["portfolio_statistics"]
    total_exposure = N_ASSETS * EXPOSURE

    def asrf_var(q):
        return LGD * nd.cdf((nd.inv_cdf(PD) + math.sqrt(RHO) * nd.inv_cdf(q)) / math.sqrt(1 - RHO))

    # Expected loss rate is exactly PD * LGD, independent of granularity.
    assert stats["mean"] / total_exposure == pytest.approx(PD * LGD, rel=0.05)

    # Tail VaR must match the ASRF closed form. Finite N biases the simulated
    # VaR slightly high (granularity fattens the tail), so the tolerance is
    # one-sidedly generous; it is still far tighter than parity's 30–40%.
    for q, key, tol in [(0.95, "var_95", 0.08), (0.99, "var_99", 0.10)]:
        sim_rate = stats[key] / total_exposure
        analytic = asrf_var(q)
        assert sim_rate == pytest.approx(analytic, rel=tol), (
            f"{key}: simulated loss-rate {sim_rate:.4f} vs ASRF {analytic:.4f} "
            f"(rel err {abs(sim_rate - analytic) / analytic:.1%})"
        )
