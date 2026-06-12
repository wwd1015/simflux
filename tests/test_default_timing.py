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
    port = sf.CreditPortfolio(
        assets=assets, intra_sector_correlations=intra,
        systematic_lgd_correlation=rho_lgd, sector_correlation_matrix=[[1.0]],
        config=SimulationConfig(seed=2024),
    )
    pl = 1.0 / n_periods
    timing_obj = sf.Frailty(persistence=fp) if timing == "frailty" else sf.Copula()
    kw = dict(n_simulations=n_sims, n_periods=n_periods, period_length=pl,
              default_timing=timing_obj)
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
    port = sf.CreditPortfolio(assets=assets, sector_correlation_matrix=[[1.0]])
    # "hazard" was removed as a named mode.
    with pytest.raises(ValueError, match="default_timing must be"):
        port.simulate(n_simulations=10, default_timing="hazard")
    # The deprecated kwarg channel still validates persistence (via Frailty).
    with pytest.raises(ValueError, match="factor_persistence must be"), \
         warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        port.simulate(n_simulations=10, default_timing="frailty", factor_persistence=1.5)
    # Timing objects validate at construction — before any simulation work.
    with pytest.raises(ValueError, match="factor_persistence must be"):
        sf.Frailty(persistence=1.5)


# ---------------------------------------------------------------------------
# Timing objects, string sugar, and the deprecation shim
# ---------------------------------------------------------------------------


def test_string_sugar_matches_objects():
    """"copula"/"frailty" strings and default-configured objects agree exactly
    (same seed, same backend => same numbers)."""
    s = _run_with(default_timing="copula")
    o = _run_with(default_timing=sf.Copula())
    assert s["portfolio_statistics"]["var_99"] == o["portfolio_statistics"]["var_99"]
    s = _run_with(default_timing="frailty")
    o = _run_with(default_timing=sf.Frailty())  # persistence=0.5 default
    assert s["portfolio_statistics"]["var_99"] == o["portfolio_statistics"]["var_99"]


def _run_with(**kw):
    assets = [sf.AssetData(i, 0, 0.08, 0.5, 0.1, 1.0, "A") for i in range(50)]
    port = sf.CreditPortfolio(
        assets=assets, intra_sector_correlations=0.4,
        sector_correlation_matrix=[[1.0]], config=SimulationConfig(seed=7),
    )
    kw.setdefault("n_simulations", 2_000)
    kw.setdefault("n_periods", 4)
    kw.setdefault("period_length", 0.25)
    return port.simulate(**kw)


def test_factor_persistence_kwarg_deprecated_but_honored():
    with pytest.warns(DeprecationWarning, match="factor_persistence is deprecated"):
        res = _run_with(default_timing="frailty", factor_persistence=0.6)
    assert res["factor_persistence"] == 0.6


def test_factor_persistence_kwarg_rejected_with_timing_object():
    """Persistence can never be specified twice."""
    with pytest.raises(ValueError, match="cannot be combined with a timing object"):
        _run_with(default_timing=sf.Frailty(persistence=0.6), factor_persistence=0.3)


def test_timing_plan_invariants():
    """Plan misuse fails loudly at plan time, not inside a backend."""
    import numpy as np
    from simflux.portfolio.default_timing import TimingPlan

    ok = np.array([[-2.0, -2.0], [-1.5, -1.4]])
    # Closed kernel set.
    with pytest.raises(ValueError, match="kernel must be one of"):
        TimingPlan(kernel="hawkes", thresholds=ok, factor_phi=0.0)
    # Copula staircase must be non-decreasing down each column.
    bad = np.array([[-1.0, -2.0], [-1.5, -1.9]])
    with pytest.raises(ValueError, match="non-decreasing"):
        TimingPlan(kernel="copula", thresholds=bad, factor_phi=0.0)
    # ...but frailty barriers need not be monotone.
    TimingPlan(kernel="frailty", thresholds=bad, factor_phi=0.5)
    # Copula implies phi == 0.
    with pytest.raises(ValueError, match="factor_phi == 0"):
        TimingPlan(kernel="copula", thresholds=ok, factor_phi=0.5)
    # The plan is frozen: thresholds are read-only.
    plan = TimingPlan(kernel="copula", thresholds=ok, factor_phi=0.0)
    with pytest.raises(ValueError):
        plan.thresholds[0, 0] = 0.0


def test_extreme_pds_simulate_without_warnings():
    """A book containing PD of exactly 0 and 1 must run warning-free in
    multi-period copula mode (regression: the staircase monotonicity check
    emitted RuntimeWarning on the inf-inf diff, crashing under -W error)."""
    import numpy as np

    assets = [
        sf.AssetData(0, 0, 0.0, 0.5, 0.1, 1.0, "A"),
        sf.AssetData(1, 0, 1.0, 0.5, 0.1, 1.0, "A"),
        sf.AssetData(2, 0, 0.05, 0.5, 0.1, 1.0, "A"),
    ]
    port = sf.CreditPortfolio(
        assets=assets, intra_sector_correlations=0.4,
        sector_correlation_matrix=[[1.0]], config=SimulationConfig(seed=11),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = port.simulate(n_simulations=500, n_periods=4, period_length=0.25)
    # PD=0 never defaults, PD=1 always does: mean loss per unit exposure ~ LGD/3.
    assert 0.0 < res["portfolio_statistics"]["mean"] < 3.0
    assert np.isfinite(res["portfolio_statistics"]["var_99"])


def test_nan_cumulative_pds_rejected_by_both_models():
    """NaN cumulative PDs must fail loudly in BOTH timing models (regression:
    frailty calibration silently converged to never-defaults barriers)."""
    import numpy as np

    cum = np.array([[np.nan], [0.5]])
    rhos = np.array([0.4])
    with pytest.raises(ValueError, match="must not contain NaN"):
        sf.Copula().plan(cumulative_pds=cum, intra_correlations=rhos, period_length=0.5)
    with pytest.raises(ValueError, match="must not contain NaN"):
        sf.Frailty().plan(cumulative_pds=cum, intra_correlations=rhos, period_length=0.5)


def test_frailty_plan_preserves_marginal_pd():
    """The plan's barriers invert to the input cumulative PDs (deterministic,
    no Monte Carlo) — the calibration invariant, checked at the plan seam."""
    import numpy as np
    from simflux.portfolio.frailty import survival_curve

    cum = np.array([[0.02, 0.05], [0.05, 0.11], [0.09, 0.18]])
    rhos = np.array([0.3, 0.5])
    plan = sf.Frailty(persistence=0.7).plan(
        cumulative_pds=cum, intra_correlations=rhos, period_length=0.5
    )
    for j in range(2):
        implied = 1.0 - survival_curve(plan.thresholds[:, j], rhos[j], plan.factor_phi)
        assert np.allclose(implied, cum[:, j], atol=1e-3)


@pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
@pytest.mark.parametrize("timing,fp", [("copula", 0.5), ("frailty", 0.6)])
def test_rust_numpy_parity(timing, fp):
    """Rust and NumPy agree distributionally (different RNG streams)."""
    r = _run(4, timing, fp=fp, force_numpy=False)["portfolio_statistics"]
    n = _run(4, timing, fp=fp, force_numpy=True)["portfolio_statistics"]
    assert abs(r["mean"] - n["mean"]) / max(n["mean"], 1.0) < 0.30
    assert abs(r["var_99"] - n["var_99"]) / max(n["var_99"], 1.0) < 0.30
