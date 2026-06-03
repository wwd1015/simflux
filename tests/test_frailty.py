"""Unit tests for the frailty barrier calibration (no Monte Carlo).

The calibration must make the marginal cumulative PD exact for ANY persistence —
this is the property that distinguishes a correct dynamic-frailty model from a
naive AR(1) (which biases the cumulative PD). We check it deterministically via
the survival-curve recursion, the analytic inverse of the calibration.
"""

import numpy as np
import pytest

from simflux.portfolio.frailty import (
    calibrate_barriers,
    per_period_phi,
    survival_curve,
)


CUM = np.array([0.02, 0.05, 0.08, 0.10])


@pytest.mark.parametrize("phi", [0.0, 0.3, 0.6, 0.9, 1.0])
@pytest.mark.parametrize("rho", [0.1, 0.4])
def test_calibration_reproduces_marginal_cumulative_pd(phi, rho):
    b = calibrate_barriers(CUM, rho, phi)
    realized = 1.0 - survival_curve(b, rho, phi)
    assert np.allclose(realized, CUM, atol=1e-3)


def test_period_zero_barrier_is_phi_independent():
    """b_0 = Phi^{-1}(cum_0) regardless of persistence (period 0 is marginal)."""
    from statistics import NormalDist
    expected = NormalDist().inv_cdf(CUM[0])
    for phi in (0.0, 0.5, 1.0):
        b = calibrate_barriers(CUM, 0.4, phi)
        assert b[0] == pytest.approx(expected, abs=1e-3)


def test_per_period_phi_scaling():
    # Annual autocorrelation 0.5 over quarterly steps -> 0.5 ** 0.25.
    assert per_period_phi(0.5, 0.25) == pytest.approx(0.5 ** 0.25)
    assert per_period_phi(0.5, 1.0) == pytest.approx(0.5)
    # Clamped to [0, 1].
    assert per_period_phi(1.5, 1.0) == pytest.approx(1.0)


def test_zero_persistence_barriers_match_forward_pd_thresholds():
    """At phi=0 the calibrated barriers equal the independent-period forward-PD
    thresholds: Phi^{-1}(forward_pd_k)."""
    from statistics import NormalDist
    nd = NormalDist()
    b = calibrate_barriers(CUM, 0.4, 0.0)
    cum = np.concatenate([[0.0], CUM])
    forward = (cum[1:] - cum[:-1]) / (1.0 - cum[:-1])
    expected = [nd.inv_cdf(p) for p in forward]
    assert np.allclose(b, expected, atol=2e-3)
