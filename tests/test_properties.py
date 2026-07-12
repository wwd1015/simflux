"""Property-based tests (hypothesis): invariants over the whole input space.

The fixed-point suites pin behavior at chosen inputs; these tests assert the
same invariants for *arbitrary* valid inputs — random correlation matrices,
random PD term structures, random Beta shapes — where hand-picked cases can't
reach. All tests are derandomized so CI is deterministic.

Tolerances are set from measured implementation accuracy with a wide margin:
`approx_norm_ppf` measures ≤ 6e-9 absolute error (asserted at 1e-7), and the
scipy-free `beta_ppf` tabulated inversion measures ≤ 2e-5 (asserted at 1e-3).
"""

from statistics import NormalDist

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

from simflux.portfolio.default_timing import Copula, Frailty
from simflux.portfolio.frailty import survival_curve
from simflux.utils.random_utils import (
    CorrelationMatrix,
    approx_norm_ppf,
    correlation_matrix_diagnostics,
    safe_cholesky,
)

_SETTINGS = settings(max_examples=50, deadline=None, derandomize=True)
_SLOW_SETTINGS = settings(max_examples=20, deadline=None, derandomize=True)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def correlation_matrices(draw, max_dim: int = 6) -> np.ndarray:
    """Random valid (symmetric, unit-diagonal, positive-definite) matrices.

    Built as a normalized Gram matrix of random loadings plus a diagonal
    ridge, so positive-definiteness holds by construction for any draw.
    """
    n = draw(st.integers(min_value=2, max_value=max_dim))
    flat = draw(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
            min_size=n * n,
            max_size=n * n,
        )
    )
    loadings = np.asarray(flat).reshape(n, n)
    gram = loadings @ loadings.T + 0.5 * np.eye(n)
    d = np.sqrt(np.diag(gram))
    corr = gram / np.outer(d, d)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return corr


@st.composite
def cumulative_pd_matrices(draw, max_periods: int = 5, max_assets: int = 4):
    """Random valid cumulative-PD term structures: (n_periods, n_assets),
    values in (0, 1), non-decreasing along the period axis."""
    n_periods = draw(st.integers(min_value=1, max_value=max_periods))
    n_assets = draw(st.integers(min_value=1, max_value=max_assets))
    increments = draw(
        st.lists(
            st.floats(min_value=1e-4, max_value=0.2, allow_nan=False),
            min_size=n_periods * n_assets,
            max_size=n_periods * n_assets,
        )
    )
    inc = np.asarray(increments).reshape(n_periods, n_assets)
    cum = np.minimum(np.cumsum(inc, axis=0), 0.99)
    return cum


# ---------------------------------------------------------------------------
# Correlation-matrix invariants
# ---------------------------------------------------------------------------


@_SETTINGS
@given(corr=correlation_matrices())
def test_valid_correlation_matrices_are_accepted_and_factor_exactly(corr):
    value = CorrelationMatrix(corr, name="prop", check_positive_definite=True)
    L = value.cholesky()
    # The factorization must reconstruct the matrix (no silent repair for a
    # genuinely positive-definite input).
    assert np.allclose(L @ L.T, corr, atol=1e-10)
    assert all(correlation_matrix_diagnostics(corr).values())


@_SETTINGS
@given(corr=correlation_matrices())
def test_safe_cholesky_output_is_lower_triangular(corr):
    L = safe_cholesky(corr)
    assert np.allclose(L, np.tril(L))
    assert np.all(np.diag(L) > 0)


# ---------------------------------------------------------------------------
# Special-function accuracy (tolerances from measured error, wide margin)
# ---------------------------------------------------------------------------


@_SETTINGS
@given(
    p=st.floats(min_value=1e-9, max_value=1.0 - 1e-9, allow_nan=False),
)
def test_approx_norm_ppf_matches_reference(p):
    got = float(approx_norm_ppf(np.asarray([p]))[0])
    expected = NormalDist().inv_cdf(p)
    assert got == pytest.approx(expected, abs=1e-7)


@_SETTINGS
@given(
    a=st.floats(min_value=0.3, max_value=30.0, allow_nan=False),
    b=st.floats(min_value=0.3, max_value=30.0, allow_nan=False),
    u=st.floats(min_value=1e-6, max_value=1.0 - 1e-6, allow_nan=False),
)
def test_fallback_beta_ppf_matches_scipy(a, b, u):
    scipy_stats = pytest.importorskip("scipy.stats")
    from simflux.utils.special import beta_ppf

    got = float(np.asarray(beta_ppf(np.asarray([u]), a, b))[0])
    expected = float(scipy_stats.beta.ppf(u, a, b))
    assert got == pytest.approx(expected, abs=1e-3)


@_SETTINGS
@given(
    a=st.floats(min_value=0.3, max_value=30.0, allow_nan=False),
    b=st.floats(min_value=0.3, max_value=30.0, allow_nan=False),
)
def test_fallback_beta_ppf_is_monotone_and_bounded(a, b):
    from simflux.utils.special import beta_ppf

    u = np.linspace(1e-6, 1.0 - 1e-6, 64)
    x = np.asarray(beta_ppf(u, a, b))
    assert np.all(np.diff(x) >= -1e-12)  # non-decreasing quantile
    assert np.all((x >= 0.0) & (x <= 1.0))


# ---------------------------------------------------------------------------
# Timing-plan invariants
# ---------------------------------------------------------------------------


@_SETTINGS
@given(cum=cumulative_pd_matrices())
def test_copula_plan_thresholds_are_staircase_quantiles(cum):
    n_assets = cum.shape[1]
    rhos = np.full(n_assets, 0.3)
    plan = Copula().plan(cumulative_pds=cum, intra_correlations=rhos, period_length=1.0)
    assert plan.kernel == "copula"
    assert plan.factor_phi == 0.0
    # Thresholds are the Phi^-1 staircase: non-decreasing along periods and
    # an exact quantile round-trip on the interior.
    assert np.all(np.diff(plan.thresholds, axis=0) >= -1e-12)
    nd = NormalDist()
    for j in range(n_assets):
        for k in range(cum.shape[0]):
            assert nd.cdf(plan.thresholds[k, j]) == pytest.approx(cum[k, j], abs=1e-9)


@_SLOW_SETTINGS
@given(
    cum=cumulative_pd_matrices(max_periods=4, max_assets=2),
    rho=st.floats(min_value=0.05, max_value=0.8, allow_nan=False),
    persistence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    period_length=st.sampled_from([0.25, 0.5, 1.0]),
)
def test_frailty_calibration_preserves_marginal_pd_for_any_input(
    cum, rho, persistence, period_length
):
    """THE frailty invariant, generalized: for arbitrary valid PD curves,
    correlations, persistences, and grids, the calibrated barriers must
    reproduce the marginal cumulative PD (checked deterministically via the
    survival-curve recursion, the calibration's analytic inverse)."""
    n_assets = cum.shape[1]
    rhos = np.full(n_assets, rho)
    plan = Frailty(persistence=persistence).plan(
        cumulative_pds=cum, intra_correlations=rhos, period_length=period_length
    )
    assert plan.kernel == "frailty"
    for j in range(n_assets):
        realized = 1.0 - survival_curve(plan.thresholds[:, j], rho, plan.factor_phi)
        assert np.allclose(realized, cum[:, j], atol=3e-3)


@_SETTINGS
@given(
    cum=cumulative_pd_matrices(max_periods=3, max_assets=2),
    scale=st.floats(min_value=1.01, max_value=1.5, allow_nan=False),
)
def test_copula_thresholds_monotone_in_pd(cum, scale):
    """Riskier books get looser thresholds: scaling every cumulative PD up
    must not lower any threshold."""
    assume(np.all(cum * scale <= 0.999))
    rhos = np.full(cum.shape[1], 0.3)
    base = Copula().plan(cumulative_pds=cum, intra_correlations=rhos, period_length=1.0)
    scaled = Copula().plan(
        cumulative_pds=cum * scale, intra_correlations=rhos, period_length=1.0
    )
    assert np.all(scaled.thresholds >= base.thresholds - 1e-12)
