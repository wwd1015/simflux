"""Guard the *sign* of the systematic LGD coupling (wrong-way risk).

This is the regression the cross-validation suite structurally cannot catch:
both backends share the same LGD formula, so they agreed with each other even
when the sign was economically backwards.  Here we pin behaviour, not parity —
a positive ``systematic_lgd_correlation`` must make losses *worse*, because
defaults cluster in the low tail of the sector factor and LGD must rise there.

Mechanism: defaults oversample bad systematic states.  With wrong-way risk on,
LGD is high in exactly those states, so E[LGD | default] > E[LGD] and the mean
portfolio loss rises with the correlation.  With the old (flipped) sign the mean
would *fall* — so the directional assertion below is a one-sided sign trap.
"""

import warnings
from unittest.mock import patch

import pytest

import simflux as sf
from simflux.core.backend import Backend
from simflux.core.base import SimulationConfig


def _portfolio(rho_lgd: float, seed: int) -> sf.CreditPortfolio:
    # One highly-correlated sector maximises the clustering of defaults with the
    # systematic factor, so the conditioning effect on LGD is large and the test
    # sits well above the Monte Carlo noise floor.
    assets = [
        sf.AssetData(i, 0, 0.08, 0.5, 0.2, 1_000_000, "A")
        for i in range(50)
    ]
    return sf.CreditPortfolio(
        assets=assets,
        intra_sector_correlations=0.6,
        systematic_lgd_correlation=rho_lgd,
        config=SimulationConfig(seed=seed),
    )


def _mean_loss_numpy(rho_lgd: float, seed: int, n_sims: int = 20_000) -> float:
    with patch.object(Backend, "is_available", return_value=False), \
         patch.object(Backend, "get_rust", return_value=None), \
         warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _portfolio(rho_lgd, seed).simulate(n_simulations=n_sims)
    return res["portfolio_statistics"]["mean"]


def test_wrong_way_lgd_raises_expected_loss_numpy():
    """NumPy fallback: positive systematic LGD correlation must raise mean loss."""
    base = _mean_loss_numpy(0.0, seed=11)
    wrong_way = _mean_loss_numpy(0.7, seed=11)
    # A flipped sign would make `wrong_way` LOWER than `base`; require a clear,
    # one-sided increase (the modelled effect is ~30%+, so 15% is conservative).
    assert wrong_way > base * 1.15, (
        f"expected wrong-way LGD to raise mean loss: base={base:.0f}, "
        f"rho=0.7 -> {wrong_way:.0f}"
    )


def test_zero_lgd_correlation_is_neutral_numpy():
    """rho_lgd = 0 must leave the marginal LGD mean unbiased (sanity baseline)."""
    # With no systematic coupling, mean loss ≈ n*pd*lgd_mean*exposure for this
    # symmetric book; we only check it is in a sane band, not exact.
    mean = _mean_loss_numpy(0.0, seed=11)
    naive = 50 * 0.08 * 0.5 * 1_000_000  # E[defaults] * E[LGD] * exposure
    assert 0.5 * naive < mean < 1.5 * naive


@pytest.mark.skipif(not Backend.is_available(), reason="Rust backend required")
def test_wrong_way_lgd_raises_expected_loss_rust():
    """Rust backend: same sign guard. Requires a rebuild after the sign fix
    (`maturin develop --release`) — an unrebuilt binary carries the old sign and
    will fail here, which is the intended signal."""
    base = _portfolio(0.0, seed=11).simulate(n_simulations=20_000)
    wrong_way = _portfolio(0.7, seed=11).simulate(n_simulations=20_000)
    assert (
        wrong_way["portfolio_statistics"]["mean"]
        > base["portfolio_statistics"]["mean"] * 1.15
    )
