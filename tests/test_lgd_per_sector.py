"""Per-sector ``systematic_lgd_correlation`` and the ``"match_intra"`` sentinel.

Covers the resolution of the four accepted input forms (scalar / list / dict /
``"match_intra"``), the validation rules, and that both backends accept and run
a per-sector vector.  Sector names are sorted, so list inputs map in sorted
order; dict inputs are keyed by name and order-independent.
"""

import math
import warnings
from unittest.mock import patch

import pytest

import simflux as sf
from simflux.core.backend import Backend
from simflux.core.base import SimulationConfig


def _assets(n=20):
    # Two sectors, "A" and "B" (already sorted).
    return [
        sf.AssetData(i, i % 2, 0.06, 0.5, 0.1, 1_000_000, "A" if i % 2 == 0 else "B")
        for i in range(n)
    ]


def test_scalar_broadcasts_to_every_sector():
    p = sf.TwoFactorPortfolio(_assets(), systematic_lgd_correlation=0.3)
    assert p.systematic_lgd_correlations == [0.3, 0.3]
    # Backward-compatible scalar view stays scalar when uniform.
    assert p.systematic_lgd_correlation == 0.3


def test_list_is_per_sector_in_sorted_order():
    p = sf.TwoFactorPortfolio(_assets(), systematic_lgd_correlation=[0.2, 0.5])
    assert p.systematic_lgd_correlations == [0.2, 0.5]
    # Non-uniform => the compat view returns the full list.
    assert p.systematic_lgd_correlation == [0.2, 0.5]


def test_dict_maps_by_name_and_defaults_missing():
    p = sf.TwoFactorPortfolio(_assets(), systematic_lgd_correlation={"B": 0.6})
    # "A" missing -> default 0.3; "B" -> 0.6 (sorted order [A, B]).
    assert p.systematic_lgd_correlations == [0.3, 0.6]


def test_match_intra_sets_sqrt_of_intra_per_sector():
    p = sf.TwoFactorPortfolio(
        _assets(),
        intra_sector_correlations={"A": 0.36, "B": 0.16},
        systematic_lgd_correlation="match_intra",
    )
    assert p.systematic_lgd_correlations == pytest.approx([math.sqrt(0.36), math.sqrt(0.16)])
    assert p.systematic_lgd_correlations == pytest.approx([0.6, 0.4])


def test_list_wrong_length_rejected():
    with pytest.raises(ValueError, match="one value per sector"):
        sf.TwoFactorPortfolio(_assets(), systematic_lgd_correlation=[0.3])


def test_per_sector_value_out_of_range_rejected():
    with pytest.raises(ValueError, match="systematic_lgd_correlation must be between -1 and 1"):
        sf.TwoFactorPortfolio(_assets(), systematic_lgd_correlation=[0.3, 1.5])


def test_unknown_string_rejected():
    with pytest.raises(ValueError, match="match_intra"):
        sf.TwoFactorPortfolio(_assets(), systematic_lgd_correlation="match_pd")


def test_summary_reports_per_sector_lgd():
    p = sf.TwoFactorPortfolio(_assets(), systematic_lgd_correlation={"A": 0.1, "B": 0.6})
    structure = p.get_portfolio_summary()["correlation_structure"]
    assert structure["systematic_lgd"] == {"A": 0.1, "B": 0.6}


def _mean_loss(rho_lgd, force_numpy):
    p = sf.TwoFactorPortfolio(
        _assets(), intra_sector_correlations=0.5,
        systematic_lgd_correlation=rho_lgd, config=SimulationConfig(seed=7),
    )
    if force_numpy:
        with patch.object(Backend, "is_available", return_value=False), \
             patch.object(Backend, "get_rust", return_value=None), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = p.simulate(n_simulations=15_000)
    else:
        r = p.simulate(n_simulations=15_000)
    return r["portfolio_statistics"]["mean"]


@pytest.mark.parametrize("force_numpy", [True, False])
def test_per_sector_runs_and_respects_sign_on_both_backends(force_numpy):
    if not force_numpy and not Backend.is_available():
        pytest.skip("Rust backend required")
    # Sector A gets strong wrong-way LGD, B gets none. Mean loss must exceed the
    # zero-correlation baseline (A's wrong-way coupling lifts it).
    base = _mean_loss([0.0, 0.0], force_numpy)
    mixed = _mean_loss([0.7, 0.0], force_numpy)
    assert base > 0 and mixed > base
