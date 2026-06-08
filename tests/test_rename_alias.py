"""CreditPortfolio rename and the deprecated TwoFactorPortfolio alias."""

import warnings

import pytest

import simflux as sf


def _sample(cls):
    return cls.create_sample_portfolio(
        n_assets_per_sector=5,
        sectors=["Technology", "Finance"],
        inter_sector_correlation=0.1,
    )


def test_credit_portfolio_is_canonical():
    """CreditPortfolio is exported and constructs without a warning."""
    assert hasattr(sf, "CreditPortfolio")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no DeprecationWarning on the new name
        pf = _sample(sf.CreditPortfolio)
    assert pf.simulate(n_simulations=50)["portfolio_statistics"]["mean"] >= 0


def test_two_factor_portfolio_is_deprecated_alias():
    """TwoFactorPortfolio still works, subclasses CreditPortfolio, and warns."""
    assert issubclass(sf.TwoFactorPortfolio, sf.CreditPortfolio)
    with pytest.warns(DeprecationWarning, match="renamed to CreditPortfolio"):
        pf = sf.TwoFactorPortfolio.create_sample_portfolio(
            n_assets_per_sector=5,
            sectors=["Technology", "Finance"],
            inter_sector_correlation=0.1,
        )
    assert isinstance(pf, sf.CreditPortfolio)
    # The alias produces the same result contract as the canonical class.
    assert "portfolio_statistics" in pf.simulate(n_simulations=50)
