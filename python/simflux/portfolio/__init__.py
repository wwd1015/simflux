"""Portfolio simulation modules."""

from .two_factor_model import CreditPortfolio, TwoFactorPortfolio, AssetData
from .correlation import TwoFactorCorrelationStructure

__all__ = [
    "CreditPortfolio",
    "TwoFactorPortfolio",  # deprecated alias for CreditPortfolio
    "AssetData",
    "TwoFactorCorrelationStructure",
]
