"""The portfolio backend seam's input contract.

A :class:`PortfolioInputs` is assembled exactly once per
:meth:`CreditPortfolio.simulate` call and handed to whichever backend adapter
runs (Rust or NumPy).  It is the *whole* contract: the two adapters take the
same single value, so their signatures cannot drift apart, and adding a model
input is one field here rather than a kwarg threaded through three signatures.

It carries the validated internal currency from the other seams — the
:class:`~simflux.utils.random_utils.CorrelationMatrix` (validated once at
construction, cached Cholesky for the NumPy kernel, list form for the FFI) and
the :class:`~simflux.portfolio.default_timing.TimingPlan` (the sole timing
input) — plus the per-obligor arrays and run/storage parameters.  Construction
validates the cross-field shape invariants, so a hand-built instance (e.g. in a
benchmark or test) gets the same guarantees as one built by ``simulate()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np

from ..core.base import SimulationConfig
from ..utils.random_utils import CorrelationMatrix
from .default_timing import TimingPlan

if TYPE_CHECKING:  # avoid a runtime cycle with two_factor_model
    from .two_factor_model import AssetData


@dataclass(frozen=True)
class PortfolioInputs:
    """Everything a portfolio backend adapter needs to run one simulation."""

    config: SimulationConfig
    assets: Sequence["AssetData"]
    sector_names: List[str]
    sector_correlation: CorrelationMatrix
    intra_sector_correlations: List[float]  # per sector (rides the Rust config)
    systematic_lgd_correlations: List[float]  # per sector
    asset_intra_correlations: np.ndarray  # per obligor (what both backends load on)
    asset_sector_ids: np.ndarray
    asset_lgd_stds: np.ndarray
    asset_exposures: np.ndarray
    lgd_means_by_period: np.ndarray  # (n_periods, n_assets)
    n_simulations: int
    n_periods: int
    period_length: float
    plan: TimingPlan
    store_interim: bool = False
    output_path: Optional[str] = None
    batch_size: Optional[int] = None

    def __post_init__(self) -> None:
        n_assets = len(self.assets)
        if self.n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if self.plan.thresholds.shape != (self.n_periods, n_assets):
            raise ValueError(
                f"timing plan thresholds shape {self.plan.thresholds.shape} must be "
                f"(n_periods, n_assets) = ({self.n_periods}, {n_assets})"
            )
        if self.lgd_means_by_period.shape != (self.n_periods, n_assets):
            raise ValueError(
                f"lgd_means_by_period shape {self.lgd_means_by_period.shape} must be "
                f"(n_periods, n_assets) = ({self.n_periods}, {n_assets})"
            )
        for name in (
            "asset_intra_correlations",
            "asset_sector_ids",
            "asset_lgd_stds",
            "asset_exposures",
        ):
            if len(getattr(self, name)) != n_assets:
                raise ValueError(f"{name} must have one value per asset ({n_assets})")
        if self.sector_correlation.n != len(self.sector_names):
            raise ValueError(
                "sector_correlation must match the number of sectors "
                f"({len(self.sector_names)})"
            )
