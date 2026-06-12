"""Default timing models: how multi-period default *timing* is generated.

This module is the single home for timing semantics (ADR-0002).  A timing model
(:class:`Copula`, :class:`Frailty`) is a validated-at-construction recipe; its
:meth:`~DefaultTiming.plan` derives the **timing plan** for a book — the
per-period, per-asset threshold matrix plus the per-period AR(1) coefficient —
which is the *sole* timing input the simulation backends consume.  Backends
never derive thresholds themselves:

* ``Copula.plan``  — thresholds are the cumulative-PD staircase quantiles
  ``Phi^{-1}(cum_k)`` (Li 2000), previously re-derived inside *both* backends.
* ``Frailty.plan`` — thresholds are the calibrated barriers preserving the
  marginal cumulative PD for any persistence (Duffie et al. 2009), via
  :mod:`simflux.portfolio.frailty` (which stays the implementation layer and
  the home of the deterministic inverse, ``survival_curve``).

The plan's **kernel** names the simulation dynamic a backend runs — a closed
two-member set: ``"copula"`` (one frozen latent per obligor, first crossing, no
fresh idiosyncratic shocks after t=0) or ``"frailty"`` (AR(1) systematic factor
with fresh idiosyncratic shocks each period, first passage).  A new *derivation*
over an existing kernel is a pure-Python addition here; a new *kernel* requires
an inner loop in both backends plus cross-validation.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Literal, Optional, Union

import numpy as np

from ..utils.random_utils import approx_norm_ppf

Kernel = Literal["copula", "frailty"]
_KERNELS = ("copula", "frailty")


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Standard-normal quantile (scipy when present, Acklam fallback).

    One quantile implementation now feeds both backends' copula thresholds, so
    threshold derivation can no longer drift between Rust and NumPy.
    """
    try:
        from scipy import stats as scipy_stats

        return np.asarray(scipy_stats.norm.ppf(p), dtype=float)
    except ImportError:
        return approx_norm_ppf(p)


def _clean_cumulative_pds(cumulative_pds: np.ndarray) -> np.ndarray:
    """Defensive input hygiene shared by every timing model.

    Clips to ``[0, 1]`` and monotonizes tiny cumulative-PD decreases along the
    period axis (matching what the frailty calibration always did); the caller
    is responsible for genuine term-structure validation.
    """
    cum = np.asarray(cumulative_pds, dtype=float)
    if cum.ndim != 2:
        raise ValueError(
            f"cumulative_pds must be (n_periods, n_assets), got ndim={cum.ndim}"
        )
    # Reject NaN here, once, for every timing model: np.clip would propagate it,
    # the copula plan would fail late, and the frailty calibration would silently
    # converge to "never defaults" — one loud error site instead.
    if np.isnan(cum).any():
        raise ValueError("cumulative_pds must not contain NaN")
    return np.maximum.accumulate(np.clip(cum, 0.0, 1.0), axis=0)


@dataclass(frozen=True, eq=False)
class TimingPlan:
    """The frozen artifact a default timing model derives for a given book.

    ``thresholds`` is ``(n_periods, n_assets)``: the latent-scale value each
    obligor's latent is compared against (``<=`` means default) in each period.
    ``factor_phi`` is the *per-period* AR(1) coefficient (exactly ``0.0`` for the
    copula kernel).  Both backends consume the same plan, so threshold parity
    across backends holds by construction.

    Invariants are enforced at construction so plan misuse fails loudly here
    rather than inside a backend: the kernel must be one of the closed two, and
    copula thresholds must be non-decreasing down each column (first-crossing
    semantics is meaningless otherwise).
    """

    kernel: Kernel
    thresholds: np.ndarray
    factor_phi: float

    def __post_init__(self) -> None:
        if self.kernel not in _KERNELS:
            raise ValueError(
                f"kernel must be one of {_KERNELS}, got {self.kernel!r} — new "
                "kernels require an inner loop in both backends (see ADR-0002)"
            )
        t = np.array(self.thresholds, dtype=float, copy=True)
        if t.ndim != 2:
            raise ValueError(
                f"thresholds must be (n_periods, n_assets), got ndim={t.ndim}"
            )
        if np.isnan(t).any():
            raise ValueError("thresholds must not contain NaN")
        if not 0.0 <= self.factor_phi <= 1.0:
            raise ValueError(
                f"factor_phi must be between 0 and 1, got {self.factor_phi}"
            )
        if self.kernel == "copula":
            if self.factor_phi != 0.0:
                raise ValueError("the copula kernel implies factor_phi == 0.0")
            # ±inf endpoints (cum PD of exactly 0/1) make diff produce NaN for
            # repeated infinities; NaN compares False, which is the right answer —
            # errstate suppresses the spurious invalid-subtract warning so a
            # legitimate PD-0/1 book doesn't warn (or crash under -W error).
            with np.errstate(invalid="ignore"):
                staircase_decreases = np.any(np.diff(t, axis=0) < 0.0)
            if staircase_decreases:
                raise ValueError(
                    "copula thresholds must be non-decreasing down each column "
                    "(the cumulative-PD staircase)"
                )
        t.setflags(write=False)
        object.__setattr__(self, "thresholds", t)

    @property
    def n_periods(self) -> int:
        return self.thresholds.shape[0]

    @property
    def n_assets(self) -> int:
        return self.thresholds.shape[1]


class DefaultTiming(ABC):
    """A default timing model: a named, validated-at-construction recipe that
    derives the :class:`TimingPlan` for a book.  Immutable after construction.
    """

    kernel: ClassVar[str]

    @abstractmethod
    def plan(
        self,
        *,
        cumulative_pds: np.ndarray,
        intra_correlations: np.ndarray,
        period_length: float,
    ) -> TimingPlan:
        """Derive the timing plan.

        ``cumulative_pds`` is ``(n_periods, n_assets)`` — cumulative PD by each
        period end (``n_periods`` is its first axis; it never rides separately).
        ``intra_correlations`` is ``(n_assets,)`` per-obligor loadings (read only
        by calibrating models).  Deterministic and side-effect-free: same inputs,
        same plan.
        """

    @abstractmethod
    def stamp(self) -> Dict[str, Any]:
        """The mode's contribution to the result contract
        (``default_timing`` / ``factor_persistence`` keys)."""

    def _validate_plan_inputs(
        self,
        cumulative_pds: np.ndarray,
        intra_correlations: np.ndarray,
        period_length: float,
    ) -> None:
        if period_length <= 0:
            raise ValueError("period_length must be positive")
        n_assets = np.asarray(cumulative_pds).shape[-1]
        if len(np.asarray(intra_correlations)) != n_assets:
            raise ValueError(
                "intra_correlations must have one value per asset "
                f"({n_assets}), got {len(np.asarray(intra_correlations))}"
            )


@dataclass(frozen=True)
class Copula(DefaultTiming):
    """One-factor Gaussian copula of default *times* (Li 2000).

    One frozen latent per obligor for the whole horizon, compared against the
    cumulative-PD staircase; default at first crossing.  All uncertainty
    resolves at t=0; the loss distribution is grid-invariant.  No parameters.
    """

    kernel: ClassVar[str] = "copula"

    def plan(
        self,
        *,
        cumulative_pds: np.ndarray,
        intra_correlations: np.ndarray,
        period_length: float,
    ) -> TimingPlan:
        self._validate_plan_inputs(cumulative_pds, intra_correlations, period_length)
        cum = _clean_cumulative_pds(cumulative_pds)
        return TimingPlan(kernel="copula", thresholds=_norm_ppf(cum), factor_phi=0.0)

    def stamp(self) -> Dict[str, Any]:
        return {"default_timing": "copula", "factor_persistence": None}


@dataclass(frozen=True)
class Frailty(DefaultTiming):
    """Dynamic frailty (Duffie, Eckner, Horel & Saita 2009).

    A persistent AR(1) systematic factor with fresh idiosyncratic shocks each
    period, against per-period barriers *calibrated* so the marginal cumulative
    PD is preserved exactly for any persistence.  ``persistence`` is the
    **annual** autocorrelation of the systematic credit-cycle factor, in
    ``[0, 1]``; the per-period coefficient is ``persistence ** period_length``.
    The default ``0.5`` is illustrative — calibrate to data for production use.
    """

    persistence: float = 0.5
    kernel: ClassVar[str] = "frailty"

    def __post_init__(self) -> None:
        if not 0.0 <= self.persistence <= 1.0:
            raise ValueError(
                f"factor_persistence must be between 0 and 1, got {self.persistence}"
            )

    def plan(
        self,
        *,
        cumulative_pds: np.ndarray,
        intra_correlations: np.ndarray,
        period_length: float,
    ) -> TimingPlan:
        from .frailty import barrier_matrix, per_period_phi

        self._validate_plan_inputs(cumulative_pds, intra_correlations, period_length)
        cum = _clean_cumulative_pds(cumulative_pds)
        n_periods = cum.shape[0]
        # n_periods == 1 forces phi = 0: with a single period there is no
        # cross-period dependence and the modes coincide.
        phi = per_period_phi(self.persistence, period_length) if n_periods > 1 else 0.0
        thresholds = barrier_matrix(
            cum, np.asarray(intra_correlations, dtype=float), phi
        )
        return TimingPlan(kernel="frailty", thresholds=thresholds, factor_phi=phi)

    def stamp(self) -> Dict[str, Any]:
        return {"default_timing": "frailty", "factor_persistence": self.persistence}


def resolve_timing(
    spec: Union[str, DefaultTiming],
    *,
    factor_persistence: Optional[float] = None,
) -> DefaultTiming:
    """Resolve the ``default_timing`` argument to a timing model.

    ``"copula"`` / ``"frailty"`` are string sugar for default-configured
    instances; a :class:`DefaultTiming` instance passes through unchanged.
    ``factor_persistence`` is the deprecated keyword channel: honored (with a
    ``DeprecationWarning``) only alongside string sugar, and rejected alongside
    a timing object so persistence can never be specified twice.
    """
    if isinstance(spec, DefaultTiming):
        if factor_persistence is not None:
            raise ValueError(
                "factor_persistence cannot be combined with a timing object; "
                "set it on the object instead (e.g. Frailty(persistence=...))"
            )
        return spec
    if isinstance(spec, str):
        if factor_persistence is not None:
            warnings.warn(
                "factor_persistence is deprecated; pass "
                "default_timing=Frailty(persistence=...) instead",
                DeprecationWarning,
                stacklevel=3,
            )
        if spec == "copula":
            return Copula()
        if spec == "frailty":
            if factor_persistence is not None:
                return Frailty(persistence=factor_persistence)
            return Frailty()
    raise ValueError(
        "default_timing must be 'copula', 'frailty', or a timing object "
        f"(e.g. Frailty(persistence=0.7)); got {spec!r}"
    )
