"""Two-factor portfolio loss simulation model (Merton framework)."""

import numpy as np
import pandas as pd
import warnings
from math import sqrt
from typing import List, Optional, Dict, Any, Union, TypedDict, NotRequired, cast
from dataclasses import dataclass
from ..core.base import BaseSimulator, SimulationConfig
from ..core.backend import Backend
from ..utils.storage import StorageConfig, ParquetResultsAnalyzer
from ..utils.random_utils import CorrelationMatrix
from .default_timing import DefaultTiming, resolve_timing
from .inputs import PortfolioInputs
from .numpy_simulation import simulate_portfolio_numpy


@dataclass
class AssetData:
    """
    Asset data container for portfolio simulation.

    Attributes
    ----------
    asset_id : int
        Unique asset identifier
    sector_id : int
        Sector identifier (0-indexed)
    pd : float
        Probability of default (0-1)
    lgd_mean : float
        Loss given default mean (0-1)
    lgd_std : float
        Loss given default standard deviation
    exposure : float
        Exposure at default
    sector_name : str
        Sector name
    intra_sector_correlation : Optional[float]
        Optional per-asset intra-sector correlation metadata
    """

    asset_id: int
    sector_id: int
    pd: float
    lgd_mean: float
    lgd_std: float
    exposure: float
    sector_name: str
    intra_sector_correlation: Optional[float] = None
    pd_term_structure: Optional[List[float]] = None
    lgd_term_structure: Optional[List[float]] = None

    def __post_init__(self) -> None:
        """Validate asset data after initialization."""
        if not 0 <= self.pd <= 1:
            raise ValueError(f"PD must be between 0 and 1, got {self.pd}")

        if not 0 <= self.lgd_mean <= 1:
            raise ValueError(f"LGD mean must be between 0 and 1, got {self.lgd_mean}")

        if self.lgd_std <= 0:
            raise ValueError(f"LGD std must be positive, got {self.lgd_std}")

        if self.exposure < 0:
            raise ValueError(f"Exposure must be non-negative, got {self.exposure}")

        if self.intra_sector_correlation is not None:
            if not 0 <= self.intra_sector_correlation <= 1:
                raise ValueError(
                    f"intra_sector_correlation must be between 0 and 1, got {self.intra_sector_correlation}"
                )

        if self.pd_term_structure is not None:
            for i, p in enumerate(self.pd_term_structure):
                if not 0 <= p <= 1:
                    raise ValueError(
                        f"pd_term_structure[{i}] must be between 0 and 1, got {p}"
                    )
            for i in range(1, len(self.pd_term_structure)):
                if self.pd_term_structure[i] < self.pd_term_structure[i - 1] - 1e-10:
                    raise ValueError(
                        "pd_term_structure must be non-decreasing (cumulative PDs)"
                    )

        # Validate Beta distribution parameters are feasible
        max_lgd_std = sqrt(self.lgd_mean * (1 - self.lgd_mean))
        if self.lgd_std >= max_lgd_std:
            raise ValueError(
                f"lgd_std ({self.lgd_std:.4f}) must be less than "
                f"sqrt(lgd_mean * (1 - lgd_mean)) = {max_lgd_std:.4f} "
                f"for valid Beta distribution parameters"
            )

        # A per-period LGD mean term structure: the obligor's realized LGD is drawn
        # from a Beta with the mean for its *default* period (lgd_std is constant).
        # Each per-period mean must be a valid mean and feasible with lgd_std.
        if self.lgd_term_structure is not None:
            for i, m in enumerate(self.lgd_term_structure):
                if not 0 <= m <= 1:
                    raise ValueError(
                        f"lgd_term_structure[{i}] must be between 0 and 1, got {m}"
                    )
                if self.lgd_std >= sqrt(m * (1 - m)):
                    raise ValueError(
                        f"lgd_std ({self.lgd_std:.4f}) is infeasible for "
                        f"lgd_term_structure[{i}] mean {m:.4f} "
                        f"(must be < sqrt(mean*(1-mean)) = {sqrt(m * (1 - m)):.4f})"
                    )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,  # type: ignore[name-defined]
        sector_mapping: Optional[Dict[str, int]] = None,
    ) -> List["AssetData"]:
        """
        Create list of AssetData from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns: asset_id, sector, pd, lgd_mean, lgd_std, exposure
        sector_mapping : Dict[str, int], optional
            Mapping from sector names to sector IDs

        Returns
        -------
        List[AssetData]
            List of AssetData objects
        """
        required_columns = [
            "asset_id",
            "sector",
            "pd",
            "lgd_mean",
            "lgd_std",
            "exposure",
        ]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if sector_mapping is None:
            unique_sectors = df["sector"].unique()
            sector_mapping = {sector: i for i, sector in enumerate(unique_sectors)}

        assets = []
        for _, row in df.iterrows():
            sector_name = str(row["sector"])
            sector_id = sector_mapping.get(sector_name)
            if sector_id is None:
                raise ValueError(f"Unknown sector: {sector_name}")

            intra_corr = None
            if "intra_sector_correlation" in df.columns:
                value = row["intra_sector_correlation"]
                if pd.notna(value):
                    intra_corr = float(value)

            asset = cls(
                asset_id=int(row["asset_id"]),
                sector_id=sector_id,
                pd=float(row["pd"]),
                lgd_mean=float(row["lgd_mean"]),
                lgd_std=float(row["lgd_std"]),
                exposure=float(row["exposure"]),
                sector_name=sector_name,
                intra_sector_correlation=intra_corr,
            )
            assets.append(asset)

        return assets


class PortfolioResult(TypedDict):
    """Result contract returned by :meth:`CreditPortfolio.simulate`.

    Both the Rust backend and the NumPy fallback populate exactly these keys, so
    a caller never has to know which backend ran.  ``analyzer`` is the only
    storage-conditional key: it is present only when interim results were
    persisted (``store_interim=True`` with an ``output_path`` and the Rust
    backend available).
    """

    portfolio_statistics: Dict[str, float]
    sector_statistics: Dict[str, Dict[str, float]]
    n_trials: int
    n_assets: int
    n_sectors: int
    sector_names: List[str]
    n_periods: int
    period_length: float
    time_horizon: float
    default_timing: str
    factor_persistence: Optional[float]
    analyzer: NotRequired[ParquetResultsAnalyzer]


def _complete_result(
    backend_results: Dict[str, Any],
    *,
    inputs: PortfolioInputs,
    timing: DefaultTiming,
) -> PortfolioResult:
    """Complete the :class:`PortfolioResult` contract — THE one place result
    keys are written.

    Both backends return only their statistics keys; everything else is stamped
    here, uniformly, so the key set cannot depend on which backend ran (and
    ``n_trials`` can never go missing on the NumPy fallback).  The timing model
    stamps its own keys.  Lives next to the TypedDict so the contract and its
    construction cannot drift apart; a contract test pins the emitted keys
    against ``PortfolioResult.__annotations__``.
    """
    results = dict(backend_results)
    results["n_trials"] = inputs.n_simulations
    results["n_assets"] = len(inputs.assets)
    results["n_sectors"] = len(inputs.sector_names)
    results["sector_names"] = inputs.sector_names
    results["n_periods"] = inputs.n_periods
    results["period_length"] = inputs.period_length
    results["time_horizon"] = inputs.n_periods * inputs.period_length
    results.update(timing.stamp())

    # The only storage-conditional key: present when interim results were
    # actually persisted (Rust-only feature).
    if inputs.store_interim and inputs.output_path and Backend.is_available():
        results["analyzer"] = ParquetResultsAnalyzer(inputs.output_path)

    return cast(PortfolioResult, results)


class CreditPortfolio(BaseSimulator):
    """
    Two-factor portfolio loss simulation model.

    Implements a Merton-style credit risk model with:
    - Global systematic factor affecting all assets
    - Sector-specific systematic factors
    - Asset-specific idiosyncratic factors
    - Beta-distributed loss given default with systematic correlation
    """

    def __init__(
        self,
        assets: Union[List[AssetData], pd.DataFrame],
        intra_sector_correlations: Optional[Union[float, Dict[str, float]]] = None,
        sector_correlation_matrix: Optional[
            Union[np.ndarray, List[List[float]]]
        ] = None,
        systematic_lgd_correlation: Union[
            float, List[float], Dict[str, float], str
        ] = 0.3,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        """Construct a two-factor credit portfolio.

        ``systematic_lgd_correlation`` accepts, in addition to a scalar:
        a per-sector ``list``, a ``{sector_name: value}`` ``dict`` (missing
        sectors default to 0.3), or the string ``"match_intra"`` — which sets
        ``rho_lgd_s = sqrt(rho_intra_s)`` per sector so LGD's cycle-sensitivity
        matches the default driver's (see methodology §2.10).
        """
        super().__init__(config)

        if not Backend.is_available():
            warnings.warn(
                "Rust backend not available. Portfolio simulation will use basic fallback. "
                "For full functionality, install from binary wheel.",
                RuntimeWarning,
            )

        # Process assets input
        if isinstance(assets, pd.DataFrame):
            self.assets = AssetData.from_dataframe(assets)
        else:
            self.assets = assets

        if not self.assets:
            raise ValueError("No assets provided")

        # Build sector information
        self.sector_names = sorted(
            list(set(asset.sector_name for asset in self.assets))
        )
        self.sector_mapping = {name: i for i, name in enumerate(self.sector_names)}

        # Ensure consistent sector IDs
        for asset in self.assets:
            asset.sector_id = self.sector_mapping[asset.sector_name]

        # Process correlation inputs
        if intra_sector_correlations is None:
            inferred_intra = self._infer_intra_correlations_from_assets()
            if inferred_intra is not None:
                self.intra_sector_correlations = inferred_intra
            else:
                self.intra_sector_correlations = [0.4] * len(self.sector_names)
        elif isinstance(intra_sector_correlations, (int, float)):
            self.intra_sector_correlations = [float(intra_sector_correlations)] * len(
                self.sector_names
            )
        elif isinstance(intra_sector_correlations, dict):
            self.intra_sector_correlations = [
                intra_sector_correlations.get(sector, 0.4)
                for sector in self.sector_names
            ]
        else:
            raise ValueError("intra_sector_correlations must be float or dict")

        # Per-obligor sector loading rho_i: an asset's own intra_sector_correlation
        # when set, else its sector's value. This is what both backends actually
        # load on, so obligors within a sector may have heterogeneous loadings.
        # The per-sector list above remains the fallback for obligors without an
        # explicit value (and the value carried on the Rust seam).
        self.asset_intra_correlations = [
            (
                a.intra_sector_correlation
                if a.intra_sector_correlation is not None
                else self.intra_sector_correlations[a.sector_id]
            )
            for a in self.assets
        ]

        # Resolve LGD-systematic correlation to one value per sector.  Must run
        # after intra_sector_correlations, because "match_intra" reads them.
        self.systematic_lgd_correlations = self._resolve_systematic_lgd_correlations(
            systematic_lgd_correlation
        )

        self._sector_corr = self._prepare_sector_correlation_matrix(
            sector_correlation_matrix
        )

        self.validate_inputs()

    @property
    def sector_correlation_matrix(self) -> np.ndarray:
        """The validated sector correlation matrix (read-only ndarray view).

        A property rather than a plain attribute so reassignment raises
        ``AttributeError`` instead of being silently ignored — the simulation
        reads the validated value object, not this view, so an accepted
        reassignment would be a silent no-op.
        """
        return self._sector_corr.values

    @property
    def systematic_lgd_correlation(self) -> Union[float, List[float]]:
        """Backward-compatible scalar view of the per-sector LGD correlation.

        Returns the single value when it is uniform across sectors (the common
        case), otherwise the full per-sector list.  The canonical attribute is
        ``systematic_lgd_correlations`` (plural).
        """
        vals = self.systematic_lgd_correlations
        first = vals[0]
        if all(v == first for v in vals):
            return first
        return list(vals)

    def _resolve_systematic_lgd_correlations(
        self, raw: Union[float, List[float], Dict[str, float], str]
    ) -> List[float]:
        """Resolve the LGD-systematic correlation input to one value per sector.

        Accepts a scalar (broadcast), a per-sector list, a ``{sector: value}``
        dict (missing sectors default to 0.3), or ``"match_intra"`` which sets
        ``rho_lgd_s = sqrt(rho_intra_s)`` so LGD's cycle-sensitivity equals the
        default driver's, sector by sector.
        """
        n = len(self.sector_names)
        if isinstance(raw, str):
            if raw != "match_intra":
                raise ValueError(
                    f"systematic_lgd_correlation string must be 'match_intra', got {raw!r}"
                )
            return [float(np.sqrt(max(0.0, c))) for c in self.intra_sector_correlations]
        if isinstance(raw, bool):
            raise ValueError("systematic_lgd_correlation must be numeric, not bool")
        if isinstance(raw, (int, float)):
            return [float(raw)] * n
        if isinstance(raw, dict):
            return [float(raw.get(sector, 0.3)) for sector in self.sector_names]
        if isinstance(raw, (list, tuple, np.ndarray)):
            vals = [float(v) for v in raw]
            if len(vals) != n:
                raise ValueError(
                    "systematic_lgd_correlation list must have one value per "
                    f"sector ({n}), got {len(vals)}"
                )
            return vals
        raise ValueError(
            "systematic_lgd_correlation must be a float, list, dict, or 'match_intra'"
        )

    def validate_inputs(self) -> None:
        """Validate portfolio configuration."""
        for corr in self.systematic_lgd_correlations:
            if not -1 <= corr <= 1:
                raise ValueError("systematic_lgd_correlation must be between -1 and 1")

        for corr in self.intra_sector_correlations:
            if not 0 <= corr <= 1:
                raise ValueError(
                    "All intra_sector_correlations must be between 0 and 1"
                )

        # The sector correlation matrix invariants (symmetry, unit diagonal,
        # bounds, strict positive-definiteness) are enforced once, by the
        # CorrelationMatrix constructed in _prepare_sector_correlation_matrix.

        # Check that we have assets in each sector
        sector_counts: Dict[str, int] = {}
        for asset in self.assets:
            sector_counts[asset.sector_name] = (
                sector_counts.get(asset.sector_name, 0) + 1
            )

        if len(sector_counts) != len(self.sector_names):
            raise ValueError("Some sectors have no assets")

    def _infer_intra_correlations_from_assets(self) -> Optional[List[float]]:
        """Derive a per-sector *representative* intra-correlation from asset metadata.

        Used only as the sector-level fallback when no ``intra_sector_correlations``
        argument is given; the true per-obligor loadings live in
        ``asset_intra_correlations``, so heterogeneity within a sector is allowed.
        Returns the per-sector mean of the assets that carry a value (a homogeneous
        sector collapses to that single value); ``None`` if no asset carries one.
        """
        sector_values: Dict[str, List[float]] = {}
        for asset in self.assets:
            if asset.intra_sector_correlation is not None:
                sector_values.setdefault(asset.sector_name, []).append(
                    asset.intra_sector_correlation
                )

        if not sector_values:
            return None

        inferred: List[float] = []
        for sector in self.sector_names:
            values = sector_values.get(sector)
            # A sector whose assets carry no value falls back to the 0.4 default.
            inferred.append(float(np.mean(values)) if values else 0.4)

        return inferred

    def _prepare_sector_correlation_matrix(
        self, matrix: Optional[Union[np.ndarray, List[List[float]]]]
    ) -> CorrelationMatrix:
        """Build the validated sector correlation matrix value.

        Requires strict positive-definiteness (not merely PSD): the Rust
        backend's Cholesky rejects a singular matrix, so admitting one here
        would make the two backends diverge (Rust raises, NumPy repairs).
        Validating PD up front keeps the seam backend-independent and matches
        CorrelatedGBM / TimeVaryingCorrelatedGBM, which also require PD.
        """
        n_sectors = len(self.sector_names)
        if matrix is None:
            arr = np.eye(n_sectors)
        else:
            arr = np.asarray(matrix, dtype=float)
            if arr.shape != (n_sectors, n_sectors):
                raise ValueError(
                    "sector_correlation_matrix must match number of sectors"
                )
        return CorrelationMatrix(
            arr, name="sector_correlation_matrix", check_positive_definite=True
        )

    def simulate(
        self,
        n_simulations: int,
        n_periods: int = 1,
        period_length: float = 1.0,
        default_timing: Union[str, DefaultTiming] = "copula",
        factor_persistence: Optional[float] = None,
        storage_config: Optional[StorageConfig] = None,
    ) -> "PortfolioResult":
        """
        Run portfolio loss simulation.

        Parameters
        ----------
        n_simulations : int
            Number of Monte Carlo simulations.
        n_periods : int, default=1
            Number of discrete time periods.  An asset can default at most once
            across all periods.  ``n_periods=1`` reproduces the classic
            single-period model.  For quarterly simulation over 2 years use
            ``n_periods=8``.
        period_length : float, default=1.0
            Length of each period in years.  ``period_length=0.25`` for
            quarterly steps.  The total time horizon is
            ``n_periods * period_length``.
        default_timing : Copula, Frailty, or {"copula", "frailty"}, default="copula"
            The default timing model — how default *timing* is generated across
            periods (no effect when ``n_periods == 1``, where the two coincide).
            Pass a timing object (``Copula()``, ``Frailty(persistence=0.5)``) or
            its string sugar, which resolves to a default-configured object.
            Both reproduce the marginal cumulative PD term structure exactly;
            they differ in the cross-period dependence of the systematic factor,
            and therefore in the tail.

            * ``"copula"`` — one-factor Gaussian copula of default *times*
              (Li, 2000).  A single latent ``V = sqrt(rho)*F + sqrt(1-rho)*eps``
              is drawn per obligor for the whole horizon and compared against the
              *cumulative* threshold staircase ``tau_k = Phi^{-1}(cum_k)``;
              default is the first crossing.  All uncertainty resolves at t=0
              (maximal cross-period dependence; grid-invariant loss distribution).
            * ``"frailty"`` — dynamic frailty (Duffie et al., 2009).  The
              systematic factor is *persistent* (an AR(1) with annual
              autocorrelation ``factor_persistence``), and fresh idiosyncratic
              shocks arrive each period; defaults are conditionally independent
              given the factor path.  The per-period default *barrier* is
              calibrated so the marginal PD is preserved for any persistence (a
              naive AR(1) on forward thresholds would bias it).  ``factor_persistence
              = 0`` reduces to independent periods; ``= 1`` freezes the factor.

            See methodology §2.6.
        factor_persistence : float, optional
            **Deprecated** — pass ``default_timing=Frailty(persistence=...)``
            instead.  Honored (with a ``DeprecationWarning``) only alongside
            string sugar; rejected alongside a timing object so persistence can
            never be specified twice.  See :class:`Frailty` for the parameter's
            meaning and calibration guidance.
        storage_config : StorageConfig, optional
            Configuration for storing interim results.

        Returns
        -------
        Dict[str, Any]
            Simulation results including portfolio statistics.

        Notes
        -----
        The cumulative PD at each period end comes from the asset's
        ``pd_term_structure`` when provided, else from the flat ``pd`` under a
        constant hazard (``cum_k = 1 - (1 - pd)^((k+1)/n_periods)``).  ``"copula"``
        thresholds the cumulative directly; ``"frailty"`` calibrates a barrier per
        period (see :mod:`simflux.portfolio.frailty`).
        """
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if n_periods <= 0:
            raise ValueError("n_periods must be positive")
        if period_length <= 0:
            raise ValueError("period_length must be positive")
        # Resolve string sugar first so a bad mode fails before any work; the
        # timing object validates its own parameters at construction.
        timing = resolve_timing(default_timing, factor_persistence=factor_persistence)

        # A pd_term_structure longer than n_periods is a valid *sub-horizon* run:
        # the simulation walks the first n_periods points of the curve, so a
        # 4-period structure run at n_periods=1 measures default by the end of
        # period 1 (cum[0]), not the full-horizon cumulative PD. Sweeping n_periods
        # from 1..len over one fixed term structure is the intended use. A structure
        # *shorter* than n_periods is under-specified — its last point is repeated
        # for the remaining periods, which is a genuine approximation.
        n_longer = sum(
            1
            for a in self.assets
            if a.pd_term_structure is not None and len(a.pd_term_structure) > n_periods
        )
        n_shorter = sum(
            1
            for a in self.assets
            if a.pd_term_structure is not None and len(a.pd_term_structure) < n_periods
        )
        if n_shorter:
            raise RuntimeError(
                f"pd_term_structure shorter than n_periods ({n_periods}) for {n_shorter} "
                "asset(s): there is no cumulative PD for the later periods, so the horizon "
                "is under-specified. Provide one cumulative PD per period "
                "(len(pd_term_structure) >= n_periods)."
            )
        if n_longer:
            warnings.warn(
                f"pd_term_structure longer than n_periods ({n_periods}) for {n_longer} "
                f"asset(s): running a sub-horizon, only the first {n_periods} cumulative-PD "
                "point(s) are used. This is expected when sweeping n_periods over a fixed "
                "term structure.",
                UserWarning,
            )

        # Assemble the backend seam's whole contract exactly once: the timing
        # plan (the sole timing input — thresholds for both kernels, derived
        # from the same per-obligor loadings each backend simulates on, so
        # threshold parity holds by construction), the per-obligor arrays, and
        # the run/storage parameters.  Both adapters take this one value.
        inputs = self._build_inputs(
            timing=timing,
            n_simulations=n_simulations,
            n_periods=n_periods,
            period_length=period_length,
            storage_config=storage_config,
        )

        # One dispatch seam, shared with the GBM engine: pick the active backend
        # adapter (Rust or NumPy) and run it.
        run = Backend.choose(
            self._simulate_portfolio_rust, self._simulate_portfolio_numpy
        )
        backend_results = run(inputs)

        # Both adapters return only their statistics keys; _complete_result is
        # the single place the rest of the contract is written.
        return _complete_result(backend_results, inputs=inputs, timing=timing)

    def _build_inputs(
        self,
        *,
        timing: DefaultTiming,
        n_simulations: int,
        n_periods: int,
        period_length: float,
        storage_config: Optional[StorageConfig] = None,
    ) -> PortfolioInputs:
        """Assemble the backend seam's input contract for one run."""
        plan = timing.plan(
            cumulative_pds=self._get_cumulative_pds_by_period(n_periods),
            intra_correlations=np.asarray(self.asset_intra_correlations),
            period_length=period_length,
        )
        return PortfolioInputs(
            config=self.config,
            assets=self.assets,
            sector_names=self.sector_names,
            sector_correlation=self._sector_corr,
            intra_sector_correlations=self.intra_sector_correlations,
            systematic_lgd_correlations=self.systematic_lgd_correlations,
            asset_intra_correlations=np.asarray(self.asset_intra_correlations),
            asset_sector_ids=np.array([a.sector_id for a in self.assets]),
            asset_lgd_stds=np.array([a.lgd_std for a in self.assets]),
            asset_exposures=np.array([a.exposure for a in self.assets]),
            lgd_means_by_period=self._get_lgd_means_by_period(n_periods),
            n_simulations=n_simulations,
            n_periods=n_periods,
            period_length=period_length,
            plan=plan,
            store_interim=storage_config.store_interim if storage_config else False,
            output_path=storage_config.output_path if storage_config else None,
            batch_size=storage_config.batch_size if storage_config else None,
        )

    def _convert_asset_to_rust(self, asset: AssetData) -> Any:
        """Convert Python AssetData to Rust AssetData."""
        _rust = Backend.get_rust()
        return _rust.AssetData(
            asset_id=asset.asset_id,
            sector_id=asset.sector_id,
            pd=asset.pd,
            lgd_mean=asset.lgd_mean,
            lgd_std=asset.lgd_std,
            exposure=asset.exposure,
            sector_name=asset.sector_name,
            pd_term_structure=asset.pd_term_structure,
            intra_sector_correlation=asset.intra_sector_correlation,
            lgd_term_structure=asset.lgd_term_structure,
        )

    # ------------------------------------------------------------------
    # Backend adapters (selected by Backend.choose in simulate)
    # ------------------------------------------------------------------

    def _simulate_portfolio_rust(self, inputs: PortfolioInputs) -> Dict[str, Any]:
        """Run the simulation on the Rust backend.

        The inputs flatten to primitives at the FFI: the timing plan crosses as
        the kernel name, the per-period AR(1) coefficient, and the threshold
        matrix (transposed to the per-asset row orientation the trial loop
        indexes, crossing as a NumPy array rather than nested lists to avoid
        boxing every threshold).
        """
        _rust = Backend.get_rust()
        rust_assets = [self._convert_asset_to_rust(asset) for asset in inputs.assets]

        # The full sector correlation matrix is the single source of truth for
        # cross-sector coupling; no scalar summary rides the seam.
        rust_config = _rust.PortfolioConfig(
            intra_sector_correlations=inputs.intra_sector_correlations,
            systematic_lgd_correlations=inputs.systematic_lgd_correlations,
            sector_names=inputs.sector_names,
            sector_correlation_matrix=inputs.sector_correlation.tolist(),
        )

        try:
            return _rust.simulate_portfolio(
                config=rust_config,
                assets=rust_assets,
                n_simulations=inputs.n_simulations,
                kernel=inputs.plan.kernel,
                factor_phi=inputs.plan.factor_phi,
                thresholds=np.ascontiguousarray(inputs.plan.thresholds.T),
                n_periods=inputs.n_periods,
                period_length=inputs.period_length,
                seed=inputs.config.seed,
                store_interim=inputs.store_interim,
                output_path=inputs.output_path,
                batch_size=inputs.batch_size,
            )
        except RuntimeError as exc:
            if inputs.store_interim:
                # Keep the underlying Rust error visible: every Rust failure
                # arrives as RuntimeError, and not all of them are storage
                # problems — masking the message misattributes config errors.
                raise RuntimeError(
                    "Rust backend failed with store_interim=True (interim storage "
                    "requires the Rust backend; disable store_interim or run in "
                    f"Python fallback mode). Underlying error: {exc}"
                ) from exc
            raise

    def _simulate_portfolio_numpy(self, inputs: PortfolioInputs) -> Dict[str, Any]:
        """Run the simulation on the NumPy fallback adapter.

        Delegates to :func:`simflux.portfolio.numpy_simulation.simulate_portfolio_numpy`,
        which consumes the same inputs value.  Interim storage is Rust-only, so
        it is not honored here.
        """
        if inputs.store_interim:
            warnings.warn(
                "Interim storage requires the Rust backend; proceeding without persistence.",
                RuntimeWarning,
            )
        return simulate_portfolio_numpy(inputs)

    # ------------------------------------------------------------------
    # Conditional PD helpers (used by frailty barrier calibration)
    # ------------------------------------------------------------------

    @staticmethod
    def _conditional_pd_from_term_structure(ts: List[float], period: int) -> float:
        idx = min(period, len(ts) - 1)
        if period == 0:
            return max(0.0, min(1.0, ts[0]))
        prev_idx = min(period - 1, len(ts) - 1)
        cum_prev = ts[prev_idx]
        cum_curr = ts[idx]
        if cum_prev >= 1.0:
            return 0.0
        return max(0.0, min(1.0, (cum_curr - cum_prev) / (1.0 - cum_prev)))

    @staticmethod
    def _conditional_pd_flat(pd: float, n_periods: int) -> float:
        if n_periods == 1:
            return pd
        return 1.0 - (1.0 - pd) ** (1.0 / n_periods)

    def _get_conditional_pds(self, period: int, n_periods: int) -> np.ndarray:
        """Return an array of per-asset conditional (forward) PDs for a period."""
        result = np.empty(len(self.assets))
        for i, a in enumerate(self.assets):
            if a.pd_term_structure is not None and len(a.pd_term_structure) > 0:
                result[i] = self._conditional_pd_from_term_structure(
                    a.pd_term_structure, period
                )
            else:
                result[i] = self._conditional_pd_flat(a.pd, n_periods)
        return result

    @staticmethod
    def _cumulative_pd_from_term_structure(ts: List[float], period: int) -> float:
        idx = min(period, len(ts) - 1)
        return max(0.0, min(1.0, ts[idx]))

    @staticmethod
    def _cumulative_pd_flat(pd: float, n_periods: int, period: int) -> float:
        # Cumulative PD by the end of period ``period`` (0-indexed) under a
        # constant hazard, so that the final period returns ``pd`` exactly.
        return 1.0 - (1.0 - pd) ** ((period + 1) / n_periods)

    def _get_cumulative_pds(self, period: int, n_periods: int) -> np.ndarray:
        """Return an array of per-asset *cumulative* PDs by a period's end."""
        result = np.empty(len(self.assets))
        for i, a in enumerate(self.assets):
            if a.pd_term_structure is not None and len(a.pd_term_structure) > 0:
                result[i] = self._cumulative_pd_from_term_structure(
                    a.pd_term_structure, period
                )
            else:
                result[i] = self._cumulative_pd_flat(a.pd, n_periods, period)
        return result

    def _get_cumulative_pds_by_period(self, n_periods: int) -> np.ndarray:
        """Per-asset cumulative PD by each period end: ``(n_periods, n_assets)``."""
        return np.stack(
            [self._get_cumulative_pds(k, n_periods) for k in range(n_periods)]
        )

    def _get_lgd_means_by_period(self, n_periods: int) -> np.ndarray:
        """Per-asset LGD Beta mean at each period: ``(n_periods, n_assets)``.

        An asset's ``lgd_term_structure`` value for the period (clamped to its last
        entry) when provided, else its constant ``lgd_mean``.
        """
        out = np.empty((n_periods, len(self.assets)))
        for j, a in enumerate(self.assets):
            ts = a.lgd_term_structure
            if ts:
                out[:, j] = [ts[min(k, len(ts) - 1)] for k in range(n_periods)]
            else:
                out[:, j] = a.lgd_mean
        return out

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the portfolio."""
        total_exposure = sum(asset.exposure for asset in self.assets)
        avg_pd = np.mean([asset.pd for asset in self.assets])
        avg_lgd = np.mean([asset.lgd_mean for asset in self.assets])

        sector_summary: Dict[str, Dict[str, Any]] = {}
        for sector_name in self.sector_names:
            sector_assets = [a for a in self.assets if a.sector_name == sector_name]
            sector_exposure = sum(a.exposure for a in sector_assets)
            sector_avg_pd = np.mean([a.pd for a in sector_assets])

            sector_summary[sector_name] = {
                "n_assets": len(sector_assets),
                "total_exposure": sector_exposure,
                "exposure_pct": sector_exposure / total_exposure * 100,
                "avg_pd": sector_avg_pd,
                "avg_lgd": np.mean([a.lgd_mean for a in sector_assets]),
            }

        return {
            "n_assets": len(self.assets),
            "n_sectors": len(self.sector_names),
            "total_exposure": total_exposure,
            "avg_pd": avg_pd,
            "avg_lgd": avg_lgd,
            "sectors": sector_summary,
            "correlation_structure": {
                "sector_matrix": self.sector_correlation_matrix.tolist(),
                "intra_sector": dict(
                    zip(self.sector_names, self.intra_sector_correlations)
                ),
                "systematic_lgd": dict(
                    zip(self.sector_names, self.systematic_lgd_correlations)
                ),
            },
        }

    @classmethod
    def create_sample_portfolio(
        cls,
        n_assets_per_sector: Union[int, List[int]] = 100,
        sectors: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "CreditPortfolio":
        """
        Create a sample portfolio for testing.

        Parameters
        ----------
        n_assets_per_sector : int or List[int], default=100
            Number of assets per sector
        sectors : List[str], optional
            Sector names. Default=['Technology', 'Finance', 'Healthcare']
        **kwargs
            Additional arguments for CreditPortfolio constructor.
            Supports ``inter_sector_correlation`` (float) to build a uniform
            sector correlation matrix when ``sector_correlation_matrix`` is
            not provided.
        """
        if sectors is None:
            sectors = ["Technology", "Finance", "Healthcare"]

        if isinstance(n_assets_per_sector, int):
            n_assets_per_sector = [n_assets_per_sector] * len(sectors)

        if len(n_assets_per_sector) != len(sectors):
            raise ValueError("n_assets_per_sector must match number of sectors")

        inter_sector_correlation = kwargs.pop("inter_sector_correlation", None)

        assets = []
        asset_id = 0

        sector_params = {
            "Technology": {"pd_mean": 0.02, "pd_std": 0.01, "lgd_mean": 0.65},
            "Finance": {"pd_mean": 0.05, "pd_std": 0.02, "lgd_mean": 0.45},
            "Healthcare": {"pd_mean": 0.03, "pd_std": 0.015, "lgd_mean": 0.55},
            "Energy": {"pd_mean": 0.08, "pd_std": 0.03, "lgd_mean": 0.70},
            "Utilities": {"pd_mean": 0.025, "pd_std": 0.01, "lgd_mean": 0.40},
        }

        rng = np.random.default_rng(42)

        for sector_idx, (sector, n_assets) in enumerate(
            zip(sectors, n_assets_per_sector)
        ):
            params = sector_params.get(
                sector, {"pd_mean": 0.04, "pd_std": 0.02, "lgd_mean": 0.60}
            )

            for i in range(n_assets):
                pd_val = max(
                    0.001, min(0.999, rng.normal(params["pd_mean"], params["pd_std"]))
                )
                lgd_mean = max(0.1, min(0.9, rng.normal(params["lgd_mean"], 0.15)))
                lgd_std = rng.uniform(0.05, 0.15)
                exposure = rng.lognormal(13, 1)

                asset = AssetData(
                    asset_id=asset_id,
                    sector_id=sector_idx,
                    pd=pd_val,
                    lgd_mean=lgd_mean,
                    lgd_std=lgd_std,
                    exposure=exposure,
                    sector_name=sector,
                )
                assets.append(asset)
                asset_id += 1

        if "sector_correlation_matrix" not in kwargs:
            base_corr = (
                inter_sector_correlation
                if inter_sector_correlation is not None
                else 0.15
            )
            matrix = np.full((len(sectors), len(sectors)), base_corr)
            np.fill_diagonal(matrix, 1.0)
            kwargs["sector_correlation_matrix"] = matrix

        return cls(assets, **kwargs)
