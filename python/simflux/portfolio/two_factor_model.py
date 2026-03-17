"""Two-factor portfolio loss simulation model (Merton framework)."""

import numpy as np
import pandas as pd
import warnings
from math import erf, sqrt
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from ..core.base import BaseSimulator, SimulationConfig
from ..core.backend import Backend, CORRELATION_TOLERANCE
from ..utils.storage import StorageConfig, ParquetResultsAnalyzer
from ..utils.random_utils import validate_correlation_matrix_strict, approx_norm_ppf


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
                    raise ValueError(f"pd_term_structure[{i}] must be between 0 and 1, got {p}")
            for i in range(1, len(self.pd_term_structure)):
                if self.pd_term_structure[i] < self.pd_term_structure[i - 1] - 1e-10:
                    raise ValueError("pd_term_structure must be non-decreasing (cumulative PDs)")

        # Validate Beta distribution parameters are feasible
        max_lgd_std = sqrt(self.lgd_mean * (1 - self.lgd_mean))
        if self.lgd_std >= max_lgd_std:
            raise ValueError(
                f"lgd_std ({self.lgd_std:.4f}) must be less than "
                f"sqrt(lgd_mean * (1 - lgd_mean)) = {max_lgd_std:.4f} "
                f"for valid Beta distribution parameters"
            )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame,
                      sector_mapping: Optional[Dict[str, int]] = None) -> List['AssetData']:
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
        required_columns = ['asset_id', 'sector', 'pd', 'lgd_mean', 'lgd_std', 'exposure']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if sector_mapping is None:
            unique_sectors = df['sector'].unique()
            sector_mapping = {sector: i for i, sector in enumerate(unique_sectors)}

        assets = []
        for _, row in df.iterrows():
            sector_name = str(row['sector'])
            sector_id = sector_mapping.get(sector_name)
            if sector_id is None:
                raise ValueError(f"Unknown sector: {sector_name}")

            intra_corr = None
            if 'intra_sector_correlation' in df.columns:
                value = row['intra_sector_correlation']
                if pd.notna(value):
                    intra_corr = float(value)

            asset = cls(
                asset_id=int(row['asset_id']),
                sector_id=sector_id,
                pd=float(row['pd']),
                lgd_mean=float(row['lgd_mean']),
                lgd_std=float(row['lgd_std']),
                exposure=float(row['exposure']),
                sector_name=sector_name,
                intra_sector_correlation=intra_corr,
            )
            assets.append(asset)

        return assets


class TwoFactorPortfolio(BaseSimulator):
    """
    Two-factor portfolio loss simulation model.

    Implements a Merton-style credit risk model with:
    - Global systematic factor affecting all assets
    - Sector-specific systematic factors
    - Asset-specific idiosyncratic factors
    - Beta-distributed loss given default with systematic correlation
    """

    def __init__(self,
                 assets: Union[List[AssetData], pd.DataFrame],
                 intra_sector_correlations: Optional[Union[float, Dict[str, float]]] = None,
                 sector_correlation_matrix: Optional[Union[np.ndarray, List[List[float]]]] = None,
                 systematic_lgd_correlation: float = 0.3,
                 config: Optional[SimulationConfig] = None) -> None:
        super().__init__(config)

        if not Backend.is_available():
            warnings.warn(
                "Rust backend not available. Portfolio simulation will use basic fallback. "
                "For full functionality, install from binary wheel.",
                RuntimeWarning
            )

        # Process assets input
        if isinstance(assets, pd.DataFrame):
            self.assets = AssetData.from_dataframe(assets)
        else:
            self.assets = assets

        if not self.assets:
            raise ValueError("No assets provided")

        # Build sector information
        self.sector_names = sorted(list(set(asset.sector_name for asset in self.assets)))
        self.sector_mapping = {name: i for i, name in enumerate(self.sector_names)}

        # Ensure consistent sector IDs
        for asset in self.assets:
            asset.sector_id = self.sector_mapping[asset.sector_name]

        # Process correlation inputs
        self.systematic_lgd_correlation = systematic_lgd_correlation

        if intra_sector_correlations is None:
            inferred_intra = self._infer_intra_correlations_from_assets()
            if inferred_intra is not None:
                self.intra_sector_correlations = inferred_intra
            else:
                self.intra_sector_correlations = [0.4] * len(self.sector_names)
        elif isinstance(intra_sector_correlations, (int, float)):
            self.intra_sector_correlations = [float(intra_sector_correlations)] * len(self.sector_names)
        elif isinstance(intra_sector_correlations, dict):
            self.intra_sector_correlations = [
                intra_sector_correlations.get(sector, 0.4)
                for sector in self.sector_names
            ]
        else:
            raise ValueError("intra_sector_correlations must be float or dict")

        self.sector_correlation_matrix = self._prepare_sector_correlation_matrix(sector_correlation_matrix)

        self.validate_inputs()

    def validate_inputs(self) -> None:
        """Validate portfolio configuration."""
        if not -1 <= self.systematic_lgd_correlation <= 1:
            raise ValueError("systematic_lgd_correlation must be between -1 and 1")

        for corr in self.intra_sector_correlations:
            if not 0 <= corr <= 1:
                raise ValueError("All intra_sector_correlations must be between 0 and 1")

        validate_correlation_matrix_strict(
            self.sector_correlation_matrix,
            name="sector_correlation_matrix",
            check_positive_definite=False,
            pd_tolerance=CORRELATION_TOLERANCE,
        )

        if self.sector_correlation_matrix.shape != (len(self.sector_names), len(self.sector_names)):
            raise ValueError("sector_correlation_matrix must match number of sectors")

        # Check that we have assets in each sector
        sector_counts: Dict[str, int] = {}
        for asset in self.assets:
            sector_counts[asset.sector_name] = sector_counts.get(asset.sector_name, 0) + 1

        if len(sector_counts) != len(self.sector_names):
            raise ValueError("Some sectors have no assets")

    def _infer_intra_correlations_from_assets(self) -> Optional[List[float]]:
        """Derive intra-sector correlations from asset metadata when available."""

        sector_values: Dict[str, List[float]] = {}
        for asset in self.assets:
            if asset.intra_sector_correlation is not None:
                sector_values.setdefault(asset.sector_name, []).append(asset.intra_sector_correlation)

        if not sector_values:
            return None

        inferred: List[float] = []
        for sector in self.sector_names:
            values = sector_values.get(sector)
            if not values:
                raise ValueError(
                    f"No intra-sector correlation provided for sector '{sector}'"
                )
            inferred.append(self._resolve_unique_value(values, f"intra correlation ({sector})"))

        return inferred

    @staticmethod
    def _resolve_unique_value(values: List[float], label: str) -> float:
        """Ensure a set of values collapses to a single correlation."""

        base = float(values[0])
        for value in values[1:]:
            if abs(value - base) > 1e-6:
                raise ValueError(
                    f"Inconsistent values supplied for {label}: {values}"
                )
        if not -1 <= base <= 1:
            raise ValueError(f"{label} must lie between -1 and 1, got {base}")
        return base

    def _prepare_sector_correlation_matrix(self, matrix: Optional[Union[np.ndarray, List[List[float]]]]) -> np.ndarray:
        """Return a valid sector correlation matrix."""

        n_sectors = len(self.sector_names)
        if matrix is None:
            return np.eye(n_sectors)

        arr = np.asarray(matrix, dtype=float)
        if arr.shape != (n_sectors, n_sectors):
            raise ValueError("sector_correlation_matrix must match number of sectors")
        return arr

    def simulate(self,
                 n_simulations: int,
                 n_periods: int = 1,
                 period_length: float = 1.0,
                 storage_config: Optional[StorageConfig] = None) -> Dict[str, Any]:
        """
        Run portfolio loss simulation.

        Parameters
        ----------
        n_simulations : int
            Number of Monte Carlo simulations.
        n_periods : int, default=1
            Number of discrete time periods.  Each period gets independent
            sector factor draws; an asset can default at most once across
            all periods.  ``n_periods=1`` reproduces the classic single-period
            model.  For quarterly simulation over 2 years use ``n_periods=8``.
        period_length : float, default=1.0
            Length of each period in years.  ``period_length=0.25`` for
            quarterly steps.  The total time horizon is
            ``n_periods * period_length``.
        storage_config : StorageConfig, optional
            Configuration for storing interim results.

        Returns
        -------
        Dict[str, Any]
            Simulation results including portfolio statistics.

        Notes
        -----
        When ``n_periods > 1`` the per-period conditional PD is derived
        automatically:

        * If an asset has ``pd_term_structure`` (cumulative PDs at each period
          end) the forward PD for each period is computed as
          ``(cum[k] - cum[k-1]) / (1 - cum[k-1])``.
        * Otherwise the flat ``pd`` field is spread across all periods assuming
          a constant hazard rate: ``1 - (1 - pd)^(1/n_periods)``.
        """
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if n_periods <= 0:
            raise ValueError("n_periods must be positive")
        if period_length <= 0:
            raise ValueError("period_length must be positive")

        store_interim = storage_config.store_interim if storage_config else False
        output_path = storage_config.output_path if storage_config else None

        if Backend.is_available():
            _rust = Backend.get_rust()
            rust_assets = [self._convert_asset_to_rust(asset) for asset in self.assets]
            off_diag = self.sector_correlation_matrix[np.triu_indices(len(self.sector_names), 1)]
            avg_inter_sector = float(np.mean(off_diag)) if off_diag.size > 0 else 0.0

            rust_config = _rust.PortfolioConfig(
                inter_sector_correlation=avg_inter_sector,
                intra_sector_correlations=self.intra_sector_correlations,
                systematic_lgd_correlation=self.systematic_lgd_correlation,
                sector_names=self.sector_names,
                sector_correlation_matrix=self.sector_correlation_matrix.tolist(),
            )

            try:
                results = _rust.simulate_portfolio(
                    config=rust_config,
                    assets=rust_assets,
                    n_simulations=n_simulations,
                    n_periods=n_periods,
                    period_length=period_length,
                    seed=self.config.seed,
                    store_interim=store_interim,
                    output_path=output_path
                )
            except RuntimeError as exc:
                if store_interim:
                    raise RuntimeError(
                        "Interim storage via the Rust backend is not available; disable "
                        "store_interim or run in Python fallback mode.") from exc
                raise
        else:
            if store_interim:
                warnings.warn(
                    "Interim storage requires the Rust backend; proceeding without persistence.",
                    RuntimeWarning
                )
            results = self._fallback_simulate_portfolio(
                n_simulations=n_simulations,
                n_periods=n_periods,
                period_length=period_length,
            )

        # Add metadata
        results['n_assets'] = len(self.assets)
        results['n_sectors'] = len(self.sector_names)
        results['sector_names'] = self.sector_names
        results['n_periods'] = n_periods
        results['period_length'] = period_length
        results['time_horizon'] = n_periods * period_length

        # Create analyzer if interim data is available
        if store_interim and output_path and Backend.is_available():
            results['analyzer'] = ParquetResultsAnalyzer(output_path)

        return results

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
        )

    # ------------------------------------------------------------------
    # Conditional PD helpers (used by NumPy fallback)
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
        """Return an array of per-asset conditional PDs for a given period."""
        result = np.empty(len(self.assets))
        for i, a in enumerate(self.assets):
            if a.pd_term_structure is not None and len(a.pd_term_structure) > 0:
                result[i] = self._conditional_pd_from_term_structure(a.pd_term_structure, period)
            else:
                result[i] = self._conditional_pd_flat(a.pd, n_periods)
        return result

    # ------------------------------------------------------------------
    # NumPy fallback
    # ------------------------------------------------------------------

    def _fallback_simulate_portfolio(self, n_simulations: int,
                                     n_periods: int = 1,
                                     period_length: float = 1.0) -> Dict[str, Any]:
        """Fallback portfolio simulation using vectorized NumPy (no Rust)."""
        try:
            from scipy import stats as scipy_stats
            from scipy.special import erf as scipy_erf
            norm_ppf = scipy_stats.norm.ppf
            beta_ppf = scipy_stats.beta.ppf
            erf_func = scipy_erf
        except ImportError:
            warnings.warn("scipy not available, using approximate calculations", UserWarning)

            norm_ppf = approx_norm_ppf
            beta_ppf = None
            erf_func = np.vectorize(erf)

        rng = np.random.default_rng(self.config.seed)

        n_assets = len(self.assets)
        n_sectors = len(self.sector_names)
        if n_sectors == 0:
            raise ValueError("Portfolio must contain at least one sector")

        self._check_memory(n_simulations * n_periods * (n_sectors + 2 * n_assets))

        try:
            sector_cholesky = np.linalg.cholesky(self.sector_correlation_matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Inter-sector correlation matrix is not positive semi-definite") from exc

        asset_sector_ids = np.array([a.sector_id for a in self.assets])
        asset_lgd_means = np.array([a.lgd_mean for a in self.assets])
        asset_lgd_stds = np.array([a.lgd_std for a in self.assets])
        asset_exposures = np.array([a.exposure for a in self.assets])

        intra_corrs = np.array(self.intra_sector_correlations)
        asset_intra_corrs = intra_corrs[asset_sector_ids]
        sector_loadings = np.sqrt(asset_intra_corrs)
        idio_loadings = np.sqrt(np.maximum(0.0, 1 - asset_intra_corrs))

        lgd_alpha = asset_lgd_means * ((asset_lgd_means * (1 - asset_lgd_means)) / (asset_lgd_stds**2) - 1)
        lgd_beta_param = (1 - asset_lgd_means) * ((asset_lgd_means * (1 - asset_lgd_means)) / (asset_lgd_stds**2) - 1)

        # Track per-simulation, per-asset state
        ever_defaulted = np.zeros((n_simulations, n_assets), dtype=bool)
        losses = np.zeros((n_simulations, n_assets))

        for period in range(n_periods):
            cond_pds = self._get_conditional_pds(period, n_periods)
            thresholds = norm_ppf(cond_pds)

            # Independent sector factors per period
            independent = rng.normal(size=(n_simulations, n_sectors))
            sector_factor_all = independent @ sector_cholesky.T
            asset_sector_factors = sector_factor_all[:, asset_sector_ids]

            idiosyncratic = rng.normal(size=(n_simulations, n_assets))
            asset_values = (
                sector_loadings[np.newaxis, :] * asset_sector_factors +
                idio_loadings[np.newaxis, :] * idiosyncratic
            )

            # Default check: only for surviving assets
            new_defaults = (asset_values <= thresholds[np.newaxis, :]) & ~ever_defaulted

            if np.any(new_defaults):
                # Compute LGD for newly defaulted assets
                lgd_sys = self.systematic_lgd_correlation * asset_sector_factors
                lgd_idio_scale = max(0.0, 1 - self.systematic_lgd_correlation ** 2) ** 0.5
                lgd_idio = lgd_idio_scale * rng.normal(size=(n_simulations, n_assets))
                lgd_normal = lgd_sys + lgd_idio
                lgd_uniform = np.clip(0.5 * (1 + erf_func(lgd_normal / np.sqrt(2))), 1e-12, 1 - 1e-12)

                if beta_ppf is not None:
                    lgd_realized = beta_ppf(lgd_uniform, lgd_alpha[np.newaxis, :], lgd_beta_param[np.newaxis, :])
                else:
                    lgd_realized = np.clip(lgd_uniform, 1e-3, 1 - 1e-3)

                period_losses = lgd_realized * asset_exposures[np.newaxis, :]
                losses = np.where(new_defaults, period_losses, losses)

            ever_defaulted |= new_defaults

        total_losses = losses.sum(axis=1)

        sector_losses: Dict[str, np.ndarray] = {}
        for s_idx, sector in enumerate(self.sector_names):
            mask = asset_sector_ids == s_idx
            sector_losses[sector] = losses[:, mask].sum(axis=1)

        def calculate_stats(loss_arr: np.ndarray) -> Dict[str, float]:
            p95, p99, p999 = np.percentile(loss_arr, [95, 99, 99.9])
            return {
                'mean': float(np.mean(loss_arr)),
                'std_dev': float(np.std(loss_arr)),
                'var_95': float(p95),
                'var_99': float(p99),
                'var_999': float(p999),
                'expected_shortfall_95': float(np.mean(loss_arr[loss_arr >= p95])),
                'expected_shortfall_99': float(np.mean(loss_arr[loss_arr >= p99])),
                'max_loss': float(np.max(loss_arr))
            }

        return {
            'portfolio_statistics': calculate_stats(total_losses),
            'sector_statistics': {s: calculate_stats(l) for s, l in sector_losses.items()},
        }

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
                'n_assets': len(sector_assets),
                'total_exposure': sector_exposure,
                'exposure_pct': sector_exposure / total_exposure * 100,
                'avg_pd': sector_avg_pd,
                'avg_lgd': np.mean([a.lgd_mean for a in sector_assets])
            }

        return {
            'n_assets': len(self.assets),
            'n_sectors': len(self.sector_names),
            'total_exposure': total_exposure,
            'avg_pd': avg_pd,
            'avg_lgd': avg_lgd,
            'sectors': sector_summary,
            'correlation_structure': {
                'sector_matrix': self.sector_correlation_matrix.tolist(),
                'intra_sector': dict(zip(self.sector_names, self.intra_sector_correlations)),
                'systematic_lgd': self.systematic_lgd_correlation
            }
        }

    @classmethod
    def create_sample_portfolio(cls,
                              n_assets_per_sector: Union[int, List[int]] = 100,
                              sectors: List[str] = None,
                              **kwargs: Any) -> 'TwoFactorPortfolio':
        """
        Create a sample portfolio for testing.

        Parameters
        ----------
        n_assets_per_sector : int or List[int], default=100
            Number of assets per sector
        sectors : List[str], optional
            Sector names. Default=['Technology', 'Finance', 'Healthcare']
        **kwargs
            Additional arguments for TwoFactorPortfolio constructor.
            Supports ``inter_sector_correlation`` (float) to build a uniform
            sector correlation matrix when ``sector_correlation_matrix`` is
            not provided.
        """
        if sectors is None:
            sectors = ['Technology', 'Finance', 'Healthcare']

        if isinstance(n_assets_per_sector, int):
            n_assets_per_sector = [n_assets_per_sector] * len(sectors)

        if len(n_assets_per_sector) != len(sectors):
            raise ValueError("n_assets_per_sector must match number of sectors")

        inter_sector_correlation = kwargs.pop('inter_sector_correlation', None)

        assets = []
        asset_id = 0

        sector_params = {
            'Technology': {'pd_mean': 0.02, 'pd_std': 0.01, 'lgd_mean': 0.65},
            'Finance': {'pd_mean': 0.05, 'pd_std': 0.02, 'lgd_mean': 0.45},
            'Healthcare': {'pd_mean': 0.03, 'pd_std': 0.015, 'lgd_mean': 0.55},
            'Energy': {'pd_mean': 0.08, 'pd_std': 0.03, 'lgd_mean': 0.70},
            'Utilities': {'pd_mean': 0.025, 'pd_std': 0.01, 'lgd_mean': 0.40}
        }

        rng = np.random.default_rng(42)

        for sector_idx, (sector, n_assets) in enumerate(zip(sectors, n_assets_per_sector)):
            params = sector_params.get(sector, {'pd_mean': 0.04, 'pd_std': 0.02, 'lgd_mean': 0.60})

            for i in range(n_assets):
                pd_val = max(0.001, min(0.999, rng.normal(params['pd_mean'], params['pd_std'])))
                lgd_mean = max(0.1, min(0.9, rng.normal(params['lgd_mean'], 0.15)))
                lgd_std = rng.uniform(0.05, 0.15)
                exposure = rng.lognormal(13, 1)

                asset = AssetData(
                    asset_id=asset_id,
                    sector_id=sector_idx,
                    pd=pd_val,
                    lgd_mean=lgd_mean,
                    lgd_std=lgd_std,
                    exposure=exposure,
                    sector_name=sector
                )
                assets.append(asset)
                asset_id += 1

        if 'sector_correlation_matrix' not in kwargs:
            base_corr = inter_sector_correlation if inter_sector_correlation is not None else 0.15
            matrix = np.full((len(sectors), len(sectors)), base_corr)
            np.fill_diagonal(matrix, 1.0)
            kwargs['sector_correlation_matrix'] = matrix

        return cls(assets, **kwargs)
