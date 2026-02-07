"""Pure-Python two-factor correlation helpers used by the fallback backend."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


class CorrelationError(ValueError):
    """Raised when a correlation structure cannot be constructed."""


@dataclass(frozen=True)
class SystematicFactors:
    """Container for sector-level systematic factors."""

    sector_factors: np.ndarray

    def get_sector_factor(self, sector_id: int) -> float:
        """Return the factor value associated with ``sector_id``."""

        return float(self.sector_factors[sector_id])


class TwoFactorCorrelationStructure:
    """Replicates the Rust two-factor correlation loading in pure Python."""

    def __init__(
        self,
        inter_sector_correlation: float,
        intra_sector_correlations: Sequence[float],
        sector_sizes: Sequence[int],
        sector_names: Optional[Sequence[str]] = None,
        sector_correlation_matrix: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self.inter_sector_correlation = inter_sector_correlation
        self.intra_sector_correlations = list(map(float, intra_sector_correlations))
        self.sector_sizes = list(map(int, sector_sizes))

        self._validate_inputs()

        if sector_names is None:
            self.sector_names = [f"Sector {i}" for i in range(len(self.sector_sizes))]
        else:
            if len(sector_names) != len(self.sector_sizes):
                raise ValueError("sector_names must match number of sectors")
            self.sector_names = list(sector_names)

        self.n_sectors = len(self.sector_sizes)
        self.n_assets = sum(self.sector_sizes)

        self._sector_corr_matrix = self._build_sector_correlation_matrix(sector_correlation_matrix)
        self._sector_cholesky = np.linalg.cholesky(self._sector_corr_matrix)

        self._asset_indices = self._build_asset_index_map()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _validate_inputs(self) -> None:
        if not -1.0 <= self.inter_sector_correlation <= 1.0:
            raise ValueError("inter_sector_correlation must be between -1 and 1")

        if len(self.intra_sector_correlations) != len(self.sector_sizes):
            raise ValueError("intra_sector_correlations and sector_sizes must have same length")

        for corr in self.intra_sector_correlations:
            if not -1.0 <= corr <= 1.0:
                raise ValueError("intra_sector_correlations must be between -1 and 1")

        for size in self.sector_sizes:
            if size <= 0:
                raise ValueError("All sector sizes must be positive")

    def _build_sector_correlation_matrix(
        self, sector_correlation_matrix: Optional[Sequence[Sequence[float]]]
    ) -> np.ndarray:
        n = len(self.sector_sizes)
        if sector_correlation_matrix is None:
            matrix = np.full((n, n), float(self.inter_sector_correlation), dtype=float)
            np.fill_diagonal(matrix, 1.0)
        else:
            matrix = np.asarray(sector_correlation_matrix, dtype=float)
            if matrix.shape != (n, n):
                raise ValueError("sector_correlation_matrix must match number of sectors")

        if not np.allclose(matrix, matrix.T, atol=1e-8):
            raise ValueError("sector_correlation_matrix must be symmetric")

        eigenvalues = np.linalg.eigvalsh(matrix)
        if np.min(eigenvalues) < -1e-8:
            raise ValueError("sector_correlation_matrix must be positive semi-definite")

        return matrix

    def _build_asset_index_map(self) -> Dict[int, slice]:
        offsets = {}
        cursor = 0
        for sector_id, size in enumerate(self.sector_sizes):
            offsets[sector_id] = slice(cursor, cursor + size)
            cursor += size
        return offsets

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_correlation_matrix(self) -> np.ndarray:
        """Return the full asset-by-asset correlation matrix."""

        total_assets = self.n_assets
        matrix = np.eye(total_assets)

        for sector_id, corr in enumerate(self.intra_sector_correlations):
            sector_slice = self._asset_indices[sector_id]
            block = self._build_intra_block(corr, sector_slice.stop - sector_slice.start)
            matrix[sector_slice, sector_slice] = block

        for i in range(self.n_sectors):
            for j in range(i + 1, self.n_sectors):
                block = self._build_cross_block(i, j)
                slice_i = self._asset_indices[i]
                slice_j = self._asset_indices[j]
                matrix[slice_i, slice_j] = block
                matrix[slice_j, slice_i] = block.T

        return matrix

    def _build_intra_block(self, corr: float, size: int) -> np.ndarray:
        block = np.full((size, size), corr, dtype=float)
        np.fill_diagonal(block, 1.0)
        return block

    def _build_cross_block(self, sector_i: int, sector_j: int) -> np.ndarray:
        corr = self._sector_corr_matrix[sector_i, sector_j]
        intra_i = self.intra_sector_correlations[sector_i]
        intra_j = self.intra_sector_correlations[sector_j]
        cross_value = sqrt(max(0.0, intra_i)) * sqrt(max(0.0, intra_j)) * corr

        block = np.full(
            (
                self._asset_indices[sector_i].stop - self._asset_indices[sector_i].start,
                self._asset_indices[sector_j].stop - self._asset_indices[sector_j].start,
            ),
            cross_value,
            dtype=float,
        )
        return block

    def get_sector_correlation_block(self, sector_id: int) -> np.ndarray:
        """Return the correlation sub-matrix for a single sector."""

        sector_slice = self._asset_indices[sector_id]
        size = sector_slice.stop - sector_slice.start
        return self._build_intra_block(self.intra_sector_correlations[sector_id], size)

    def get_cross_sector_correlation(self, sector_i: int, sector_j: int) -> np.ndarray:
        """Return the correlation block between two sectors."""

        return self._build_cross_block(sector_i, sector_j)

    def get_factor_loadings(self) -> np.ndarray:
        """Return factor loadings for each asset and sector factors."""

        total_assets = self.n_assets
        loadings = np.zeros((total_assets, self.n_sectors + 1))

        for sector_id in range(self.n_sectors):
            sector_slice = self._asset_indices[sector_id]
            sector_loading = sqrt(max(0.0, self.intra_sector_correlations[sector_id]))
            idio_loading = sqrt(max(0.0, 1.0 - self.intra_sector_correlations[sector_id]))

            loadings[sector_slice, sector_id] = sector_loading
            loadings[sector_slice, -1] = idio_loading

        return loadings

    def generate_factors(self, n_simulations: int, seed: Optional[int] = None) -> List[SystematicFactors]:
        """Generate correlated sector factors via Cholesky sampling."""

        rng = np.random.default_rng(seed)
        draws = rng.standard_normal(size=(n_simulations, self.n_sectors))
        factors = draws @ self._sector_cholesky.T
        return [SystematicFactors(factors[i]) for i in range(n_simulations)]

    def validate_structure(self) -> Dict[str, bool]:
        """Run structural validation checks on the correlation matrix."""

        matrix = self.get_correlation_matrix()
        symmetric = np.allclose(matrix, matrix.T, atol=1e-8)
        diagonal_ones = np.allclose(np.diag(matrix), 1.0, atol=1e-8)
        bounds_valid = np.all((matrix >= -1.0 - 1e-8) & (matrix <= 1.0 + 1e-8))

        try:
            np.linalg.cholesky(matrix)
            positive_definite = True
        except np.linalg.LinAlgError:
            positive_definite = False

        return {
            "is_symmetric": bool(symmetric),
            "diagonal_ones": bool(diagonal_ones),
            "bounds_valid": bool(bounds_valid),
            "positive_definite": positive_definite,
            "structure_valid": bool(symmetric and diagonal_ones and bounds_valid and positive_definite),
        }

    def summary(self) -> Dict[str, object]:
        """Produce diagnostic information about the structure."""

        matrix = self.get_correlation_matrix()
        eigenvalues = np.linalg.eigvalsh(matrix)
        validation = self.validate_structure()

        u, s, _ = np.linalg.svd(matrix)
        effective_rank = float(np.sum(s) / s[0]) if s.size > 0 else 0.0
        condition_number = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")

        return {
            "n_assets": self.n_assets,
            "n_sectors": self.n_sectors,
            "sector_names": list(self.sector_names),
            "sector_sizes": list(self.sector_sizes),
            "inter_sector_correlation": self.inter_sector_correlation,
            "intra_sector_correlations": list(self.intra_sector_correlations),
            "matrix_rank": int(np.linalg.matrix_rank(matrix)),
            "effective_rank": effective_rank,
            "condition_number": condition_number,
            "largest_eigenvalue": float(eigenvalues[-1]),
            "smallest_eigenvalue": float(eigenvalues[0]),
            "validation": validation,
        }

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------
    @classmethod
    def create_uniform(
        cls,
        n_sectors: int,
        assets_per_sector: int,
        inter_sector_correlation: float,
        intra_sector_correlation: float,
        sector_names: Optional[Iterable[str]] = None,
    ) -> "TwoFactorCorrelationStructure":
        """Construct a structure with uniform sector sizes and correlations."""

        sizes = [assets_per_sector] * n_sectors
        intra = [intra_sector_correlation] * n_sectors

        if sector_names is not None:
            names_list = list(sector_names)
        else:
            names_list = None

        return cls(
            inter_sector_correlation=inter_sector_correlation,
            intra_sector_correlations=intra,
            sector_sizes=sizes,
            sector_names=names_list,
        )
