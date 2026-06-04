"""Scipy-free special functions for the NumPy fallback path.

The portfolio fallback needs an inverse Beta CDF (``beta.ppf``) to map a
Gaussian-copula LGD driver onto a Beta-distributed realized LGD. When scipy is
installed we use ``scipy.stats.beta.ppf``; when it is not, these routines provide
an accurate, dependency-free replacement so the fallback produces a *correct* LGD
rather than a silently-wrong one.

Implementation:
- :func:`reg_incomplete_beta` is the regularized incomplete beta ``I_x(a, b)``
  via the Lentz continued fraction (Numerical Recipes ``betacf``/``betai``),
  using ``math.lgamma`` for the normalization. Fully vectorized over ``x``.
- :func:`beta_ppf` inverts it. The Beta parameters are constant per column
  (one ``(alpha, beta)`` per obligor), so we tabulate ``I_x`` on a fine ``x``
  grid once per distinct ``(alpha, beta)`` and invert by linear interpolation
  (``np.interp``). Accuracy is ~1e-4 with the default grid — far finer than any
  meaningful LGD resolution — and the cost is O(n_obs) interpolation rather than
  per-element root-finding.
"""

from __future__ import annotations

from math import lgamma
from typing import Dict, Tuple

import numpy as np

_TINY = 1e-30


def _betacf(
    a: float, b: float, x: np.ndarray, maxit: int = 400, eps: float = 1e-14
) -> np.ndarray:
    """Continued fraction for the incomplete beta (Lentz), vectorized over ``x``.

    ``a``, ``b`` are scalars; ``x`` is an array. Returns the continued-fraction
    factor used by :func:`reg_incomplete_beta`.
    """
    x = np.asarray(x, dtype=float)
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = np.where(np.abs(d) < _TINY, _TINY, d)
    d = 1.0 / d
    h = d.copy()
    for m in range(1, maxit + 1):
        m2 = 2 * m
        # Even step.
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < _TINY, _TINY, d)
        d = 1.0 / d
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < _TINY, _TINY, c)
        h = h * d * c
        # Odd step.
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < _TINY, _TINY, d)
        d = 1.0 / d
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < _TINY, _TINY, c)
        delta = d * c
        h = h * delta
        if np.all(np.abs(delta - 1.0) < eps):
            break
    return h


def reg_incomplete_beta(a: float, b: float, x: np.ndarray) -> np.ndarray:
    """Regularized incomplete beta ``I_x(a, b)`` for scalar ``a, b`` over array ``x``.

    Monotone increasing from ``I_0 = 0`` to ``I_1 = 1``. This is the Beta CDF.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    lo = x <= 0.0
    hi = x >= 1.0
    mid = ~(lo | hi)
    out[lo] = 0.0
    out[hi] = 1.0
    if np.any(mid):
        xm = x[mid]
        ln_norm = lgamma(a + b) - lgamma(a) - lgamma(b)
        bt = np.exp(ln_norm + a * np.log(xm) + b * np.log1p(-xm))
        cutoff = (a + 1.0) / (a + b + 2.0)
        left = xm < cutoff
        res = np.empty_like(xm)
        if np.any(left):
            res[left] = bt[left] * _betacf(a, b, xm[left]) / a
        right = ~left
        if np.any(right):
            # Symmetry: I_x(a,b) = 1 - I_{1-x}(b,a).
            res[right] = 1.0 - bt[right] * _betacf(b, a, 1.0 - xm[right]) / b
        out[mid] = res
    return out


def _column_params(p, shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce a Beta parameter (scalar, ``(ncol,)``, or ``(1, ncol)``) to one value
    per last-axis column. Parameters are assumed constant along all but the last
    axis (one obligor per column)."""
    big = np.broadcast_to(np.asarray(p, dtype=float), shape)
    idx0 = (0,) * (len(shape) - 1)
    return np.asarray(big[idx0]).reshape(-1)


def beta_ppf(u, a, b, grid_points: int = 8192) -> np.ndarray:
    """Inverse Beta CDF (quantile) without scipy.

    Mirrors the ``scipy.stats.beta.ppf(q, a, b)`` calling convention used in the
    portfolio fallback: ``u`` has shape ``(..., ncol)`` and ``a``, ``b`` are
    broadcastable, constant per column. Tabulates ``I_x(a, b)`` once per distinct
    ``(a, b)`` and inverts by interpolation.
    """
    u = np.asarray(u, dtype=float)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    x_grid = np.linspace(0.0, 1.0, grid_points)

    def _table(av: float, bv: float) -> np.ndarray:
        cdf = reg_incomplete_beta(av, bv, x_grid)
        np.maximum.accumulate(cdf, out=cdf)  # guard monotonicity for np.interp
        return cdf

    # Fast path: a single (alpha, beta) applies to the whole array.
    if a_arr.size == 1 and b_arr.size == 1:
        return np.interp(u, _table(float(a_arr), float(b_arr)), x_grid)

    # Per-column path: parameters are constant along all but the last axis.
    ncol = u.shape[-1]
    a_col = _column_params(a_arr, u.shape)
    b_col = _column_params(b_arr, u.shape)
    cache: Dict[Tuple[float, float], np.ndarray] = {}
    out = np.empty_like(u)
    for j in range(ncol):
        key = (a_col[j], b_col[j])
        cdf = cache.get(key)
        if cdf is None:
            cdf = _table(*key)
            cache[key] = cdf
        out[..., j] = np.interp(u[..., j], cdf, x_grid)
    return out
