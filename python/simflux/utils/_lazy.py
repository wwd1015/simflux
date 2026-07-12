"""Deferred loading for heavy dependencies.

pandas and polars together account for the large majority of ``import
simflux`` time but are needed only by the DataFrame factory and the Parquet
analyzer. A :class:`LazyModule` stands in for the real module and imports it
on first attribute access, so the cost is paid by the features that use it,
not by every import of the package.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


class LazyModule:
    """Module proxy that imports the real module on first attribute access."""

    def __init__(self, name: str) -> None:
        self._lazy_name = name
        self._lazy_module: ModuleType | None = None

    def __getattr__(self, attr: str) -> Any:
        module = self._lazy_module
        if module is None:
            module = importlib.import_module(self._lazy_name)
            self._lazy_module = module
        return getattr(module, attr)
