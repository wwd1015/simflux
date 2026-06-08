"""Centralized Rust backend detection and access."""

from typing import Optional, TypeVar

# Shared tolerance for correlation matrix validation across Python and Rust.
CORRELATION_TOLERANCE = 1e-8

T = TypeVar("T")


class Backend:
    """Registry for the Rust extension module.

    All modules should query ``Backend.is_available()`` instead of maintaining
    their own ``RUST_AVAILABLE`` flag.  Use ``Backend.get_rust()`` to obtain a
    reference to the ``_rust`` extension.
    """

    _rust = None
    _available: Optional[bool] = None

    @classmethod
    def is_available(cls) -> bool:
        if cls._available is None:
            try:
                from simflux import _rust  # type: ignore[attr-defined]

                cls._rust = _rust
                cls._available = True
            except ImportError:
                cls._available = False
        return cls._available

    @classmethod
    def get_rust(cls):
        """Return the ``_rust`` extension module, or ``None``."""
        cls.is_available()
        return cls._rust

    @classmethod
    def choose(cls, rust: T, numpy: T) -> T:
        """Return ``rust`` when the Rust extension is active, else ``numpy``.

        The single backend-dispatch decision.  Callers pass the two adapters for
        one operation (a bound method, callable, or object) and receive the active
        one, instead of open-coding ``if Backend.is_available()`` at each call
        site.  Resolution is per call, so patching ``is_available`` (as the tests
        do) still flips the backend after construction.
        """
        return rust if cls.is_available() else numpy

    @classmethod
    def force(cls, use_rust: bool) -> None:
        """Override backend availability (useful for testing)."""
        cls._available = use_rust
