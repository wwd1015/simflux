"""SimFlux exception hierarchy.

Every error SimFlux raises deliberately derives from :class:`SimfluxError`,
so callers can catch "anything SimFlux" with one clause — while each concrete
type also subclasses the builtin it historically was (``ValueError``,
``RuntimeError``, ``MemoryError``), so existing ``except ValueError`` /
``except RuntimeError`` code keeps working unchanged.

- :class:`ValidationError` — invalid model or run parameters (bad correlation
  matrix, negative sigma, inconsistent term structures, out-of-range
  probabilities). Raised at the validation seams before any simulation work.
- :class:`BackendError` — the backend seam misbehaved: the Rust extension
  returned an unexpected shape, or failed in a way the Python layer wraps
  with context. Indicates a bug or an environment problem, not bad inputs.
- :class:`MemoryLimitError` — the configured ``memory_limit_gb`` guard
  tripped *before* allocation; reduce the run size or raise the limit.
"""


class SimfluxError(Exception):
    """Base class for all errors SimFlux raises deliberately."""


class ValidationError(SimfluxError, ValueError):
    """Invalid model or run parameters, rejected at a validation seam."""


class BackendError(SimfluxError, RuntimeError):
    """The simulation backend violated its contract or failed internally."""


class MemoryLimitError(SimfluxError, MemoryError):
    """The configured ``memory_limit_gb`` guard rejected the run up front."""
