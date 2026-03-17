"""Tests for the centralized Backend registry and SimulationResults."""

import pytest
import numpy as np
import warnings
import tempfile
import os
from unittest.mock import patch, MagicMock

from simflux.core.backend import Backend, CORRELATION_TOLERANCE
from simflux.core.base import BaseSimulator, SimulationConfig, SimulationResults


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

class TestBackend:
    """Test Backend class methods."""

    def teardown_method(self):
        """Reset backend state between tests."""
        Backend._available = None
        Backend._rust = None

    def test_is_available_returns_bool(self):
        result = Backend.is_available()
        assert isinstance(result, bool)

    def test_is_available_cached(self):
        """Second call returns cached value without re-importing."""
        first = Backend.is_available()
        second = Backend.is_available()
        assert first == second

    def test_get_rust_returns_module_or_none(self):
        rust = Backend.get_rust()
        if Backend.is_available():
            assert rust is not None
            assert hasattr(rust, "simulate_gbm")
        else:
            assert rust is None

    def test_force_true(self):
        Backend.force(True)
        assert Backend.is_available() is True

    def test_force_false(self):
        Backend.force(False)
        assert Backend.is_available() is False

    def test_force_resets_cache(self):
        Backend.force(True)
        assert Backend.is_available() is True
        Backend.force(False)
        assert Backend.is_available() is False

    def test_correlation_tolerance_is_float(self):
        assert isinstance(CORRELATION_TOLERANCE, float)
        assert CORRELATION_TOLERANCE > 0
        assert CORRELATION_TOLERANCE == 1e-8


# ---------------------------------------------------------------------------
# SimulationResults
# ---------------------------------------------------------------------------

class TestSimulationResults:
    """Test SimulationResults lazy-loading container."""

    def test_init_without_interim(self):
        stats = {"mean": 1.0, "std": 0.5}
        results = SimulationResults(summary_stats=stats)
        assert results.summary_stats == stats
        assert results.interim_data_path is None
        assert results.has_interim_data is False

    def test_init_with_interim(self):
        results = SimulationResults(
            summary_stats={"mean": 1.0},
            interim_data_path="/tmp/results.parquet",
        )
        assert results.has_interim_data is True

    def test_get_summary(self):
        stats = {"mean": 1.0, "var_95": 2.5}
        results = SimulationResults(summary_stats=stats)
        assert results.get_summary() is stats

    def test_load_interim_data_raises_when_none(self):
        results = SimulationResults(summary_stats={})
        with pytest.raises(ValueError, match="No interim data available"):
            results.load_interim_data()

    def test_load_interim_data_with_valid_parquet(self):
        """Round-trip: write a parquet file, then load it via SimulationResults."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.parquet")
            table = pa.table({"trial_id": [0, 1], "asset_id": [10, 20]})
            pq.write_table(table, path)

            results = SimulationResults(
                summary_stats={"mean": 0.0},
                interim_data_path=path,
            )
            analyzer = results.load_interim_data()
            assert analyzer is not None
            # calling again returns cached instance
            assert results.load_interim_data() is analyzer


# ---------------------------------------------------------------------------
# SimulationConfig validation
# ---------------------------------------------------------------------------

class TestSimulationConfig:
    """Test SimulationConfig edge cases."""

    def test_default_config(self):
        cfg = SimulationConfig()
        assert cfg.seed is None
        assert cfg.batch_size == 10000
        assert cfg.memory_limit_gb is None
        assert cfg.progress_callback is None

    def test_negative_batch_size(self):
        """Both zero and negative batch_size should fail."""
        class Sim(BaseSimulator):
            def simulate(self, *a, **kw): pass
            def validate_inputs(self, *a, **kw): pass

        with pytest.raises(ValueError, match="batch_size must be positive"):
            Sim(config=SimulationConfig(batch_size=-1))

    def test_negative_memory_limit(self):
        class Sim(BaseSimulator):
            def simulate(self, *a, **kw): pass
            def validate_inputs(self, *a, **kw): pass

        with pytest.raises(ValueError, match="memory_limit_gb must be positive"):
            Sim(config=SimulationConfig(memory_limit_gb=-0.5))


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:
    """Test _report_progress helper on BaseSimulator."""

    def test_callback_invoked(self):
        calls = []
        cfg = SimulationConfig(progress_callback=lambda f: calls.append(f))

        class DummySim(BaseSimulator):
            def simulate(self, *a, **kw):
                pass
            def validate_inputs(self, *a, **kw):
                pass

        sim = DummySim(config=cfg)
        sim._report_progress(0.5)
        sim._report_progress(1.0)
        assert calls == [0.5, 1.0]

    def test_no_callback_is_noop(self):
        class DummySim(BaseSimulator):
            def simulate(self, *a, **kw):
                pass
            def validate_inputs(self, *a, **kw):
                pass

        sim = DummySim()
        sim._report_progress(0.5)  # should not raise


# ---------------------------------------------------------------------------
# Memory check
# ---------------------------------------------------------------------------

class TestMemoryCheck:
    """Test BaseSimulator._check_memory edge cases."""

    def _make_sim(self, limit_gb=None):
        class DummySim(BaseSimulator):
            def simulate(self, *a, **kw):
                pass
            def validate_inputs(self, *a, **kw):
                pass
        return DummySim(config=SimulationConfig(memory_limit_gb=limit_gb))

    def test_no_limit_passes(self):
        sim = self._make_sim(limit_gb=None)
        sim._check_memory(10**12)  # no error

    def test_within_limit_passes(self):
        sim = self._make_sim(limit_gb=1.0)
        sim._check_memory(1000)  # ~8 KB, well within 1 GB

    def test_exceeds_limit_raises(self):
        sim = self._make_sim(limit_gb=0.001)  # ~1 MB
        with pytest.raises(MemoryError, match="exceeds limit"):
            sim._check_memory(10**9)  # ~8 GB
