"""Tests for bench.py -- system benchmark suite."""
import json
import math
import os
import tempfile
import time
import pytest


# ---------------------------------------------------------------------------
# Dependency detection
# ---------------------------------------------------------------------------

def test_numpy_is_available():
    """bench.py requires numpy -- verify it imports."""
    from bench import HAS_NUMPY
    assert HAS_NUMPY is True


def test_mlx_detection():
    """HAS_MLX is a bool reflecting whether mlx is installed."""
    from bench import HAS_MLX
    assert isinstance(HAS_MLX, bool)


def test_psutil_detection():
    """HAS_PSUTIL is a bool reflecting whether psutil is installed."""
    from bench import HAS_PSUTIL
    assert isinstance(HAS_PSUTIL, bool)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

def test_test_timeout_exception():
    from bench import TestTimeout
    exc = TestTimeout("too slow")
    assert str(exc) == "too slow"


def test_test_crashed_exception():
    from bench import TestCrashed
    exc = TestCrashed("segfault", -11)
    assert exc.exit_code == -11


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def test_classify_bench_error():
    from bench import (
        classify_bench_error, TestTimeout, TestCrashed,
    )
    assert classify_bench_error(TestTimeout("x")) == "timeout"
    assert classify_bench_error(TestCrashed("x", 1)) == "crashed"
    assert classify_bench_error(MemoryError("x")) == "out_of_memory"
    assert classify_bench_error(ImportError("x")) == "missing_dependency"
    assert classify_bench_error(NotImplementedError("x")) == "not_supported"
    assert classify_bench_error(PermissionError("x")) == "permission_denied"
    assert classify_bench_error(OSError("x")) == "io_error"
    assert classify_bench_error(RuntimeError("x")) == "unexpected"
