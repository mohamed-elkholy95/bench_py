#!/usr/bin/env python3
"""System Benchmark -- comprehensive cross-platform performance scoring.

Measures CPU (single + multi-core), GPU compute, memory bandwidth, and
storage I/O. Produces a normalized composite score (baseline = 1000).

Usage:
    python bench.py                    # full benchmark, all outputs
    python bench.py --json-only        # JSON only (for piping)
    python bench.py --quick            # fast preview (reduced iterations)
    python bench.py --skip gpu         # skip GPU tests
    python bench.py --calibrate        # print raw values for baseline
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import multiprocessing
import os
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    mx = None  # type: ignore[assignment]
    HAS_MLX = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("bench")

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class TestTimeout(Exception):
    """Benchmark iteration exceeded its timeout."""

class TestCrashed(Exception):
    """Benchmark subprocess exited abnormally."""
    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code

# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def classify_bench_error(exc: Exception) -> str:
    """Map an exception to an error category string."""
    if isinstance(exc, TestTimeout):
        return "timeout"
    if isinstance(exc, TestCrashed):
        return "crashed"
    if isinstance(exc, MemoryError):
        return "out_of_memory"
    if isinstance(exc, ImportError):
        return "missing_dependency"
    if isinstance(exc, NotImplementedError):
        return "not_supported"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, OSError):
        return "io_error"
    return "unexpected"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    name: str
    category: str
    raw_value: float
    unit: str
    score: float
    iterations: int
    warmups: int
    median_time: float
    std_dev: float
    times: List[float]
    degraded: bool = False
    resource_summary: Optional[Dict[str, Any]] = None


@dataclass
class CategoryScore:
    name: str
    score: float
    weight: float
    tests: List[BenchmarkResult]
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class BenchmarkError:
    test: str
    category: str
    error_type: str
    message: str
    suggestion: str
    retries_attempted: int = 0


@dataclass
class ReportIntegrity:
    complete: bool
    degraded_tests: List[str]
    cpu_fallback_tests: List[str]
    retried_tests: List[str]
    partial: bool
    constrained: bool


@dataclass
class ExecutionMetadata:
    phases_completed: int
    phases_total: int
    total_cooldown_seconds: float
    peak_cpu_temp_c: Optional[float]
    peak_ram_usage_mb: float
    pre_flight: Dict[str, Any]
    execution_mode: str


@dataclass
class BenchmarkReport:
    overall_score: float
    categories: List[CategoryScore]
    baseline_machine: str
    baseline_version: str
    system: Optional[Dict[str, Any]]
    skipped: List[str]
    errors: List[BenchmarkError]
    integrity: ReportIntegrity
    execution: ExecutionMetadata
    duration_seconds: float
    timestamp: str


@dataclass
class BenchConfig:
    iterations: int = 5
    warmups: int = 3
    test_timeout: int = 30
    timeout: int = 60
    skip_categories: List[str] = field(default_factory=list)
    only_categories: List[str] = field(default_factory=list)
    quick: bool = False
    no_cooldown: bool = False
    calibrate: bool = False
    json_only: bool = False
    no_color: bool = False
    verbose: bool = False
    output_dir: str = "."
    system_report_path: str = "./system_report.json"


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def _clean_none(d: Any) -> Any:
    """Recursively remove None values from dicts for clean JSON."""
    if isinstance(d, dict):
        return {k: _clean_none(v) for k, v in d.items() if v is not None}
    if isinstance(d, list):
        return [_clean_none(i) for i in d]
    return d


def report_to_dict(report: BenchmarkReport) -> Dict[str, Any]:
    """Convert a BenchmarkReport to a JSON-friendly dict with None values removed."""
    return _clean_none(asdict(report))

# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

def compute_test_score(raw_value: float, baseline_value: float) -> float:
    """Normalize a raw measurement against the baseline.
    Returns a score where 1000 = baseline machine.
    """
    if baseline_value <= 0:
        return 0.0
    return (raw_value / baseline_value) * 1000.0


def geometric_mean(scores: List[float]) -> float:
    """Geometric mean of scores. Returns 0.0 if empty or any score <= 0."""
    if not scores or any(s <= 0 for s in scores):
        return 0.0
    log_sum = sum(math.log(s) for s in scores)
    return math.exp(log_sum / len(scores))


def compute_median(times: List[float]) -> float:
    """Median of a list of floats."""
    s = sorted(times)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def redistribute_weights(categories: List[CategoryScore]) -> Dict[str, float]:
    """Redistribute weights proportionally among active categories."""
    active = [c for c in categories if not c.skipped]
    total_active_weight = sum(c.weight for c in active)
    if total_active_weight <= 0:
        return {}
    return {c.name: c.weight / total_active_weight for c in active}


def compute_overall_score(categories: List[CategoryScore]) -> float:
    """Weighted geometric mean of category scores."""
    weights = redistribute_weights(categories)
    active = [c for c in categories if not c.skipped and c.score > 0]
    if not active:
        return 0.0
    log_sum = sum(weights[c.name] * math.log(c.score) for c in active)
    total_weight = sum(weights[c.name] for c in active)
    if total_weight <= 0:
        return 0.0
    return math.exp(log_sum / total_weight)

# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    max_retries: int = 2
    backoff_seconds: float = 1.0
    _retry_types: Tuple[type, ...] = (TestTimeout, TestCrashed)
    _no_retry_types: Tuple[type, ...] = (
        ImportError, NotImplementedError, PermissionError,
    )

    def should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, self._no_retry_types):
            return False
        return isinstance(exc, self._retry_types) or isinstance(exc, MemoryError)


# ---------------------------------------------------------------------------
# Subprocess target (module-level for pickling)
# ---------------------------------------------------------------------------

def _subprocess_target(fn: Callable, args: tuple, result_queue: multiprocessing.Queue) -> None:
    """Target function for multiprocessing.Process. Must be module-level."""
    try:
        value = fn(*args)
        result_queue.put(("ok", value))
    except Exception as exc:
        result_queue.put(("error", repr(exc)))


# ---------------------------------------------------------------------------
# TestExecutor
# ---------------------------------------------------------------------------

class TestExecutor:
    """Run a single benchmark function in an isolated subprocess."""

    def __init__(self) -> None:
        self._active_process: Optional[multiprocessing.Process] = None

    def run_single(self, fn: Callable, args: tuple = (), timeout: int = 30) -> float:
        """Run fn(*args) in a subprocess. Returns the float result.
        Raises TestTimeout or TestCrashed on failure.
        """
        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_subprocess_target,
            args=(fn, args, result_queue),
        )
        self._active_process = proc
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
                proc.join()
            self._active_process = None
            raise TestTimeout(f"Benchmark timed out after {timeout}s")

        self._active_process = None

        if proc.exitcode != 0:
            raise TestCrashed(
                f"Benchmark subprocess exited with code {proc.exitcode}",
                proc.exitcode or -1,
            )

        if result_queue.empty():
            raise TestCrashed("Benchmark returned no result", -1)

        status, value = result_queue.get_nowait()
        if status == "error":
            raise TestCrashed(f"Benchmark raised: {value}", 1)

        return value

    def kill_active(self) -> None:
        """Kill the currently running subprocess, if any."""
        if self._active_process and self._active_process.is_alive():
            self._active_process.terminate()
            self._active_process.join(timeout=3)
            if self._active_process.is_alive():
                self._active_process.kill()
                self._active_process.join()
