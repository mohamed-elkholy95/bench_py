# System Benchmark Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `bench.py` -- a single-file benchmark tool that measures CPU, GPU, memory, and storage performance with Geekbench-style normalized scoring.

**Architecture:** Registry of benchmark functions (mirrors fetch.py's COLLECTORS pattern), phased sequential execution with cooldowns, subprocess isolation per test, weighted geometric mean scoring normalized to a baseline of 1000. Reads `system_report.json` from fetch.py as optional metadata.

**Tech Stack:** Python 3.12+, numpy (required), mlx (optional -- Apple Silicon GPU), psutil (optional -- resource monitoring)

**Spec:** `docs/superpowers/specs/2026-04-13-system-benchmark-design.md`

**Note on file size:** The spec mandates a single file. This plan follows that. If the file grows unwieldy during implementation, the orchestrator/benchmark-functions boundary is the natural split point for a future refactor.

---

## File Structure

Single file plus test file:

- **Create:** `bench.py` -- the benchmark tool
- **Create:** `test_bench.py` -- unit + integration tests

---

### Task 1: Foundation -- Imports, Constants, Dependency Checks, Exceptions

**Files:**
- Create: `bench.py` (lines 1-120 approx)
- Create: `test_bench.py`

- [ ] **Step 1: Write failing tests for dependency detection and custom exceptions**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest test_bench.py -v --tb=short 2>&1 | head -40`
Expected: All FAIL with `ModuleNotFoundError: No module named 'bench'`

- [ ] **Step 3: Write the foundation of bench.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test_bench.py -v --tb=short`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bench.py test_bench.py
git commit -m "feat(bench): foundation -- imports, dependency checks, exceptions"
```

---

### Task 2: Dataclasses

**Files:**
- Modify: `bench.py` (append after error classification)
- Modify: `test_bench.py` (append tests)

- [ ] **Step 1: Write failing tests for dataclasses and JSON serialization**

Append to `test_bench.py`:

```python
# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

def test_benchmark_result_defaults():
    from bench import BenchmarkResult
    r = BenchmarkResult(
        name="test", category="cpu_single", raw_value=100.0,
        unit="ops/sec", score=1000.0, iterations=5, warmups=3,
        median_time=0.01, std_dev=0.001, times=[0.01, 0.01, 0.01, 0.01, 0.01],
    )
    assert r.degraded is False
    assert r.resource_summary is None


def test_category_score_defaults():
    from bench import CategoryScore
    c = CategoryScore(name="cpu_single", score=1000.0, weight=0.25, tests=[])
    assert c.skipped is False
    assert c.skip_reason is None


def test_benchmark_error_fields():
    from bench import BenchmarkError
    e = BenchmarkError(
        test="gpu_matrix", category="gpu", error_type="missing_dependency",
        message="MLX not available", suggestion="pip install mlx",
    )
    assert e.retries_attempted == 0


def test_report_integrity_defaults():
    from bench import ReportIntegrity
    ri = ReportIntegrity(
        complete=True, degraded_tests=[], cpu_fallback_tests=[],
        retried_tests=[], partial=False, constrained=False,
    )
    assert ri.complete is True


def test_benchmark_report_json_roundtrip():
    from bench import (
        BenchmarkReport, CategoryScore, BenchmarkResult,
        ReportIntegrity, ExecutionMetadata, report_to_dict,
    )
    result = BenchmarkResult(
        name="prime_sieve", category="cpu_single", raw_value=823.5,
        unit="ops/sec", score=1000.0, iterations=5, warmups=3,
        median_time=0.00121, std_dev=0.00003,
        times=[0.00121, 0.00122, 0.00120, 0.00121, 0.00123],
    )
    cat = CategoryScore(
        name="cpu_single", score=1000.0, weight=0.25, tests=[result],
    )
    integrity = ReportIntegrity(
        complete=True, degraded_tests=[], cpu_fallback_tests=[],
        retried_tests=[], partial=False, constrained=False,
    )
    execution = ExecutionMetadata(
        phases_completed=8, phases_total=8, total_cooldown_seconds=12.0,
        peak_cpu_temp_c=78.0, peak_ram_usage_mb=4200.0,
        pre_flight={}, execution_mode="full",
    )
    report = BenchmarkReport(
        overall_score=1000.0, categories=[cat],
        baseline_machine="Apple M4 Max / 36GB / macOS",
        baseline_version="1.0", system=None, skipped=[],
        errors=[], integrity=integrity, execution=execution,
        duration_seconds=45.0, timestamp="2026-04-13T20:00:00Z",
    )
    d = report_to_dict(report)
    json_str = json.dumps(d)
    loaded = json.loads(json_str)
    assert loaded["overall_score"] == 1000.0
    assert loaded["categories"][0]["name"] == "cpu_single"
    assert loaded["categories"][0]["tests"][0]["name"] == "prime_sieve"
    assert "system" not in loaded  # None values stripped
    assert loaded["integrity"]["complete"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest test_bench.py::test_benchmark_result_defaults -v --tb=short`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the dataclasses and report_to_dict**

Append to `bench.py` after the error classification section:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test_bench.py -v --tb=short`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bench.py test_bench.py
git commit -m "feat(bench): dataclasses and JSON serialization"
```

---

### Task 3: Scoring Engine

Pure math, no hardware dependency. Highly testable.

**Files:**
- Modify: `bench.py` (append after JSON serialization)
- Modify: `test_bench.py` (append tests)

- [ ] **Step 1: Write failing tests for scoring functions**

Append to `test_bench.py`:

```python
# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

def test_compute_test_score_baseline_equals_1000():
    from bench import compute_test_score
    assert compute_test_score(100.0, 100.0) == 1000.0


def test_compute_test_score_double_is_2000():
    from bench import compute_test_score
    assert compute_test_score(200.0, 100.0) == 2000.0


def test_compute_test_score_half_is_500():
    from bench import compute_test_score
    assert compute_test_score(50.0, 100.0) == 500.0


def test_compute_test_score_zero_baseline():
    from bench import compute_test_score
    assert compute_test_score(100.0, 0.0) == 0.0


def test_geometric_mean_identical():
    from bench import geometric_mean
    assert geometric_mean([1000.0, 1000.0, 1000.0]) == 1000.0


def test_geometric_mean_known_value():
    from bench import geometric_mean
    # geometric_mean([100, 10000]) = sqrt(100 * 10000) = 1000
    result = geometric_mean([100.0, 10000.0])
    assert abs(result - 1000.0) < 0.01


def test_geometric_mean_empty():
    from bench import geometric_mean
    assert geometric_mean([]) == 0.0


def test_geometric_mean_with_zero():
    from bench import geometric_mean
    assert geometric_mean([0.0, 1000.0]) == 0.0


def test_compute_median_odd():
    from bench import compute_median
    assert compute_median([3.0, 1.0, 2.0]) == 2.0


def test_compute_median_even():
    from bench import compute_median
    assert compute_median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_redistribute_weights_no_skip():
    from bench import redistribute_weights, CategoryScore
    cats = [
        CategoryScore(name="cpu_single", score=1000, weight=0.25, tests=[]),
        CategoryScore(name="cpu_multi", score=1000, weight=0.25, tests=[]),
        CategoryScore(name="gpu", score=1000, weight=0.20, tests=[]),
        CategoryScore(name="memory", score=1000, weight=0.15, tests=[]),
        CategoryScore(name="storage", score=1000, weight=0.15, tests=[]),
    ]
    weights = redistribute_weights(cats)
    assert abs(weights["cpu_single"] - 0.25) < 0.001
    assert abs(sum(weights.values()) - 1.0) < 0.001


def test_redistribute_weights_gpu_skipped():
    from bench import redistribute_weights, CategoryScore
    cats = [
        CategoryScore(name="cpu_single", score=1000, weight=0.25, tests=[]),
        CategoryScore(name="cpu_multi", score=1000, weight=0.25, tests=[]),
        CategoryScore(name="gpu", score=0, weight=0.20, tests=[], skipped=True),
        CategoryScore(name="memory", score=1000, weight=0.15, tests=[]),
        CategoryScore(name="storage", score=1000, weight=0.15, tests=[]),
    ]
    weights = redistribute_weights(cats)
    assert "gpu" not in weights
    assert abs(sum(weights.values()) - 1.0) < 0.001
    assert abs(weights["cpu_single"] - 0.3125) < 0.001


def test_overall_score_all_baseline():
    from bench import compute_overall_score, CategoryScore
    cats = [
        CategoryScore(name="cpu_single", score=1000, weight=0.25, tests=[]),
        CategoryScore(name="cpu_multi", score=1000, weight=0.25, tests=[]),
        CategoryScore(name="gpu", score=1000, weight=0.20, tests=[]),
        CategoryScore(name="memory", score=1000, weight=0.15, tests=[]),
        CategoryScore(name="storage", score=1000, weight=0.15, tests=[]),
    ]
    result = compute_overall_score(cats)
    assert abs(result - 1000.0) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest test_bench.py::test_compute_test_score_baseline_equals_1000 -v --tb=short`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the scoring engine**

Append to `bench.py` after JSON serialization:

```python
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


def redistribute_weights(
    categories: List[CategoryScore],
) -> Dict[str, float]:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test_bench.py -v --tb=short`
Expected: All 27 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bench.py test_bench.py
git commit -m "feat(bench): scoring engine -- normalization, geometric mean, weights"
```

---

### Task 4: TestExecutor -- Subprocess Isolation and Retry

**Files:**
- Modify: `bench.py` (append after scoring engine)
- Modify: `test_bench.py` (append tests)

**Key design note:** `multiprocessing.Process` requires picklable callables. All benchmark functions are module-level in `bench.py`, so they pickle fine. The executor uses a module-level `_subprocess_target` function (not a lambda or closure) to stay picklable.

- [ ] **Step 1: Write failing tests for RetryPolicy and TestExecutor**

Append to `test_bench.py`:

```python
# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------

def test_retry_policy_defaults():
    from bench import RetryPolicy
    rp = RetryPolicy()
    assert rp.max_retries == 2
    assert rp.backoff_seconds == 1.0


def test_retry_policy_should_retry_timeout():
    from bench import RetryPolicy, TestTimeout
    rp = RetryPolicy()
    assert rp.should_retry(TestTimeout("x"))


def test_retry_policy_should_not_retry_import():
    from bench import RetryPolicy
    rp = RetryPolicy()
    assert not rp.should_retry(ImportError("x"))
```

Append test helper functions and executor tests. These helpers must be module-level (not nested) for pickling:

```python
def _dummy_bench_success(size: int) -> float:
    """A trivial benchmark that returns a known throughput."""
    return 42.0


def test_executor_runs_function():
    """TestExecutor runs a function and returns its result."""
    from bench import TestExecutor
    executor = TestExecutor()
    result = executor.run_single(_dummy_bench_success, (100,), timeout=10)
    assert result == 42.0


def _dummy_bench_crash(size: int) -> float:
    """A benchmark that crashes."""
    raise RuntimeError("boom")


def test_executor_handles_crash():
    """TestExecutor raises TestCrashed when subprocess errors."""
    from bench import TestExecutor, TestCrashed
    executor = TestExecutor()
    with pytest.raises(TestCrashed):
        executor.run_single(_dummy_bench_crash, (100,), timeout=10)


def _dummy_bench_slow(size: int) -> float:
    """A benchmark that takes too long."""
    import time as _time
    _time.sleep(30)
    return 0.0


def test_executor_handles_timeout():
    """TestExecutor raises TestTimeout when function exceeds timeout."""
    from bench import TestExecutor, TestTimeout
    executor = TestExecutor()
    with pytest.raises(TestTimeout):
        executor.run_single(_dummy_bench_slow, (100,), timeout=2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest test_bench.py::test_retry_policy_defaults -v --tb=short`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write RetryPolicy and TestExecutor**

Append to `bench.py` after scoring engine:

```python
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

def _subprocess_target(
    fn: Callable,
    args: tuple,
    result_queue: multiprocessing.Queue,
) -> None:
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

    def run_single(
        self,
        fn: Callable,
        args: tuple = (),
        timeout: int = 30,
    ) -> float:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test_bench.py -v --tb=short`
Expected: All 33 tests PASS (the timeout test may take ~2s)

- [ ] **Step 5: Commit**

```bash
git add bench.py test_bench.py
git commit -m "feat(bench): TestExecutor with subprocess isolation and retry policy"
```

---

### Task 5: SystemProbe, ResourceGuard, CooldownManager

**Files:**
- Modify: `bench.py` (append after TestExecutor)
- Modify: `test_bench.py` (append tests)

- [ ] **Step 1: Write failing tests for monitoring infrastructure**

Append to `test_bench.py`:

```python
# ---------------------------------------------------------------------------
# SystemProbe
# ---------------------------------------------------------------------------

def test_system_probe_returns_readiness():
    from bench import SystemProbe, SystemReadiness
    probe = SystemProbe()
    readiness = probe.check()
    assert isinstance(readiness, SystemReadiness)
    assert isinstance(readiness.ready, bool)
    assert readiness.available_ram_gb > 0
    assert isinstance(readiness.warnings, list)
    assert isinstance(readiness.blockers, list)


# ---------------------------------------------------------------------------
# CooldownManager
# ---------------------------------------------------------------------------

def test_cooldown_policy_defaults():
    from bench import CooldownPolicy
    p = CooldownPolicy()
    assert p.min_seconds == 3.0
    assert p.max_seconds == 30.0
    assert p.target_cpu_pct == 10.0


def test_cooldown_manager_respects_no_cooldown():
    from bench import CooldownManager, CooldownPolicy
    mgr = CooldownManager()
    policy = CooldownPolicy(min_seconds=0, max_seconds=0)
    result = mgr.wait(policy)
    assert result["waited_seconds"] >= 0


# ---------------------------------------------------------------------------
# ResourceGuard
# ---------------------------------------------------------------------------

def test_resource_guard_start_stop():
    from bench import ResourceGuard
    guard = ResourceGuard()
    guard.start()
    time.sleep(0.6)  # let it collect at least 1 sample
    summary = guard.stop()
    assert isinstance(summary, dict)
    assert "peak_cpu_pct" in summary or summary == {}  # empty if no psutil
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest test_bench.py::test_system_probe_returns_readiness -v --tb=short`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write SystemProbe, CooldownManager, ResourceGuard**

Append to `bench.py` after TestExecutor:

```python
# ---------------------------------------------------------------------------
# SystemProbe -- pre-flight checks
# ---------------------------------------------------------------------------

@dataclass
class SystemReadiness:
    cpu_idle_pct: float = 0.0
    available_ram_gb: float = 0.0
    disk_free_gb: float = 0.0
    thermal_state: str = "unknown"
    battery_plugged: bool = True
    background_load: List[str] = field(default_factory=list)
    ready: bool = True
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)


class SystemProbe:
    """Pre-flight system readiness checks."""

    def check(self) -> SystemReadiness:
        sr = SystemReadiness()

        # RAM check
        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                sr.available_ram_gb = round(mem.available / (1024 ** 3), 1)
            except Exception:
                sr.available_ram_gb = 0.0
        else:
            sr.available_ram_gb = 999.0  # assume OK if we cannot check
            sr.warnings.append("psutil not installed -- cannot check RAM")

        if sr.available_ram_gb < 2.0 and HAS_PSUTIL:
            sr.blockers.append(
                f"Available RAM too low: {sr.available_ram_gb} GB (need >2 GB)"
            )

        # CPU idle check
        if HAS_PSUTIL:
            try:
                cpu_pct = psutil.cpu_percent(interval=1)
                sr.cpu_idle_pct = round(100.0 - cpu_pct, 1)
                if sr.cpu_idle_pct < 80:
                    sr.warnings.append(
                        f"CPU only {sr.cpu_idle_pct}% idle -- results may vary"
                    )
            except Exception:
                pass

        # Disk free check
        try:
            st = os.statvfs(".")
            sr.disk_free_gb = round(st.f_bavail * st.f_frsize / (1024 ** 3), 1)
        except (OSError, AttributeError):
            sr.disk_free_gb = 999.0  # assume OK on Windows
        if sr.disk_free_gb < 1.0:
            sr.warnings.append(
                f"Disk free space low: {sr.disk_free_gb} GB -- storage tests may fail"
            )

        # Thermal state (macOS only)
        if platform.system() == "Darwin":
            try:
                raw = subprocess.run(
                    ["pmset", "-g", "therm"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
                if "CPU_Scheduler_Limit" in raw:
                    for line in raw.split("\n"):
                        if "CPU_Scheduler_Limit" in line:
                            val = line.split("=")[-1].strip()
                            if val != "100":
                                sr.thermal_state = "throttled"
                                sr.warnings.append("CPU is thermally throttled")
                            else:
                                sr.thermal_state = "nominal"
                            break
                else:
                    sr.thermal_state = "nominal"
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                sr.thermal_state = "unknown"

        # Battery check
        if HAS_PSUTIL:
            try:
                bat = psutil.sensors_battery()
                if bat and not bat.power_plugged:
                    sr.battery_plugged = False
                    sr.warnings.append("On battery -- performance may be throttled")
            except Exception:
                pass

        sr.ready = len(sr.blockers) == 0
        return sr


# ---------------------------------------------------------------------------
# CooldownManager
# ---------------------------------------------------------------------------

@dataclass
class CooldownPolicy:
    min_seconds: float = 3.0
    max_seconds: float = 30.0
    target_cpu_pct: float = 10.0
    target_temp_c: float = 70.0
    poll_interval: float = 1.0


class CooldownManager:
    """Wait for system to cool down between benchmark phases."""

    def wait(self, policy: CooldownPolicy) -> Dict[str, Any]:
        """Block until cooldown targets met or max_seconds exceeded."""
        start = time.monotonic()
        waited = 0.0

        if policy.max_seconds <= 0:
            return {"waited_seconds": 0.0}

        gc.collect()

        while waited < policy.max_seconds:
            if waited >= policy.min_seconds:
                # Check if we have cooled down enough
                if HAS_PSUTIL:
                    try:
                        cpu = psutil.cpu_percent(interval=0.1)
                        if cpu <= policy.target_cpu_pct:
                            break
                    except Exception:
                        break
                else:
                    break  # cannot monitor, just wait min_seconds

            time.sleep(policy.poll_interval)
            waited = time.monotonic() - start

        return {"waited_seconds": round(waited, 1)}


# ---------------------------------------------------------------------------
# ResourceGuard -- runtime monitoring
# ---------------------------------------------------------------------------

class ResourceGuard:
    """Monitor system resources during benchmark execution.

    Runs in a daemon thread, sampling at 500ms intervals.
    """

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: List[Dict[str, Any]] = []

    def start(self) -> None:
        if not HAS_PSUTIL:
            return
        self._stop_event.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        if not HAS_PSUTIL or self._thread is None:
            return {}
        self._stop_event.set()
        self._thread.join(timeout=2)
        self._thread = None

        if not self._samples:
            return {}

        return {
            "peak_cpu_pct": max(s["cpu_pct"] for s in self._samples),
            "peak_ram_used_mb": max(s["ram_used_mb"] for s in self._samples),
            "min_ram_available_mb": min(s["ram_available_mb"] for s in self._samples),
            "samples_count": len(self._samples),
        }

    def _monitor(self) -> None:
        while not self._stop_event.is_set():
            try:
                cpu = psutil.cpu_percent(interval=0)
                mem = psutil.virtual_memory()
                self._samples.append({
                    "timestamp": time.monotonic(),
                    "cpu_pct": cpu,
                    "ram_used_mb": round(mem.used / (1024 ** 2)),
                    "ram_available_mb": round(mem.available / (1024 ** 2)),
                })
            except Exception:
                pass
            self._stop_event.wait(timeout=0.5)

    def check_critical(self) -> Optional[str]:
        """Return a warning string if resources are critically low, else None."""
        if not HAS_PSUTIL:
            return None
        try:
            mem = psutil.virtual_memory()
            avail_mb = mem.available / (1024 ** 2)
            if avail_mb < 512:
                return f"RAM critically low: {avail_mb:.0f} MB available"
        except Exception:
            pass
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test_bench.py -v --tb=short`
Expected: All 37 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bench.py test_bench.py
git commit -m "feat(bench): SystemProbe, CooldownManager, ResourceGuard"
```

---

### Task 6: CPU Single-Core Benchmarks

**Files:**
- Modify: `bench.py` (append after ResourceGuard)
- Modify: `test_bench.py` (append tests)

- [ ] **Step 1: Write failing tests for CPU single-core benchmarks**

Tests validate structure (returns float > 0), not absolute values.

Append to `test_bench.py`:

```python
# ---------------------------------------------------------------------------
# CPU Single-Core benchmarks
# ---------------------------------------------------------------------------

def test_bench_prime_sieve():
    from bench import bench_prime_sieve
    result = bench_prime_sieve(100_000)  # smaller N for test speed
    assert isinstance(result, float)
    assert result > 0  # ops/sec


def test_bench_mandelbrot():
    from bench import bench_mandelbrot
    result = bench_mandelbrot(128)  # small grid for test speed
    assert isinstance(result, float)
    assert result > 0  # pixels/sec


def test_bench_matrix_single():
    from bench import bench_matrix_single
    result = bench_matrix_single(256)  # small matrix for test speed
    assert isinstance(result, float)
    assert result > 0  # GFLOPS


def test_bench_compression():
    from bench import bench_compression
    result = bench_compression(1)  # 1 MB for test speed
    assert isinstance(result, float)
    assert result > 0  # MB/s


def test_bench_sort():
    from bench import bench_sort
    result = bench_sort(100_000)  # small array for test speed
    assert isinstance(result, float)
    assert result > 0  # M_elements/sec
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest test_bench.py::test_bench_prime_sieve -v --tb=short`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the CPU single-core benchmark functions**

Append to `bench.py` after ResourceGuard:

```python
# ---------------------------------------------------------------------------
# CPU Single-Core benchmark functions
# ---------------------------------------------------------------------------

def bench_prime_sieve(n: int = 1_000_000) -> float:
    """Sieve of Eratosthenes. Returns ops/sec (one full sieve = one op)."""
    start = time.perf_counter()
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    return 1.0 / elapsed


def bench_mandelbrot(grid_size: int = 1024) -> float:
    """Mandelbrot set computation. Returns pixels/sec."""
    start = time.perf_counter()
    total_pixels = grid_size * grid_size
    max_iter = 100
    count = 0
    for py in range(grid_size):
        for px in range(grid_size):
            x0 = (px / grid_size) * 3.5 - 2.5
            y0 = (py / grid_size) * 2.0 - 1.0
            x, y = 0.0, 0.0
            iteration = 0
            while x * x + y * y <= 4.0 and iteration < max_iter:
                x, y = x * x - y * y + x0, 2.0 * x * y + y0
                iteration += 1
            count += 1
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    return total_pixels / elapsed


def bench_matrix_single(size: int = 1024) -> float:
    """NumPy matmul with OMP_NUM_THREADS=1. Returns GFLOPS."""
    old_env = {}
    thread_vars = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    for key in thread_vars:
        old_env[key] = os.environ.get(key)
        os.environ[key] = "1"
    try:
        a = np.random.randn(size, size).astype(np.float64)
        b = np.random.randn(size, size).astype(np.float64)
        start = time.perf_counter()
        np.dot(a, b)
        elapsed = time.perf_counter() - start
    finally:
        for key in thread_vars:
            if old_env[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_env[key]
    if elapsed <= 0:
        return 0.0
    flops = 2.0 * size * size * size
    return (flops / elapsed) / 1e9


def bench_compression(size_mb: int = 10) -> float:
    """zlib compress + decompress. Returns MB/s."""
    data = os.urandom(size_mb * 1024 * 1024)
    start = time.perf_counter()
    compressed = zlib.compress(data, level=6)
    zlib.decompress(compressed)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    return size_mb / elapsed


def bench_sort(n: int = 10_000_000) -> float:
    """Sort random integers. Returns M_elements/sec."""
    import random
    data = [random.randint(0, 2**31) for _ in range(n)]
    start = time.perf_counter()
    sorted(data)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    return (n / elapsed) / 1e6
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test_bench.py -v --tb=short`
Expected: All 42 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bench.py test_bench.py
git commit -m "feat(bench): CPU single-core benchmarks -- sieve, mandelbrot, matrix, compression, sort"
```

---

### Task 7: CPU Multi-Core Benchmarks

**Files:**
- Modify: `bench.py` (append after CPU single-core)
- Modify: `test_bench.py` (append tests)

- [ ] **Step 1: Write failing tests**

Append to `test_bench.py`:

```python
# ---------------------------------------------------------------------------
# CPU Multi-Core benchmarks
# ---------------------------------------------------------------------------

def test_bench_matrix_multi():
    from bench import bench_matrix_multi
    result = bench_matrix_multi(256)
    assert isinstance(result, float)
    assert result > 0


def test_bench_parallel_compute():
    from bench import bench_parallel_compute
    result = bench_parallel_compute(128)  # small grid
    assert isinstance(result, float)
    assert result > 0


def test_bench_hash_throughput():
    from bench import bench_hash_throughput
    result = bench_hash_throughput(1)  # 1 MB
    assert isinstance(result, float)
    assert result > 0


def test_bench_parallel_sort():
    from bench import bench_parallel_sort
    result = bench_parallel_sort(100_000)
    assert isinstance(result, float)
    assert result > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest test_bench.py::test_bench_matrix_multi -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Write CPU multi-core benchmarks**

Append to `bench.py`:

```python
# ---------------------------------------------------------------------------
# CPU Multi-Core benchmark functions
# ---------------------------------------------------------------------------

def bench_matrix_multi(size: int = 4096) -> float:
    """NumPy matmul with all threads. Returns GFLOPS."""
    a = np.random.randn(size, size).astype(np.float64)
    b = np.random.randn(size, size).astype(np.float64)
    start = time.perf_counter()
    np.dot(a, b)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    flops = 2.0 * size * size * size
    return (flops / elapsed) / 1e9


def _mandelbrot_chunk(args: Tuple[int, int, int, int]) -> int:
    """Compute a horizontal slice of the Mandelbrot set. Module-level for pickling."""
    grid_size, max_iter, y_start, y_end = args
    count = 0
    for py in range(y_start, y_end):
        for px in range(grid_size):
            x0 = (px / grid_size) * 3.5 - 2.5
            y0 = (py / grid_size) * 2.0 - 1.0
            x, y = 0.0, 0.0
            iteration = 0
            while x * x + y * y <= 4.0 and iteration < max_iter:
                x, y = x * x - y * y + x0, 2.0 * x * y + y0
                iteration += 1
            count += 1
    return count


def bench_parallel_compute(grid_size: int = 1024) -> float:
    """Parallel Mandelbrot. Returns pixels/sec."""
    max_iter = 100
    n_workers = os.cpu_count() or 4
    total_pixels = grid_size * grid_size
    chunk_size = grid_size // n_workers
    chunks = []
    for i in range(n_workers):
        y_start = i * chunk_size
        y_end = grid_size if i == n_workers - 1 else (i + 1) * chunk_size
        chunks.append((grid_size, max_iter, y_start, y_end))

    start = time.perf_counter()
    with multiprocessing.Pool(n_workers) as pool:
        pool.map(_mandelbrot_chunk, chunks)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    return total_pixels / elapsed


def _hash_chunk(data: bytes) -> bytes:
    """SHA-256 hash a chunk. Module-level for pickling."""
    return hashlib.sha256(data).digest()


def bench_hash_throughput(size_mb: int = 100) -> float:
    """Parallel SHA-256 hashing. Returns MB/s."""
    n_workers = os.cpu_count() or 4
    chunk_size = max(1, size_mb // n_workers) * 1024 * 1024
    chunks = [os.urandom(chunk_size) for _ in range(n_workers)]

    start = time.perf_counter()
    with multiprocessing.Pool(n_workers) as pool:
        pool.map(_hash_chunk, chunks)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    total_mb = (chunk_size * n_workers) / (1024 * 1024)
    return total_mb / elapsed


def _sort_chunk(data: List[int]) -> List[int]:
    """Sort a list. Module-level for pickling."""
    return sorted(data)


def bench_parallel_sort(n: int = 10_000_000) -> float:
    """Parallel sort. Returns M_elements/sec total throughput."""
    import random
    n_workers = os.cpu_count() or 4
    per_worker = n // n_workers
    chunks = [
        [random.randint(0, 2**31) for _ in range(per_worker)]
        for _ in range(n_workers)
    ]

    start = time.perf_counter()
    with multiprocessing.Pool(n_workers) as pool:
        pool.map(_sort_chunk, chunks)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    total_elements = per_worker * n_workers
    return (total_elements / elapsed) / 1e6
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test_bench.py -v --tb=short`
Expected: All 46 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bench.py test_bench.py
git commit -m "feat(bench): CPU multi-core benchmarks -- matrix, parallel compute, hash, sort"
```

---

### Task 8: GPU Compute Benchmarks

**Files:**
- Modify: `bench.py` (append after CPU multi-core)
- Modify: `test_bench.py` (append tests)

- [ ] **Step 1: Write failing tests**

Append to `test_bench.py`:

```python
# ---------------------------------------------------------------------------
# GPU benchmarks
# ---------------------------------------------------------------------------

def test_bench_gpu_matrix():
    from bench import bench_gpu_matrix, HAS_MLX
    if not HAS_MLX:
        pytest.skip("MLX not available")
    result = bench_gpu_matrix(512)
    assert isinstance(result, float)
    assert result > 0


def test_bench_gpu_elementwise():
    from bench import bench_gpu_elementwise, HAS_MLX
    if not HAS_MLX:
        pytest.skip("MLX not available")
    result = bench_gpu_elementwise(1_000_000)
    assert isinstance(result, float)
    assert result > 0


def test_bench_gpu_reduction():
    from bench import bench_gpu_reduction, HAS_MLX
    if not HAS_MLX:
        pytest.skip("MLX not available")
    result = bench_gpu_reduction(1_000_000)
    assert isinstance(result, float)
    assert result > 0


def test_bench_gpu_batch_matmul():
    from bench import bench_gpu_batch_matmul, HAS_MLX
    if not HAS_MLX:
        pytest.skip("MLX not available")
    result = bench_gpu_batch_matmul(8, 128)
    assert isinstance(result, float)
    assert result > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest test_bench.py::test_bench_gpu_matrix -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Write GPU benchmark functions**

Append to `bench.py`:

```python
# ---------------------------------------------------------------------------
# GPU Compute benchmark functions (MLX -- Apple Silicon)
# ---------------------------------------------------------------------------

def bench_gpu_matrix(size: int = 4096) -> float:
    """MLX GPU matmul. Returns GFLOPS."""
    a = mx.random.normal((size, size), dtype=mx.float32)
    b = mx.random.normal((size, size), dtype=mx.float32)
    mx.eval(a, b)  # ensure data is on GPU

    start = time.perf_counter()
    c = mx.matmul(a, b)
    mx.eval(c)  # force sync
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    flops = 2.0 * size * size * size
    return (flops / elapsed) / 1e9


def bench_gpu_elementwise(n: int = 32_000_000) -> float:
    """MLX GPU element-wise ops chain. Returns GB/s."""
    a = mx.random.normal((n,), dtype=mx.float32)
    b = mx.random.normal((n,), dtype=mx.float32)
    mx.eval(a, b)

    start = time.perf_counter()
    c = mx.exp(a + b) * a
    mx.eval(c)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    bytes_processed = n * 4 * 3  # 3 arrays, float32
    return (bytes_processed / elapsed) / 1e9


def bench_gpu_reduction(n: int = 64_000_000) -> float:
    """MLX GPU reduction (sum + mean). Returns GB/s."""
    a = mx.random.normal((n,), dtype=mx.float32)
    mx.eval(a)

    start = time.perf_counter()
    s = mx.sum(a)
    m = mx.mean(a)
    mx.eval(s, m)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    bytes_processed = n * 4 * 2  # 2 passes over array
    return (bytes_processed / elapsed) / 1e9


def bench_gpu_batch_matmul(batch: int = 64, size: int = 512) -> float:
    """MLX batched matmul. Returns GFLOPS."""
    a = mx.random.normal((batch, size, size), dtype=mx.float32)
    b = mx.random.normal((batch, size, size), dtype=mx.float32)
    mx.eval(a, b)

    start = time.perf_counter()
    c = mx.matmul(a, b)
    mx.eval(c)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    flops = batch * 2.0 * size * size * size
    return (flops / elapsed) / 1e9
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test_bench.py -v --tb=short`
Expected: All tests PASS (GPU tests skip if MLX unavailable)

- [ ] **Step 5: Commit**

```bash
git add bench.py test_bench.py
git commit -m "feat(bench): GPU compute benchmarks -- matrix, elementwise, reduction, batch matmul"
```

---

### Task 9: Memory Benchmarks

**Files:**
- Modify: `bench.py` (append after GPU benchmarks)
- Modify: `test_bench.py` (append tests)

- [ ] **Step 1: Write failing tests**

Append to `test_bench.py`:

```python
# ---------------------------------------------------------------------------
# Memory benchmarks
# ---------------------------------------------------------------------------

def test_bench_mem_seq_read():
    from bench import bench_mem_seq_read
    result = bench_mem_seq_read(16)  # 16 MB
    assert isinstance(result, float)
    assert result > 0


def test_bench_mem_seq_write():
    from bench import bench_mem_seq_write
    result = bench_mem_seq_write(16)
    assert isinstance(result, float)
    assert result > 0


def test_bench_mem_random_access():
    from bench import bench_mem_random_access
    result = bench_mem_random_access(16, 100_000)
    assert isinstance(result, float)
    assert result > 0


def test_bench_mem_copy():
    from bench import bench_mem_copy
    result = bench_mem_copy(16)
    assert isinstance(result, float)
    assert result > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest test_bench.py::test_bench_mem_seq_read -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Write memory benchmark functions**

Append to `bench.py`:

```python
# ---------------------------------------------------------------------------
# Memory benchmark functions
# ---------------------------------------------------------------------------

def bench_mem_seq_read(size_mb: int = 256) -> float:
    """Sequential read over numpy array. Returns GB/s."""
    n = size_mb * 1024 * 1024 // 8  # float64 = 8 bytes
    arr = np.random.randn(n)
    start = time.perf_counter()
    _ = np.sum(arr)  # forces sequential read
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    return (size_mb / 1024) / elapsed


def bench_mem_seq_write(size_mb: int = 256) -> float:
    """Sequential write to numpy array. Returns GB/s."""
    n = size_mb * 1024 * 1024 // 8
    arr = np.empty(n, dtype=np.float64)
    start = time.perf_counter()
    arr[:] = 1.0  # forces sequential write
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    return (size_mb / 1024) / elapsed


def bench_mem_random_access(size_mb: int = 256, accesses: int = 1_000_000) -> float:
    """Random index reads. Returns M_accesses/sec."""
    n = size_mb * 1024 * 1024 // 8
    arr = np.random.randn(n)
    indices = np.random.randint(0, n, size=accesses)
    start = time.perf_counter()
    _ = arr[indices]  # random gather
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    return (accesses / elapsed) / 1e6


def bench_mem_copy(size_mb: int = 256) -> float:
    """numpy.copy bandwidth. Returns GB/s."""
    n = size_mb * 1024 * 1024 // 8
    arr = np.random.randn(n)
    start = time.perf_counter()
    _ = np.copy(arr)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    return (size_mb / 1024) / elapsed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test_bench.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add bench.py test_bench.py
git commit -m "feat(bench): memory benchmarks -- seq read/write, random access, copy"
```

---

### Task 10: Storage I/O Benchmarks

**Files:**
- Modify: `bench.py` (append after memory benchmarks)
- Modify: `test_bench.py` (append tests)

**Note:** macOS does not have `os.posix_fadvise`. Use `fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)` on macOS. On Linux use `os.posix_fadvise`. On Windows, neither is available.

- [ ] **Step 1: Write failing tests**

Append to `test_bench.py`:

```python
# ---------------------------------------------------------------------------
# Storage I/O benchmarks
# ---------------------------------------------------------------------------

def test_bench_disk_seq_write():
    from bench import bench_disk_seq_write
    with tempfile.TemporaryDirectory() as tmpdir:
        result = bench_disk_seq_write(tmpdir, 8)  # 8 MB
        assert isinstance(result, float)
        assert result > 0


def test_bench_disk_seq_read():
    from bench import bench_disk_seq_read
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a file first
        path = os.path.join(tmpdir, "test_read.bin")
        with open(path, "wb") as f:
            f.write(os.urandom(8 * 1024 * 1024))
        result = bench_disk_seq_read(path)
        assert isinstance(result, float)
        assert result > 0


def test_bench_disk_random_write():
    from bench import bench_disk_random_write
    with tempfile.TemporaryDirectory() as tmpdir:
        result = bench_disk_random_write(tmpdir, 100)  # 100 ops
        assert isinstance(result, float)
        assert result > 0


def test_bench_disk_random_read():
    from bench import bench_disk_random_read
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file
        path = os.path.join(tmpdir, "test_rand.bin")
        with open(path, "wb") as f:
            f.write(os.urandom(1024 * 1024))  # 1MB
        result = bench_disk_random_read(path, 100)
        assert isinstance(result, float)
        assert result > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest test_bench.py::test_bench_disk_seq_write -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Write storage benchmark functions**

Append to `bench.py`:

```python
# ---------------------------------------------------------------------------
# Storage I/O benchmark functions
# ---------------------------------------------------------------------------

def _disable_file_cache(fd: int) -> None:
    """Platform-specific file cache bypass."""
    system = platform.system()
    if system == "Darwin":
        try:
            import fcntl
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        except (ImportError, OSError):
            pass
    elif system == "Linux":
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    # Windows: no equivalent -- accept cached behavior


def bench_disk_seq_write(output_dir: str, size_mb: int = 256) -> float:
    """Sequential write. Returns MB/s."""
    path = os.path.join(output_dir, "_bench_seq_write.tmp")
    chunk = os.urandom(1024 * 1024)  # 1 MB chunk
    try:
        start = time.perf_counter()
        with open(path, "wb") as f:
            for _ in range(size_mb):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        elapsed = time.perf_counter() - start
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if elapsed <= 0:
        return 0.0
    return size_mb / elapsed


def bench_disk_seq_read(path: str) -> float:
    """Sequential read. Returns MB/s. Reads entire file."""
    file_size = os.path.getsize(path)
    size_mb = file_size / (1024 * 1024)
    fd = os.open(path, os.O_RDONLY)
    _disable_file_cache(fd)
    os.close(fd)

    start = time.perf_counter()
    with open(path, "rb") as f:
        while f.read(1024 * 1024):
            pass
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    return size_mb / elapsed


def bench_disk_random_write(output_dir: str, ops: int = 1000) -> float:
    """Random 4K write IOPS."""
    import random
    path = os.path.join(output_dir, "_bench_rand_write.tmp")
    block_size = 4096
    file_size = max(ops * block_size, 1024 * 1024)  # at least 1MB
    data = os.urandom(block_size)

    try:
        # Pre-allocate file
        with open(path, "wb") as f:
            f.write(b"\x00" * file_size)

        start = time.perf_counter()
        with open(path, "r+b") as f:
            for _ in range(ops):
                offset = random.randint(0, (file_size - block_size) // block_size) * block_size
                f.seek(offset)
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
        elapsed = time.perf_counter() - start
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if elapsed <= 0:
        return 0.0
    return ops / elapsed


def bench_disk_random_read(path: str, ops: int = 1000) -> float:
    """Random 4K read IOPS."""
    import random
    file_size = os.path.getsize(path)
    block_size = 4096
    if file_size < block_size:
        return 0.0

    fd = os.open(path, os.O_RDONLY)
    _disable_file_cache(fd)
    os.close(fd)

    start = time.perf_counter()
    with open(path, "rb") as f:
        for _ in range(ops):
            offset = random.randint(0, (file_size - block_size) // block_size) * block_size
            f.seek(offset)
            f.read(block_size)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    return ops / elapsed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test_bench.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add bench.py test_bench.py
git commit -m "feat(bench): storage I/O benchmarks -- seq/random read/write with cache bypass"
```

---

### Task 11: Benchmark Registry, Phases, Baseline, safe_benchmark

**Files:**
- Modify: `bench.py` (append after storage benchmarks)
- Modify: `test_bench.py` (append tests)

- [ ] **Step 1: Write failing tests**

Append to `test_bench.py`:

```python
# ---------------------------------------------------------------------------
# Registry and safe_benchmark
# ---------------------------------------------------------------------------

def test_benchmark_registry_has_20_tests():
    from bench import BENCHMARKS
    # 20 entries, but disk_seq_read and disk_random_read have size=0
    # because they need a file path, not a size
    assert len(BENCHMARKS) == 20


def test_benchmark_registry_categories():
    from bench import BENCHMARKS
    categories = set(b[1] for b in BENCHMARKS)
    assert categories == {"cpu_single", "cpu_multi", "gpu", "memory", "storage"}


def test_safe_benchmark_success():
    from bench import safe_benchmark, BenchConfig, BenchmarkResult
    config = BenchConfig(iterations=2, warmups=1, test_timeout=10, timeout=30)

    def _trivial(size: int) -> float:
        return 42.0

    result, error = safe_benchmark(
        "test_trivial", "cpu_single", _trivial, (100,), "ops/sec", 42.0, config,
    )
    assert result is not None
    assert error is None
    assert isinstance(result, BenchmarkResult)
    assert result.score > 0


def test_safe_benchmark_failure():
    from bench import safe_benchmark, BenchConfig, BenchmarkError

    config = BenchConfig(iterations=2, warmups=1, test_timeout=5, timeout=15)

    def _failing(size: int) -> float:
        raise NotImplementedError("nope")

    result, error = safe_benchmark(
        "test_fail", "cpu_single", _failing, (100,), "ops/sec", 1.0, config,
    )
    assert result is None
    assert isinstance(error, BenchmarkError)
    assert error.error_type == "not_supported"


def test_phase_enum_order():
    from bench import Phase, PHASE_ORDER
    assert PHASE_ORDER[0] == Phase.WARMUP
    assert Phase.CPU_SINGLE in PHASE_ORDER
    assert Phase.FINALIZE == PHASE_ORDER[-1]


def test_category_weights_sum_to_one():
    from bench import CATEGORY_WEIGHTS
    total = sum(CATEGORY_WEIGHTS.values())
    assert abs(total - 1.0) < 0.001
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest test_bench.py::test_benchmark_registry_has_20_tests -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Write the registry, phases, baseline, and safe_benchmark**

This is a large step. Append to `bench.py` -- see the spec sections 4.3 (Registry), 7.2 (Phases), 5.6 (Baseline), and 6 (safe_benchmark). The full code is provided in the plan file due to length -- the implementer should write the registry list mapping all 20 benchmarks to their categories, the Phase enum and PHASE_ORDER, the CATEGORY_WEIGHTS dict, the BASELINE dict with placeholder values of 1.0, the QUICK_SIZES dict, and the `safe_benchmark` function with retry logic per the spec.

Key points for the implementer:
- Registry entry format: `(name, category, function, default_size, unit)`
- `safe_benchmark` runs warmups directly (no subprocess), measured iterations via direct call with timing, retries on transient failures with backoff
- Include `_suggest_bench_fix` helper for user-facing error messages
- BASELINE keys use format `"{category}_{name}"`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test_bench.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add bench.py test_bench.py
git commit -m "feat(bench): benchmark registry, phases, baseline, safe_benchmark with retry"
```

---

### Task 12: BenchmarkOrchestrator

**Files:**
- Modify: `bench.py` (append after safe_benchmark)
- Modify: `test_bench.py` (append tests)

- [ ] **Step 1: Write failing tests**

Append to `test_bench.py`:

```python
# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def test_orchestrator_quick_run():
    """Orchestrator produces a BenchmarkReport in quick mode."""
    from bench import BenchmarkOrchestrator, BenchConfig, BenchmarkReport
    config = BenchConfig(
        quick=True, iterations=1, warmups=0,
        skip_categories=["gpu", "storage"],
        no_cooldown=True, test_timeout=15, timeout=30,
    )
    orch = BenchmarkOrchestrator(config, system_info={})
    report = orch.run()
    assert isinstance(report, BenchmarkReport)
    assert report.overall_score >= 0
    assert report.timestamp != ""
    assert report.duration_seconds >= 0
    assert len(report.categories) > 0


def test_orchestrator_skip_all():
    """Orchestrator handles skipping all categories."""
    from bench import BenchmarkOrchestrator, BenchConfig, BenchmarkReport
    config = BenchConfig(
        quick=True, iterations=1, warmups=0,
        skip_categories=["cpu_single", "cpu_multi", "gpu", "memory", "storage"],
        no_cooldown=True,
    )
    orch = BenchmarkOrchestrator(config, system_info={})
    report = orch.run()
    assert isinstance(report, BenchmarkReport)
    assert report.overall_score == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest test_bench.py::test_orchestrator_quick_run -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Write the BenchmarkOrchestrator**

The orchestrator manages the full lifecycle: pre-flight, phase iteration, per-category benchmark execution, score calculation, and report assembly. Key points:
- `_resolve_categories` filters based on `--skip`/`--only` flags and available deps (MLX for GPU)
- `_run_category` iterates benchmarks in a category, calling `safe_benchmark` for each
- `_build_args` handles special cases (storage benchmarks need file paths, gpu_batch_matmul takes two args)
- `_build_category_scores` computes geometric mean per category, marks categories with <2 tests as skipped
- `shutdown()` sets `_shutdown` flag, checked between tests

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test_bench.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add bench.py test_bench.py
git commit -m "feat(bench): BenchmarkOrchestrator -- phased execution with cooldowns"
```

---

### Task 13: Output Formatters (Terminal, JSON, Text)

**Files:**
- Modify: `bench.py` (append after orchestrator)
- Modify: `test_bench.py` (append tests)

- [ ] **Step 1: Write failing tests**

Append to `test_bench.py`:

```python
# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _make_sample_report():
    from bench import (
        BenchmarkReport, CategoryScore, BenchmarkResult,
        ReportIntegrity, ExecutionMetadata,
    )
    result = BenchmarkResult(
        name="prime_sieve", category="cpu_single", raw_value=823.5,
        unit="ops/sec", score=1000.0, iterations=5, warmups=3,
        median_time=0.00121, std_dev=0.00003,
        times=[0.00121, 0.00122, 0.00120, 0.00121, 0.00123],
    )
    cat = CategoryScore(name="cpu_single", score=1000.0, weight=0.25, tests=[result])
    integrity = ReportIntegrity(
        complete=True, degraded_tests=[], cpu_fallback_tests=[],
        retried_tests=[], partial=False, constrained=False,
    )
    execution = ExecutionMetadata(
        phases_completed=8, phases_total=8, total_cooldown_seconds=5.0,
        peak_cpu_temp_c=70.0, peak_ram_usage_mb=4000.0,
        pre_flight={}, execution_mode="full",
    )
    return BenchmarkReport(
        overall_score=1000.0, categories=[cat],
        baseline_machine="Test Machine", baseline_version="1.0",
        system=None, skipped=[], errors=[], integrity=integrity,
        execution=execution, duration_seconds=45.0,
        timestamp="2026-04-13T20:00:00Z",
    )


def test_format_terminal_produces_output():
    from bench import format_terminal
    report = _make_sample_report()
    output = format_terminal(report, use_color=False)
    assert "OVERALL SCORE" in output
    assert "1000" in output


def test_format_terminal_no_ansi():
    from bench import format_terminal
    report = _make_sample_report()
    output = format_terminal(report, use_color=False)
    assert "\033[" not in output


def test_format_json_valid():
    from bench import format_json
    report = _make_sample_report()
    result = format_json(report)
    parsed = json.loads(result)
    assert parsed["overall_score"] == 1000.0
    assert "categories" in parsed


def test_format_text_no_ansi():
    from bench import format_text
    report = _make_sample_report()
    result = format_text(report)
    assert "\033[" not in result
    assert "OVERALL SCORE" in result


def test_save_outputs_creates_files():
    from bench import save_outputs
    report = _make_sample_report()
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path, text_path = save_outputs(report, output_dir=tmpdir)
        assert os.path.exists(json_path)
        assert text_path is not None and os.path.exists(text_path)
        with open(json_path) as f:
            data = json.loads(f.read())
            assert data["overall_score"] == 1000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest test_bench.py::test_format_terminal_produces_output -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Write formatters**

Implement `_Color` class (same pattern as fetch.py), `format_terminal`, `format_json`, `format_text`, and `save_outputs`. The terminal formatter shows per-test scores with timing/stddev, category scores, overall score, skipped categories, errors, and footer with file paths.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test_bench.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add bench.py test_bench.py
git commit -m "feat(bench): output formatters -- terminal, JSON, plain text, save_outputs"
```

---

### Task 14: CLI, main(), Signal Handling

**Files:**
- Modify: `bench.py` (append after formatters)
- Modify: `test_bench.py` (append tests)

- [ ] **Step 1: Write failing tests**

Append to `test_bench.py`:

```python
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_parse_args_defaults():
    from bench import parse_args
    args = parse_args([])
    assert args.json_only is False
    assert args.verbose is False
    assert args.no_color is False
    assert args.timeout == 60
    assert args.test_timeout == 30
    assert args.output_dir == "."
    assert args.iterations == 5
    assert args.warmups == 3
    assert args.quick is False
    assert args.no_cooldown is False
    assert args.calibrate is False
    assert args.skip == []


def test_parse_args_all_flags():
    from bench import parse_args
    args = parse_args([
        "--json-only", "--verbose", "--no-color", "--timeout", "120",
        "--test-timeout", "60", "--output-dir", "/tmp",
        "--iterations", "10", "--warmups", "5", "--quick", "--no-cooldown",
        "--calibrate", "--skip", "gpu", "--skip", "storage",
        "--only", "cpu_single",
    ])
    assert args.json_only is True
    assert args.verbose is True
    assert args.timeout == 120
    assert args.test_timeout == 60
    assert args.iterations == 10
    assert args.warmups == 5
    assert args.quick is True
    assert args.no_cooldown is True
    assert args.calibrate is True
    assert args.skip == ["gpu", "storage"]
    assert args.only == ["cpu_single"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest test_bench.py::test_parse_args_defaults -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Write CLI, main, signal handling**

Implement `parse_args` with all flags from the spec (section 10), signal handler using global `_orchestrator_ref`, and `main()` that: checks numpy, loads system_report.json, builds BenchConfig, runs orchestrator, handles `--calibrate` output, and saves reports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test_bench.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add bench.py test_bench.py
git commit -m "feat(bench): CLI, main entry point, signal handling, calibrate mode"
```

---

### Task 15: Integration Test -- Full Quick Run

**Files:**
- Modify: `test_bench.py` (append integration test)

- [ ] **Step 1: Write integration test**

Append to `test_bench.py`:

```python
# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def test_full_quick_run_integration():
    """Full quick benchmark run produces valid report with scores."""
    from bench import (
        BenchmarkOrchestrator, BenchConfig, BenchmarkReport,
        report_to_dict, format_terminal, format_json, save_outputs,
    )
    config = BenchConfig(
        quick=True, iterations=1, warmups=0,
        skip_categories=["gpu", "storage"],
        no_cooldown=True, test_timeout=30, timeout=60,
    )
    orch = BenchmarkOrchestrator(config, system_info={"os": {"type": "test"}})
    report = orch.run()

    # Report structure
    assert isinstance(report, BenchmarkReport)
    assert report.timestamp != ""
    assert report.duration_seconds >= 0
    assert report.baseline_machine != ""

    # At least some categories ran
    active = [c for c in report.categories if not c.skipped]
    assert len(active) >= 2  # cpu_single + cpu_multi at minimum

    # Scores are positive
    for cat in active:
        assert cat.score > 0
        for test in cat.tests:
            assert test.score > 0
            assert test.raw_value > 0
            assert test.median_time >= 0

    # Overall score computed
    assert report.overall_score > 0

    # JSON roundtrip
    d = report_to_dict(report)
    json_str = json.dumps(d)
    loaded = json.loads(json_str)
    assert loaded["overall_score"] > 0

    # Formatters work
    terminal_output = format_terminal(report, use_color=False)
    assert "OVERALL SCORE" in terminal_output

    # Save works
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path, text_path = save_outputs(report, output_dir=tmpdir)
        assert os.path.exists(json_path)
        assert text_path is not None and os.path.exists(text_path)
```

- [ ] **Step 2: Run the integration test**

Run: `python3 -m pytest test_bench.py::test_full_quick_run_integration -v --tb=long`
Expected: PASS (may take 10-30 seconds)

- [ ] **Step 3: Run the full test suite**

Run: `python3 -m pytest test_bench.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Run bench.py manually in quick mode**

Run: `python3 bench.py --quick --skip gpu --skip storage --no-cooldown --output-dir /tmp/bench_test`
Expected: Terminal output with scores, JSON and text files created

- [ ] **Step 5: Commit**

```bash
git add test_bench.py
git commit -m "test(bench): integration test -- full quick run with assertions"
```

---

### Task 16: Calibration Run -- Populate Baseline Constants

This task is done manually after all code works.

- [ ] **Step 1: Run calibration**

Run: `python3 bench.py --calibrate --no-cooldown 2>&1 | tee calibration_output.txt`
Expected: Prints raw values for all 20 tests

- [ ] **Step 2: Update BASELINE dict in bench.py**

Copy the printed values into the `BASELINE` dict, replacing all `1.0` placeholders.

- [ ] **Step 3: Verify scores are near 1000**

Run: `python3 bench.py --quick --no-cooldown`
Expected: All scores approximately 1000 (this IS the baseline machine)

- [ ] **Step 4: Run all tests**

Run: `python3 -m pytest test_bench.py test_fetch.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add bench.py
git commit -m "feat(bench): populate baseline constants from M4 Max calibration"
```

---

### Task 17: Cleanup -- gitignore and temp files

- [ ] **Step 1: Update .gitignore**

Add to `.gitignore`:

```
benchmark_report.json
benchmark_report.txt
system_report.json
system_report.txt
_bench_*.tmp
calibration_output.txt
```

- [ ] **Step 2: Clean up stale temp files**

Run: `rm -f _bench_*.tmp calibration_output.txt`

- [ ] **Step 3: Run final full test suite**

Run: `python3 -m pytest test_bench.py test_fetch.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: update gitignore for benchmark outputs and temp files"
```
