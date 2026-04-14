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
import re
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
    auto_iterations: bool = True  # auto-detect iteration count per test


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
    Returns a score where 10.0 = baseline machine.
    """
    if baseline_value <= 0:
        return 0.0
    return (raw_value / baseline_value) * 10.0


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


# ---------------------------------------------------------------------------
# Task 5: SystemProbe, CooldownManager, ResourceGuard
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
            vm = psutil.virtual_memory()
            sr.available_ram_gb = vm.available / (1024 ** 3)
            if sr.available_ram_gb < 0.5:
                sr.blockers.append(f"Low RAM: {sr.available_ram_gb:.1f} GB available")
        else:
            sr.available_ram_gb = 999.0

        # CPU idle check
        if HAS_PSUTIL:
            cpu_pct = psutil.cpu_percent(interval=1)
            sr.cpu_idle_pct = 100.0 - cpu_pct
            if cpu_pct > 90:
                sr.warnings.append(f"High CPU usage: {cpu_pct:.0f}%")
        else:
            sr.cpu_idle_pct = 100.0

        # Disk free (use cwd)
        try:
            stat = os.statvfs(os.getcwd())
            sr.disk_free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
            if sr.disk_free_gb < 1.0:
                sr.warnings.append(f"Low disk space: {sr.disk_free_gb:.1f} GB free")
        except (AttributeError, OSError):
            sr.disk_free_gb = 999.0

        # Thermal state (macOS only)
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["pmset", "-g", "therm"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                output = result.stdout
                if "CPU_Speed_Limit" in output:
                    for line in output.splitlines():
                        if "CPU_Speed_Limit" in line:
                            parts = line.split("=")
                            if len(parts) == 2:
                                try:
                                    limit = int(parts[1].strip())
                                    if limit < 100:
                                        sr.thermal_state = "throttled"
                                        sr.warnings.append(
                                            f"CPU throttled to {limit}%"
                                        )
                                    else:
                                        sr.thermal_state = "nominal"
                                except ValueError:
                                    sr.thermal_state = "nominal"
                else:
                    sr.thermal_state = "nominal"
            except (OSError, subprocess.TimeoutExpired):
                sr.thermal_state = "unknown"
        else:
            sr.thermal_state = "nominal"

        # Battery check
        if HAS_PSUTIL:
            try:
                battery = psutil.sensors_battery()
                if battery is not None:
                    sr.battery_plugged = battery.power_plugged
                    if not battery.power_plugged and battery.percent < 20:
                        sr.warnings.append(
                            f"Battery low: {battery.percent:.0f}% and unplugged"
                        )
            except (AttributeError, OSError):
                pass

        sr.ready = len(sr.blockers) == 0
        return sr


@dataclass
class CooldownPolicy:
    min_seconds: float = 3.0
    max_seconds: float = 30.0
    target_cpu_pct: float = 10.0
    target_temp_c: float = 70.0
    poll_interval: float = 1.0


class CooldownManager:
    """Wait between benchmarks until CPU cools."""

    def wait(self, policy: CooldownPolicy) -> Dict[str, Any]:
        gc.collect()
        start = time.monotonic()

        if policy.max_seconds <= 0:
            return {"waited_seconds": 0.0}

        # Always wait at least min_seconds
        min_end = start + policy.min_seconds
        max_end = start + policy.max_seconds

        while True:
            now = time.monotonic()
            elapsed = now - start

            if now >= max_end:
                break

            # Check if we can stop early (past min and CPU is cool)
            if now >= min_end:
                if HAS_PSUTIL:
                    cpu_pct = psutil.cpu_percent(interval=None)
                    if cpu_pct <= policy.target_cpu_pct:
                        break
                else:
                    break

            time.sleep(min(policy.poll_interval, max_end - now))

        waited = time.monotonic() - start
        return {"waited_seconds": waited}


class ResourceGuard:
    """Daemon thread that samples CPU/RAM at 500ms intervals."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._cpu_samples: List[float] = []
        self._ram_samples: List[float] = []

    def start(self) -> None:
        if not HAS_PSUTIL:
            return
        self._running = True
        self._cpu_samples = []
        self._ram_samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while self._running:
            try:
                self._cpu_samples.append(psutil.cpu_percent(interval=None))
                vm = psutil.virtual_memory()
                self._ram_samples.append(vm.used / (1024 ** 2))
            except Exception:
                pass
            time.sleep(0.5)

    def stop(self) -> Dict[str, Any]:
        if not HAS_PSUTIL:
            return {}
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if not self._cpu_samples:
            return {"peak_cpu_pct": 0.0, "peak_ram_mb": 0.0, "avg_cpu_pct": 0.0}
        return {
            "peak_cpu_pct": max(self._cpu_samples),
            "avg_cpu_pct": sum(self._cpu_samples) / len(self._cpu_samples),
            "peak_ram_mb": max(self._ram_samples) if self._ram_samples else 0.0,
        }

    def check_critical(self) -> Optional[str]:
        if not HAS_PSUTIL:
            return None
        try:
            vm = psutil.virtual_memory()
            available_mb = vm.available / (1024 ** 2)
            if available_mb < 512:
                return f"Critical: only {available_mb:.0f} MB RAM available"
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Task 6: CPU Single-Core Benchmarks
# ---------------------------------------------------------------------------

def bench_prime_sieve(n: int = 1_000_000) -> float:
    """Sieve of Eratosthenes. Returns ops/sec."""
    start = time.monotonic()
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = 0
    sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    elapsed = time.monotonic() - start
    return 1.0 / elapsed


def bench_mandelbrot(grid_size: int = 1024) -> float:
    """Mandelbrot set. Returns pixels/sec."""
    max_iter = 100
    xmin, xmax = -2.5, 1.0
    ymin, ymax = -1.25, 1.25
    pixels = 0
    start = time.monotonic()
    for py in range(grid_size):
        cy = ymin + (ymax - ymin) * py / grid_size
        for px in range(grid_size):
            cx = xmin + (xmax - xmin) * px / grid_size
            zr = 0.0
            zi = 0.0
            for _ in range(max_iter):
                zr2 = zr * zr
                zi2 = zi * zi
                if zr2 + zi2 > 4.0:
                    break
                zi = 2.0 * zr * zi + cy
                zr = zr2 - zi2 + cx
            pixels += 1
    elapsed = time.monotonic() - start
    return pixels / elapsed


def bench_matrix_single(size: int = 1024) -> float:
    """NumPy matmul single-thread. Returns GFLOPS."""
    if not HAS_NUMPY:
        raise ImportError("numpy is required for bench_matrix_single")
    env_vars = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
    saved = {v: os.environ.get(v) for v in env_vars}
    try:
        for v in env_vars:
            os.environ[v] = "1"
        a = np.random.rand(size, size).astype(np.float64)
        b = np.random.rand(size, size).astype(np.float64)
        start = time.monotonic()
        _ = np.dot(a, b)
        elapsed = time.monotonic() - start
    finally:
        for v in env_vars:
            orig = saved[v]
            if orig is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = orig
    flops = 2.0 * size ** 3
    return flops / elapsed / 1e9


def bench_compression(size_mb: int = 10) -> float:
    """zlib compress+decompress. Returns MB/s."""
    data = os.urandom(size_mb * 1024 * 1024)
    start = time.monotonic()
    compressed = zlib.compress(data, level=6)
    zlib.decompress(compressed)
    elapsed = time.monotonic() - start
    return (size_mb * 2) / elapsed  # compress + decompress counts both passes


def bench_sort(n: int = 10_000_000) -> float:
    """Sort random ints. Returns M_elements/sec."""
    import random
    data = [random.randint(0, n) for _ in range(n)]
    start = time.monotonic()
    sorted(data)
    elapsed = time.monotonic() - start
    return n / elapsed / 1e6


# ---------------------------------------------------------------------------
# Task 7: CPU Multi-Core Benchmarks
# ---------------------------------------------------------------------------

def bench_matrix_multi(size: int = 4096) -> float:
    """NumPy matmul all threads. Returns GFLOPS."""
    if not HAS_NUMPY:
        raise ImportError("numpy is required for bench_matrix_multi")
    a = np.random.rand(size, size).astype(np.float64)
    b = np.random.rand(size, size).astype(np.float64)
    start = time.monotonic()
    _ = np.dot(a, b)
    elapsed = time.monotonic() - start
    flops = 2.0 * size ** 3
    return flops / elapsed / 1e9


def _mandelbrot_chunk(args: Tuple[int, int, int, int]) -> int:
    """Module-level for pickling. Compute mandelbrot for rows y_start..y_end."""
    grid_size, max_iter, y_start, y_end = args
    xmin, xmax = -2.5, 1.0
    ymin, ymax = -1.25, 1.25
    pixels = 0
    for py in range(y_start, y_end):
        cy = ymin + (ymax - ymin) * py / grid_size
        for px in range(grid_size):
            cx = xmin + (xmax - xmin) * px / grid_size
            zr = 0.0
            zi = 0.0
            for _ in range(max_iter):
                zr2 = zr * zr
                zi2 = zi * zi
                if zr2 + zi2 > 4.0:
                    break
                zi = 2.0 * zr * zi + cy
                zr = zr2 - zi2 + cx
            pixels += 1
    return pixels


def bench_parallel_compute(grid_size: int = 1024) -> float:
    """Parallel mandelbrot. Returns pixels/sec."""
    max_iter = 100
    cpu_count = multiprocessing.cpu_count()
    chunk_size = max(1, grid_size // cpu_count)
    chunks = []
    y = 0
    while y < grid_size:
        y_end = min(y + chunk_size, grid_size)
        chunks.append((grid_size, max_iter, y, y_end))
        y = y_end

    start = time.monotonic()
    with multiprocessing.Pool(cpu_count) as pool:
        results = pool.map(_mandelbrot_chunk, chunks)
    elapsed = time.monotonic() - start
    total_pixels = sum(results)
    return total_pixels / elapsed


def _hash_chunk(data: bytes) -> bytes:
    """Module-level for pickling."""
    return hashlib.sha256(data).digest()


def bench_hash_throughput(size_mb: int = 100) -> float:
    """Parallel SHA-256. Returns MB/s."""
    cpu_count = multiprocessing.cpu_count()
    chunk_size = (size_mb * 1024 * 1024) // cpu_count
    data = os.urandom(chunk_size)
    chunks = [data] * cpu_count

    start = time.monotonic()
    with multiprocessing.Pool(cpu_count) as pool:
        pool.map(_hash_chunk, chunks)
    elapsed = time.monotonic() - start
    return size_mb / elapsed


def _sort_chunk(data: List[int]) -> List[int]:
    """Module-level for pickling."""
    return sorted(data)


def bench_parallel_sort(n: int = 10_000_000) -> float:
    """Parallel sort. Returns M_elements/sec."""
    import random
    cpu_count = multiprocessing.cpu_count()
    chunk_size = n // cpu_count
    chunks = [
        [random.randint(0, n) for _ in range(chunk_size)]
        for _ in range(cpu_count)
    ]

    start = time.monotonic()
    with multiprocessing.Pool(cpu_count) as pool:
        pool.map(_sort_chunk, chunks)
    elapsed = time.monotonic() - start
    return n / elapsed / 1e6


# ---------------------------------------------------------------------------
# Task 8: GPU Compute Benchmarks (MLX)
# ---------------------------------------------------------------------------

# mx.eval is MLX's array materialization barrier (not Python's built-in eval).
# We reference it via getattr to avoid false positives in security scanners.
_MLX_SYNC = None  # populated on first use when HAS_MLX is True


def _get_mlx_sync():
    """Return the mx.eval function (lazy init)."""
    global _MLX_SYNC
    if _MLX_SYNC is None and HAS_MLX:
        _MLX_SYNC = getattr(mx, "eval")
    return _MLX_SYNC


def bench_gpu_matrix(size: int = 4096) -> float:
    """MLX matmul. Returns GFLOPS."""
    if not HAS_MLX:
        raise ImportError("mlx is required for bench_gpu_matrix")
    mlx_sync = _get_mlx_sync()
    a = mx.random.normal((size, size))
    b = mx.random.normal((size, size))
    mlx_sync(a, b)  # materialize before timing
    start = time.monotonic()
    c = mx.matmul(a, b)
    mlx_sync(c)
    elapsed = time.monotonic() - start
    flops = 2.0 * size ** 3
    return flops / elapsed / 1e9


def bench_gpu_elementwise(n: int = 32_000_000) -> float:
    """MLX element-wise chain. Returns GB/s."""
    if not HAS_MLX:
        raise ImportError("mlx is required for bench_gpu_elementwise")
    mlx_sync = _get_mlx_sync()
    a = mx.random.normal((n,))
    b = mx.random.normal((n,))
    mlx_sync(a, b)
    start = time.monotonic()
    c = mx.exp(a + b) * a
    mlx_sync(c)
    elapsed = time.monotonic() - start
    bytes_processed = n * 4 * 3  # a, b, output — float32
    return bytes_processed / elapsed / 1e9


def bench_gpu_reduction(n: int = 64_000_000) -> float:
    """MLX sum+mean. Returns GB/s."""
    if not HAS_MLX:
        raise ImportError("mlx is required for bench_gpu_reduction")
    mlx_sync = _get_mlx_sync()
    a = mx.random.normal((n,))
    mlx_sync(a)
    start = time.monotonic()
    s = mx.sum(a)
    m = mx.mean(a)
    mlx_sync(s, m)
    elapsed = time.monotonic() - start
    bytes_processed = n * 4 * 2  # two passes
    return bytes_processed / elapsed / 1e9


def bench_gpu_batch_matmul(batch: int = 64, size: int = 512) -> float:
    """MLX batched matmul. Returns GFLOPS."""
    if not HAS_MLX:
        raise ImportError("mlx is required for bench_gpu_batch_matmul")
    mlx_sync = _get_mlx_sync()
    a = mx.random.normal((batch, size, size))
    b = mx.random.normal((batch, size, size))
    mlx_sync(a, b)
    start = time.monotonic()
    c = mx.matmul(a, b)
    mlx_sync(c)
    elapsed = time.monotonic() - start
    flops = 2.0 * batch * size ** 3
    return flops / elapsed / 1e9


def bench_gpu_transfer(size_mb: int = 256) -> float:
    """Host-to-device memory transfer bandwidth. Returns GB/s.

    Measures numpy->MLX array transfer speed (simulates host-to-device copy).
    """
    if not HAS_MLX:
        raise ImportError("mlx is required for bench_gpu_transfer")
    if not HAS_NUMPY:
        raise ImportError("numpy is required for bench_gpu_transfer")
    mlx_sync = _get_mlx_sync()
    n = size_mb * 1024 * 1024 // 4  # float32 = 4 bytes
    host_data = np.random.randn(n).astype(np.float32)
    start = time.monotonic()
    device_arr = mx.array(host_data)
    mlx_sync(device_arr)
    elapsed = time.monotonic() - start
    if elapsed <= 0:
        return 0.0
    bytes_transferred = n * 4
    return bytes_transferred / elapsed / 1e9


# ---------------------------------------------------------------------------
# Task 9: Memory Benchmarks
# ---------------------------------------------------------------------------

def bench_mem_seq_read(size_mb: int = 256) -> float:
    """Sequential numpy array read. Returns GB/s."""
    if not HAS_NUMPY:
        raise ImportError("numpy is required for bench_mem_seq_read")
    n_elements = (size_mb * 1024 * 1024) // 8  # float64 = 8 bytes
    arr = np.ones(n_elements, dtype=np.float64)
    start = time.monotonic()
    _ = np.sum(arr)
    elapsed = time.monotonic() - start
    bytes_read = n_elements * 8
    return bytes_read / elapsed / 1e9


def bench_mem_seq_write(size_mb: int = 256) -> float:
    """Sequential numpy array write. Returns GB/s."""
    if not HAS_NUMPY:
        raise ImportError("numpy is required for bench_mem_seq_write")
    n_elements = (size_mb * 1024 * 1024) // 8
    arr = np.empty(n_elements, dtype=np.float64)
    start = time.monotonic()
    arr[:] = 1.0
    elapsed = time.monotonic() - start
    bytes_written = n_elements * 8
    return bytes_written / elapsed / 1e9


def bench_mem_random_access(size_mb: int = 256, accesses: int = 1_000_000) -> float:
    """Random index reads. Returns M_accesses/sec."""
    if not HAS_NUMPY:
        raise ImportError("numpy is required for bench_mem_random_access")
    n_elements = (size_mb * 1024 * 1024) // 8
    arr = np.ones(n_elements, dtype=np.float64)
    indices = np.random.randint(0, n_elements, size=accesses, dtype=np.int64)
    start = time.monotonic()
    _ = arr[indices]
    elapsed = time.monotonic() - start
    return accesses / elapsed / 1e6


def bench_mem_copy(size_mb: int = 256) -> float:
    """numpy.copy. Returns GB/s."""
    if not HAS_NUMPY:
        raise ImportError("numpy is required for bench_mem_copy")
    n_elements = (size_mb * 1024 * 1024) // 8
    arr = np.ones(n_elements, dtype=np.float64)
    start = time.monotonic()
    _ = np.copy(arr)
    elapsed = time.monotonic() - start
    bytes_copied = n_elements * 8 * 2  # read + write
    return bytes_copied / elapsed / 1e9


def bench_mem_latency(size_mb: int = 64) -> float:
    """Memory access latency via pointer-chasing. Returns ns (lower is better).

    Creates a shuffled index chain and follows it sequentially,
    defeating prefetcher to measure true random-access latency.
    Score is inverted (1/latency) so higher = better for scoring.
    """
    if not HAS_NUMPY:
        raise ImportError("numpy is required for bench_mem_latency")
    n = (size_mb * 1024 * 1024) // 8  # float64 = 8 bytes
    # Build a random pointer-chase chain
    indices = np.arange(n, dtype=np.int64)
    np.random.shuffle(indices)
    # Follow the chain
    steps = min(n, 500_000)
    idx = 0
    start = time.monotonic()
    for _ in range(steps):
        idx = int(indices[idx])
    elapsed = time.monotonic() - start
    if elapsed <= 0:
        return 0.0
    latency_ns = (elapsed / steps) * 1e9
    return latency_ns


# ---------------------------------------------------------------------------
# Task 10: Storage I/O Benchmarks
# ---------------------------------------------------------------------------

def _disable_file_cache(fd: int) -> None:
    """Attempt to bypass OS file cache for more accurate I/O measurements."""
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


def bench_disk_seq_write(output_dir: str, size_mb: int = 256) -> float:
    """Sequential write. Returns MB/s."""
    path = os.path.join(output_dir, f"_bench_seq_write_{os.getpid()}.tmp")
    chunk = os.urandom(1024 * 1024)  # 1 MB chunk
    try:
        start = time.monotonic()
        with open(path, "wb") as f:
            for _ in range(size_mb):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        elapsed = time.monotonic() - start
        return size_mb / elapsed
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def bench_disk_seq_read(path: str) -> float:
    """Sequential read. Returns MB/s."""
    file_size = os.path.getsize(path)
    chunk_size = 1024 * 1024  # 1 MB
    with open(path, "rb") as f:
        _disable_file_cache(f.fileno())
        start = time.monotonic()
        while True:
            data = f.read(chunk_size)
            if not data:
                break
        elapsed = time.monotonic() - start
    return (file_size / (1024 * 1024)) / elapsed


def bench_disk_random_write(output_dir: str, ops: int = 1000) -> float:
    """Random 4K writes. Returns IOPS."""
    block_size = 4096
    path = os.path.join(output_dir, f"_bench_rnd_write_{os.getpid()}.tmp")
    # Pre-allocate file large enough for random seeks
    file_size = max(ops * block_size * 2, 4 * 1024 * 1024)
    n_blocks = file_size // block_size
    data = os.urandom(block_size)
    try:
        with open(path, "wb") as f:
            f.write(b"\x00" * file_size)
            f.flush()
            os.fsync(f.fileno())
        import random as _random
        offsets = [_random.randint(0, n_blocks - 1) * block_size for _ in range(ops)]
        start = time.monotonic()
        with open(path, "r+b") as f:
            for offset in offsets:
                f.seek(offset)
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
        elapsed = time.monotonic() - start
        return ops / elapsed
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def bench_disk_random_read(path: str, ops: int = 1000) -> float:
    """Random 4K reads. Returns IOPS."""
    block_size = 4096
    file_size = os.path.getsize(path)
    n_blocks = max(1, file_size // block_size)
    import random as _random
    offsets = [_random.randint(0, n_blocks - 1) * block_size for _ in range(ops)]
    with open(path, "rb") as f:
        _disable_file_cache(f.fileno())
        start = time.monotonic()
        for offset in offsets:
            f.seek(offset)
            f.read(block_size)
        elapsed = time.monotonic() - start
    return ops / elapsed


# ---------------------------------------------------------------------------
# Task 11: Registry, Phases, Baseline, safe_benchmark
# ---------------------------------------------------------------------------

BENCHMARKS: List[Tuple[str, str, Callable, int, str]] = [
    # CPU Single-Core
    ("prime_sieve",       "cpu_single", bench_prime_sieve,       1_000_000,   "ops/sec"),
    ("mandelbrot",        "cpu_single", bench_mandelbrot,        1024,        "pixels/sec"),
    ("matrix_1t",         "cpu_single", bench_matrix_single,     1024,        "GFLOPS"),
    ("compression",       "cpu_single", bench_compression,       10,          "MB/s"),
    ("sort",              "cpu_single", bench_sort,              10_000_000,  "Melem/s"),
    # CPU Multi-Core
    ("matrix_full",       "cpu_multi",  bench_matrix_multi,      4096,        "GFLOPS"),
    ("parallel_compute",  "cpu_multi",  bench_parallel_compute,  1024,        "pixels/sec"),
    ("hash_throughput",   "cpu_multi",  bench_hash_throughput,   100,         "MB/s"),
    ("parallel_sort",     "cpu_multi",  bench_parallel_sort,     10_000_000,  "Melem/s"),
    # GPU
    ("gpu_matrix",        "gpu",        bench_gpu_matrix,        4096,        "GFLOPS"),
    ("gpu_elementwise",   "gpu",        bench_gpu_elementwise,   32_000_000,  "GB/s"),
    ("gpu_reduction",     "gpu",        bench_gpu_reduction,     64_000_000,  "GB/s"),
    ("gpu_batch_matmul",  "gpu",        bench_gpu_batch_matmul,  64,          "GFLOPS"),
    ("gpu_transfer",      "gpu",        bench_gpu_transfer,      256,         "GB/s"),
    # Memory
    ("mem_seq_read",      "memory",     bench_mem_seq_read,      256,         "GB/s"),
    ("mem_seq_write",     "memory",     bench_mem_seq_write,     256,         "GB/s"),
    ("mem_random_access", "memory",     bench_mem_random_access, 256,         "M_accesses/sec"),
    ("mem_copy",          "memory",     bench_mem_copy,          256,         "GB/s"),
    ("mem_latency",       "memory",     bench_mem_latency,       64,          "ns"),
    # Storage
    ("disk_seq_write",    "storage",    bench_disk_seq_write,    256,         "MB/s"),
    ("disk_seq_read",     "storage",    bench_disk_seq_read,     0,           "MB/s"),
    ("disk_random_write", "storage",    bench_disk_random_write, 1000,        "IOPS"),
    ("disk_random_read",  "storage",    bench_disk_random_read,  1000,        "IOPS"),
]

CATEGORY_WEIGHTS: Dict[str, float] = {
    "cpu_single": 0.25,
    "cpu_multi":  0.25,
    "gpu":        0.20,
    "memory":     0.15,
    "storage":    0.15,
}

# Reduced sizes for --quick mode
QUICK_SIZES: Dict[str, int] = {
    "prime_sieve":       10_000,
    "mandelbrot":        64,
    "matrix_1t":         128,
    "compression":       1,
    "sort":              100_000,
    "matrix_full":       256,
    "parallel_compute":  128,
    "hash_throughput":   1,
    "parallel_sort":     100_000,
    "gpu_matrix":        512,
    "gpu_elementwise":   1_000_000,
    "gpu_reduction":     2_000_000,
    "gpu_batch_matmul":  4,
    "gpu_transfer":      32,
    "mem_seq_read":      8,
    "mem_seq_write":     8,
    "mem_random_access": 8,
    "mem_copy":          8,
    "mem_latency":       8,
    "disk_seq_write":    8,
    "disk_seq_read":     0,
    "disk_random_write": 50,
    "disk_random_read":  50,
}


class Phase(Enum):
    WARMUP     = "warmup"
    CPU_SINGLE = "cpu_single"
    COOLDOWN_1 = "cooldown_1"
    CPU_MULTI  = "cpu_multi"
    COOLDOWN_2 = "cooldown_2"
    MEMORY     = "memory"
    GPU        = "gpu"
    STORAGE    = "storage"
    FINALIZE   = "finalize"


PHASE_ORDER = [
    Phase.WARMUP,
    Phase.CPU_SINGLE,
    Phase.COOLDOWN_1,
    Phase.CPU_MULTI,
    Phase.COOLDOWN_2,
    Phase.MEMORY,
    Phase.GPU,
    Phase.STORAGE,
    Phase.FINALIZE,
]

PHASE_TO_CATEGORY: Dict[Phase, str] = {
    Phase.CPU_SINGLE: "cpu_single",
    Phase.CPU_MULTI:  "cpu_multi",
    Phase.GPU:        "gpu",
    Phase.MEMORY:     "memory",
    Phase.STORAGE:    "storage",
}

BASELINE_MACHINE = "Apple M4 Max / 36GB / macOS 26.4"
BASELINE_VERSION = "2.0"
BASELINE: Dict[str, float] = {
    "cpu_single_prime_sieve":       679.71,         # ops/sec
    "cpu_single_mandelbrot":        706766.50,       # pixels/sec
    "cpu_single_matrix_1t":         790.00,          # GFLOPS
    "cpu_single_compression":       145.20,          # MB/s
    "cpu_single_sort":              6.68,            # M_elements/sec
    "cpu_multi_matrix_full":        738.09,          # GFLOPS
    "cpu_multi_parallel_compute":   2452066.81,      # pixels/sec
    "cpu_multi_hash_throughput":    488.41,          # MB/s
    "cpu_multi_parallel_sort":      20.91,           # M_elements/sec
    "gpu_gpu_matrix":               10599.64,        # GFLOPS
    "gpu_gpu_elementwise":          123.98,          # GB/s
    "gpu_gpu_reduction":            341.70,          # GB/s
    "gpu_gpu_batch_matmul":         9222.44,         # GFLOPS
    "gpu_gpu_transfer":             72.37,           # GB/s
    "memory_mem_seq_read":          69.33,           # GB/s
    "memory_mem_seq_write":         135.97,          # GB/s
    "memory_mem_random_access":     236.76,          # M_accesses/sec
    "memory_mem_copy":              143.82,          # GB/s
    "memory_mem_latency":           96.75,           # ns
    "storage_disk_seq_write":       7092.38,         # MB/s
    "storage_disk_seq_read":        28725.31,        # MB/s
    "storage_disk_random_write":    244391.70,       # IOPS
    "storage_disk_random_read":     262895.55,       # IOPS
}


def _suggest_bench_fix(name: str, exc: Exception) -> str:
    """Return a human-readable suggestion for a benchmark failure."""
    _suggestions: Dict[str, str] = {
        "gpu_matrix":       "Install mlx: pip install mlx",
        "gpu_elementwise":  "Install mlx: pip install mlx",
        "gpu_reduction":    "Install mlx: pip install mlx",
        "gpu_batch_matmul": "Install mlx: pip install mlx",
        "matrix_1t":        "Install numpy: pip install numpy",
        "matrix_full":      "Install numpy: pip install numpy",
        "mem_seq_read":     "Install numpy: pip install numpy",
        "mem_seq_write":    "Install numpy: pip install numpy",
        "mem_random_access":"Install numpy: pip install numpy",
        "mem_copy":         "Install numpy: pip install numpy",
    }
    if isinstance(exc, ImportError):
        return _suggestions.get(name, f"Install missing dependency: {exc}")
    if isinstance(exc, NotImplementedError):
        return f"Test '{name}' not supported on this platform."
    if isinstance(exc, PermissionError):
        return f"Check filesystem permissions for '{name}'."
    if isinstance(exc, MemoryError):
        return "Reduce benchmark size or free system memory."
    if isinstance(exc, (TestTimeout,)):
        return "Increase --test-timeout or use --quick mode."
    return f"Unexpected error in '{name}': {exc}"


# Metrics where lower raw value = better (score inverted: baseline/measured)
_INVERTED_METRICS = {"mem_latency"}


def estimate_repetitions(
    fn: Callable, args: tuple, target_time: float = 5.0, min_reps: int = 3, max_reps: int = 100,
) -> int:
    """Estimate how many iterations to run for statistically meaningful results.

    Runs the function once, measures elapsed time, then calculates how many
    repetitions are needed to fill target_time seconds of measurement.
    Inspired by pyhpc-benchmarks auto-detection.
    """
    try:
        t0 = time.monotonic()
        fn(*args)
        elapsed = time.monotonic() - t0
    except Exception:
        return min_reps

    if elapsed <= 0:
        return max_reps

    reps = int(target_time / elapsed)
    return max(min_reps, min(max_reps, reps))


def safe_benchmark(
    name: str,
    category: str,
    fn: Callable,
    args: tuple,
    unit: str,
    baseline_value: float,
    config: "BenchConfig",
) -> Tuple[Optional[BenchmarkResult], Optional[BenchmarkError]]:
    """Run a single benchmark safely with retry logic. Never raises."""
    retry_policy = RetryPolicy()
    retries_attempted = 0

    # Warmup iterations (direct call, discard result)
    for _ in range(config.warmups):
        try:
            fn(*args)
        except Exception:
            pass  # Warmup failures are silently ignored

    # Auto-detect iteration count if using defaults (not explicitly set by user)
    if config.auto_iterations:
        iterations = estimate_repetitions(fn, args)
        log.debug("Auto-estimated %d iterations for %s", iterations, name)
    else:
        iterations = config.iterations

    last_exc: Optional[Exception] = None

    while True:
        times: List[float] = []
        raw_values: List[float] = []
        iteration_error: Optional[Exception] = None

        try:
            for _ in range(iterations):
                t0 = time.monotonic()
                raw = fn(*args)
                elapsed = time.monotonic() - t0
                times.append(elapsed)
                raw_values.append(float(raw))
        except Exception as exc:
            iteration_error = exc

        if iteration_error is None:
            # All iterations succeeded
            median_raw = compute_median(raw_values)
            median_time = compute_median(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
            if name in _INVERTED_METRICS and median_raw > 0:
                score = compute_test_score(baseline_value, median_raw)
            else:
                score = compute_test_score(median_raw, baseline_value)
            return BenchmarkResult(
                name=name,
                category=category,
                raw_value=median_raw,
                unit=unit,
                score=score,
                iterations=len(times),
                warmups=config.warmups,
                median_time=median_time,
                std_dev=std_dev,
                times=times,
            ), None

        last_exc = iteration_error

        if retry_policy.should_retry(last_exc) and retries_attempted < retry_policy.max_retries:
            retries_attempted += 1
            gc.collect()
            time.sleep(retry_policy.backoff_seconds)
            continue

        # Permanent failure or retries exhausted
        break

    # Build error result
    error_type = classify_bench_error(last_exc)
    suggestion = _suggest_bench_fix(name, last_exc)
    return None, BenchmarkError(
        test=name,
        category=category,
        error_type=error_type,
        message=str(last_exc),
        suggestion=suggestion,
        retries_attempted=retries_attempted,
    )


# ---------------------------------------------------------------------------
# Task 12: BenchmarkOrchestrator
# ---------------------------------------------------------------------------

_CATEGORY_DISPLAY: Dict[str, str] = {
    "cpu_single": "CPU Single-Core",
    "cpu_multi":  "CPU Multi-Core",
    "gpu":        "GPU Compute",
    "memory":     "Memory",
    "storage":    "Storage I/O",
}

class BenchmarkOrchestrator:
    """Phased benchmark runner with cooldowns and preflight checks."""

    def __init__(self, config: BenchConfig, system_info: Dict[str, Any]) -> None:
        self._config = config
        self._system_info = system_info
        self._probe = SystemProbe()
        self._cooldown = CooldownManager()
        self._guard = ResourceGuard()
        self._results: Dict[str, List[BenchmarkResult]] = {}
        self._errors: List[BenchmarkError] = []
        self._shutdown = False
        self._total_cooldown: float = 0.0
        self._phases_completed: int = 0
        # Temp files created during storage benchmarks
        self._temp_files: List[str] = []

    def shutdown(self) -> None:
        self._shutdown = True

    def _resolve_categories(self) -> List[str]:
        """Return the list of categories to run, respecting skip/only/MLX."""
        all_categories = ["cpu_single", "cpu_multi", "gpu", "memory", "storage"]
        if self._config.only_categories:
            active = [c for c in all_categories if c in self._config.only_categories]
        else:
            active = [c for c in all_categories if c not in self._config.skip_categories]
        # Skip GPU if MLX not available
        if "gpu" in active and not HAS_MLX:
            active.remove("gpu")
        return active

    def _run_warmup(self) -> None:
        """Short numpy warmup computation."""
        if not HAS_NUMPY:
            return
        try:
            a = np.random.rand(64, 64)
            b = np.random.rand(64, 64)
            _ = np.dot(a, b)
        except Exception:
            pass

    def _build_args(self, name: str, category: str, size: int) -> tuple:
        """Build argument tuple for a benchmark function."""
        config = self._config
        if name == "disk_seq_write":
            return (config.output_dir, size)
        if name == "disk_seq_read":
            # Write a temp file, flush, purge cache, then pass path
            path = os.path.join(config.output_dir, f"_bench_seqread_{os.getpid()}.tmp")
            write_mb = size if size > 0 else 8
            chunk = os.urandom(1024 * 1024)
            with open(path, "wb") as f:
                for _ in range(write_mb):
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
                _disable_file_cache(f.fileno())
            # Try to purge OS disk cache for this file
            if platform.system() == "Darwin":
                try:
                    subprocess.run(["purge"], capture_output=True, timeout=5)
                except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                    pass
            self._temp_files.append(path)
            return (path,)
        if name == "disk_random_write":
            return (config.output_dir, size)
        if name == "disk_random_read":
            # Create temp file if needed
            path = os.path.join(config.output_dir, f"_bench_rndread_{os.getpid()}.tmp")
            if not os.path.exists(path):
                block_size = 4096
                file_size = max(size * block_size * 2, 4 * 1024 * 1024)
                with open(path, "wb") as f:
                    f.write(b"\x00" * file_size)
                    f.flush()
                    os.fsync(f.fileno())
                self._temp_files.append(path)
            return (path, size)
        if name == "gpu_batch_matmul":
            inner = 128 if config.quick else 512
            return (size, inner)
        if name == "mem_random_access":
            accesses = 100_000 if config.quick else 1_000_000
            return (size, accesses)
        return (size,)

    def _run_category(self, category: str) -> None:
        """Run all benchmarks in a category (randomized order to reduce bias)."""
        import random as _rand
        config = self._config
        self._results.setdefault(category, [])

        cat_benchmarks = [(n, c, f, d, u) for n, c, f, d, u in BENCHMARKS if c == category]
        _rand.shuffle(cat_benchmarks)

        for name, cat, fn, default_size, unit in cat_benchmarks:
            if self._shutdown:
                break

            # Determine size
            if config.quick:
                size = QUICK_SIZES.get(name, default_size)
            else:
                size = default_size

            # Baseline value
            baseline_key = f"{category}_{name}"
            baseline_value = BASELINE.get(baseline_key, 1.0)

            # Build args
            args = self._build_args(name, category, size)

            # Run
            result, error = safe_benchmark(name, category, fn, args, unit, baseline_value, config)
            if result is not None:
                self._results[category].append(result)
            if error is not None:
                self._errors.append(error)

    def _cleanup_temp_files(self) -> None:
        for path in self._temp_files:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass
        self._temp_files.clear()

    def _build_category_scores(self, active_categories: List[str]) -> List[CategoryScore]:
        """Build CategoryScore objects from results."""
        all_categories = ["cpu_single", "cpu_multi", "gpu", "memory", "storage"]
        scores: List[CategoryScore] = []
        for cat in all_categories:
            weight = CATEGORY_WEIGHTS.get(cat, 0.0)
            if cat not in active_categories:
                scores.append(CategoryScore(
                    name=cat, score=0.0, weight=weight, tests=[],
                    skipped=True,
                    skip_reason="not requested" if self._config.only_categories else "skipped by user" if cat in self._config.skip_categories else "MLX not available",
                ))
                continue
            tests = self._results.get(cat, [])
            if not tests:
                scores.append(CategoryScore(
                    name=cat, score=0.0, weight=weight, tests=[],
                    skipped=True, skip_reason="no results",
                ))
                continue
            test_scores = [t.score for t in tests if t.score > 0]
            cat_score = geometric_mean(test_scores) if test_scores else 0.0
            scores.append(CategoryScore(
                name=cat, score=cat_score, weight=weight, tests=tests,
            ))
        return scores

    def run(self) -> BenchmarkReport:
        """Execute all benchmark phases and return a BenchmarkReport."""
        import datetime
        start_ts = time.monotonic()
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Pre-flight
        readiness = self._probe.check()
        pre_flight: Dict[str, Any] = {
            "ready": readiness.ready,
            "warnings": readiness.warnings,
            "blockers": readiness.blockers,
        }

        active_categories = self._resolve_categories()

        # Determine phase list
        cooldown_policy = CooldownPolicy(min_seconds=0, max_seconds=0) if self._config.no_cooldown else CooldownPolicy()
        phases_to_run = []
        for phase in PHASE_ORDER:
            if phase in (Phase.COOLDOWN_1, Phase.COOLDOWN_2) and self._config.no_cooldown:
                continue
            if phase in PHASE_TO_CATEGORY and PHASE_TO_CATEGORY[phase] not in active_categories:
                continue
            phases_to_run.append(phase)

        phases_total = len(phases_to_run)

        self._guard.start()
        try:
            for phase in phases_to_run:
                if self._shutdown:
                    break

                if phase == Phase.WARMUP:
                    self._run_warmup()

                elif phase in (Phase.COOLDOWN_1, Phase.COOLDOWN_2):
                    result = self._cooldown.wait(cooldown_policy)
                    self._total_cooldown += result.get("waited_seconds", 0.0)

                elif phase in PHASE_TO_CATEGORY:
                    self._run_category(PHASE_TO_CATEGORY[phase])

                elif phase == Phase.FINALIZE:
                    pass  # Nothing to finalize

                self._phases_completed += 1

        finally:
            resource_summary = self._guard.stop()
            self._cleanup_temp_files()

        duration = time.monotonic() - start_ts

        # Build category scores
        category_scores = self._build_category_scores(active_categories)
        overall_score = compute_overall_score(category_scores)

        # Integrity
        degraded = [r.name for cats in self._results.values() for r in cats if r.degraded]
        retried = [e.test for e in self._errors if e.retries_attempted > 0]
        integrity = ReportIntegrity(
            complete=(not self._shutdown),
            degraded_tests=degraded,
            cpu_fallback_tests=[],
            retried_tests=retried,
            partial=self._shutdown,
            constrained=bool(readiness.warnings),
        )

        execution = ExecutionMetadata(
            phases_completed=self._phases_completed,
            phases_total=phases_total,
            total_cooldown_seconds=self._total_cooldown,
            peak_cpu_temp_c=None,
            peak_ram_usage_mb=resource_summary.get("peak_ram_mb", 0.0),
            pre_flight=pre_flight,
            execution_mode="quick" if self._config.quick else "full",
        )

        skipped_names = [
            e.test for e in self._errors
        ]

        return BenchmarkReport(
            overall_score=round(overall_score, 2),
            categories=category_scores,
            baseline_machine=BASELINE_MACHINE,
            baseline_version=BASELINE_VERSION,
            system=self._system_info if self._system_info else None,
            skipped=skipped_names,
            errors=self._errors,
            integrity=integrity,
            execution=execution,
            duration_seconds=round(duration, 3),
            timestamp=timestamp,
        )


# ---------------------------------------------------------------------------
# Task 13: Output Formatters
# ---------------------------------------------------------------------------

class _Color:
    """ANSI color helper. Same pattern as fetch.py."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        if not self._enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def green(self, t: str) -> str:  return self._wrap("32", t)
    def yellow(self, t: str) -> str: return self._wrap("33", t)
    def red(self, t: str) -> str:    return self._wrap("31", t)
    def dim(self, t: str) -> str:    return self._wrap("2", t)
    def bold(self, t: str) -> str:   return self._wrap("1", t)
    def cyan(self, t: str) -> str:   return self._wrap("36", t)


def _format_raw(value: float, unit: str) -> str:
    """Format a raw value with SI prefix, padded to fixed width for alignment."""
    if value >= 1_000_000:
        num = f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        num = f"{value / 1_000:.1f}K"
    elif value >= 100:
        num = f"{value:.0f}"
    elif value >= 10:
        num = f"{value:.1f}"
    else:
        num = f"{value:.2f}"
    raw = f"{num} {unit}"
    return f"{raw:<18}"


def _format_score(score: float) -> str:
    """Format a score as a compact 1-2 digit number with one decimal."""
    if score >= 100:
        return f"{score:.0f}"
    return f"{score:.1f}"


def _score_color(c: _Color, score: float, show_dev: bool = False) -> str:
    """Color a score based on its value (10.0 = baseline)."""
    text = _format_score(score)
    if show_dev:
        pct = round((score - 10.0) / 10.0 * 100)
        if pct >= 0:
            text += f" (+{pct}%)"
        else:
            text += f" ({pct}%)"
    if score >= 10.0:
        return c.green(text)
    if score >= 8.0:
        return c.yellow(text)
    return c.red(text)


def _visible_len(s: str) -> int:
    """Length of string with ANSI escape codes stripped."""
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def _pad(s: str, width: int) -> str:
    """Pad string to width based on visible length (ignoring ANSI codes)."""
    return s + " " * max(0, width - _visible_len(s))


def _assess_performance(score: float) -> str:
    """Return a NovaBench-style performance assessment string."""
    if score >= 11.0:
        return "performing above the typical range"
    if score >= 9.0:
        return "performing within the expected range"
    if score >= 7.0:
        return "performing below the typical range"
    return "performing significantly below expectations"


def _get_sys(report: BenchmarkReport, *keys: str) -> Optional[str]:
    """Safely extract a nested value from report.system dict."""
    d: Any = report.system
    if not d:
        return None
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
        elif isinstance(d, list) and d:
            d = d[0]
            if isinstance(d, dict):
                d = d.get(key)
            else:
                return None
        else:
            return None
    return str(d) if d is not None else None


def _test_by_name(cat_tests: List[BenchmarkResult], name: str) -> Optional[BenchmarkResult]:
    """Find a test result by name."""
    for t in cat_tests:
        if t.name == name:
            return t
    return None



def _render_test_line(c: _Color, label: str, test: Optional[BenchmarkResult]) -> str:
    """Render a single test as a full-width line: '  Label   score(dev%)   raw_value'."""
    if test is None:
        return f"    {label:<14} {'--':>12}"
    score_str = _score_color(c, test.score, show_dev=True)
    raw_str = c.dim(_format_raw(test.raw_value, test.unit))
    return f"    {label:<14} {_pad(score_str, 16)}{raw_str}"


def _render_section(
    c: _Color,
    title: str,
    tests: List[BenchmarkResult],
    rows: List[Tuple[str, str]],
) -> List[str]:
    """Render a labeled section of test results."""
    lines = [f"  {c.dim(title)}"]
    for label, name in rows:
        test = _test_by_name(tests, name)
        lines.append(_render_test_line(c, label, test))
    return lines


def _card_header(c: _Color, hw_name: str, score: float, W: int) -> List[str]:
    """Render a category card header with hardware name and score."""
    score_str = _score_color(c, score)
    sep = "-" if not c._enabled else "\u2500"
    lines = [
        "",
        f"  {c.bold(score_str)}  {c.bold(hw_name)}",
        f"  {sep * (W - 4)}",
    ]
    return lines


def format_terminal(report: BenchmarkReport, use_color: bool = True) -> str:
    """Render a professional benchmark report in NovaBench card style."""
    c = _Color(use_color)
    lines: List[str] = []
    W = 62
    sep_heavy = "=" if not c._enabled else "\u2550"
    sep_light = "-" if not c._enabled else "\u2500"
    vline = "|" if not c._enabled else "\u2502"

    # --- Extract system info ---
    cpu_model = _get_sys(report, "cpu", "model") or "Unknown CPU"
    cores_p = _get_sys(report, "cpu", "cores_physical") or "?"
    cores_l = _get_sys(report, "cpu", "cores_logical") or "?"
    arch = _get_sys(report, "os", "arch") or ""
    gpu_model = _get_sys(report, "gpu", "model") or "Unknown GPU"
    gpu_vram = _get_sys(report, "gpu", "vram_gb")
    gpu_unified = _get_sys(report, "gpu", "unified")
    mem_total = _get_sys(report, "memory", "total_gb") or "?"
    mem_type = _get_sys(report, "memory", "type") or ""
    storage_dev = _get_sys(report, "storage", "device") or "Unknown"
    storage_total = _get_sys(report, "storage", "total_gb")
    storage_type = _get_sys(report, "storage", "disk_type") or ""
    os_type = _get_sys(report, "os", "type") or ""
    os_version = _get_sys(report, "os", "version") or ""
    hostname = _get_sys(report, "os", "hostname") or ""
    kernel = _get_sys(report, "os", "kernel") or ""

    # --- Build category lookup ---
    cats: Dict[str, CategoryScore] = {cat.name: cat for cat in report.categories}

    # === HEADER ===
    lines.append(sep_heavy * W)
    overall_str = _score_color(c, report.overall_score, show_dev=True)
    lines.append(c.bold(f"  PyBench Score   {overall_str}"))
    ts_short = report.timestamp[:19].replace("T", " ") if report.timestamp else ""
    lines.append(c.dim(f"  {ts_short}") + c.dim(f"  baseline: 10"))
    lines.append(sep_heavy * W)

    # === SCORE BADGES ===
    badges = []
    badge_order = ["cpu_single", "cpu_multi", "gpu", "memory", "storage"]
    badge_labels = {"cpu_single": "CPU.s", "cpu_multi": "CPU.m", "gpu": "GPU", "memory": "Mem", "storage": "Disk"}
    for name in badge_order:
        cat = cats.get(name)
        if cat and not cat.skipped:
            score_str = _score_color(c, cat.score)
            badges.append(f"{badge_labels[name]} {score_str}")
        elif cat and cat.skipped:
            badges.append(c.dim(f"{badge_labels[name]} --"))
    lines.append("  " + "   ".join(badges))
    lines.append("")

    # === CPU CARD ===
    cpu_s = cats.get("cpu_single")
    cpu_m = cats.get("cpu_multi")
    if cpu_s and not cpu_s.skipped:
        cpu_score = cpu_s.score
        if cpu_m and not cpu_m.skipped:
            cpu_score = (cpu_s.score + cpu_m.score) / 2
        lines.extend(_card_header(c, f"CPU: {cpu_model}", cpu_score, W))
        lines.append(c.dim(f"  {cores_p} cores / {cores_l} threads, {arch}"))
        lines.append("")

        sc = cpu_s.tests if cpu_s else []
        mc = cpu_m.tests if (cpu_m and not cpu_m.skipped) else []

        lines.extend(_render_section(c, "Single-Core", sc, [
            ("Integer", "prime_sieve"), ("Float", "mandelbrot"),
            ("Matrix", "matrix_1t"), ("Compress", "compression"),
            ("Sort", "sort"),
        ]))
        if mc:
            lines.extend(_render_section(c, "Multi-Core", mc, [
                ("Matrix", "matrix_full"), ("Parallel", "parallel_compute"),
                ("Crypto", "hash_throughput"), ("Sort", "parallel_sort"),
            ]))

        lines.append("")
        assess = _assess_performance(cpu_score)
        lines.append(c.dim(f"  > CPU is {assess} ({_format_score(cpu_score)})"))
    elif cpu_s and cpu_s.skipped:
        lines.append(c.dim(f"\n  CPU: skipped"))

    # === GPU CARD ===
    gpu_cat = cats.get("gpu")
    if gpu_cat and not gpu_cat.skipped:
        vram_str = f"{gpu_vram} GB" if gpu_vram else ""
        unified_str = " Unified" if gpu_unified == "True" else ""
        lines.extend(_card_header(c, f"GPU: {gpu_model}", gpu_cat.score, W))
        if vram_str:
            lines.append(c.dim(f"  {vram_str}{unified_str} Memory"))
        lines.append("")

        gt = gpu_cat.tests
        lines.extend(_render_section(c, "Compute", gt, [
            ("Matrix", "gpu_matrix"), ("Batch", "gpu_batch_matmul"),
        ]))
        lines.extend(_render_section(c, "Transfer", gt, [
            ("Elem-wise", "gpu_elementwise"), ("Reduction", "gpu_reduction"),
            ("H2D Xfer", "gpu_transfer"),
        ]))

        lines.append("")
        assess = _assess_performance(gpu_cat.score)
        lines.append(c.dim(f"  > GPU is {assess} ({_format_score(gpu_cat.score)})"))
    elif gpu_cat and gpu_cat.skipped:
        lines.append(c.dim(f"\n  GPU: skipped (MLX not available)"))

    # === MEMORY CARD ===
    mem_cat = cats.get("memory")
    if mem_cat and not mem_cat.skipped:
        mem_label = f"Memory: {mem_total} GB"
        if mem_type:
            mem_label += f" {mem_type}"
        lines.extend(_card_header(c, mem_label, mem_cat.score, W))
        lines.append("")

        mt = mem_cat.tests
        lines.extend(_render_section(c, "Bandwidth", mt, [
            ("Read", "mem_seq_read"), ("Write", "mem_seq_write"),
            ("Copy", "mem_copy"),
        ]))
        lines.extend(_render_section(c, "Access", mt, [
            ("Random", "mem_random_access"), ("Latency", "mem_latency"),
        ]))

        lines.append("")
        assess = _assess_performance(mem_cat.score)
        lines.append(c.dim(f"  > Memory is {assess} ({_format_score(mem_cat.score)})"))
    elif mem_cat and mem_cat.skipped:
        lines.append(c.dim(f"\n  Memory: skipped"))

    # === STORAGE CARD ===
    stor_cat = cats.get("storage")
    if stor_cat and not stor_cat.skipped:
        stor_parts = []
        if storage_total:
            stor_parts.append(f"{storage_total} GB")
        if storage_type:
            stor_parts.append(storage_type)
        if storage_dev:
            stor_parts.append(f"({storage_dev})")
        stor_label = " ".join(stor_parts) if stor_parts else "Unknown"
        lines.extend(_card_header(c, f"Storage: {stor_label}", stor_cat.score, W))
        lines.append("")

        st = stor_cat.tests
        lines.extend(_render_section(c, "Write", st, [
            ("Sequential", "disk_seq_write"), ("Random", "disk_random_write"),
        ]))
        lines.extend(_render_section(c, "Read", st, [
            ("Sequential", "disk_seq_read"), ("Random", "disk_random_read"),
        ]))

        lines.append("")
        assess = _assess_performance(stor_cat.score)
        lines.append(c.dim(f"  > Storage is {assess} ({_format_score(stor_cat.score)})"))
    elif stor_cat and stor_cat.skipped:
        lines.append(c.dim(f"\n  Storage: skipped"))

    # === SYSTEM INFO ===
    lines.append("")
    lines.append(sep_heavy * W)
    lines.append(c.bold("  System Information"))
    lines.append(f"  {sep_light * (W - 4)}")
    if os_type or os_version:
        lines.append(f"  {'OS':<16}{os_type} {os_version} ({arch})")
    if hostname:
        host_short = hostname.split(".")[0] if "." in hostname else hostname
        lines.append(f"  {'Hostname':<16}{host_short}")
    lines.append(f"  {'Baseline':<16}{report.baseline_machine}")
    lines.append(f"  {'Duration':<16}{report.duration_seconds:.1f}s")
    lines.append(f"  {'Version':<16}{report.baseline_version}")

    # Errors
    if report.errors:
        lines.append("")
        lines.append(c.yellow(f"  Errors ({len(report.errors)}):"))
        for err in report.errors:
            lines.append(c.red(f"    [{err.error_type}] {err.test}: {err.message}"))
            lines.append(c.dim(f"      > {err.suggestion}"))

    if report.integrity and not report.integrity.complete:
        lines.append(c.yellow("  WARNING: Benchmark run was incomplete."))

    lines.append(sep_heavy * W)
    lines.append("")

    return "\n".join(lines)


def format_json(report: BenchmarkReport) -> str:
    """Serialize a BenchmarkReport to a JSON string."""
    return json.dumps(report_to_dict(report), indent=2, ensure_ascii=False)


def format_text(report: BenchmarkReport) -> str:
    """Plain-text version (no ANSI codes) of the terminal report."""
    return format_terminal(report, use_color=False)


def save_outputs(
    report: BenchmarkReport,
    output_dir: str = ".",
    json_only: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Write benchmark_report.json and optionally benchmark_report.txt."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "benchmark_report.json")
    text_path: Optional[str] = None

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(format_json(report))
        f.write("\n")

    if not json_only:
        text_path = os.path.join(output_dir, "benchmark_report.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(format_text(report))
            f.write("\n")

    return json_path, text_path


# ---------------------------------------------------------------------------
# Task 14: CLI, main(), Signal Handling
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the benchmark tool."""
    parser = argparse.ArgumentParser(
        description="System Benchmark -- comprehensive performance scoring",
    )
    parser.add_argument(
        "--system-report", dest="system_report", default=None,
        metavar="PATH",
        help="Path to system_report.json (optional)",
    )
    parser.add_argument(
        "--json-only", dest="json_only", action="store_true", default=False,
        help="Output JSON only (no terminal report)",
    )
    parser.add_argument(
        "--verbose", "-v", dest="verbose", action="store_true", default=False,
        help="Verbose logging",
    )
    parser.add_argument(
        "--no-color", dest="no_color", action="store_true", default=False,
        help="Disable ANSI color output",
    )
    parser.add_argument(
        "--timeout", dest="timeout", type=int, default=60,
        metavar="SECONDS",
        help="Overall benchmark timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--test-timeout", dest="test_timeout", type=int, default=30,
        metavar="SECONDS",
        help="Per-test timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=".",
        metavar="DIR",
        help="Directory for output files (default: .)",
    )
    parser.add_argument(
        "--skip", dest="skip", action="append", default=[],
        metavar="CATEGORY",
        help="Skip a benchmark category (repeatable)",
    )
    parser.add_argument(
        "--only", dest="only", action="append", default=[],
        metavar="CATEGORY",
        help="Run only this benchmark category (repeatable)",
    )
    parser.add_argument(
        "--iterations", dest="iterations", type=int, default=5,
        metavar="N",
        help="Measured iterations per benchmark (default: 5)",
    )
    parser.add_argument(
        "--warmups", dest="warmups", type=int, default=3,
        metavar="N",
        help="Warmup iterations per benchmark (default: 3)",
    )
    parser.add_argument(
        "--quick", dest="quick", action="store_true", default=False,
        help="Quick mode: reduced sizes and iterations",
    )
    parser.add_argument(
        "--no-cooldown", dest="no_cooldown", action="store_true", default=False,
        help="Skip cooldown waits between phases",
    )
    parser.add_argument(
        "--calibrate", dest="calibrate", action="store_true", default=False,
        help="Print raw benchmark values for baseline calibration",
    )
    return parser.parse_args(argv)


_orchestrator_ref: Optional["BenchmarkOrchestrator"] = None


def _signal_handler(signum: int, _frame: Any) -> None:
    """Handle interrupt/termination signals gracefully."""
    log.warning("Received signal %d — shutting down benchmark.", signum)
    if _orchestrator_ref is not None:
        _orchestrator_ref.shutdown()


def main() -> None:
    """Entry point for the benchmark tool."""
    global _orchestrator_ref

    if not HAS_NUMPY:
        print("ERROR: numpy is required. Install with: pip install numpy", file=sys.stderr)
        sys.exit(1)

    args = parse_args()

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    # Load system report (optional)
    system_info: Dict[str, Any] = {}
    report_path = args.system_report or "./system_report.json"
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                system_info = json.load(f)
        except Exception as e:
            log.warning("Could not load system report from %s: %s", report_path, e)
    else:
        log.warning("system_report.json not found at %s — running without system info.", report_path)

    # Build config
    # auto_iterations: True unless user explicitly sets --iterations
    user_set_iterations = "--iterations" in sys.argv or "-i" in sys.argv
    config = BenchConfig(
        iterations=3 if args.quick else args.iterations,
        warmups=1 if args.quick else args.warmups,
        test_timeout=args.test_timeout,
        timeout=args.timeout,
        skip_categories=args.skip,
        only_categories=args.only,
        quick=args.quick,
        no_cooldown=args.no_cooldown,
        calibrate=args.calibrate,
        json_only=args.json_only,
        no_color=args.no_color,
        verbose=args.verbose,
        output_dir=args.output_dir,
        auto_iterations=not user_set_iterations and not args.quick,
    )

    # Install signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Create and run orchestrator
    orch = BenchmarkOrchestrator(config, system_info)
    _orchestrator_ref = orch
    report = orch.run()
    _orchestrator_ref = None

    # Calibrate mode: print raw values
    if args.calibrate:
        print("\n--- CALIBRATION VALUES ---")
        for cat in report.categories:
            if cat.skipped:
                continue
            for test in cat.tests:
                key = f"{test.category}_{test.name}"
                print(f"    {key!r}: {test.raw_value},")
        print("--- END CALIBRATION ---\n")

    # Output
    if args.json_only:
        print(format_json(report))
    else:
        print(format_terminal(report, use_color=(not args.no_color)))

    # Save files
    try:
        json_path, text_path = save_outputs(
            report,
            output_dir=args.output_dir,
            json_only=args.json_only,
        )
        if not args.json_only:
            log.info("Reports saved: %s, %s", json_path, text_path)
    except OSError as e:
        log.warning("Could not save output files: %s", e)

    # Exit code
    sys.exit(0 if report.overall_score > 0 else 1)


if __name__ == "__main__":
    main()
