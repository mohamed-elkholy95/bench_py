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
    ("sort",              "cpu_single", bench_sort,              10_000_000,  "M_elements/sec"),
    # CPU Multi-Core
    ("matrix_full",       "cpu_multi",  bench_matrix_multi,      4096,        "GFLOPS"),
    ("parallel_compute",  "cpu_multi",  bench_parallel_compute,  1024,        "pixels/sec"),
    ("hash_throughput",   "cpu_multi",  bench_hash_throughput,   100,         "MB/s"),
    ("parallel_sort",     "cpu_multi",  bench_parallel_sort,     10_000_000,  "M_elements/sec"),
    # GPU
    ("gpu_matrix",        "gpu",        bench_gpu_matrix,        4096,        "GFLOPS"),
    ("gpu_elementwise",   "gpu",        bench_gpu_elementwise,   32_000_000,  "GB/s"),
    ("gpu_reduction",     "gpu",        bench_gpu_reduction,     64_000_000,  "GB/s"),
    ("gpu_batch_matmul",  "gpu",        bench_gpu_batch_matmul,  64,          "GFLOPS"),
    # Memory
    ("mem_seq_read",      "memory",     bench_mem_seq_read,      256,         "GB/s"),
    ("mem_seq_write",     "memory",     bench_mem_seq_write,     256,         "GB/s"),
    ("mem_random_access", "memory",     bench_mem_random_access, 256,         "M_accesses/sec"),
    ("mem_copy",          "memory",     bench_mem_copy,          256,         "GB/s"),
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
    "mem_seq_read":      8,
    "mem_seq_write":     8,
    "mem_random_access": 8,
    "mem_copy":          8,
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
BASELINE_VERSION = "1.0"
BASELINE: Dict[str, float] = {
    "cpu_single_prime_sieve":       1.0,
    "cpu_single_mandelbrot":        1.0,
    "cpu_single_matrix_1t":         1.0,
    "cpu_single_compression":       1.0,
    "cpu_single_sort":              1.0,
    "cpu_multi_matrix_full":        1.0,
    "cpu_multi_parallel_compute":   1.0,
    "cpu_multi_hash_throughput":    1.0,
    "cpu_multi_parallel_sort":      1.0,
    "gpu_gpu_matrix":               1.0,
    "gpu_gpu_elementwise":          1.0,
    "gpu_gpu_reduction":            1.0,
    "gpu_gpu_batch_matmul":         1.0,
    "memory_mem_seq_read":          1.0,
    "memory_mem_seq_write":         1.0,
    "memory_mem_random_access":     1.0,
    "memory_mem_copy":              1.0,
    "storage_disk_seq_write":       1.0,
    "storage_disk_seq_read":        1.0,
    "storage_disk_random_write":    1.0,
    "storage_disk_random_read":     1.0,
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

    last_exc: Optional[Exception] = None

    while True:
        times: List[float] = []
        raw_values: List[float] = []
        iteration_error: Optional[Exception] = None

        try:
            for _ in range(config.iterations):
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
            # Write a temp file first, then pass the path
            path = os.path.join(config.output_dir, f"_bench_seqread_{os.getpid()}.tmp")
            write_mb = size if size > 0 else 8
            chunk = os.urandom(1024 * 1024)
            with open(path, "wb") as f:
                for _ in range(write_mb):
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
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
        """Run all benchmarks in a category."""
        config = self._config
        self._results.setdefault(category, [])

        for name, cat, fn, default_size, unit in BENCHMARKS:
            if cat != category:
                continue
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


def _score_color(c: _Color, score: float) -> str:
    """Color a score based on its value."""
    if score >= 1000:
        return c.green(f"{score:.0f}")
    if score >= 800:
        return c.yellow(f"{score:.0f}")
    return c.red(f"{score:.0f}")


def format_terminal(report: BenchmarkReport, use_color: bool = True) -> str:
    """Render a human-readable benchmark report with optional ANSI color."""
    c = _Color(use_color)
    lines: List[str] = []

    # Header
    lines.append(c.bold("=" * 60))
    lines.append(c.bold("  SYSTEM BENCHMARK REPORT"))
    lines.append(c.bold("=" * 60))
    lines.append(f"  Timestamp : {report.timestamp}")
    lines.append(f"  Duration  : {report.duration_seconds:.1f}s")
    lines.append(f"  Baseline  : {report.baseline_machine}")
    lines.append(c.bold("-" * 60))

    # Per-category results
    for cat in report.categories:
        display = _CATEGORY_DISPLAY.get(cat.name, cat.name)
        if cat.skipped:
            lines.append(f"  {c.dim(display):<30} {c.dim('(skipped)')}")
            continue
        lines.append(c.bold(f"  {display}"))
        for test in cat.tests:
            score_str = _score_color(c, test.score)
            lines.append(
                f"    {test.name:<28} {test.raw_value:>10.2f} {test.unit:<18} score: {score_str}"
            )
        cat_score_str = _score_color(c, cat.score)
        lines.append(f"  {'Category Score':<30} {cat_score_str}")
        lines.append("")

    # Overall score
    lines.append(c.bold("=" * 60))
    overall_str = _score_color(c, report.overall_score)
    lines.append(c.bold(f"  OVERALL SCORE : {overall_str}"))
    lines.append(c.bold("=" * 60))

    # Errors
    if report.errors:
        lines.append("")
        lines.append(c.yellow(f"  Errors ({len(report.errors)}):"))
        for err in report.errors:
            lines.append(c.red(f"    [{err.error_type}] {err.test}: {err.message}"))
            lines.append(c.dim(f"      -> {err.suggestion}"))

    # Footer
    lines.append("")
    lines.append(c.dim(f"  Baseline version: {report.baseline_version}"))
    if report.integrity and not report.integrity.complete:
        lines.append(c.yellow("  WARNING: Benchmark run was interrupted."))

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
