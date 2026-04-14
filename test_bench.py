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
    assert abs(geometric_mean([1000.0, 1000.0, 1000.0]) - 1000.0) < 0.01

def test_geometric_mean_known_value():
    from bench import geometric_mean
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


# ---------------------------------------------------------------------------
# TestExecutor — subprocess isolation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Task 5: SystemProbe, CooldownManager, ResourceGuard
# ---------------------------------------------------------------------------

def test_system_probe_returns_readiness():
    from bench import SystemProbe, SystemReadiness
    probe = SystemProbe()
    readiness = probe.check()
    assert isinstance(readiness, SystemReadiness)
    assert isinstance(readiness.ready, bool)
    assert readiness.available_ram_gb > 0


def test_cooldown_policy_defaults():
    from bench import CooldownPolicy
    p = CooldownPolicy()
    assert p.min_seconds == 3.0
    assert p.max_seconds == 30.0


def test_cooldown_manager_respects_no_cooldown():
    from bench import CooldownManager, CooldownPolicy
    mgr = CooldownManager()
    policy = CooldownPolicy(min_seconds=0, max_seconds=0)
    result = mgr.wait(policy)
    assert result["waited_seconds"] >= 0


def test_resource_guard_start_stop():
    from bench import ResourceGuard
    guard = ResourceGuard()
    guard.start()
    time.sleep(0.6)
    summary = guard.stop()
    assert isinstance(summary, dict)
    assert "peak_cpu_pct" in summary or summary == {}


# ---------------------------------------------------------------------------
# Task 6: CPU single-core benchmarks
# ---------------------------------------------------------------------------

def test_bench_prime_sieve():
    from bench import bench_prime_sieve
    assert bench_prime_sieve(100_000) > 0


def test_bench_mandelbrot():
    from bench import bench_mandelbrot
    assert bench_mandelbrot(128) > 0


def test_bench_matrix_single():
    from bench import bench_matrix_single
    assert bench_matrix_single(256) > 0


def test_bench_compression():
    from bench import bench_compression
    assert bench_compression(1) > 0


def test_bench_sort():
    from bench import bench_sort
    assert bench_sort(100_000) > 0


# ---------------------------------------------------------------------------
# Task 7: CPU multi-core benchmarks
# ---------------------------------------------------------------------------

def test_bench_matrix_multi():
    from bench import bench_matrix_multi
    assert bench_matrix_multi(256) > 0


def test_bench_parallel_compute():
    from bench import bench_parallel_compute
    assert bench_parallel_compute(128) > 0


def test_bench_hash_throughput():
    from bench import bench_hash_throughput
    assert bench_hash_throughput(1) > 0


def test_bench_parallel_sort():
    from bench import bench_parallel_sort
    assert bench_parallel_sort(100_000) > 0


# ---------------------------------------------------------------------------
# Task 8: GPU compute benchmarks
# ---------------------------------------------------------------------------

def test_bench_gpu_matrix():
    from bench import bench_gpu_matrix, HAS_MLX
    if not HAS_MLX:
        pytest.skip("MLX not available")
    assert bench_gpu_matrix(512) > 0


def test_bench_gpu_elementwise():
    from bench import bench_gpu_elementwise, HAS_MLX
    if not HAS_MLX:
        pytest.skip("MLX not available")
    assert bench_gpu_elementwise(1_000_000) > 0


def test_bench_gpu_reduction():
    from bench import bench_gpu_reduction, HAS_MLX
    if not HAS_MLX:
        pytest.skip("MLX not available")
    assert bench_gpu_reduction(1_000_000) > 0


def test_bench_gpu_batch_matmul():
    from bench import bench_gpu_batch_matmul, HAS_MLX
    if not HAS_MLX:
        pytest.skip("MLX not available")
    assert bench_gpu_batch_matmul(4, 128) > 0


# ---------------------------------------------------------------------------
# Task 9: Memory benchmarks
# ---------------------------------------------------------------------------

def test_bench_mem_seq_read():
    from bench import bench_mem_seq_read
    assert bench_mem_seq_read(16) > 0


def test_bench_mem_seq_write():
    from bench import bench_mem_seq_write
    assert bench_mem_seq_write(16) > 0


def test_bench_mem_random_access():
    from bench import bench_mem_random_access
    assert bench_mem_random_access(16, 100_000) > 0


def test_bench_mem_copy():
    from bench import bench_mem_copy
    assert bench_mem_copy(16) > 0


# ---------------------------------------------------------------------------
# Task 10: Storage I/O benchmarks
# ---------------------------------------------------------------------------

def test_bench_disk_seq_write():
    from bench import bench_disk_seq_write
    with tempfile.TemporaryDirectory() as tmpdir:
        assert bench_disk_seq_write(tmpdir, 8) > 0


def test_bench_disk_seq_read():
    from bench import bench_disk_seq_read
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.bin")
        with open(path, "wb") as f:
            f.write(os.urandom(8 * 1024 * 1024))
        assert bench_disk_seq_read(path) > 0


def test_bench_disk_random_write():
    from bench import bench_disk_random_write
    with tempfile.TemporaryDirectory() as tmpdir:
        assert bench_disk_random_write(tmpdir, 100) > 0


def test_bench_disk_random_read():
    from bench import bench_disk_random_read
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.bin")
        with open(path, "wb") as f:
            f.write(os.urandom(1024 * 1024))
        assert bench_disk_random_read(path, 100) > 0
