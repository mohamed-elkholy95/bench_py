# System Benchmark Suite — Design Spec

**Date:** 2026-04-13
**Status:** Draft
**Scope:** Comprehensive system benchmark tool (`bench.py`) — CPU, GPU, Memory, Storage scoring with Geekbench-style normalized scoring
**Prerequisite:** `fetch.py` system inventory (provides `system_report.json`)

---

## 1. Purpose

A single-file Python benchmark suite that measures CPU (single + multi-core), GPU compute, memory bandwidth, and storage I/O performance. Produces a normalized composite score calibrated against a baseline machine (Apple M4 Max = 1000). Designed as the companion to `fetch.py` — run fetch first to inventory the hardware, then bench to measure it.

**Use cases:**
- Compare machines against each other via a single score
- Track performance changes over time (OS updates, hardware swaps)
- Validate hardware is performing to spec
- Feed structured benchmark data into future analysis tools

The JSON output (`benchmark_report.json`) is the stable contract for downstream tools.

---

## 2. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| File structure | Single file (`bench.py`) | Matches fetch.py — personal tool, just run it |
| Input | Reads `system_report.json` | fetch.py's JSON is the stable contract (per fetch spec) |
| Dependencies | numpy required; MLX, psutil optional | numpy gives BLAS-backed matrix ops. MLX for Apple Silicon GPU. psutil for resource monitoring. |
| Benchmark pattern | Registry of benchmark functions | Flat, isolated, easy to extend — mirrors fetch.py COLLECTORS |
| Scoring | Weighted geometric mean, baseline-normalized to 1000 | Industry standard per Geekbench/PassMark. Geometric mean is invariant to normalization (ACM benchmark statistics). |
| Output | Terminal + JSON + plain text | Consistency with fetch.py |
| Execution model | Phased sequential with cooldowns | Reproducibility — concurrent tests invalidate each other |
| Process isolation | subprocess per test via multiprocessing | Crash/OOM in one test cannot take down the suite |
| Baseline machine | Apple M4 Max / 36GB / macOS = 1000 | Author's machine — calibrated once, frozen |

---

## 3. Architecture

### 3.1 File Layout (single file, top-to-bottom)

```
bench.py
  1. Imports & constants
  2. Dependency checks (numpy required, MLX/psutil optional)
  3. Dataclasses (BenchmarkResult, CategoryScore, BenchmarkReport, etc.)
  4. Error classification & retry logic
  5. SystemProbe (pre-flight checks)
  6. ResourceGuard (runtime monitoring)
  7. CooldownManager (thermal management)
  8. TestExecutor (subprocess isolation, CPU affinity)
  9. Benchmark functions (20 tests across 5 categories)
 10. Benchmark registry (ordered list of name -> function -> category)
 11. BenchmarkOrchestrator (phased scheduler, runs the suite)
 12. Scoring engine (normalization, geometric mean, weight distribution)
 13. Baseline constants (hardcoded from M4 Max calibration)
 14. Output formatters (terminal, JSON, text)
 15. CLI argument parsing
 16. main()
```

### 3.2 Execution Flow

```
bench.py invoked
    |
    v
Load system_report.json (optional — warn if missing, continue)
    |
    v
Dependency check (numpy required, detect MLX/psutil)
    |
    v
Pre-flight SystemProbe
  - Sample CPU idle % over 3s (require >80%)
  - Check available RAM (require >2GB)
  - Check disk free space (require >1GB for I/O tests)
  - Check thermal state on macOS via pmset
  - Check if plugged in (warn on battery)
  - Detect heavy background processes
  - Report: ready / warnings / blockers
    |
    v
Build phase schedule (skip categories based on --skip/--only/missing deps)
    |
    v
For each phase:
  1. WARMUP — short throwaway computation to stabilize CPU clocks
  2. CPU_SINGLE — 5 tests, single-threaded
  3. COOLDOWN — wait for thermal recovery
  4. CPU_MULTI — 4 tests, all cores
  5. COOLDOWN — wait for thermal recovery
  6. MEMORY — 4 tests
  7. GPU — 4 tests (if MLX available)
  8. STORAGE — 4 tests
  9. FINALIZE — calculate scores, write reports
    |
    v
For each test within a phase:
  1. ResourceGuard starts monitoring (daemon thread)
  2. TestExecutor forks subprocess
  3. Run warmup iterations (discard)
  4. Run measured iterations (collect times)
  5. ResourceGuard stops, attaches ResourceSummary
  6. Retry on transient failure (up to 2 retries)
  7. Record BenchmarkResult or BenchmarkError
  8. 1s pause + gc.collect() before next test
    |
    v
Score calculation
  - Raw throughput -> normalize against BASELINE -> per-test score
  - Geometric mean of test scores -> category score
  - Weighted geometric mean of category scores -> overall score
    |
    v
Output: terminal display + benchmark_report.json + benchmark_report.txt
```

### 3.3 Orchestrator

```python
class BenchmarkOrchestrator:
    """Top-level controller. Owns the full lifecycle."""

    def __init__(self, config: BenchConfig, system_info: Dict):
        self.config = config
        self.system_info = system_info
        self.probe = SystemProbe()
        self.guard = ResourceGuard()
        self.cooldown = CooldownManager()
        self.executor = TestExecutor()
        self.results: List[BenchmarkResult] = []
        self.errors: List[BenchmarkError] = []
        self._shutdown = False

    def run(self) -> BenchmarkReport:
        """Execute the full benchmark suite."""
        # 1. Pre-flight
        # 2. Iterate phases
        # 3. Score
        # 4. Assemble report

    def shutdown(self) -> None:
        """Signal graceful stop — called from signal handler."""
        self._shutdown = True
        self.executor.kill_active()
```

---

## 4. Benchmark Categories & Tests

### 4.1 CPU Single-Core (weight: 25%)

| Test | Measures | Method | Unit |
|------|----------|--------|------|
| `prime_sieve` | Integer ALU, branch prediction | Sieve of Eratosthenes to N=1,000,000 | ops/sec |
| `mandelbrot` | FPU pipeline, precision | Mandelbrot set 1024x1024 grid, max 100 iterations, pure Python single thread | pixels/sec |
| `matrix_1t` | BLAS single-thread | numpy matmul 1024x1024 float64 with `OMP_NUM_THREADS=1` | GFLOPS |
| `compression` | Mixed integer + memory | zlib compress then decompress 10MB random buffer | MB/s |
| `sort` | Cache efficiency, branching | `sorted()` on 10M random integers | M_elements/sec |

### 4.2 CPU Multi-Core (weight: 25%)

| Test | Measures | Method | Unit |
|------|----------|--------|------|
| `matrix_full` | Parallel BLAS throughput | numpy matmul 4096x4096 float64 (all threads) | GFLOPS |
| `parallel_compute` | Process scaling efficiency | Mandelbrot split across `multiprocessing.Pool(N)`, N=logical cores | pixels/sec |
| `hash_throughput` | Crypto pipeline | SHA-256 hash 100MB split across N processes | MB/s |
| `parallel_sort` | Memory bandwidth under contention | Sort 10M integers in each of N parallel processes | M_elements/sec |

### 4.3 GPU Compute (weight: 20%)

Runs only if MLX is available. Otherwise skipped, weights redistributed.

| Test | Measures | Method | Unit |
|------|----------|--------|------|
| `gpu_matrix` | GPU FLOPS | `mx.matmul` 4096x4096 float32, `mx.eval()` to force sync | GFLOPS |
| `gpu_elementwise` | GPU memory bandwidth | Large array (32M elements) add + multiply + exp chain | GB/s |
| `gpu_reduction` | GPU reduction pipeline | `mx.sum` + `mx.mean` over 64M element array | GB/s |
| `gpu_batch_matmul` | Sustained GPU throughput | 64x batched 512x512 float32 matmuls | GFLOPS |

### 4.4 Memory (weight: 15%)

| Test | Measures | Method | Unit |
|------|----------|--------|------|
| `mem_seq_read` | Read bandwidth | Iterate over 256MB numpy array sequentially | GB/s |
| `mem_seq_write` | Write bandwidth | Write to 256MB numpy array | GB/s |
| `mem_random_access` | Latency | Random index reads across 256MB array (1M accesses) | M_accesses/sec |
| `mem_copy` | memcpy throughput | `numpy.copy()` on 256MB array | GB/s |

### 4.5 Storage I/O (weight: 15%)

Uses `tempfile.mkdtemp()` in the output directory. Cleans up after.

| Test | Measures | Method | Unit |
|------|----------|--------|------|
| `disk_seq_write` | Sequential write throughput | Write 256MB file in 1MB chunks | MB/s |
| `disk_seq_read` | Sequential read throughput | Read that file back (with `os.posix_fadvise` to bypass cache on Linux) | MB/s |
| `disk_random_write` | Write IOPS | Write 1000 random 4KB blocks to a pre-allocated file | IOPS |
| `disk_random_read` | Read IOPS | Read 1000 random 4KB blocks | IOPS |

### 4.6 Weight Redistribution

When a category is skipped (missing dependency, all tests failed, or `--skip`):

```python
def redistribute_weights(categories: List[CategoryScore]) -> Dict[str, float]:
    """Redistribute weights proportionally among active categories."""
    active = [c for c in categories if not c.skipped]
    total_active_weight = sum(c.weight for c in active)
    return {c.name: c.weight / total_active_weight for c in active}
```

Example: GPU skipped (0.20 removed) -> CPU Single 0.3125, CPU Multi 0.3125, Memory 0.1875, Storage 0.1875.

---

## 5. Scoring System

### 5.1 Per-Test Scoring

Every test produces a raw throughput value (higher = better). All time-based measurements are converted to throughput first.

```python
def compute_test_score(raw_value: float, baseline_value: float) -> float:
    """Normalize a raw measurement against the baseline.
    
    Returns a score where 1000 = baseline machine.
    """
    if baseline_value <= 0:
        return 0.0
    return (raw_value / baseline_value) * 1000.0
```

### 5.2 Per-Category Scoring

Geometric mean of all completed test scores within the category:

```python
import math

def geometric_mean(scores: List[float]) -> float:
    """Geometric mean — the correct way to summarize benchmark scores.
    
    Invariant to normalization, prevents one outlier from dominating.
    Requires all scores > 0.
    """
    if not scores or any(s <= 0 for s in scores):
        return 0.0
    log_sum = sum(math.log(s) for s in scores)
    return math.exp(log_sum / len(scores))
```

If fewer than 2 tests completed in a category, the category is marked as skipped (insufficient data for a meaningful geometric mean).

### 5.3 Overall Score

Weighted geometric mean across active categories:

```python
def overall_score(categories: List[CategoryScore]) -> float:
    """Weighted geometric mean of category scores."""
    weights = redistribute_weights(categories)
    active = [c for c in categories if not c.skipped]
    if not active:
        return 0.0
    log_sum = sum(weights[c.name] * math.log(c.score) for c in active if c.score > 0)
    total_weight = sum(weights[c.name] for c in active if c.score > 0)
    if total_weight <= 0:
        return 0.0
    return math.exp(log_sum / total_weight)
```

### 5.4 Score Interpretation

| Score | Meaning |
|-------|---------|
| 1000 | Equal to baseline (M4 Max) |
| 2000 | 2x faster than baseline |
| 500 | Half the speed of baseline |
| >1000 | Faster than baseline |
| <1000 | Slower than baseline |

### 5.5 Statistical Rigor

Each test runs:
- **3 warmup iterations** (discarded — stabilize CPU clocks, fill caches)
- **5 measured iterations** (or 3 in `--quick` mode)

Score uses the **median** of measured runs (not mean — resistant to outliers from background OS activity). Standard deviation is recorded in JSON for reproducibility assessment.

```python
def compute_median(times: List[float]) -> float:
    s = sorted(times)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2
```

### 5.6 Baseline Constants

Hardcoded values measured on the author's M4 Max. Populated by running `bench.py --calibrate`:

```python
BASELINE_MACHINE = "Apple M4 Max / 36GB / macOS 26.4"
BASELINE_VERSION = "1.0"

BASELINE = {
    # CPU Single-Core
    "cpu_single_prime_sieve": 0.0,      # ops/sec — filled by calibration
    "cpu_single_mandelbrot": 0.0,       # pixels/sec
    "cpu_single_matrix_1t": 0.0,        # GFLOPS
    "cpu_single_compression": 0.0,      # MB/s
    "cpu_single_sort": 0.0,             # M_elements/sec

    # CPU Multi-Core
    "cpu_multi_matrix_full": 0.0,       # GFLOPS
    "cpu_multi_parallel_compute": 0.0,  # pixels/sec
    "cpu_multi_hash_throughput": 0.0,   # MB/s
    "cpu_multi_parallel_sort": 0.0,     # M_elements/sec

    # GPU Compute
    "gpu_matrix": 0.0,                  # GFLOPS
    "gpu_elementwise": 0.0,             # GB/s
    "gpu_reduction": 0.0,               # GB/s
    "gpu_batch_matmul": 0.0,            # GFLOPS

    # Memory
    "mem_seq_read": 0.0,                # GB/s
    "mem_seq_write": 0.0,               # GB/s
    "mem_random_access": 0.0,           # M_accesses/sec
    "mem_copy": 0.0,                    # GB/s

    # Storage
    "disk_seq_write": 0.0,              # MB/s
    "disk_seq_read": 0.0,               # MB/s
    "disk_random_write": 0.0,           # IOPS
    "disk_random_read": 0.0,            # IOPS
}
```

These are populated once via `--calibrate` and then frozen in the source. The `BASELINE_VERSION` is bumped only if baselines are recalibrated (which invalidates prior scores).

---

## 6. Error Handling — Three Layers

### Layer 1: TestExecutor (innermost)

Each benchmark iteration runs in a forked subprocess via `multiprocessing.Process`:

- Hard timeout per iteration (default 30s, configurable via `--test-timeout`)
- Subprocess crash (segfault, OOM) returns `TestCrashed` — orchestrator is unaffected
- Result passed back via `multiprocessing.Queue` (serialized, no shared state)
- Orphan process cleanup in `finally` block

Error types:
- `TestTimeout` — iteration exceeded timeout, subprocess killed
- `TestCrashed` — subprocess exited non-zero or was killed by OS
- `ResourceWarning` — ResourceGuard detected critical resource state

### Layer 2: Benchmark Functions (per-test)

Degradation waterfall within each test:

```
Primary method -> Fallback (reduced size) -> Skip with BenchmarkError
```

Examples:
- GPU matrix: `MLX 4096x4096` -> `MLX 2048x2048` -> `Skip (no MLX)`
- Multi-core hash: `N processes` -> `N/2 processes` -> `Single process (degraded flag)`
- Memory sequential: `256MB` -> `128MB (low RAM)` -> `Skip`

### Layer 3: Orchestrator (outermost)

```python
def safe_benchmark(
    name: str, fn: Callable, config: BenchConfig
) -> Tuple[Optional[BenchmarkResult], Optional[BenchmarkError]]:
    """Run a benchmark safely with retry logic. Never raises."""
```

### Retry Logic

```python
@dataclass
class RetryPolicy:
    max_retries: int = 2              # up to 3 total attempts
    backoff_seconds: float = 1.0      # wait between retries
    retry_on: Tuple[type, ...] = (
        TestTimeout,                  # might be transient load
        TestCrashed,                  # might be resource contention
    )
    no_retry_on: Tuple[type, ...] = (
        ImportError,                  # missing dependency will not fix itself
        NotImplementedError,          # platform not supported
        PermissionError,              # will not change between retries
    )
```

Retry flow:
```
Attempt 1 -> failed (transient?)
  | gc.collect() + 1s backoff
Attempt 2 -> failed again?
  | gc.collect() + 2s backoff + reduce problem size 50%
Attempt 3 -> failed again?
  | Record BenchmarkError, skip test, continue suite
```

### Error Classification

```python
def classify_bench_error(exc: Exception) -> str:
    if isinstance(exc, TestTimeout):       return "timeout"
    if isinstance(exc, TestCrashed):       return "crashed"
    if isinstance(exc, MemoryError):       return "out_of_memory"
    if isinstance(exc, ImportError):       return "missing_dependency"
    if isinstance(exc, NotImplementedError): return "not_supported"
    if isinstance(exc, PermissionError):   return "permission_denied"
    if isinstance(exc, OSError):           return "io_error"
    return "unexpected"
```

### Graceful Degradation Matrix

| Situation | Behavior |
|-----------|----------|
| numpy not installed | Exit with clear error — numpy is required |
| MLX not installed | Skip GPU category, redistribute weights, log info |
| psutil not installed | Skip RAM monitoring during tests, warn once |
| Single test fails all retries | Skip test, category score uses remaining tests. If <2 tests remain, skip category |
| Entire category skipped | Redistribute weights proportionally |
| system_report.json missing | Run with `system: null` — benchmarks do not need it, just metadata |
| system_report.json corrupt | Same as missing — warn and continue |
| Low RAM detected (<2GB) | Reduce all matrix/array sizes by 50%, flag `integrity.constrained = true` |
| SIGINT during test | Kill test subprocess, save completed results, skip remaining |
| SIGINT between tests | Save all completed results, skip remaining, report as partial |
| Disk full during I/O test | Skip storage category, clean up temp files, continue |
| Thermal critical (macOS) | Pause up to 60s, if still critical: warn and continue with `integrity.constrained` flag |

---

## 7. Execution Engine

### 7.1 Pre-Flight SystemProbe

```python
@dataclass
class SystemReadiness:
    cpu_idle_pct: float          # require >80% idle
    available_ram_gb: float      # require >2GB free
    disk_free_gb: float          # require >1GB for I/O tests
    thermal_state: str           # "nominal" | "fair" | "serious" | "critical" (macOS)
    battery_plugged: bool        # warn if on battery
    background_load: List[str]   # heavy processes detected
    ready: bool                  # all checks pass
    warnings: List[str]          # non-blocking concerns
    blockers: List[str]          # must fix before running
```

| Check | Threshold | On Fail |
|-------|-----------|---------|
| CPU idle | >80% over 3s sample | Wait up to 30s for load to drop, then warn |
| Available RAM | >2GB | Block — not safe to benchmark |
| Disk free | >1GB | Skip storage tests |
| Thermal state (macOS) | Not "serious"/"critical" | Wait up to 60s for cooldown |
| On battery | Plugged in preferred | Warn — results may be throttled |
| Heavy processes | Check for compilers, VMs, Docker builds | Warn with process names |

Thermal state on macOS: `pmset -g therm`
Thermal state on Linux: `/sys/class/thermal/thermal_zone*/temp`

### 7.2 Phased Scheduler

```python
class Phase(Enum):
    WARMUP = "warmup"
    CPU_SINGLE = "cpu_single"
    COOLDOWN_1 = "cooldown_1"
    CPU_MULTI = "cpu_multi"
    COOLDOWN_2 = "cooldown_2"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    FINALIZE = "finalize"
```

Phase ordering rationale:
- CPU single-core first (before thermal buildup from multi-core)
- Cooldown between CPU phases (Apple Silicon throttles aggressively under sustained load)
- Memory tests after CPU (avoid cache pollution from prior tests)
- GPU after CPU cooldown (shared thermal envelope on Apple Silicon — unified architecture)
- Storage last (least affected by thermal state, most affected by OS caching)

### 7.3 CooldownManager

```python
@dataclass
class CooldownPolicy:
    min_seconds: float = 3.0         # always wait at least 3s between phases
    max_seconds: float = 30.0        # never wait more than 30s
    target_cpu_pct: float = 10.0     # wait until CPU drops below 10%
    target_temp_c: float = 70.0      # wait until temp drops below 70C (if readable)
    poll_interval: float = 1.0       # check every 1s during cooldown
```

Between phases: full cooldown policy.
Between tests within a phase: 1s pause + `gc.collect()`.

### 7.4 ResourceGuard

Daemon thread sampling at 500ms intervals during test execution:

```python
@dataclass
class ResourceSample:
    timestamp: float
    cpu_pct: float
    ram_available_mb: float
    temp_c: Optional[float]

@dataclass
class ResourceSummary:
    peak_cpu_pct: float
    peak_ram_used_mb: float
    peak_temp_c: Optional[float]
    min_ram_available_mb: float
    samples: List[ResourceSample]
```

Intervention thresholds:
- RAM < 512MB available: abort current test (risk of OOM-killing)
- CPU temp > 100C: pause and cool down
- Disk < 100MB free: abort storage tests

ResourceSummary is attached to each BenchmarkResult in JSON — useful for diagnosing unreliable results. When psutil is not available, `resource_summary` is `null` for all tests (monitoring is best-effort, not required).

### 7.5 TestExecutor

```python
class TestExecutor:
    """Run a single benchmark in an isolated subprocess."""

    def execute(self, fn, args, timeout, retry_policy):
        """
        1. Fork subprocess via multiprocessing.Process
        2. Pin to specific cores if possible (single-core tests)
        3. Set process priority (nice level)
        4. Run warmup iterations (discard results)
        5. Run measured iterations (collect times)
        6. Return results via multiprocessing.Queue
        7. Kill subprocess on timeout
        8. Cleanup in finally block
        """
```

CPU affinity for single-core tests (if psutil available):
- Pin to core 0 (typically a performance core)
- Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` in subprocess environment

### 7.6 Concurrency Rules

**Within a phase:** Tests run sequentially. Concurrent benchmarks invalidate each other.

**Daemon threads only:** ResourceGuard monitoring runs as a daemon thread (<1% CPU overhead). No other concurrency during measurement.

**Between phases:** Cleanup tasks (temp file removal, gc) can overlap with cooldown monitoring.

---

## 8. Data Model

### 8.1 Core Types

```python
@dataclass
class BenchmarkResult:
    name: str                    # "cpu_single_prime_sieve"
    category: str                # "cpu_single"
    raw_value: float             # measured throughput (higher = better)
    unit: str                    # "ops/sec", "GFLOPS", "MB/s", "IOPS", etc.
    score: float                 # normalized: 1000 = baseline
    iterations: int              # number of measured iterations
    warmups: int                 # number of warmup iterations
    median_time: float           # median seconds per iteration
    std_dev: float               # std dev across measured iterations
    times: List[float]           # all measured iteration times
    degraded: bool = False       # ran at reduced size
    resource_summary: Optional[Dict] = None  # ResourceSummary as dict

@dataclass
class CategoryScore:
    name: str                    # "cpu_single"
    score: float                 # geometric mean of test scores
    weight: float                # 0.25
    tests: List[BenchmarkResult]
    skipped: bool = False
    skip_reason: Optional[str] = None

@dataclass
class BenchmarkError:
    test: str                    # "gpu_matrix"
    category: str                # "gpu_compute"
    error_type: str              # "missing_dependency"
    message: str                 # "MLX not available"
    suggestion: str              # "pip install mlx"
    retries_attempted: int = 0

@dataclass
class ReportIntegrity:
    complete: bool               # all tests ran successfully
    degraded_tests: List[str]    # tests that ran at reduced size
    cpu_fallback_tests: List[str] # GPU tests that fell back to CPU
    retried_tests: List[str]     # tests that needed retries
    partial: bool                # interrupted before completion
    constrained: bool            # reduced sizes due to low resources

@dataclass
class ExecutionMetadata:
    phases_completed: int
    phases_total: int
    total_cooldown_seconds: float
    peak_cpu_temp_c: Optional[float]
    peak_ram_usage_mb: float
    pre_flight: Dict[str, Any]   # SystemReadiness as dict
    execution_mode: str          # "full", "quick", "partial"

@dataclass
class BenchmarkReport:
    overall_score: float
    categories: List[CategoryScore]
    baseline_machine: str        # "Apple M4 Max / 36GB / macOS"
    baseline_version: str        # "1.0"
    system: Optional[Dict]       # loaded from system_report.json (or null)
    skipped: List[str]           # skipped category names
    errors: List[BenchmarkError]
    integrity: ReportIntegrity
    execution: ExecutionMetadata
    duration_seconds: float
    timestamp: str               # ISO 8601
```

### 8.2 JSON Output Structure

File: `benchmark_report.json`

```json
{
  "overall_score": 1000,
  "baseline_machine": "Apple M4 Max / 36GB / macOS",
  "baseline_version": "1.0",
  "categories": [
    {
      "name": "cpu_single",
      "score": 1015,
      "weight": 0.25,
      "skipped": false,
      "tests": [
        {
          "name": "prime_sieve",
          "category": "cpu_single",
          "raw_value": 823.5,
          "unit": "ops/sec",
          "score": 1000,
          "iterations": 5,
          "warmups": 3,
          "median_time": 0.00121,
          "std_dev": 0.00003,
          "times": [0.00121, 0.00122, 0.00120, 0.00121, 0.00123],
          "degraded": false
        }
      ]
    }
  ],
  "skipped": [],
  "errors": [],
  "integrity": {
    "complete": true,
    "degraded_tests": [],
    "cpu_fallback_tests": [],
    "retried_tests": [],
    "partial": false,
    "constrained": false
  },
  "execution": {
    "phases_completed": 8,
    "phases_total": 8,
    "total_cooldown_seconds": 12.4,
    "peak_cpu_temp_c": 78.2,
    "peak_ram_usage_mb": 4200,
    "execution_mode": "full"
  },
  "system": {},
  "duration_seconds": 45.2,
  "timestamp": "2026-04-13T20:00:00Z"
}
```

---

## 9. Terminal Output

### 9.1 Progress During Execution

```
System Benchmark v1.0
------------------------------------------------------------
Pre-flight ............ OK (CPU 4%, 28.3 GB free, plugged in)

[1/5] CPU Single-Core
  prime_sieve ......... 1023  (0.34s, std=0.002)
  mandelbrot .......... 987   (1.21s, std=0.015)
  matrix_1t ........... 1005  (0.89s, std=0.008)
  compression ......... 1102  (0.45s, std=0.004)
  sort ................ 965   (0.72s, std=0.011)
  -> Category score: 1015

  Cooling down... 4.2s (CPU 3%, 62C)

[2/5] CPU Multi-Core
  matrix_full ......... 1044  (2.13s, std=0.031)
  parallel_compute .... 998   (3.45s, std=0.042)
  hash_throughput ..... 1021  (1.87s, std=0.019)
  parallel_sort ....... 976   (2.89s, std=0.037)
  -> Category score: 1010

  Cooling down... 5.1s (CPU 5%, 65C)

[3/5] Memory
  seq_read ............ 1050  (0.12s, std=0.001)
  seq_write ........... 980   (0.15s, std=0.002)
  random_access ....... 1010  (0.43s, std=0.008)
  copy ................ 1005  (0.13s, std=0.001)
  -> Category score: 1011

[4/5] GPU Compute
  gpu_matrix .......... 1030  (0.08s, std=0.001)
  gpu_elementwise ..... 995   (0.05s, std=0.001)
  gpu_reduction ....... 1015  (0.03s, std=0.001)
  gpu_batch_matmul .... 1008  (0.11s, std=0.002)
  -> Category score: 1012

[5/5] Storage I/O
  disk_seq_write ...... 990   (1.24s, std=0.045)
  disk_seq_read ....... 1010  (0.98s, std=0.032)
  disk_random_write ... 985   (2.34s, std=0.089)
  disk_random_read .... 1005  (1.87s, std=0.067)
  -> Category score: 997

------------------------------------------------------------
                    OVERALL SCORE: 1010
------------------------------------------------------------

  CPU Single-Core    1015   (25%)
  CPU Multi-Core     1010   (25%)
  Memory             1011   (15%)
  GPU Compute        1012   (20%)
  Storage I/O         997   (15%)

  Baseline: Apple M4 Max / 36GB / macOS (= 1000)

Completed in 45.2s -- 20 tests, 0 errors
------------------------------------------------------------

  JSON   benchmark_report.json
  Text   benchmark_report.txt
```

### 9.2 Color Rules

Same as fetch.py:
- ANSI escape codes only — no external dependency
- Auto-disabled when stdout is not a TTY
- `--no-color` forces plain output
- Box-drawing characters degrade to ASCII dashes in non-TTY
- Scores colored: green (>1000), white (=1000), yellow (<800), red (<500)

---

## 10. CLI Interface

```
usage: bench.py [options]

Options:
  --system-report PATH   Path to system_report.json (default: ./system_report.json)
  --json-only            JSON output only (no terminal, no text file)
  --verbose, -v          Debug logging
  --no-color             Disable colored output
  --timeout SECS         Per-test total timeout (default: 60)
  --test-timeout SECS    Per-iteration timeout (default: 30)
  --output-dir DIR       Output directory (default: current)
  --skip CATEGORY        Skip a category (repeatable: --skip gpu --skip storage)
  --only CATEGORY        Run only these categories (repeatable)
  --iterations N         Measured iterations per test (default: 5)
  --warmups N            Warmup iterations per test (default: 3)
  --quick                Quick mode: 3 iterations, 1 warmup, reduced problem sizes
  --no-cooldown          Disable cooldown pauses (less accurate, faster)
  --calibrate            Run suite and print raw values for populating BASELINE
```

---

## 11. Signal Handling & Graceful Shutdown

Same pattern as fetch.py:

- `SIGINT` (Ctrl+C) and `SIGTERM` set `_shutdown` flag on orchestrator
- Active test subprocess is terminated via `proc.terminate()` then `proc.wait(timeout=3)`, escalating to `proc.kill()`
- Orchestrator checks `_shutdown` between tests — stops running new tests
- All completed results are saved to output files (partial report)
- Terminal shows which tests were skipped
- Exit code: 0 if overall score computed, 1 if nothing completed

---

## 12. Dependencies

| Package | Required? | Purpose |
|---------|-----------|---------|
| numpy | **Required** | BLAS matrix ops, array operations for memory/CPU benchmarks |
| mlx | Optional | Apple Silicon GPU compute benchmarks |
| psutil | Optional | Resource monitoring, CPU affinity, thermal monitoring |

If numpy is missing, bench.py exits immediately with:
```
Error: numpy is required. Install with: pip install numpy
```

---

## 13. Non-Goals

- **Not a stress test** — measures performance, not stability. Tests are seconds, not hours.
- **Not a monitoring tool** — runs once, exits. No daemon mode.
- **Not a package** — no setup.py, no pyproject.toml. Just `python bench.py`.
- **No network benchmarks** — pure local computation and I/O.
- **No disk-to-disk comparison** — storage tests use whatever disk the output dir is on.
- **No historical database** — each run produces a standalone JSON. Comparison is done by diffing JSONs externally.
