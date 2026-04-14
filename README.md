# python_info

A comprehensive, cross-platform system information and benchmarking toolkit written in pure Python. Discover your hardware, software, and peripherals with `fetch.py`, then measure real-world performance across CPU, GPU, memory, and storage with `bench.py`.

---

## Features

### System Fetch (`fetch.py`)

Collects a full inventory of your machine in seconds:

| Category | Details |
|----------|---------|
| **OS & Hardware** | OS version, kernel, architecture, hostname |
| **CPU** | Model, physical/logical cores, frequency, features |
| **Memory** | Total capacity, type (DDR4/DDR5/LPDDR5), speed |
| **GPU** | Model, VRAM, unified memory detection (Apple Silicon) |
| **Storage** | Devices, mount points, capacity, free space, disk type |
| **Network** | Interfaces, IPv4/IPv6, MAC addresses, link speed, status |
| **Battery** | Charge level, power source, time remaining |
| **Peripherals** | Displays (resolution/refresh), audio devices, Bluetooth |
| **Sensors** | CPU temperature, fan speeds |
| **Python** | Installations, conda/venv environments, active env |
| **Dev Tools** | git, docker, node, gcc, clang, cmake, curl, and more |
| **Packages** | Homebrew, apt, pip, npm — package counts |
| **Services** | Running services (Ollama, Docker, databases, etc.) |

### System Benchmark (`bench.py`)

Runs isolated, scored benchmarks across five categories:

| Category | Weight | Tests |
|----------|--------|-------|
| **CPU Single-Core** | 25% | Prime sieve, Mandelbrot set, matrix multiply, zlib compression, array sort |
| **CPU Multi-Core** | 25% | Parallel matrix multiply, parallel compute, hash throughput, parallel sort |
| **GPU Compute** | 20% | Matrix multiply, elementwise ops, reduction, batch matmul (Apple MLX) |
| **Memory** | 15% | Sequential read/write, random access, memory copy |
| **Storage I/O** | 15% | Sequential read/write, random read/write (with OS cache bypass) |

Scores are normalized against a baseline machine (10.0 = baseline). The overall score is a weighted geometric mean across all categories.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mohamed-elkholy95/bench_py.git
cd bench_py

# Run system fetch (zero required dependencies)
python fetch.py

# Install benchmark dependencies
pip install numpy

# Run full benchmark suite
python bench.py
```

---

## Requirements

| Tool | Required | Notes |
|------|----------|-------|
| **Python 3.10+** | Yes | Uses dataclasses, type hints, `match` patterns |
| **numpy** | `bench.py` only | Core benchmark computations |
| **psutil** | Optional | Enables CPU temps, battery, RAM details, cooldown monitoring |
| **mlx** | Optional | Apple Silicon GPU benchmarks (skipped if unavailable) |

```bash
# Install all optional dependencies
pip install numpy psutil mlx
```

---

## Usage

### System Fetch

```bash
# Full scan — colored terminal output + JSON + text files
python fetch.py

# JSON only (pipe to jq, store in CI, etc.)
python fetch.py --json-only

# Verbose logging for debugging
python fetch.py --verbose

# Disable ANSI colors
python fetch.py --no-color

# Custom timeout per collector (default: 15s)
python fetch.py --timeout 30

# Save output files to a specific directory
python fetch.py --output-dir /tmp/reports
```

**Output files:**
- `system_report.json` — machine-readable full report
- `system_report.txt` — plain-text version of the terminal output

### System Benchmark

```bash
# Full benchmark — all categories
python bench.py

# Quick mode — reduced iterations for a fast preview
python bench.py --quick

# JSON output only
python bench.py --json-only

# Skip specific categories
python bench.py --skip gpu storage

# Run only specific categories
python bench.py --only cpu_single memory

# Calibration mode — print raw values for baseline tuning
python bench.py --calibrate

# Custom iterations and warmups
python bench.py --iterations 10 --warmups 5

# Skip cooldown between phases
python bench.py --no-cooldown

# Save to a specific directory
python bench.py --output-dir ./results
```

**Output files:**
- `benchmark_report.json` — full results with per-test scores, timings, and metadata
- `benchmark_report.txt` — plain-text summary

---

## Sample Output

### System Fetch

```
                      System Report
     2026-04-13T23:13:28 | Mohameds-MacBook-Pro.local
══════════════════════════════════════════════════════════

 OS & Hardware
  OS              macOS 26.4.1 (arm64)
  CPU             Apple M4 Max -- 14 cores / 14 threads
  Memory          36.0 GB LPDDR5
  GPU             Apple M4 Max (36.0 GB VRAM (unified))

 Storage
  /dev/disk3s1s1  926.0 GB SSD -- 459.0 GB free

 Battery
  Battery         80.0% | Plugged in

 Peripherals
  Display         Color LCD (3456x2234)
  Audio           MacBook Pro Speakers (output)
  Bluetooth       Logitech Pebble (paired) [mouse]

 Python
  Python 3.14.3   /opt/homebrew/bin/python3
  Conda envs      ai, datascience, mlx, pythinker

 Dev Tools
  git 2.53.0  docker 28.5.2  node 25.9.0  bun 1.3.11

══════════════════════════════════════════════════════════
Completed in 1.2s -- 15 succeeded, 0 failed
══════════════════════════════════════════════════════════
```

### System Benchmark

```
============================================================
  SYSTEM BENCHMARK REPORT
============================================================
  Timestamp : 2026-04-13T23:15:42
  Duration  : 87.3s
  Baseline  : Apple M4 Max / 36GB / macOS 26.4
------------------------------------------------------------
  CPU Single-Core  (10.0)
    prime_sieve              10.0  679.7 ops/s
    mandelbrot               10.0  706.8 Kpx/s
    matrix_1t                10.0  790.0 GFLOPS
    compression              10.0  145.2 MB/s
    sort                     10.0  6.7 Melem/s

  CPU Multi-Core  (10.0)
    matrix_full              10.0  738.1 GFLOPS
    parallel_compute         10.0  2.5 Mpx/s
    hash_throughput          10.0  488.4 MB/s
    parallel_sort            10.0  20.9 Melem/s

  GPU Compute  (10.0)
    gpu_matrix               10.0  10.6 TFLOPS
    gpu_elementwise          10.0  124.0 GB/s
    gpu_reduction            10.0  341.7 GB/s
    gpu_batch_matmul         10.0  9.2 TFLOPS

  Memory  (10.0)
    mem_seq_read             10.0  69.3 GB/s
    ...

  Storage I/O  (10.0)
    disk_seq_write           10.0  ...
    ...

------------------------------------------------------------
  OVERALL SCORE : 10.0  /10

  CPU Single-Core      10.0 ==============================
  CPU Multi-Core       10.0 ==============================
  GPU Compute          10.0 ==============================
  Memory               10.0 ==============================
  Storage I/O          10.0 ==============================
============================================================
```

---

## Architecture

```
python_info/
├── fetch.py           # System information collector
├── bench.py           # Benchmark suite
├── test_fetch.py      # Tests for fetch.py
├── test_bench.py      # Tests for bench.py
└── docs/              # Design specs and implementation plans
```

### How Fetch Works

1. **OS Detection** — identifies macOS, Linux, or Windows
2. **CommandRunner** — executes shell commands with timeout, cleanup, and graceful shutdown
3. **Collectors** — 15 independent collectors run sequentially, each wrapped in `safe_collect()` for fault isolation
4. **Error Handling** — failures are classified (timeout, permission denied, command not found, parse error) with actionable suggestions
5. **Output** — colored terminal display, JSON, and plain text — all generated from a single `SystemReport` dataclass

### How Bench Works

1. **Pre-flight Checks** — `SystemProbe` verifies RAM, CPU load, thermal state, battery, and disk space
2. **Phased Execution** — benchmarks run in ordered phases (CPU single → CPU multi → GPU → Memory → Storage) with cooldown periods between phases
3. **Subprocess Isolation** — each benchmark runs in a separate `multiprocessing.Process` to prevent crashes from affecting other tests
4. **Retry Policy** — transient failures (timeouts, crashes) are retried up to 2 times with backoff; permanent failures (missing dependencies, permissions) fail immediately
5. **Scoring** — raw measurements are normalized against baseline values; category scores use geometric mean; overall score uses weighted geometric mean
6. **Resource Monitoring** — `ResourceGuard` samples CPU and RAM usage at 500ms intervals during each test
7. **Cooldown Management** — `CooldownManager` waits between phases until CPU usage drops below threshold, ensuring consistent results

### Signal Handling

Both tools handle `SIGINT` and `SIGTERM` gracefully — active processes are terminated, partial results are preserved, and output files are written with whatever data was collected.

---

## Scoring System

The benchmark uses a **relative scoring model** where **10.0 = baseline machine performance**.

- **> 10.0** — faster than baseline
- **= 10.0** — matches baseline
- **< 10.0** — slower than baseline

The baseline is calibrated against an **Apple M4 Max / 36GB / macOS** configuration. Category scores are computed using the geometric mean of individual test scores. The overall score is a weighted geometric mean across categories, with weights redistributed proportionally when categories are skipped.

---

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest test_fetch.py
pytest test_bench.py
```

---

## Cross-Platform Support

| Feature | macOS | Linux | Windows |
|---------|-------|-------|---------|
| OS info | Full | Full | Full |
| CPU details | Full | Full | Full |
| Memory type/speed | Full | Full | Partial |
| GPU detection | Full (Metal/MLX) | Via lspci | Via wmic |
| Storage | Full | Full | Full |
| Network | Full | Full | Full |
| Battery | Full | Full | Full |
| Displays | Full | Via xrandr | Via wmic |
| Bluetooth | Full | Via bluetoothctl | Partial |
| Sensors | Via psutil | Via psutil | Via psutil |
| GPU Benchmarks | MLX (Apple Silicon) | Skipped | Skipped |

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Run tests (`pytest`)
4. Submit a pull request

---

## Author

**Mohamed Elkholy** — [@mohamed-elkholy95](https://github.com/mohamed-elkholy95)

---

## License

MIT
