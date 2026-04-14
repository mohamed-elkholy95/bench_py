<div align="center">

# Bench_py

### System Discovery & Performance Scoring Toolkit

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-00C853?style=for-the-badge&logo=apple&logoColor=white)](.)
[![License](https://img.shields.io/badge/License-MIT-F5C518?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/mohamed-elkholy95/bench_py?style=for-the-badge&logo=github&color=E040FB)](https://github.com/mohamed-elkholy95/bench_py/stargazers)

<br/>

**Discover your hardware. Measure your performance. Compare your score.**

Bench_py is a zero-config, cross-platform toolkit that collects a complete hardware & software inventory of your machine, then benchmarks CPU, GPU, memory, and storage — producing a single normalized score you can compare across systems.

<br/>

[Get Started](#-quick-start) · [Features](#-what-it-does) · [Sample Output](#-sample-output) · [Architecture](#-architecture)

---

</div>

<br/>

## **Why Bench_py?**

> Most system tools do one thing — either fetch specs _or_ run benchmarks. Bench_py does both in two commands, with structured JSON output, graceful error handling, and zero mandatory dependencies for system discovery.

<br/>

## **What It Does**

<table>
<tr>
<td width="50%" valign="top">

### `fetch.py` — System Discovery

Collects a **complete inventory** of your machine in ~1 second:

| | Category | Details |
|---|----------|---------|
| **1** | OS & Hardware | Version, kernel, architecture |
| **2** | CPU | Model, cores, threads, frequency |
| **3** | Memory | Capacity, type, speed |
| **4** | GPU | Model, VRAM, unified memory |
| **5** | Storage | Devices, capacity, free space |
| **6** | Network | Interfaces, IPs, link status |
| **7** | Battery | Charge, power source, time left |
| **8** | Peripherals | Displays, audio, Bluetooth |
| **9** | Sensors | Temperatures, fan speeds |
| **10** | Python | Installs, envs, active env |
| **11** | Dev Tools | git, docker, node, gcc, etc. |
| **12** | Packages | brew, apt, pip, npm counts |
| **13** | Services | Running daemons & services |

</td>
<td width="50%" valign="top">

### `bench.py` — Performance Scoring

Runs **22 isolated benchmarks** across 5 categories:

| | Category | Wt. | Tests |
|---|----------|-----|-------|
| **1** | CPU Single | 25% | Prime sieve, Mandelbrot, matrix, compression, sort |
| **2** | CPU Multi | 25% | Parallel matrix, parallel compute, hash, sort |
| **3** | GPU | 20% | Matrix multiply, elementwise, reduction, batch matmul |
| **4** | Memory | 15% | Sequential R/W, random access, copy |
| **5** | Storage | 15% | Sequential R/W, random R/W |

**Scoring:** `10.0 = baseline` · Weighted geometric mean

**Isolation:** Each test runs in its own subprocess

**Safety:** Pre-flight checks, cooldowns, retry with backoff

</td>
</tr>
</table>

<br/>

## **Quick Start**

```bash
git clone https://github.com/mohamed-elkholy95/bench_py.git
cd bench_py
```

<table>
<tr>
<td width="50%">

**System Discovery** _(zero dependencies)_

```bash
python fetch.py
```

</td>
<td width="50%">

**Performance Benchmark** _(requires numpy)_

```bash
pip install numpy
python bench.py
```

</td>
</tr>
</table>

<br/>

## **Installation**

<table>
<tr>
<td>

![Python](https://img.shields.io/badge/Python_3.10+-required-3776AB?style=flat-square&logo=python&logoColor=white)

</td>
<td>Core runtime — dataclasses, type hints, subprocess isolation</td>
</tr>
<tr>
<td>

![NumPy](https://img.shields.io/badge/numpy-bench.py_only-013243?style=flat-square&logo=numpy&logoColor=white)

</td>
<td>Matrix operations, memory bandwidth, parallel compute benchmarks</td>
</tr>
<tr>
<td>

![psutil](https://img.shields.io/badge/psutil-optional-4B8BBE?style=flat-square&logo=python&logoColor=white)

</td>
<td>CPU temps, battery info, RAM details, cooldown monitoring</td>
</tr>
<tr>
<td>

![MLX](https://img.shields.io/badge/mlx-optional_(Apple_Silicon)-000000?style=flat-square&logo=apple&logoColor=white)

</td>
<td>GPU compute benchmarks on Apple Silicon (gracefully skipped otherwise)</td>
</tr>
</table>

```bash
# Everything at once
pip install numpy psutil mlx
```

<br/>

## **Usage**

<details>
<summary><b>fetch.py</b> — All flags</summary>

<br/>

| Flag | Description |
|------|-------------|
| `--json-only` | JSON output only — pipe to `jq`, store in CI |
| `--verbose` / `-v` | Debug logging with full tracebacks |
| `--no-color` | Disable ANSI color codes |
| `--timeout N` | Per-collector timeout in seconds (default: 15) |
| `--output-dir PATH` | Directory for output files |

```bash
python fetch.py                          # Full scan, colored terminal + files
python fetch.py --json-only              # JSON to stdout
python fetch.py --json-only | jq '.cpu'  # Pipe to jq
python fetch.py --verbose --timeout 30   # Debug mode, longer timeouts
python fetch.py --output-dir /tmp        # Save reports elsewhere
```

**Output:** `system_report.json` + `system_report.txt`

</details>

<details>
<summary><b>bench.py</b> — All flags</summary>

<br/>

| Flag | Description |
|------|-------------|
| `--quick` | Reduced iterations for fast preview |
| `--json-only` | JSON output only |
| `--skip CATEGORY [...]` | Skip categories (e.g. `gpu storage`) |
| `--only CATEGORY [...]` | Run only listed categories |
| `--calibrate` | Print raw values for baseline tuning |
| `--iterations N` | Measurement iterations (default: 5) |
| `--warmups N` | Warmup iterations (default: 3) |
| `--test-timeout N` | Per-test timeout in seconds (default: 30) |
| `--no-cooldown` | Skip cooldown waits between phases |
| `--no-color` | Disable ANSI color codes |
| `--verbose` / `-v` | Debug logging |
| `--output-dir PATH` | Directory for output files |

```bash
python bench.py                          # Full benchmark suite
python bench.py --quick                  # Fast preview
python bench.py --skip gpu               # Skip GPU tests
python bench.py --only cpu_single memory # Run specific categories
python bench.py --calibrate              # Baseline calibration mode
python bench.py --iterations 10          # More precise measurements
python bench.py --output-dir ./results   # Save reports elsewhere
```

**Output:** `benchmark_report.json` + `benchmark_report.txt`

</details>

<br/>

## **Sample Output**

<details open>
<summary><b>fetch.py</b> — System Report</summary>

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

</details>

<details open>
<summary><b>bench.py</b> — Benchmark Report</summary>

```
============================================================
  SYSTEM BENCHMARK REPORT
============================================================
  Timestamp : 2026-04-13T23:15:42
  Duration  : 87.3s
  Baseline  : Apple M4 Max / 36GB / macOS 26.4
------------------------------------------------------------

  CPU Single-Core  (10.0)
    prime_sieve              10.0   679.7 ops/s
    mandelbrot               10.0   706.8 Kpx/s
    matrix_1t                10.0   790.0 GFLOPS
    compression              10.0   145.2 MB/s
    sort                     10.0   6.7 Melem/s

  CPU Multi-Core  (10.0)
    matrix_full              10.0   738.1 GFLOPS
    parallel_compute         10.0   2.5 Mpx/s
    hash_throughput          10.0   488.4 MB/s
    parallel_sort            10.0   20.9 Melem/s

  GPU Compute  (10.0)
    gpu_matrix               10.0   10.6 TFLOPS
    gpu_elementwise          10.0   124.0 GB/s
    gpu_reduction            10.0   341.7 GB/s
    gpu_batch_matmul         10.0   9.2 TFLOPS

  Memory  (10.0)
    mem_seq_read             10.0   69.3 GB/s
    mem_seq_write            10.0   52.1 GB/s
    mem_random_access        10.0   0.3 Gops/s
    mem_copy                 10.0   58.7 GB/s

  Storage I/O  (10.0)
    disk_seq_write           10.0   4.8 GB/s
    disk_seq_read            10.0   5.2 GB/s
    disk_random_write        10.0   62.4 Kops/s
    disk_random_read         10.0   71.8 Kops/s

------------------------------------------------------------
  OVERALL SCORE : 10.0  /10

  CPU Single-Core      10.0 ==============================
  CPU Multi-Core       10.0 ==============================
  GPU Compute          10.0 ==============================
  Memory               10.0 ==============================
  Storage I/O          10.0 ==============================
============================================================
```

</details>

<br/>

## **Scoring System**

<table>
<tr>
<td align="center"><h3>< 10.0</h3><b>Below Baseline</b></td>
<td align="center"><h3>= 10.0</h3><b>Matches Baseline</b></td>
<td align="center"><h3>> 10.0</h3><b>Above Baseline</b></td>
</tr>
</table>

The benchmark normalizes every raw measurement against reference values from a baseline machine (**Apple M4 Max / 36GB / macOS**).

- **Per-test scores** are computed as `(raw_value / baseline_value) * 10.0`
- **Category scores** use the **geometric mean** of individual test scores
- **Overall score** is a **weighted geometric mean** across categories
- When categories are skipped, weights are **redistributed proportionally** among active categories

<br/>

## **Architecture**

```
bench_py/
├── fetch.py           System discovery — 15 collectors, 3 output formats
├── bench.py           Benchmark suite — 22 tests, 5 categories, scoring engine
├── test_fetch.py      Unit tests for fetch.py
├── test_bench.py      Unit tests for bench.py
└── docs/              Design specifications and implementation plans
```

<details>
<summary><b>How <code>fetch.py</code> works</b></summary>

<br/>

```
┌─────────────┐     ┌───────────────┐     ┌────────────────┐
│  OS Detect   │────>│ CommandRunner  │────>│  15 Collectors  │
│ macOS/Linux/ │     │ timeout+retry  │     │  (fault-isolated│
│   Windows    │     │   +cleanup     │     │   via safe_     │
└─────────────┘     └───────────────┘     │   collect)      │
                                           └───────┬────────┘
                                                   │
                              ┌─────────────────────┼─────────────────────┐
                              ▼                     ▼                     ▼
                    ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
                    │   Terminal    │     │     JSON      │     │  Plain Text  │
                    │ (ANSI color) │     │  (structured) │     │  (no ANSI)   │
                    └──────────────┘     └───────────────┘     └──────────────┘
```

1. **OS Detection** — identifies macOS, Linux, or Windows via `platform.system()`
2. **CommandRunner** — executes shell commands with per-command timeout, process tracking, and graceful shutdown
3. **15 Collectors** — each wrapped in `safe_collect()` for fault isolation; failures are classified with actionable suggestions
4. **Error Classification** — `timeout`, `permission_denied`, `command_not_found`, `parse_error`, `not_supported`
5. **Signal Handling** — `SIGINT`/`SIGTERM` trigger graceful shutdown; partial results are preserved

</details>

<details>
<summary><b>How <code>bench.py</code> works</b></summary>

<br/>

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────────┐
│  SystemProbe  │────>│  Orchestrator │────>│         Phased Execution         │
│  (pre-flight) │     │  (cooldowns)  │     │                                  │
│  RAM / CPU /  │     │               │     │  Phase 1: CPU Single ─> cooldown │
│  thermal /    │     │               │     │  Phase 2: CPU Multi  ─> cooldown │
│  battery      │     │               │     │  Phase 3: GPU        ─> cooldown │
└──────────────┘     └──────────────┘     │  Phase 4: Memory     ─> cooldown │
                                           │  Phase 5: Storage                │
                                           └────────────┬─────────────────────┘
                                                        │
                              ┌──────────────────────────┼───────────────────┐
                              ▼                          ▼                   ▼
                    ┌──────────────────┐     ┌────────────────┐    ┌──────────────┐
                    │  TestExecutor     │     │ Scoring Engine  │    │ ResourceGuard │
                    │  (subprocess      │     │ normalize ─>    │    │ (CPU/RAM      │
                    │   isolation +     │     │ geo mean ─>     │    │  sampling     │
                    │   retry policy)   │     │ weighted score  │    │  @500ms)      │
                    └──────────────────┘     └────────────────┘    └──────────────┘
```

1. **Pre-flight** — `SystemProbe` checks RAM, CPU load, thermal state, battery, disk space
2. **Phased Execution** — ordered phases with `CooldownManager` between each
3. **Subprocess Isolation** — each benchmark runs in `multiprocessing.Process`
4. **Retry Policy** — transient failures retried 2x with backoff; permanent failures fail immediately
5. **Scoring Engine** — raw values normalized against baseline; geometric mean per category; weighted overall
6. **Resource Monitoring** — `ResourceGuard` daemon thread samples CPU/RAM at 500ms intervals

</details>

<br/>

## **Cross-Platform Support**

|  | macOS | Linux | Windows |
|--|-------|-------|---------|
| OS info | **Full** | **Full** | **Full** |
| CPU | **Full** | **Full** | **Full** |
| Memory type/speed | **Full** | **Full** | Partial |
| GPU detection | **Full** (Metal/MLX) | Via `lspci` | Via `wmic` |
| Storage | **Full** | **Full** | **Full** |
| Network | **Full** | **Full** | **Full** |
| Battery | **Full** | **Full** | **Full** |
| Displays | **Full** | Via `xrandr` | Via `wmic` |
| Bluetooth | **Full** | Via `bluetoothctl` | Partial |
| Sensors | Via `psutil` | Via `psutil` | Via `psutil` |
| GPU Benchmarks | **MLX** (Apple Silicon) | _Skipped_ | _Skipped_ |

<br/>

## **Testing**

```bash
pytest              # Run all tests
pytest -v           # Verbose output
pytest test_fetch.py  # Fetch tests only
pytest test_bench.py  # Benchmark tests only
```

<br/>

## **Contributing**

Contributions are welcome. Please open an issue to discuss changes before submitting a PR.

1. **Fork** the repository
2. **Branch** — `git checkout -b feature/my-change`
3. **Test** — `pytest`
4. **PR** — submit with a clear description

<br/>

## **Author**

<a href="https://github.com/mohamed-elkholy95">
  <img src="https://img.shields.io/badge/Mohamed_Elkholy-181717?style=for-the-badge&logo=github&logoColor=white" />
</a>

<br/>
<br/>

## **License**

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

<sub>If you find Bench_py useful, consider giving it a star.</sub>

[![Star](https://img.shields.io/github/stars/mohamed-elkholy95/bench_py?style=social)](https://github.com/mohamed-elkholy95/bench_py)

</div>
