# System Fetch Redesign — Design Spec

**Date:** 2026-04-13
**Status:** Draft
**Scope:** Complete rewrite of `fetch.py` — comprehensive cross-platform system inventory tool

---

## 1. Purpose

A single-file Python script that detects everything about a machine — hardware, peripherals, software, services — across macOS, Linux, and Windows. Designed as a **pre-check tool** that feeds structured data into a future benchmark suite.

**Use case:** Personal machine inventory. Run once, get a full snapshot in three formats (colored terminal, JSON, plain text). The JSON output is the stable contract for the benchmark tool.

---

## 2. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Collector Registry (Approach 1) | Flat, readable, easy to extend. Each collector is an isolated function — no inheritance ceremony. |
| File structure | Single file | Personal tool — just `python fetch.py` and done. |
| psutil | Primary with subprocess fallback | Covers ~70% of detection cross-platform. Gracefully degrades if not installed. |
| Output | Terminal + JSON + Text | Terminal for humans, JSON for the benchmark tool, text for grepping. |
| CLI | Minimal argparse | Five flags: `--json-only`, `--verbose`, `--no-color`, `--timeout`, `--output-dir` |
| Error handling | Three-layer (CommandRunner → Collector → Runner) | No silent failures, no leaks, partial results always saved. |

---

## 3. Data Model

All dataclasses use `Optional` fields — collectors fill what they can, leave the rest as `None`.

```python
@dataclass
class OSInfo:
    type: str                          # "macOS", "Linux", "Windows"
    version: str                       # "26.4.1", "Ubuntu 24.04", "Windows 11 23H2"
    kernel: str                        # "Darwin 25.4.0", "6.8.0-generic", "10.0.26100"
    arch: str                          # "arm64", "x86_64"
    hostname: str                      # machine hostname

@dataclass
class CpuInfo:
    model: Optional[str]               # "Apple M4 Max", "AMD Ryzen 9 7950X"
    cores_physical: Optional[int]
    cores_logical: Optional[int]
    freq_mhz: Optional[float]          # max frequency
    features: List[str]                 # ISA extensions: ["avx2", "neon", "amx"] — from lscpu flags / sysctl machdep.cpu.features / registry. Empty list if unknown.

@dataclass
class MemoryInfo:
    total_gb: Optional[float]
    type: Optional[str]                # "DDR5", "DDR4", "LPDDR5"
    speed_mhz: Optional[int]

@dataclass
class GpuInfo:
    model: Optional[str]
    vram_gb: Optional[float]
    unified: bool = False              # Apple Silicon shared memory

@dataclass
class StorageInfo:
    device: str                        # "disk0s2", "/dev/sda1", "C:"
    mount_point: str                   # "/", "/home", "C:\\"
    total_gb: Optional[float]
    free_gb: Optional[float]
    fs_type: Optional[str]             # "APFS", "ext4", "NTFS"
    disk_type: Optional[str]           # "SSD", "HDD", "NVMe"

@dataclass
class NetworkInfo:
    name: str                          # "en0", "eth0", "Wi-Fi"
    type: Optional[str]                # "Wi-Fi", "Ethernet", "Loopback"
    mac: Optional[str]
    ipv4: Optional[str]
    ipv6: Optional[str]
    speed_mbps: Optional[int]
    is_up: bool = False

@dataclass
class BatteryInfo:
    percent: Optional[float]
    plugged_in: bool = False
    time_remaining_min: Optional[int]  # None if calculating or N/A

@dataclass
class DisplayInfo:
    name: Optional[str]                # "Built-in Retina Display"
    resolution: Optional[str]          # "3456x2234"
    refresh_rate_hz: Optional[int]

@dataclass
class AudioInfo:
    name: str                          # "MacBook Pro Speakers"
    type: str                          # "input" or "output"

@dataclass
class BluetoothInfo:
    name: str                          # "Magic Keyboard"
    connected: bool = False
    device_type: Optional[str]         # "keyboard", "audio", "mouse", "other"

@dataclass
class SensorInfo:
    temperatures: Dict[str, float]     # {"CPU": 42.0, "GPU": 38.5} — Celsius
    fan_speeds: Dict[str, int]         # {"Fan 1": 1200} — RPM

@dataclass
class PythonInstall:
    version: str                       # "3.14.3"
    path: str                          # "/opt/homebrew/bin/python3"

@dataclass
class VirtualEnv:
    type: str                          # "conda", "venv", "virtualenv"
    name: str                          # "mlx", ".venv"
    path: str                          # full path

@dataclass
class PythonInfo:
    installations: List[PythonInstall]
    virtual_envs: List[VirtualEnv]
    active_env: Optional[str]          # currently activated env, if any

@dataclass
class DevTool:
    name: str                          # "git"
    version: Optional[str]             # "2.44.0"
    path: Optional[str]                # "/usr/bin/git"

@dataclass
class PackageInfo:
    manager: str                       # "brew", "apt", "choco", "winget"
    count: Optional[int]               # total installed
    packages: List[str]                # list of names (no versions — keep it lean)

@dataclass
class ServiceInfo:
    name: str                          # "ollama", "docker", "postgresql"
    status: str                        # "running", "stopped", "unknown"

@dataclass
class CollectionError:
    collector: str                     # "gpu", "bluetooth"
    category: str                      # "permission_denied", "timeout", "command_not_found",
                                       # "parse_error", "not_supported", "unexpected"
    message: str                       # raw error detail
    suggestion: Optional[str]          # "Run with sudo for full access"

@dataclass
class SystemReport:
    os: OSInfo
    cpu: Optional[CpuInfo]
    memory: Optional[MemoryInfo]
    gpu: List[GpuInfo]
    storage: List[StorageInfo]
    network: List[NetworkInfo]
    battery: Optional[BatteryInfo]
    displays: List[DisplayInfo]
    audio: List[AudioInfo]
    bluetooth: List[BluetoothInfo]
    sensors: Optional[SensorInfo]
    python: Optional[PythonInfo]
    dev_tools: List[DevTool]
    packages: List[PackageInfo]
    services: List[ServiceInfo]
    errors: List[CollectionError]
    duration_seconds: float
    timestamp: str                     # ISO 8601
```

---

## 4. Architecture

### 4.1 File Layout (single file, top-to-bottom)

```
fetch.py
 1. Imports & constants
 2. Logging setup (structured, colored, TTY-aware)
 3. Dataclasses (all types from Section 3)
 4. CommandRunner class
 5. psutil availability check
 6. Collector functions (15 total)
 7. Collector registry (ordered list of name → function)
 8. Runner (iterates registry with safe_collect)
 9. Output formatters (terminal, JSON, text)
10. CLI argument parsing
11. main()
```

### 4.2 Collector Contract

Every collector follows the same signature and behavioral contract:

```python
def collect_cpu(run: CommandRunner, os_type: OSType) -> CpuInfo:
    """
    1. Try psutil first (if available)
    2. Fall back to platform-specific commands
    3. Return CpuInfo with whatever data was collected
    4. NEVER raise — always return, even if mostly empty
    """
```

- Receives `CommandRunner` (for subprocess calls) and `os_type` (for platform branching)
- Returns its typed dataclass — partial data is expected and fine
- Logs warnings internally but never raises to the runner
- Each collector is self-contained — no shared mutable state between collectors

### 4.3 Collector Registry

```python
COLLECTORS: List[Tuple[str, Callable]] = [
    ("os",         collect_os),
    ("cpu",        collect_cpu),
    ("memory",     collect_memory),
    ("gpu",        collect_gpu),
    ("storage",    collect_storage),
    ("network",    collect_network),
    ("battery",    collect_battery),
    ("displays",   collect_displays),
    ("audio",      collect_audio),
    ("bluetooth",  collect_bluetooth),
    ("sensors",    collect_sensors),
    ("python",     collect_python),
    ("dev_tools",  collect_dev_tools),
    ("packages",   collect_packages),
    ("services",   collect_services),
]
```

Order matters for terminal output grouping but not for correctness. No collector depends on another's output.

### 4.4 CommandRunner

```python
class CommandRunner:
    def __init__(self, default_timeout: int = 15):
        self.default_timeout = default_timeout
        self._processes: List[subprocess.Popen] = []
        self._shutdown = False

    def run(self, command: List[str], timeout: Optional[int] = None) -> str:
        """Execute command, return stdout. Raises on failure."""

    def run_or_none(self, command: List[str], timeout: Optional[int] = None) -> Optional[str]:
        """Execute command, return stdout or None on any failure."""

    def shutdown(self) -> None:
        """Signal shutdown — terminate all active subprocesses."""

    # Context manager for resource cleanup
    def __enter__(self): ...
    def __exit__(self): ...  # kills any remaining processes
```

Key properties:
- Context manager guarantees cleanup even on unexpected exits
- `run()` raises typed exceptions (`CommandTimeout`, `CommandNotFound`, `CommandFailed`)
- `run_or_none()` wraps `run()` for convenience — returns `None` instead of raising
- All subprocesses tracked in `_processes` list, removed in `finally` blocks
- `shutdown()` sets flag + terminates active processes (called from signal handler)

---

## 5. Error Handling — Three Layers

### Layer 1: CommandRunner (innermost)

Subprocess lifecycle is fully managed. Every `Popen` call is wrapped:

```python
proc = subprocess.Popen(command, stdout=PIPE, stderr=PIPE, text=True)
self._processes.append(proc)
try:
    stdout, stderr = proc.communicate(timeout=timeout)
finally:
    if proc in self._processes:
        self._processes.remove(proc)
```

Error types raised:
- `CommandTimeout` — subprocess killed, waited, cleaned up
- `CommandNotFound` — binary doesn't exist (`FileNotFoundError`)
- `CommandFailed` — non-zero exit code, includes stderr

### Layer 2: Collector Functions (per-category)

Waterfall pattern within each collector:

```
psutil → primary command → fallback command → empty result
```

Each step catches its own exceptions. A failure at one step means trying the next. The collector always returns a dataclass, never raises.

### Layer 3: Runner (outermost)

```python
def safe_collect(name, fn, runner, os_type, timeout):
    try:
        result = fn(runner, os_type)
        return result, None
    except Exception as e:
        error = CollectionError(
            collector=name,
            category=classify_error(e),
            message=str(e),
            suggestion=suggest_fix(name, e),
        )
        return empty_result_for(name), error
```

Error classification and suggestion mapping:

| Exception Type | Category | Suggestion Template |
|---------------|----------|-------------------|
| `CommandNotFound` | `command_not_found` | "Install {tool} for {collector} detection" |
| `PermissionError` | `permission_denied` | "Run with sudo for {collector} data" |
| `CommandTimeout` | `timeout` | "Increase timeout with --timeout flag" |
| `ValueError`, `IndexError` | `parse_error` | "Unexpected output format from {command}" |
| `NotImplementedError` | `not_supported` | "{collector} not supported on {os}" |
| Any other `Exception` | `unexpected` | "Unexpected error — run with --verbose for details" |

---

## 6. Terminal Error Reporting

### Inline Status (printed as each collector completes)

```
[✓] CPU              Apple M4 Max — 14 cores              (green)
[✓] Memory           36.0 GB unified                       (green)
[⚠] Sensors          Partial — temperatures OK, fans N/A   (yellow)
[✗] Bluetooth        Permission denied (needs sudo)        (red)
[—] Services         Skipped (shutdown requested)           (dim/gray)
```

### Summary Footer

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Completed in 4.2s — 13 succeeded, 1 partial, 1 failed

Errors:
  Bluetooth    Permission denied — run with sudo for full access

Output:
  Terminal     ✓ displayed above
  JSON         ✓ system_report.json
  Text         ✓ system_report.txt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Color & Formatting Rules

- ANSI escape codes only — no external dependency
- Auto-disabled when stdout is not a TTY (piped or redirected)
- `--no-color` flag forces plain output
- Box-drawing characters (━, ╔, ║, etc.) for structure — degrades to ASCII dashes in non-TTY

---

## 7. Platform Detection Matrix

| Collector | psutil (cross-platform) | macOS fallback | Linux fallback | Windows fallback |
|-----------|------------------------|----------------|----------------|------------------|
| OS | `platform.*` stdlib | — | — | — |
| CPU | `cpu_count()`, `cpu_freq()` | `sysctl` | `/proc/cpuinfo`, `lscpu` | `Get-CimInstance Win32_Processor` |
| Memory | `virtual_memory()` | `sysctl hw.memsize` | `/proc/meminfo` | `Get-CimInstance Win32_ComputerSystem` |
| GPU | N/A | `system_profiler SPDisplaysDataType` | `nvidia-smi`, `rocm-smi`, `lspci` | `nvidia-smi`, `Get-CimInstance Win32_VideoController` |
| Storage | `disk_partitions()`, `disk_usage()` | `diskutil info` | `lsblk --json` | `Get-CimInstance Win32_LogicalDisk` |
| Network | `net_if_addrs()`, `net_if_stats()` | `networksetup -listallhardwareports` | `ip addr` | `Get-NetAdapter` |
| Battery | `sensors_battery()` | `pmset -g batt` | `upower -i /org/freedesktop/UPower/devices/battery_BAT0` | `Get-CimInstance Win32_Battery` |
| Displays | N/A | `system_profiler SPDisplaysDataType` | `xrandr --query` | `Get-CimInstance Win32_VideoController` |
| Audio | N/A | `system_profiler SPAudioDataType` | `aplay -l`, `pactl list sinks` | `Get-CimInstance Win32_SoundDevice` |
| Bluetooth | N/A | `system_profiler SPBluetoothDataType` | `bluetoothctl devices` | `Get-CimInstance Win32_PnPEntity` (filtered) |
| Sensors | `sensors_temperatures()`, `sensors_fans()` | `powermetrics` (limited, needs sudo) | `sensors` (lm-sensors) | `Get-CimInstance Win32_TemperatureProbe` |
| Python | N/A | `which` + `--version` | `which` + `--version` | `Get-Command` + `--version` |
| Dev Tools | N/A | `which` + `--version` | `which` + `--version` | `Get-Command` + `--version` |
| Packages | N/A | `brew list`, `brew list --cask` | `dpkg --list` / `rpm -qa` / `pacman -Q` | `choco list` / `winget list` |
| Services | N/A | `launchctl list` | `systemctl list-units --type=service` | `Get-Service` |

### Dev Tools Detected

`git`, `docker`, `node`, `npm`, `bun`, `rustc`, `cargo`, `go`, `java`, `gcc`, `clang`, `make`, `cmake`, `curl`, `wget`, `ssh`

---

## 8. Output Formats

### 8.1 Terminal (colored, box-drawn)

Grouped sections: OS & Hardware, Storage, Network, Battery, Peripherals, Sensors, Python, Dev Tools, Packages, Services. Each collector result formatted inline with status indicators. Summary footer with timing, error count, and output file paths.

### 8.2 JSON (`system_report.json`)

Full `SystemReport` serialized as JSON. This is the **stable contract** for the benchmark tool. Keys match the dataclass field names exactly. `None` values are omitted. Lists are always present (empty `[]` if no data). ISO 8601 timestamp.

### 8.3 Plain Text (`system_report.txt`)

Same content as terminal but without ANSI codes or box-drawing. Suitable for grepping, diffing, or pasting into reports.

### 8.4 File Locations

Default output directory is the current working directory. Override with `--output-dir`.

---

## 9. CLI Interface

```
usage: fetch.py [options]

Options:
  --json-only       Output JSON only (no terminal display, no text file)
  --verbose, -v     Debug logging with full tracebacks
  --no-color        Disable colored output (auto-detected for non-TTY)
  --timeout SECS    Per-collector timeout in seconds (default: 15)
  --output-dir DIR  Directory for output files (default: current directory)
```

Implemented via `argparse` with a single argument group. No subcommands.

---

## 10. Graceful Shutdown

- `SIGINT` (Ctrl+C) and `SIGTERM` set `_shutdown` flag on `CommandRunner`
- Runner checks the flag between collectors — stops iteration
- Any in-flight subprocess is terminated via `proc.terminate()` then `proc.wait(timeout=3)`, escalating to `proc.kill()` if needed
- Partial results are saved to all output files
- Terminal shows `[—] Skipped` for collectors that didn't run
- Exit code: 0 if any data collected, 1 if nothing collected

---

## 11. psutil Availability

```python
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
```

- If `HAS_PSUTIL` is False, a single yellow warning is printed at startup:
  `[⚠] psutil not installed — using fallback detection (install with: pip install psutil)`
- Every collector checks `HAS_PSUTIL` before calling psutil functions
- All psutil calls are wrapped in try/except — psutil itself can fail on edge cases

---

## 12. Non-Goals

- **Not a monitoring tool** — runs once, exits. No daemon mode, no polling.
- **Not a package** — no `setup.py`, no `pyproject.toml`. Just `python fetch.py`.
- **Not a benchmark** — pure detection/inventory. The benchmark tool is a separate future project.
- **No network requests** — everything is local detection. No phoning home.
- **No root required** — runs as regular user, degrades gracefully for privileged data.
