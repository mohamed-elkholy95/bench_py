# System Fetch Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `fetch.py` as a comprehensive, cross-platform system inventory tool with 15 collectors, three output formats, and bulletproof error handling.

**Architecture:** Single-file collector registry pattern. Each collector is an isolated function that tries psutil first, falls back to platform-specific commands. A central runner iterates the registry with per-collector timeouts and error capture. Output flows to colored terminal, JSON, and plain text simultaneously.

**Tech Stack:** Python 3.10+ stdlib (dataclasses, argparse, subprocess, json, logging, signal, platform), optional psutil

**Strategy:** Complete rewrite — the new `fetch.py` replaces the existing file entirely. Output filenames change from `benchmark_results.txt` to `system_report.json` + `system_report.txt`.

---

## Task 1: Foundation — Imports, Dataclasses, Exceptions, CommandRunner

**Files:**
- Rewrite: `fetch.py` (replace entire file — start fresh)
- Test: `test_fetch.py` (create)

This task establishes the skeleton that everything else plugs into. The first ~300 lines of the file.

- [ ] **Step 1: Write tests for CommandRunner and error types**

Create `test_fetch.py`:

```python
"""Tests for fetch.py foundation: CommandRunner, exceptions, dataclass serialization."""
import json
import subprocess
import pytest


def test_command_runner_success():
    """run() returns stdout for a successful command."""
    from fetch import CommandRunner
    with CommandRunner() as runner:
        result = runner.run(["echo", "hello"])
        assert result == "hello"


def test_command_runner_or_none_success():
    """run_or_none() returns stdout on success."""
    from fetch import CommandRunner
    with CommandRunner() as runner:
        result = runner.run_or_none(["echo", "world"])
        assert result == "world"


def test_command_runner_or_none_failure():
    """run_or_none() returns None when command fails."""
    from fetch import CommandRunner
    with CommandRunner() as runner:
        result = runner.run_or_none(["false"])
        assert result is None


def test_command_not_found():
    """run() raises CommandNotFound for missing binaries."""
    from fetch import CommandRunner, CommandNotFound
    with CommandRunner() as runner:
        with pytest.raises(CommandNotFound):
            runner.run(["nonexistent_binary_xyz"])


def test_command_failed():
    """run() raises CommandFailed for non-zero exit."""
    from fetch import CommandRunner, CommandFailed
    with CommandRunner() as runner:
        with pytest.raises(CommandFailed):
            runner.run(["false"])


def test_command_timeout():
    """run() raises CommandTimeout when command exceeds timeout."""
    from fetch import CommandRunner, CommandTimeout
    with CommandRunner(default_timeout=1) as runner:
        with pytest.raises(CommandTimeout):
            runner.run(["sleep", "10"], timeout=1)


def test_runner_context_manager_cleanup():
    """Exiting context manager kills lingering processes."""
    from fetch import CommandRunner
    runner = CommandRunner()
    runner.__enter__()
    proc = subprocess.Popen(["sleep", "60"], stdout=subprocess.PIPE)
    runner._processes.append(proc)
    runner.__exit__(None, None, None)
    # Process should be terminated
    assert proc.poll() is not None


def test_shutdown_flag():
    """shutdown() sets flag and terminates active processes."""
    from fetch import CommandRunner
    with CommandRunner() as runner:
        proc = subprocess.Popen(["sleep", "60"], stdout=subprocess.PIPE)
        runner._processes.append(proc)
        runner.shutdown()
        assert runner._shutdown is True
        proc.wait(timeout=3)
        assert proc.poll() is not None


def test_error_classification():
    """classify_error maps exception types to category strings."""
    from fetch import (
        classify_error, CommandNotFound, CommandFailed,
        CommandTimeout,
    )
    assert classify_error(CommandNotFound("x")) == "command_not_found"
    assert classify_error(CommandTimeout("x")) == "timeout"
    assert classify_error(PermissionError("x")) == "permission_denied"
    assert classify_error(CommandFailed("x", 1, "")) == "command_failed"
    assert classify_error(ValueError("x")) == "parse_error"
    assert classify_error(IndexError("x")) == "parse_error"
    assert classify_error(NotImplementedError("x")) == "not_supported"
    assert classify_error(RuntimeError("x")) == "unexpected"


def test_dataclass_json_roundtrip():
    """SystemReport serializes to JSON and back without data loss."""
    from fetch import (
        SystemReport, OSInfo, CpuInfo, MemoryInfo,
        CollectionError, report_to_dict,
    )
    report = SystemReport(
        os=OSInfo(type="macOS", version="26.4", kernel="Darwin 25.4", arch="arm64", hostname="test"),
        cpu=CpuInfo(model="Test CPU", cores_physical=4, cores_logical=8, freq_mhz=3200.0, features=["avx2"]),
        memory=MemoryInfo(total_gb=16.0, type="DDR5", speed_mhz=4800),
        gpu=[], storage=[], network=[], battery=None,
        displays=[], audio=[], bluetooth=[],
        sensors=None, python=None, dev_tools=[],
        packages=[], services=[], errors=[],
        duration_seconds=1.5, timestamp="2026-04-13T18:00:00Z",
    )
    d = report_to_dict(report)
    json_str = json.dumps(d)
    loaded = json.loads(json_str)
    assert loaded["os"]["type"] == "macOS"
    assert loaded["cpu"]["model"] == "Test CPU"
    assert loaded["cpu"]["features"] == ["avx2"]
    assert loaded["memory"]["total_gb"] == 16.0
    assert loaded["timestamp"] == "2026-04-13T18:00:00Z"
    # None values should be omitted
    assert "battery" not in loaded
    assert "sensors" not in loaded
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/panda/Projects/open-source/python_test && python -m pytest test_fetch.py -v`
Expected: ImportError — `fetch` module doesn't export the new types yet.

- [ ] **Step 3: Write the foundation in fetch.py**

Replace the entire `fetch.py` with the foundation code containing:
- `from __future__ import annotations` and all stdlib imports
- Optional psutil import with `HAS_PSUTIL` flag
- Logger setup
- `OSType` enum
- Custom exceptions: `CommandNotFound`, `CommandFailed`, `CommandTimeout`
- All 17 dataclasses from the spec: `OSInfo`, `CpuInfo`, `MemoryInfo`, `GpuInfo`, `StorageInfo`, `NetworkInfo`, `BatteryInfo`, `DisplayInfo`, `AudioInfo`, `BluetoothInfo`, `SensorInfo`, `PythonInstall`, `VirtualEnv`, `PythonInfo`, `DevTool`, `PackageInfo`, `ServiceInfo`, `CollectionError`, `SystemReport`
- `classify_error()` and `suggest_fix()` functions
- `CommandRunner` class with `run()`, `run_or_none()`, `shutdown()`, `__enter__`/`__exit__`
- `_clean_none()` and `report_to_dict()` JSON helpers

```python
#!/usr/bin/env python3
"""System Fetch — comprehensive cross-platform system inventory.

Detects hardware, peripherals, software, and services on macOS, Linux,
and Windows. Outputs colored terminal display, JSON, and plain text.

Usage:
    python fetch.py                  # full scan, all outputs
    python fetch.py --json-only      # JSON only (for piping)
    python fetch.py --verbose        # debug logging
    python fetch.py --no-color       # disable color
    python fetch.py --timeout 30     # per-collector timeout
    python fetch.py --output-dir /tmp
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import time
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
# psutil (optional)
# ---------------------------------------------------------------------------
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("fetch")

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OSType(Enum):
    MACOS = "macOS"
    LINUX = "Linux"
    WINDOWS = "Windows"
    UNKNOWN = "Unknown"

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class CommandNotFound(Exception):
    """The requested binary does not exist on this system."""

class CommandFailed(Exception):
    """Command exited with a non-zero return code."""
    def __init__(self, message: str, returncode: int, stderr: str) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr

class CommandTimeout(Exception):
    """Command exceeded its timeout and was killed."""

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OSInfo:
    type: str
    version: str
    kernel: str
    arch: str
    hostname: str

@dataclass
class CpuInfo:
    model: Optional[str] = None
    cores_physical: Optional[int] = None
    cores_logical: Optional[int] = None
    freq_mhz: Optional[float] = None
    features: List[str] = field(default_factory=list)

@dataclass
class MemoryInfo:
    total_gb: Optional[float] = None
    type: Optional[str] = None
    speed_mhz: Optional[int] = None

@dataclass
class GpuInfo:
    model: Optional[str] = None
    vram_gb: Optional[float] = None
    unified: bool = False

@dataclass
class StorageInfo:
    device: str = ""
    mount_point: str = ""
    total_gb: Optional[float] = None
    free_gb: Optional[float] = None
    fs_type: Optional[str] = None
    disk_type: Optional[str] = None

@dataclass
class NetworkInfo:
    name: str = ""
    type: Optional[str] = None
    mac: Optional[str] = None
    ipv4: Optional[str] = None
    ipv6: Optional[str] = None
    speed_mbps: Optional[int] = None
    is_up: bool = False

@dataclass
class BatteryInfo:
    percent: Optional[float] = None
    plugged_in: bool = False
    time_remaining_min: Optional[int] = None

@dataclass
class DisplayInfo:
    name: Optional[str] = None
    resolution: Optional[str] = None
    refresh_rate_hz: Optional[int] = None

@dataclass
class AudioInfo:
    name: str = ""
    type: str = ""  # "input" or "output"

@dataclass
class BluetoothInfo:
    name: str = ""
    connected: bool = False
    device_type: Optional[str] = None

@dataclass
class SensorInfo:
    temperatures: Dict[str, float] = field(default_factory=dict)
    fan_speeds: Dict[str, int] = field(default_factory=dict)

@dataclass
class PythonInstall:
    version: str = ""
    path: str = ""

@dataclass
class VirtualEnv:
    type: str = ""   # "conda", "venv"
    name: str = ""
    path: str = ""

@dataclass
class PythonInfo:
    installations: List[PythonInstall] = field(default_factory=list)
    virtual_envs: List[VirtualEnv] = field(default_factory=list)
    active_env: Optional[str] = None

@dataclass
class DevTool:
    name: str = ""
    version: Optional[str] = None
    path: Optional[str] = None

@dataclass
class PackageInfo:
    manager: str = ""
    count: Optional[int] = None
    packages: List[str] = field(default_factory=list)

@dataclass
class ServiceInfo:
    name: str = ""
    status: str = "unknown"  # "running", "stopped", "unknown"

@dataclass
class CollectionError:
    collector: str = ""
    category: str = ""
    message: str = ""
    suggestion: Optional[str] = None

@dataclass
class SystemReport:
    os: Optional[OSInfo] = None
    cpu: Optional[CpuInfo] = None
    memory: Optional[MemoryInfo] = None
    gpu: List[GpuInfo] = field(default_factory=list)
    storage: List[StorageInfo] = field(default_factory=list)
    network: List[NetworkInfo] = field(default_factory=list)
    battery: Optional[BatteryInfo] = None
    displays: List[DisplayInfo] = field(default_factory=list)
    audio: List[AudioInfo] = field(default_factory=list)
    bluetooth: List[BluetoothInfo] = field(default_factory=list)
    sensors: Optional[SensorInfo] = None
    python: Optional[PythonInfo] = None
    dev_tools: List[DevTool] = field(default_factory=list)
    packages: List[PackageInfo] = field(default_factory=list)
    services: List[ServiceInfo] = field(default_factory=list)
    errors: List[CollectionError] = field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: str = ""

# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def classify_error(exc: Exception) -> str:
    """Map an exception to a human-readable error category."""
    if isinstance(exc, CommandNotFound):
        return "command_not_found"
    if isinstance(exc, CommandTimeout):
        return "timeout"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, CommandFailed):
        return "command_failed"
    if isinstance(exc, (ValueError, IndexError, KeyError)):
        return "parse_error"
    if isinstance(exc, NotImplementedError):
        return "not_supported"
    return "unexpected"


def suggest_fix(collector: str, exc: Exception) -> str:
    """Return a user-facing suggestion for how to resolve the error."""
    cat = classify_error(exc)
    suggestions = {
        "command_not_found": f"Install the required tool for {collector} detection",
        "timeout": "Increase timeout with --timeout flag",
        "permission_denied": f"Run with sudo for {collector} data",
        "command_failed": f"Command failed — run with --verbose for details",
        "parse_error": f"Unexpected output format during {collector} detection",
        "not_supported": f"{collector} detection not supported on this platform",
        "unexpected": "Unexpected error — run with --verbose for details",
    }
    return suggestions.get(cat, "Unknown error")

# ---------------------------------------------------------------------------
# CommandRunner
# ---------------------------------------------------------------------------

class CommandRunner:
    """Execute shell commands with timeout, tracking, and cleanup."""

    def __init__(self, default_timeout: int = 15) -> None:
        self.default_timeout = default_timeout
        self._processes: List[subprocess.Popen] = []
        self._shutdown = False

    def __enter__(self) -> "CommandRunner":
        return self

    def __exit__(self, *_: Any) -> None:
        for proc in self._processes[:]:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except OSError:
                pass
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._processes.clear()

    def run(self, command: List[str], timeout: Optional[int] = None) -> str:
        """Execute command and return stripped stdout. Raises on any failure."""
        if self._shutdown:
            raise InterruptedError("Shutdown requested")
        timeout = timeout or self.default_timeout
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            raise CommandNotFound(f"Command not found: {command[0]}")

        self._processes.append(proc)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise CommandTimeout(
                f"Command timed out ({timeout}s): {' '.join(command)}"
            )
        finally:
            if proc in self._processes:
                self._processes.remove(proc)

        if proc.returncode != 0:
            raise CommandFailed(
                f"Command failed (exit {proc.returncode}): {' '.join(command)}",
                proc.returncode,
                stderr,
            )
        return stdout.strip()

    def run_or_none(self, command: List[str], timeout: Optional[int] = None) -> Optional[str]:
        """Execute command, return stdout or None on any failure."""
        try:
            return self.run(command, timeout)
        except (CommandNotFound, CommandFailed, CommandTimeout,
                InterruptedError, OSError):
            return None

    def shutdown(self) -> None:
        """Signal graceful shutdown — terminate all active processes."""
        self._shutdown = True
        for proc in self._processes[:]:
            try:
                proc.terminate()
            except OSError:
                pass

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


def report_to_dict(report: SystemReport) -> Dict[str, Any]:
    """Convert a SystemReport to a JSON-friendly dict with None values removed."""
    return _clean_none(asdict(report))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/panda/Projects/open-source/python_test && python -m pytest test_fetch.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/panda/Projects/open-source/python_test
git add fetch.py test_fetch.py
git commit -m "feat: rewrite fetch.py foundation — dataclasses, CommandRunner, error handling"
```

---

## Task 2: OS Detection, Runner, and Signal Handling

**Files:**
- Modify: `fetch.py` (add detect_os, safe_collect, signal wiring)
- Modify: `test_fetch.py` (add runner and OS tests)

- [ ] **Step 1: Write tests for OS detection and safe_collect**

Add to `test_fetch.py`:

```python
def test_detect_os_returns_valid_enum():
    """detect_os() returns a known OSType member."""
    from fetch import detect_os, OSType
    result = detect_os()
    assert isinstance(result, OSType)
    assert result != OSType.UNKNOWN  # should work on any CI/dev machine


def test_collect_os_returns_os_info():
    """collect_os() returns populated OSInfo."""
    from fetch import collect_os, CommandRunner, detect_os, OSInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        info = collect_os(runner, os_type)
        assert isinstance(info, OSInfo)
        assert info.type in ("macOS", "Linux", "Windows")
        assert len(info.hostname) > 0
        assert len(info.arch) > 0
        assert len(info.kernel) > 0


def test_safe_collect_success():
    """safe_collect returns (result, None) on success."""
    from fetch import safe_collect, CommandRunner, detect_os, collect_os
    os_type = detect_os()
    with CommandRunner() as runner:
        result, error = safe_collect("os", collect_os, runner, os_type)
        assert result is not None
        assert error is None


def test_safe_collect_handles_exception():
    """safe_collect returns (None, CollectionError) when collector raises."""
    from fetch import safe_collect, CommandRunner, detect_os, CollectionError

    def bad_collector(runner, os_type):
        raise RuntimeError("boom")

    os_type = detect_os()
    with CommandRunner() as runner:
        result, error = safe_collect("test", bad_collector, runner, os_type)
        assert result is None
        assert isinstance(error, CollectionError)
        assert error.collector == "test"
        assert error.category == "unexpected"
        assert "boom" in error.message
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_fetch.py -v -k "detect_os or collect_os or safe_collect"`
Expected: ImportError — `detect_os`, `collect_os`, `safe_collect` not defined yet.

- [ ] **Step 3: Implement detect_os, collect_os, safe_collect, and signal handling**

Append to `fetch.py` after the `report_to_dict` function:

```python
# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------

def detect_os() -> OSType:
    """Detect the current operating system."""
    system = platform.system().lower()
    mapping = {"darwin": OSType.MACOS, "linux": OSType.LINUX, "windows": OSType.WINDOWS}
    return mapping.get(system, OSType.UNKNOWN)


# ---------------------------------------------------------------------------
# First collector: OS info
# ---------------------------------------------------------------------------

def collect_os(run: CommandRunner, os_type: OSType) -> OSInfo:
    """Collect operating system information. Pure stdlib — no psutil needed."""
    hostname = platform.node() or "unknown"
    arch = platform.machine() or "unknown"
    kernel = platform.platform() or "unknown"

    # Friendly version string
    version = ""
    if os_type == OSType.MACOS:
        v = platform.mac_ver()[0]
        version = v if v else kernel
    elif os_type == OSType.LINUX:
        try:
            import distro  # type: ignore[import-untyped]
            version = f"{distro.name()} {distro.version()}"
        except ImportError:
            version = platform.version()
    elif os_type == OSType.WINDOWS:
        version = platform.version()
    else:
        version = platform.version()

    return OSInfo(
        type=os_type.value,
        version=version,
        kernel=kernel,
        arch=arch,
        hostname=hostname,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def safe_collect(
    name: str,
    fn: Callable,
    runner: CommandRunner,
    os_type: OSType,
) -> Tuple[Any, Optional[CollectionError]]:
    """Run a collector safely. Returns (result, error_or_none)."""
    try:
        result = fn(runner, os_type)
        log.info("Collected %s", name)
        return result, None
    except InterruptedError:
        log.warning("Skipped %s (shutdown)", name)
        return None, CollectionError(
            collector=name,
            category="shutdown",
            message="Skipped — shutdown in progress",
        )
    except Exception as exc:
        log.warning("Failed to collect %s — %s", name, exc)
        return None, CollectionError(
            collector=name,
            category=classify_error(exc),
            message=str(exc),
            suggestion=suggest_fix(name, exc),
        )


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

_runner_ref: Optional[CommandRunner] = None


def _signal_handler(signum: int, _frame: Any) -> None:
    sig_name = signal.Signals(signum).name
    log.warning("Received %s — shutting down gracefully", sig_name)
    if _runner_ref is not None:
        _runner_ref.shutdown()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_fetch.py -v`
Expected: All tests PASS (previous 11 + new 4 = 15 total).

- [ ] **Step 5: Commit**

```bash
cd /Users/panda/Projects/open-source/python_test
git add fetch.py test_fetch.py
git commit -m "feat: add OS detection, safe_collect runner, signal handling"
```

---

## Task 3: Hardware Collectors — CPU, Memory, GPU

**Files:**
- Modify: `fetch.py` (add collect_cpu, collect_memory, collect_gpu)
- Modify: `test_fetch.py` (add integration tests)

Each collector follows the pattern established in Task 2: try psutil, fall back to platform commands, never raise.

- [ ] **Step 1: Write integration tests**

Add to `test_fetch.py`:

```python
def test_collect_cpu_returns_cpu_info():
    from fetch import collect_cpu, CommandRunner, detect_os, CpuInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        info = collect_cpu(runner, os_type)
        assert isinstance(info, CpuInfo)
        # At least one of these should be populated on any machine
        assert info.model is not None or info.cores_logical is not None


def test_collect_memory_returns_memory_info():
    from fetch import collect_memory, CommandRunner, detect_os, MemoryInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        info = collect_memory(runner, os_type)
        assert isinstance(info, MemoryInfo)
        assert info.total_gb is not None
        assert info.total_gb > 0


def test_collect_gpu_returns_list():
    from fetch import collect_gpu, CommandRunner, detect_os, GpuInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        gpus = collect_gpu(runner, os_type)
        assert isinstance(gpus, list)
        # May be empty on headless machines, but shouldn't crash
        for gpu in gpus:
            assert isinstance(gpu, GpuInfo)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_fetch.py -v -k "collect_cpu or collect_memory or collect_gpu"`
Expected: ImportError — functions not defined yet.

- [ ] **Step 3: Implement collect_cpu**

Append to `fetch.py`. The collector tries psutil for core counts and frequency, then uses platform-specific commands for the CPU model name and features:

- **macOS:** `sysctl -n machdep.cpu.brand_string` (Intel) or `sysctl -n hw.chip` (Apple Silicon), `sysctl -n hw.physicalcpu` / `hw.logicalcpu`, `sysctl -n machdep.cpu.features`
- **Linux:** `/proc/cpuinfo` for model and flags, `nproc --all` for logical cores, `lscpu` for physical cores
- **Windows:** `Get-CimInstance Win32_Processor` for model, core counts

```python
def collect_cpu(run: CommandRunner, os_type: OSType) -> CpuInfo:
    info = CpuInfo()

    if HAS_PSUTIL:
        try:
            info.cores_logical = psutil.cpu_count(logical=True)
            info.cores_physical = psutil.cpu_count(logical=False)
        except Exception:
            pass
        try:
            freq = psutil.cpu_freq()
            if freq:
                info.freq_mhz = freq.max or freq.current
        except Exception:
            pass

    if os_type == OSType.MACOS:
        model = run.run_or_none(["sysctl", "-n", "machdep.cpu.brand_string"])
        if not model:
            chip = run.run_or_none(["sysctl", "-n", "hw.chip"])
            model = f"Apple {chip}" if chip else None
        info.model = model
        if info.cores_physical is None:
            val = run.run_or_none(["sysctl", "-n", "hw.physicalcpu"])
            if val:
                info.cores_physical = int(val)
        if info.cores_logical is None:
            val = run.run_or_none(["sysctl", "-n", "hw.logicalcpu"])
            if val:
                info.cores_logical = int(val)
        feats = run.run_or_none(["sysctl", "-n", "machdep.cpu.features"])
        if feats:
            info.features = feats.lower().split()

    elif os_type == OSType.LINUX:
        cpuinfo = run.run_or_none(["cat", "/proc/cpuinfo"])
        if cpuinfo:
            for line in cpuinfo.split("\n"):
                if "model name" in line:
                    info.model = line.split(":", 1)[1].strip()
                    break
            for line in cpuinfo.split("\n"):
                if line.startswith("flags"):
                    info.features = line.split(":", 1)[1].strip().split()
                    break
        if not info.model:
            info.model = platform.processor() or None
        if info.cores_logical is None:
            val = run.run_or_none(["nproc", "--all"])
            if val:
                info.cores_logical = int(val)
        if info.cores_physical is None:
            lscpu = run.run_or_none(["lscpu"])
            if lscpu:
                cores_per = sockets = None
                for line in lscpu.split("\n"):
                    if "Core(s) per socket" in line:
                        cores_per = int(line.split(":")[1].strip())
                    elif "Socket(s)" in line:
                        sockets = int(line.split(":")[1].strip())
                if cores_per and sockets:
                    info.cores_physical = cores_per * sockets

    elif os_type == OSType.WINDOWS:
        model = run.run_or_none(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"]
        )
        if model:
            info.model = model.split("\n")[0].strip()
        if info.cores_physical is None:
            val = run.run_or_none(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).NumberOfCores"]
            )
            if val:
                info.cores_physical = int(val.strip())
        if info.cores_logical is None:
            val = run.run_or_none(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).NumberOfLogicalProcessors"]
            )
            if val:
                info.cores_logical = int(val.strip())

    return info
```

- [ ] **Step 4: Implement collect_memory**

Same pattern: psutil `virtual_memory()` first, then platform-specific for total and memory type/speed.

- **macOS:** `sysctl -n hw.memsize`, `system_profiler SPMemoryDataType` for type/speed
- **Linux:** `/proc/meminfo`, `dmidecode -t memory` for type/speed
- **Windows:** `Get-CimInstance Win32_ComputerSystem` for total, `Get-CimInstance Win32_PhysicalMemory` for type/speed

```python
def collect_memory(run: CommandRunner, os_type: OSType) -> MemoryInfo:
    info = MemoryInfo()

    if HAS_PSUTIL:
        try:
            mem = psutil.virtual_memory()
            info.total_gb = round(mem.total / (1024 ** 3), 1)
        except Exception:
            pass

    if info.total_gb is None:
        if os_type == OSType.MACOS:
            val = run.run_or_none(["sysctl", "-n", "hw.memsize"])
            if val:
                info.total_gb = round(int(val) / (1024 ** 3), 1)
        elif os_type == OSType.LINUX:
            meminfo = run.run_or_none(["cat", "/proc/meminfo"])
            if meminfo:
                for line in meminfo.split("\n"):
                    if "MemTotal" in line:
                        kb = int(line.split(":")[1].strip().split()[0])
                        info.total_gb = round(kb / (1024 ** 2), 1)
                        break
        elif os_type == OSType.WINDOWS:
            val = run.run_or_none(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"]
            )
            if val:
                info.total_gb = round(int(val.strip()) / (1024 ** 3), 1)

    if os_type == OSType.MACOS:
        sp = run.run_or_none(["system_profiler", "SPMemoryDataType"])
        if sp:
            for line in sp.split("\n"):
                stripped = line.strip()
                if stripped.startswith("Type:"):
                    info.type = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("Speed:"):
                    parts = stripped.split(":", 1)[1].strip().split()
                    if parts and parts[0].isdigit():
                        info.speed_mhz = int(parts[0])
    elif os_type == OSType.LINUX:
        dmi = run.run_or_none(["dmidecode", "-t", "memory"])
        if dmi:
            for line in dmi.split("\n"):
                stripped = line.strip()
                if stripped.startswith("Type:") and "Unknown" not in stripped:
                    info.type = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("Speed:") and "Unknown" not in stripped:
                    parts = stripped.split(":", 1)[1].strip().split()
                    if parts and parts[0].isdigit():
                        info.speed_mhz = int(parts[0])
    elif os_type == OSType.WINDOWS:
        mem_type_raw = run.run_or_none(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_PhysicalMemory)[0].SMBIOSMemoryType"]
        )
        if mem_type_raw:
            type_map = {"26": "DDR4", "34": "DDR5", "24": "DDR3"}
            info.type = type_map.get(mem_type_raw.strip(), None)
        speed = run.run_or_none(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_PhysicalMemory)[0].Speed"]
        )
        if speed and speed.strip().isdigit():
            info.speed_mhz = int(speed.strip())

    return info
```

- [ ] **Step 5: Implement collect_gpu**

No psutil support — fully platform-specific:

- **macOS:** `system_profiler SPDisplaysDataType` for model + VRAM, detect Apple Silicon unified memory
- **Linux:** `nvidia-smi` (NVIDIA), `rocm-smi` (AMD), `lspci` (fallback)
- **Windows:** `nvidia-smi` (NVIDIA), `Get-CimInstance Win32_VideoController` (fallback)

```python
def collect_gpu(run: CommandRunner, os_type: OSType) -> List[GpuInfo]:
    gpus: List[GpuInfo] = []

    if os_type == OSType.MACOS:
        sp = run.run_or_none(["system_profiler", "SPDisplaysDataType"])
        if sp:
            current_gpu: Optional[GpuInfo] = None
            for line in sp.split("\n"):
                stripped = line.strip()
                if "Chipset Model" in stripped or "Chip Model" in stripped:
                    if current_gpu:
                        gpus.append(current_gpu)
                    model_name = stripped.split(":", 1)[1].strip()
                    current_gpu = GpuInfo(model=model_name)
                elif current_gpu and "VRAM" in stripped and ":" in stripped:
                    val = stripped.split(":")[1].strip().split()
                    if len(val) >= 2:
                        num = float(val[0])
                        if "MB" in val[1].upper():
                            num /= 1024
                        current_gpu.vram_gb = round(num, 1)
            if current_gpu:
                gpus.append(current_gpu)
            # Apple Silicon unified memory
            for gpu in gpus:
                if gpu.vram_gb is None and gpu.model and "Apple" in gpu.model:
                    if HAS_PSUTIL:
                        try:
                            gpu.vram_gb = round(
                                psutil.virtual_memory().total / (1024 ** 3), 1
                            )
                        except Exception:
                            pass
                    if gpu.vram_gb is None:
                        val = run.run_or_none(["sysctl", "-n", "hw.memsize"])
                        if val:
                            gpu.vram_gb = round(int(val) / (1024 ** 3), 1)
                    gpu.unified = True

    elif os_type == OSType.LINUX:
        nv_name = run.run_or_none(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"]
        )
        if nv_name:
            nv_vram = run.run_or_none(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"]
            )
            for i, name in enumerate(nv_name.strip().split("\n")):
                gpu = GpuInfo(model=name.strip())
                if nv_vram:
                    lines = nv_vram.strip().split("\n")
                    if i < len(lines):
                        gpu.vram_gb = round(float(lines[i].strip()) / 1024, 1)
                gpus.append(gpu)
        if not gpus:
            rocm = run.run_or_none(["rocm-smi", "--showproductname"])
            if rocm:
                for line in rocm.split("\n"):
                    if "GPU" in line and ":" in line:
                        name = line.split(":", 1)[1].strip()
                        gpus.append(GpuInfo(model=name))
            rocm_vram = run.run_or_none(["rocm-smi", "--showmeminfo", "vram"])
            if rocm_vram:
                for line in rocm_vram.split("\n"):
                    if "Total" in line:
                        parts = line.split()
                        for idx, p in enumerate(parts):
                            if p == "Total" and idx + 2 < len(parts):
                                mb = float(parts[idx + 2])
                                if gpus:
                                    gpus[0].vram_gb = round(mb / 1024, 1)
                                break
                        break
        if not gpus:
            lspci = run.run_or_none(["lspci"])
            if lspci:
                for line in lspci.split("\n"):
                    if "VGA" in line or "3D" in line or "Display" in line:
                        parts = line.split(":", 2)
                        if len(parts) >= 3:
                            gpus.append(GpuInfo(model=parts[2].strip()))

    elif os_type == OSType.WINDOWS:
        nv_name = run.run_or_none(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"]
        )
        if nv_name:
            nv_vram = run.run_or_none(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"]
            )
            for i, name in enumerate(nv_name.strip().split("\n")):
                gpu = GpuInfo(model=name.strip())
                if nv_vram:
                    lines = nv_vram.strip().split("\n")
                    if i < len(lines):
                        gpu.vram_gb = round(float(lines[i].strip()) / 1024, 1)
                gpus.append(gpu)
        if not gpus:
            name = run.run_or_none(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"]
            )
            if name:
                for n in name.strip().split("\n"):
                    if n.strip():
                        gpus.append(GpuInfo(model=n.strip()))
            vram_raw = run.run_or_none(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).AdapterRAM"]
            )
            if vram_raw and gpus:
                for i, v in enumerate(vram_raw.strip().split("\n")):
                    if v.strip().isdigit() and i < len(gpus):
                        gpus[i].vram_gb = round(int(v.strip()) / (1024 ** 3), 1)

    return gpus
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest test_fetch.py -v`
Expected: All tests PASS (15 previous + 3 new = 18 total).

- [ ] **Step 7: Commit**

```bash
cd /Users/panda/Projects/open-source/python_test
git add fetch.py test_fetch.py
git commit -m "feat: add CPU, memory, GPU collectors with psutil + fallback"
```

---

## Task 4: Storage and Network Collectors

**Files:**
- Modify: `fetch.py` (add collect_storage, collect_network)
- Modify: `test_fetch.py`

- [ ] **Step 1: Write tests**

Add to `test_fetch.py`:

```python
def test_collect_storage_returns_list():
    from fetch import collect_storage, CommandRunner, detect_os, StorageInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        disks = collect_storage(runner, os_type)
        assert isinstance(disks, list)
        assert len(disks) > 0  # every machine has at least one disk
        assert isinstance(disks[0], StorageInfo)
        assert disks[0].total_gb is not None
        assert disks[0].total_gb > 0


def test_collect_network_returns_list():
    from fetch import collect_network, CommandRunner, detect_os, NetworkInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        ifaces = collect_network(runner, os_type)
        assert isinstance(ifaces, list)
        assert len(ifaces) > 0  # loopback always exists
        for iface in ifaces:
            assert isinstance(iface, NetworkInfo)
            assert len(iface.name) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_fetch.py -v -k "collect_storage or collect_network"`
Expected: ImportError.

- [ ] **Step 3: Implement collect_storage**

psutil `disk_partitions()` + `disk_usage()` primary; fallback to `df` (macOS/Linux) or `Get-CimInstance Win32_LogicalDisk` (Windows). Disk type detection via `diskutil info` on macOS.

```python
def collect_storage(run: CommandRunner, os_type: OSType) -> List[StorageInfo]:
    disks: List[StorageInfo] = []

    if HAS_PSUTIL:
        try:
            partitions = psutil.disk_partitions(all=False)
            for part in partitions:
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    disks.append(StorageInfo(
                        device=part.device,
                        mount_point=part.mountpoint,
                        total_gb=round(usage.total / (1024 ** 3), 1),
                        free_gb=round(usage.free / (1024 ** 3), 1),
                        fs_type=part.fstype or None,
                    ))
                except (PermissionError, OSError):
                    disks.append(StorageInfo(
                        device=part.device,
                        mount_point=part.mountpoint,
                        fs_type=part.fstype or None,
                    ))
        except Exception:
            pass

    if not disks:
        if os_type == OSType.MACOS:
            df = run.run_or_none(["df", "-g", "/"])
            if df:
                lines = df.split("\n")
                if len(lines) >= 2:
                    parts = lines[1].split()
                    disks.append(StorageInfo(
                        device=parts[0], mount_point="/",
                        total_gb=float(parts[1]), free_gb=float(parts[3]),
                    ))
        elif os_type == OSType.LINUX:
            df = run.run_or_none(["df", "--block-size=G", "/"])
            if df:
                lines = df.split("\n")
                if len(lines) >= 2:
                    parts = lines[1].split()
                    disks.append(StorageInfo(
                        device=parts[0],
                        mount_point=parts[5] if len(parts) > 5 else "/",
                        total_gb=float(parts[1].rstrip("G")),
                        free_gb=float(parts[3].rstrip("G")),
                    ))
        elif os_type == OSType.WINDOWS:
            raw = run.run_or_none(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_LogicalDisk -Filter \"DriveType=3\" | "
                 "Select-Object DeviceID,Size,FreeSpace | ConvertTo-Json"]
            )
            if raw:
                try:
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        data = [data]
                    for d in data:
                        disks.append(StorageInfo(
                            device=d.get("DeviceID", ""),
                            mount_point=d.get("DeviceID", "") + "\\",
                            total_gb=round(d["Size"] / (1024 ** 3), 1) if d.get("Size") else None,
                            free_gb=round(d["FreeSpace"] / (1024 ** 3), 1) if d.get("FreeSpace") else None,
                            fs_type="NTFS",
                        ))
                except (json.JSONDecodeError, KeyError):
                    pass

    for disk in disks:
        if os_type == OSType.MACOS and disk.device:
            di = run.run_or_none(["diskutil", "info", disk.device])
            if di:
                for line in di.split("\n"):
                    if "Solid State" in line:
                        disk.disk_type = "SSD" if "Yes" in line else "HDD"
                    elif "Protocol" in line and "NVMe" in line:
                        disk.disk_type = "NVMe"

    return disks
```

- [ ] **Step 4: Implement collect_network**

psutil `net_if_addrs()` + `net_if_stats()` primary; fallback to `ifconfig` (macOS) or `ip addr` (Linux).

```python
def collect_network(run: CommandRunner, os_type: OSType) -> List[NetworkInfo]:
    ifaces: List[NetworkInfo] = []

    if HAS_PSUTIL:
        try:
            addrs = psutil.net_if_addrs()
            stats = {}
            try:
                stats = psutil.net_if_stats()
            except Exception:
                pass

            for name, addr_list in addrs.items():
                iface = NetworkInfo(name=name)
                for addr in addr_list:
                    if addr.family.name == "AF_INET":
                        iface.ipv4 = addr.address
                    elif addr.family.name == "AF_INET6":
                        if not iface.ipv6:
                            iface.ipv6 = addr.address
                    elif addr.family.name in ("AF_LINK", "AF_PACKET"):
                        iface.mac = addr.address

                if name in stats:
                    st = stats[name]
                    iface.is_up = st.isup
                    if st.speed > 0:
                        iface.speed_mbps = st.speed

                lower = name.lower()
                if lower.startswith("lo") or lower == "lo0":
                    iface.type = "Loopback"
                elif "wi" in lower or "wlan" in lower or "airport" in lower:
                    iface.type = "Wi-Fi"
                elif "eth" in lower or "en" in lower or "eno" in lower:
                    iface.type = "Ethernet"
                elif "bridge" in lower or "br" in lower:
                    iface.type = "Bridge"
                elif "docker" in lower or "veth" in lower:
                    iface.type = "Virtual"

                ifaces.append(iface)
        except Exception:
            pass

    if not ifaces and os_type == OSType.MACOS:
        raw = run.run_or_none(["ifconfig"])
        if raw:
            current: Optional[NetworkInfo] = None
            for line in raw.split("\n"):
                if not line.startswith("\t") and ":" in line:
                    if current:
                        ifaces.append(current)
                    iface_name = line.split(":")[0]
                    current = NetworkInfo(name=iface_name, is_up="UP" in line)
                elif current:
                    stripped = line.strip()
                    if stripped.startswith("inet "):
                        current.ipv4 = stripped.split()[1]
                    elif stripped.startswith("inet6 "):
                        if not current.ipv6:
                            current.ipv6 = stripped.split()[1]
                    elif stripped.startswith("ether "):
                        current.mac = stripped.split()[1]
            if current:
                ifaces.append(current)

    return ifaces
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest test_fetch.py -v`
Expected: All tests PASS (18 previous + 2 new = 20 total).

- [ ] **Step 6: Commit**

```bash
cd /Users/panda/Projects/open-source/python_test
git add fetch.py test_fetch.py
git commit -m "feat: add storage and network collectors"
```

---

## Task 5: Peripheral Collectors — Battery, Displays, Audio, Bluetooth, Sensors

**Files:**
- Modify: `fetch.py`
- Modify: `test_fetch.py`

These collectors are grouped because they're all peripheral detection with similar structure.

- [ ] **Step 1: Write tests**

Add to `test_fetch.py`:

```python
def test_collect_battery_returns_optional():
    from fetch import collect_battery, CommandRunner, detect_os, BatteryInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        result = collect_battery(runner, os_type)
        assert result is None or isinstance(result, BatteryInfo)


def test_collect_displays_returns_list():
    from fetch import collect_displays, CommandRunner, detect_os, DisplayInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        result = collect_displays(runner, os_type)
        assert isinstance(result, list)
        for d in result:
            assert isinstance(d, DisplayInfo)


def test_collect_audio_returns_list():
    from fetch import collect_audio, CommandRunner, detect_os, AudioInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        result = collect_audio(runner, os_type)
        assert isinstance(result, list)
        for a in result:
            assert isinstance(a, AudioInfo)


def test_collect_bluetooth_returns_list():
    from fetch import collect_bluetooth, CommandRunner, detect_os, BluetoothInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        result = collect_bluetooth(runner, os_type)
        assert isinstance(result, list)
        for b in result:
            assert isinstance(b, BluetoothInfo)


def test_collect_sensors_returns_optional():
    from fetch import collect_sensors, CommandRunner, detect_os, SensorInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        result = collect_sensors(runner, os_type)
        assert result is None or isinstance(result, SensorInfo)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_fetch.py -v -k "battery or displays or audio or bluetooth or sensors"`
Expected: ImportError.

- [ ] **Step 3: Implement all five peripheral collectors**

Each follows the same pattern: psutil where applicable (battery, sensors), then platform-specific commands. Full implementations for each:

- **collect_battery:** psutil `sensors_battery()`, fallback to `pmset -g batt` (macOS), `upower` (Linux), `Get-CimInstance Win32_Battery` (Windows)
- **collect_displays:** `system_profiler SPDisplaysDataType` (macOS), `xrandr --query` (Linux), `Get-CimInstance Win32_VideoController` (Windows)
- **collect_audio:** `system_profiler SPAudioDataType` (macOS), `pactl list sinks/sources` + `aplay -l` (Linux), `Get-CimInstance Win32_SoundDevice` (Windows)
- **collect_bluetooth:** `system_profiler SPBluetoothDataType` (macOS), `bluetoothctl devices` (Linux), `Get-CimInstance Win32_PnPEntity` filtered by Bluetooth (Windows)
- **collect_sensors:** psutil `sensors_temperatures()` + `sensors_fans()`, fallback to `osx-cpu-temp` (macOS), `sensors` lm-sensors (Linux)

See Task 3 for the exact pattern. Each collector returns its typed dataclass, never raises.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_fetch.py -v`
Expected: All tests PASS (20 previous + 5 new = 25 total).

- [ ] **Step 5: Commit**

```bash
cd /Users/panda/Projects/open-source/python_test
git add fetch.py test_fetch.py
git commit -m "feat: add battery, display, audio, bluetooth, sensor collectors"
```

---

## Task 6: Software Collectors — Python, Dev Tools, Packages, Services

**Files:**
- Modify: `fetch.py`
- Modify: `test_fetch.py`

- [ ] **Step 1: Write tests**

Add to `test_fetch.py`:

```python
def test_collect_python_returns_info():
    from fetch import collect_python, CommandRunner, detect_os, PythonInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        info = collect_python(runner, os_type)
        assert isinstance(info, PythonInfo)
        assert len(info.installations) > 0  # we're running Python right now


def test_collect_dev_tools_returns_list():
    from fetch import collect_dev_tools, CommandRunner, detect_os, DevTool
    os_type = detect_os()
    with CommandRunner() as runner:
        tools = collect_dev_tools(runner, os_type)
        assert isinstance(tools, list)
        names = [t.name for t in tools]
        assert "git" in names


def test_collect_packages_returns_list():
    from fetch import collect_packages, CommandRunner, detect_os, PackageInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        pkgs = collect_packages(runner, os_type)
        assert isinstance(pkgs, list)
        for p in pkgs:
            assert isinstance(p, PackageInfo)


def test_collect_services_returns_list():
    from fetch import collect_services, CommandRunner, detect_os, ServiceInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        services = collect_services(runner, os_type)
        assert isinstance(services, list)
        for s in services:
            assert isinstance(s, ServiceInfo)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_fetch.py -v -k "collect_python or collect_dev_tools or collect_packages or collect_services"`
Expected: ImportError.

- [ ] **Step 3: Implement collect_python**

Detects Python installations via `which` / `Get-Command`, finds conda envs by scanning `~/miniforge3/envs`, `~/miniconda3/envs`, `~/anaconda3/envs`, and finds venvs by walking `~/Projects` and cwd looking for `pyvenv.cfg` files (max depth 4, skipping `node_modules`, `.git`, `__pycache__`). Checks `CONDA_DEFAULT_ENV` and `VIRTUAL_ENV` env vars for active env.

```python
def collect_python(run: CommandRunner, os_type: OSType) -> PythonInfo:
    info = PythonInfo()
    seen_paths: set = set()

    active = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")
    if active:
        info.active_env = active

    cmds = ["python3", "python"] if os_type != OSType.WINDOWS else ["python", "python3"]
    for cmd in cmds:
        if os_type == OSType.WINDOWS:
            path = run.run_or_none(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-Command {cmd} -ErrorAction SilentlyContinue).Source"]
            )
        else:
            path = run.run_or_none(["which", cmd])
        if not path:
            continue
        real = os.path.realpath(path.strip())
        if real in seen_paths:
            continue
        seen_paths.add(real)
        ver = run.run_or_none([cmd, "--version"])
        if ver:
            ver_str = ver.replace("Python ", "").strip()
            info.installations.append(PythonInstall(version=ver_str, path=path.strip()))

    for base in ["miniforge3", "miniconda3", "anaconda3", "Anaconda3"]:
        envs_dir = os.path.join(os.path.expanduser("~"), base, "envs")
        if not os.path.isdir(envs_dir):
            continue
        try:
            for name in sorted(os.listdir(envs_dir)):
                full = os.path.join(envs_dir, name)
                if os.path.isdir(full):
                    info.virtual_envs.append(VirtualEnv(type="conda", name=name, path=full))
        except OSError:
            pass

    search_dirs = {os.path.expanduser("~/Projects"), os.getcwd()}
    if os_type == OSType.WINDOWS:
        search_dirs.add(os.path.expanduser("~\\Projects"))
    seen_envs: set = set()
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        try:
            for root, dirs, files in os.walk(search_dir):
                depth = root.replace(search_dir, "").count(os.sep)
                if depth >= 4:
                    dirs.clear()
                    continue
                dirs[:] = [d for d in dirs if d not in {
                    "node_modules", ".git", "__pycache__", "site-packages",
                }]
                if "pyvenv.cfg" in files and root not in seen_envs:
                    seen_envs.add(root)
                    info.virtual_envs.append(VirtualEnv(
                        type="venv", name=os.path.basename(root), path=root,
                    ))
        except OSError:
            pass

    return info
```

- [ ] **Step 4: Implement collect_dev_tools**

Checks for 16 dev tools: `git`, `docker`, `node`, `npm`, `bun`, `rustc`, `cargo`, `go`, `java`, `gcc`, `clang`, `make`, `cmake`, `curl`, `wget`, `ssh`. For each: check if binary exists via `which`/`Get-Command`, then run its `--version` command (handling tools like `java -version` and `ssh -V` that print to stderr).

```python
_DEV_TOOLS = [
    ("git",    ["git", "--version"]),
    ("docker", ["docker", "--version"]),
    ("node",   ["node", "--version"]),
    ("npm",    ["npm", "--version"]),
    ("bun",    ["bun", "--version"]),
    ("rustc",  ["rustc", "--version"]),
    ("cargo",  ["cargo", "--version"]),
    ("go",     ["go", "version"]),
    ("java",   ["java", "-version"]),
    ("gcc",    ["gcc", "--version"]),
    ("clang",  ["clang", "--version"]),
    ("make",   ["make", "--version"]),
    ("cmake",  ["cmake", "--version"]),
    ("curl",   ["curl", "--version"]),
    ("wget",   ["wget", "--version"]),
    ("ssh",    ["ssh", "-V"]),
]


def _parse_version(name: str, raw: str) -> Optional[str]:
    import re
    first_line = raw.strip().split("\n")[0]
    match = re.search(r"(\d+\.\d+[\.\d]*)", first_line)
    return match.group(1) if match else first_line.strip()


def collect_dev_tools(run: CommandRunner, os_type: OSType) -> List[DevTool]:
    tools: List[DevTool] = []
    for name, version_cmd in _DEV_TOOLS:
        if os_type == OSType.WINDOWS:
            path = run.run_or_none(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-Command {name} -ErrorAction SilentlyContinue).Source"]
            )
        else:
            path = run.run_or_none(["which", name])
        if not path:
            continue
        version = None
        try:
            proc = subprocess.Popen(
                version_cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            stdout, stderr = proc.communicate(timeout=5)
            raw = stdout.strip() or stderr.strip()
            if raw:
                version = _parse_version(name, raw)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        tools.append(DevTool(name=name, version=version, path=path.strip()))
    return tools
```

- [ ] **Step 5: Implement collect_packages**

- **macOS:** `brew list --formula -1` + `brew list --cask -1`
- **Linux:** `dpkg-query` (Debian/Ubuntu), `rpm -qa` (RHEL/Fedora), `pacman -Qq` (Arch) — tries each, uses first that works
- **Windows:** `choco list`, `winget list`

```python
def collect_packages(run: CommandRunner, os_type: OSType) -> List[PackageInfo]:
    packages: List[PackageInfo] = []

    if os_type == OSType.MACOS:
        brew = run.run_or_none(["brew", "list", "--formula", "-1"])
        if brew:
            pkg_list = [p.strip() for p in brew.split("\n") if p.strip()]
            packages.append(PackageInfo(manager="brew", count=len(pkg_list), packages=pkg_list))
        casks = run.run_or_none(["brew", "list", "--cask", "-1"])
        if casks:
            cask_list = [c.strip() for c in casks.split("\n") if c.strip()]
            packages.append(PackageInfo(manager="brew-cask", count=len(cask_list), packages=cask_list))

    elif os_type == OSType.LINUX:
        dpkg = run.run_or_none(["dpkg-query", "-W", "-f", "${Package}\n"])
        if dpkg:
            pkg_list = [p.strip() for p in dpkg.split("\n") if p.strip()]
            packages.append(PackageInfo(manager="apt", count=len(pkg_list), packages=pkg_list))
        if not packages:
            rpm = run.run_or_none(["rpm", "-qa", "--qf", "%{NAME}\n"])
            if rpm:
                pkg_list = [p.strip() for p in rpm.split("\n") if p.strip()]
                packages.append(PackageInfo(manager="rpm", count=len(pkg_list), packages=pkg_list))
        if not packages:
            pac = run.run_or_none(["pacman", "-Qq"])
            if pac:
                pkg_list = [p.strip() for p in pac.split("\n") if p.strip()]
                packages.append(PackageInfo(manager="pacman", count=len(pkg_list), packages=pkg_list))

    elif os_type == OSType.WINDOWS:
        choco = run.run_or_none(["choco", "list", "--local-only", "--id-only"])
        if choco:
            pkg_list = [p.strip() for p in choco.split("\n")
                        if p.strip() and "packages installed" not in p]
            packages.append(PackageInfo(manager="choco", count=len(pkg_list), packages=pkg_list))
        winget = run.run_or_none(["winget", "list", "--source", "winget"])
        if winget:
            lines = [l for l in winget.split("\n")
                     if l.strip() and "---" not in l and "Name" not in l]
            packages.append(PackageInfo(manager="winget", count=len(lines), packages=[]))

    return packages
```

- [ ] **Step 6: Implement collect_services**

Looks for notable dev services: `ollama`, `docker`, `postgresql`, `mysql`, `redis`, `nginx`, `mongodb`, `rabbitmq`, `elasticsearch`, `grafana`, `prometheus`, `jenkins`, `sshd`, `orbstack`.

- **macOS:** `launchctl list` — parses PID column (digit = running, `-` = stopped)
- **Linux:** `systemctl list-units --type=service` — parses active column
- **Windows:** `Get-Service -Name '*svc*'` per service

```python
def collect_services(run: CommandRunner, os_type: OSType) -> List[ServiceInfo]:
    services: List[ServiceInfo] = []
    interesting = {
        "ollama", "docker", "postgresql", "postgres", "mysql", "redis",
        "nginx", "apache", "httpd", "mongodb", "mongod", "rabbitmq",
        "elasticsearch", "grafana", "prometheus", "jenkins", "sshd", "orbstack",
    }

    if os_type == OSType.MACOS:
        raw = run.run_or_none(["launchctl", "list"])
        if raw:
            for line in raw.split("\n")[1:]:
                parts = line.split("\t")
                if len(parts) >= 3:
                    label = parts[2].lower()
                    for svc in interesting:
                        if svc in label:
                            pid = parts[0].strip()
                            status = "running" if pid != "-" and pid.isdigit() else "stopped"
                            services.append(ServiceInfo(name=svc, status=status))
                            break

    elif os_type == OSType.LINUX:
        raw = run.run_or_none(
            ["systemctl", "list-units", "--type=service", "--no-pager", "--no-legend"]
        )
        if raw:
            for line in raw.split("\n"):
                parts = line.strip().split()
                if len(parts) >= 4:
                    unit = parts[0].replace(".service", "").lower()
                    for svc in interesting:
                        if svc in unit:
                            active = parts[2] if len(parts) > 2 else "unknown"
                            status = "running" if active == "active" else "stopped"
                            services.append(ServiceInfo(name=svc, status=status))
                            break

    elif os_type == OSType.WINDOWS:
        for svc in interesting:
            raw = run.run_or_none(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-Service -Name '*{svc}*' -ErrorAction SilentlyContinue).Status"]
            )
            if raw:
                status = "running" if "Running" in raw else "stopped"
                services.append(ServiceInfo(name=svc, status=status))

    return services
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest test_fetch.py -v`
Expected: All tests PASS (25 previous + 4 new = 29 total).

- [ ] **Step 8: Commit**

```bash
cd /Users/panda/Projects/open-source/python_test
git add fetch.py test_fetch.py
git commit -m "feat: add Python, dev tools, packages, services collectors"
```

---

## Task 7: Terminal Output Formatter (Colored)

**Files:**
- Modify: `fetch.py` (add ANSI color helpers and format_terminal)
- Modify: `test_fetch.py`

- [ ] **Step 1: Write tests**

Add to `test_fetch.py`:

```python
def test_format_terminal_produces_output():
    from fetch import format_terminal, SystemReport, OSInfo, CpuInfo, MemoryInfo
    report = SystemReport(
        os=OSInfo(type="macOS", version="26.4", kernel="Darwin 25.4", arch="arm64", hostname="test-mac"),
        cpu=CpuInfo(model="Apple M4 Max", cores_physical=14, cores_logical=14, freq_mhz=4050.0, features=[]),
        memory=MemoryInfo(total_gb=36.0, type="DDR5", speed_mhz=None),
        gpu=[], storage=[], network=[], battery=None,
        displays=[], audio=[], bluetooth=[],
        sensors=None, python=None, dev_tools=[],
        packages=[], services=[], errors=[],
        duration_seconds=2.5, timestamp="2026-04-13T18:00:00Z",
    )
    output = format_terminal(report, use_color=False)
    assert "System Report" in output
    assert "Apple M4 Max" in output
    assert "36.0" in output
    assert "test-mac" in output
    assert "2.5s" in output


def test_format_terminal_with_errors():
    from fetch import format_terminal, SystemReport, OSInfo, CollectionError
    report = SystemReport(
        os=OSInfo(type="Linux", version="Ubuntu 24.04", kernel="6.8.0", arch="x86_64", hostname="dev"),
        errors=[CollectionError(
            collector="bluetooth", category="permission_denied",
            message="Permission denied", suggestion="Run with sudo",
        )],
        duration_seconds=1.0, timestamp="2026-04-13T18:00:00Z",
    )
    output = format_terminal(report, use_color=False)
    assert "bluetooth" in output.lower()
    assert "Permission denied" in output or "permission_denied" in output


def test_ansi_stripped_when_no_color():
    from fetch import format_terminal, SystemReport, OSInfo
    report = SystemReport(
        os=OSInfo(type="macOS", version="26.4", kernel="Darwin", arch="arm64", hostname="test"),
        duration_seconds=0.5, timestamp="2026-04-13T18:00:00Z",
    )
    output = format_terminal(report, use_color=False)
    assert "\033[" not in output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_fetch.py -v -k "format_terminal or ansi"`
Expected: ImportError.

- [ ] **Step 3: Implement _Color helper class and format_terminal**

The `_Color` class wraps ANSI codes — all methods return plain text when `enabled=False`. The `format_terminal` function builds the output string section by section: header, OS & Hardware, Storage, Network, Battery, Peripherals, Sensors, Python, Dev Tools, Packages, Services, Errors, Footer with timing summary.

Full implementation provided in the code block. Key details:
- Status indicators: `[ok]` green, `[!!]` yellow, `[XX]` red, `[--]` dim
- Section headers in bold cyan
- 58-character display width
- Footer shows succeeded/failed counts and completion time
- Box-drawing characters in color mode, plain `=` in no-color mode

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_fetch.py -v -k "format_terminal or ansi"`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/panda/Projects/open-source/python_test
git add fetch.py test_fetch.py
git commit -m "feat: add colored terminal output formatter"
```

---

## Task 8: JSON and Plain Text Output Formatters

**Files:**
- Modify: `fetch.py` (add format_json, format_text, save_outputs)
- Modify: `test_fetch.py`

- [ ] **Step 1: Write tests**

Add to `test_fetch.py`:

```python
import tempfile
import os as _os


def test_format_json_valid():
    from fetch import format_json, SystemReport, OSInfo
    report = SystemReport(
        os=OSInfo(type="macOS", version="26.4", kernel="Darwin", arch="arm64", hostname="test"),
        duration_seconds=1.0, timestamp="2026-04-13T18:00:00Z",
    )
    result = format_json(report)
    parsed = json.loads(result)
    assert parsed["os"]["type"] == "macOS"
    assert "duration_seconds" in parsed
    assert "battery" not in parsed


def test_format_text_no_ansi():
    from fetch import format_text, SystemReport, OSInfo
    report = SystemReport(
        os=OSInfo(type="Linux", version="Ubuntu 24", kernel="6.8", arch="x86_64", hostname="srv"),
        duration_seconds=0.5, timestamp="2026-04-13T18:00:00Z",
    )
    result = format_text(report)
    assert "\033[" not in result
    assert "System Report" in result
    assert "Ubuntu 24" in result


def test_save_outputs_creates_files():
    from fetch import save_outputs, SystemReport, OSInfo
    report = SystemReport(
        os=OSInfo(type="macOS", version="26.4", kernel="Darwin", arch="arm64", hostname="test"),
        duration_seconds=0.5, timestamp="2026-04-13T18:00:00Z",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        save_outputs(report, output_dir=tmpdir)
        json_path = _os.path.join(tmpdir, "system_report.json")
        text_path = _os.path.join(tmpdir, "system_report.txt")
        assert _os.path.exists(json_path)
        assert _os.path.exists(text_path)
        with open(json_path) as f:
            data = json.loads(f.read())
            assert data["os"]["type"] == "macOS"
        with open(text_path) as f:
            content = f.read()
            assert "System Report" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_fetch.py -v -k "format_json or format_text or save_outputs"`
Expected: ImportError.

- [ ] **Step 3: Implement format_json, format_text, save_outputs**

```python
def format_json(report: SystemReport) -> str:
    """Serialize report to indented JSON string. None values omitted."""
    return json.dumps(report_to_dict(report), indent=2, ensure_ascii=False)


def format_text(report: SystemReport) -> str:
    """Plain text version — same as terminal but no color."""
    return format_terminal(report, use_color=False)


def save_outputs(
    report: SystemReport,
    output_dir: str = ".",
    json_only: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Write output files. Returns (json_path, text_path)."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "system_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(format_json(report))
    log.info("Saved JSON to %s", json_path)

    text_path = None
    if not json_only:
        text_path = os.path.join(output_dir, "system_report.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(format_text(report))
        log.info("Saved text to %s", text_path)

    return json_path, text_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_fetch.py -v`
Expected: All tests PASS (32 previous + 3 new = 35 total).

- [ ] **Step 5: Commit**

```bash
cd /Users/panda/Projects/open-source/python_test
git add fetch.py test_fetch.py
git commit -m "feat: add JSON, plain text formatters and file output"
```

---

## Task 9: CLI, Collector Registry, and main()

**Files:**
- Modify: `fetch.py` (add parse_args, COLLECTORS registry, main)
- Modify: `test_fetch.py`

This is the final wiring — connects everything together.

- [ ] **Step 1: Write tests**

Add to `test_fetch.py`:

```python
def test_parse_args_defaults():
    from fetch import parse_args
    args = parse_args([])
    assert args.json_only is False
    assert args.verbose is False
    assert args.no_color is False
    assert args.timeout == 15
    assert args.output_dir == "."


def test_parse_args_all_flags():
    from fetch import parse_args
    args = parse_args(["--json-only", "--verbose", "--no-color", "--timeout", "30", "--output-dir", "/tmp"])
    assert args.json_only is True
    assert args.verbose is True
    assert args.no_color is True
    assert args.timeout == 30
    assert args.output_dir == "/tmp"


def test_full_run_integration():
    from fetch import run_collection, detect_os, CommandRunner
    os_type = detect_os()
    with CommandRunner(default_timeout=15) as runner:
        report = run_collection(runner, os_type)
        assert report.os is not None
        assert report.os.type in ("macOS", "Linux", "Windows")
        assert report.timestamp != ""
        assert report.duration_seconds >= 0
        from fetch import report_to_dict
        d = report_to_dict(report)
        assert "os" in d
        assert "timestamp" in d
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_fetch.py -v -k "parse_args or full_run"`
Expected: ImportError.

- [ ] **Step 3: Implement COLLECTORS registry, run_collection, parse_args, and main**

The registry is an ordered list of `(name, function)` tuples. `run_collection()` iterates the registry, calls `safe_collect()` for each, and maps results to `SystemReport` fields. `parse_args()` uses argparse with five flags. `main()` wires everything: logging config, signal handlers, runner context manager, collection, terminal output, file output, exit code.

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


def run_collection(runner: CommandRunner, os_type: OSType) -> SystemReport:
    report = SystemReport(timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    start = time.monotonic()

    for name, fn in COLLECTORS:
        if runner._shutdown:
            report.errors.append(CollectionError(
                collector=name, category="shutdown",
                message="Skipped — shutdown in progress",
            ))
            continue
        result, error = safe_collect(name, fn, runner, os_type)
        if error:
            report.errors.append(error)
        # Map result to the correct report field by name
        if name == "os" and result:
            report.os = result
        elif name == "cpu":
            report.cpu = result
        elif name == "memory":
            report.memory = result
        elif name == "gpu":
            report.gpu = result or []
        elif name == "storage":
            report.storage = result or []
        elif name == "network":
            report.network = result or []
        elif name == "battery":
            report.battery = result
        elif name == "displays":
            report.displays = result or []
        elif name == "audio":
            report.audio = result or []
        elif name == "bluetooth":
            report.bluetooth = result or []
        elif name == "sensors":
            report.sensors = result
        elif name == "python":
            report.python = result
        elif name == "dev_tools":
            report.dev_tools = result or []
        elif name == "packages":
            report.packages = result or []
        elif name == "services":
            report.services = result or []

    report.duration_seconds = round(time.monotonic() - start, 2)
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="System Fetch — comprehensive system inventory",
    )
    parser.add_argument("--json-only", action="store_true",
                        help="Output JSON only (no terminal, no text file)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Debug logging with full tracebacks")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")
    parser.add_argument("--timeout", type=int, default=15,
                        help="Per-collector timeout in seconds (default: 15)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory for output files (default: current)")
    return parser.parse_args(argv)


def main() -> None:
    global _runner_ref
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    os_type = detect_os()
    use_color = not args.no_color and not args.json_only and sys.stdout.isatty()
    c = _Color(enabled=use_color)

    log.info("Detected %s (%s)", os_type.value, platform.machine())

    if not HAS_PSUTIL:
        msg = "psutil not installed — using fallback detection (pip install psutil)"
        if use_color:
            print(f"  {c.yellow('[!]')} {msg}")
        else:
            log.warning(msg)

    with CommandRunner(default_timeout=args.timeout) as runner:
        _runner_ref = runner
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        report = run_collection(runner, os_type)
    _runner_ref = None

    if not args.json_only:
        print(format_terminal(report, use_color=use_color))

    json_path, text_path = save_outputs(
        report, output_dir=args.output_dir, json_only=args.json_only,
    )

    if not args.json_only:
        print(f"  {c.green('JSON')}  {json_path}")
        if text_path:
            print(f"  {c.green('Text')}  {text_path}")
        print()

    sys.exit(0 if report.os else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_fetch.py -v`
Expected: All tests PASS (35 previous + 3 new = 38 total).

- [ ] **Step 5: Run the script end-to-end**

Run: `cd /Users/panda/Projects/open-source/python_test && python fetch.py --no-color`

Expected: Full terminal output with all 15 collectors, `system_report.json` and `system_report.txt` created.

Verify JSON: `python -c "import json; d=json.load(open('system_report.json')); print(sorted(d.keys()))"`

- [ ] **Step 6: Run with --json-only**

Run: `python fetch.py --json-only --output-dir /tmp/fetch-test`
Expected: No terminal output, `/tmp/fetch-test/system_report.json` created, no `.txt` file.

- [ ] **Step 7: Commit**

```bash
cd /Users/panda/Projects/open-source/python_test
git add fetch.py test_fetch.py
git commit -m "feat: wire up CLI, collector registry, and main entry point"
```

---

## Task 10: Final Verification and Cleanup

**Files:**
- Verify: `fetch.py`, `test_fetch.py`
- Delete: `benchmark_results.txt` (replaced by new output files)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest test_fetch.py -v --tb=short`
Expected: All 38 tests PASS, no warnings.

- [ ] **Step 2: Run the script and validate all outputs**

Run: `python fetch.py`
Expected: Colored terminal output with all sections, both output files created.

Validate JSON:
```bash
python -c "
import json
d = json.load(open('system_report.json'))
expected = {'os','cpu','memory','gpu','storage','network','displays','audio','bluetooth','dev_tools','packages','services','errors','duration_seconds','timestamp'}
present = set(d.keys())
missing = expected - present
print(f'Present: {sorted(present)}')
if missing:
    print(f'MISSING: {missing}')
else:
    print('All expected keys present')
"
```

- [ ] **Step 3: Test graceful shutdown (Ctrl+C)**

Run: `python fetch.py --timeout 30`
Press Ctrl+C after 1-2 collectors finish.

Expected: No traceback, partial results saved, remaining collectors listed as skipped, clean exit.

- [ ] **Step 4: Remove old output file**

```bash
rm -f /Users/panda/Projects/open-source/python_test/benchmark_results.txt
```

- [ ] **Step 5: Final commit**

```bash
cd /Users/panda/Projects/open-source/python_test
git add -A
git commit -m "chore: final verification, remove old benchmark_results.txt"
```
