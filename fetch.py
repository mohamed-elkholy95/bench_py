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
    type: str = ""

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
    type: str = ""
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
    status: str = "unknown"

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

    version = ""
    if os_type == OSType.MACOS:
        v = platform.mac_ver()[0]
        version = v if v else kernel
    elif os_type == OSType.LINUX:
        try:
            import distro
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
