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
import re
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


# ---------------------------------------------------------------------------
# Hardware collectors
# ---------------------------------------------------------------------------

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
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]
        )
        if model:
            info.model = model.split("\n")[0].strip()
        if info.cores_physical is None:
            val = run.run_or_none(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).NumberOfCores"]
            )
            if val:
                info.cores_physical = int(val.strip())
        if info.cores_logical is None:
            val = run.run_or_none(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).NumberOfLogicalProcessors"]
            )
            if val:
                info.cores_logical = int(val.strip())

    return info


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
            for gpu in gpus:
                if gpu.vram_gb is None and gpu.model and "Apple" in gpu.model:
                    if HAS_PSUTIL:
                        try:
                            gpu.vram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
                        except Exception:
                            pass
                    if gpu.vram_gb is None:
                        val = run.run_or_none(["sysctl", "-n", "hw.memsize"])
                        if val:
                            gpu.vram_gb = round(int(val) / (1024 ** 3), 1)
                    gpu.unified = True

    elif os_type == OSType.LINUX:
        nv_name = run.run_or_none(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"])
        if nv_name:
            nv_vram = run.run_or_none(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
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
        nv_name = run.run_or_none(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"])
        if nv_name:
            nv_vram = run.run_or_none(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            for i, name in enumerate(nv_name.strip().split("\n")):
                gpu = GpuInfo(model=name.strip())
                if nv_vram:
                    lines = nv_vram.strip().split("\n")
                    if i < len(lines):
                        gpu.vram_gb = round(float(lines[i].strip()) / 1024, 1)
                gpus.append(gpu)
        if not gpus:
            name = run.run_or_none(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"]
            )
            if name:
                for n in name.strip().split("\n"):
                    if n.strip():
                        gpus.append(GpuInfo(model=n.strip()))
            vram_raw = run.run_or_none(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).AdapterRAM"]
            )
            if vram_raw and gpus:
                for i, v in enumerate(vram_raw.strip().split("\n")):
                    if v.strip().isdigit() and i < len(gpus):
                        gpus[i].vram_gb = round(int(v.strip()) / (1024 ** 3), 1)

    return gpus


def collect_storage(run: CommandRunner, os_type: OSType) -> List[StorageInfo]:
    disks: List[StorageInfo] = []

    if HAS_PSUTIL:
        try:
            partitions = psutil.disk_partitions(all=False)
            for part in partitions:
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    disks.append(StorageInfo(
                        device=part.device, mount_point=part.mountpoint,
                        total_gb=round(usage.total / (1024 ** 3), 1),
                        free_gb=round(usage.free / (1024 ** 3), 1),
                        fs_type=part.fstype or None,
                    ))
                except (PermissionError, OSError):
                    disks.append(StorageInfo(
                        device=part.device, mount_point=part.mountpoint,
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


# ---------------------------------------------------------------------------
# Peripheral collectors
# ---------------------------------------------------------------------------

def collect_battery(run: CommandRunner, os_type: OSType) -> Optional[BatteryInfo]:
    if HAS_PSUTIL:
        try:
            bat = psutil.sensors_battery()
            if bat is not None:
                time_left = None
                if bat.secsleft > 0:
                    time_left = int(bat.secsleft / 60)
                return BatteryInfo(
                    percent=round(bat.percent, 1),
                    plugged_in=bat.power_plugged,
                    time_remaining_min=time_left,
                )
            return None
        except Exception:
            pass

    if os_type == OSType.MACOS:
        raw = run.run_or_none(["pmset", "-g", "batt"])
        if raw and "InternalBattery" in raw:
            info = BatteryInfo()
            for line in raw.split("\n"):
                if "InternalBattery" in line:
                    parts = line.split("\t")
                    for part in parts:
                        part = part.strip()
                        if "%" in part:
                            # Extract the numeric value before any "%" — handles
                            # formats like "80%" or "80; AC attached; not charging"
                            pct_str = part.split("%")[0].split(";")[0].strip()
                            try:
                                info.percent = float(pct_str)
                            except ValueError:
                                pass
                        if "charging" in part.lower() or "ac power" in part.lower():
                            info.plugged_in = True
            return info
    elif os_type == OSType.LINUX:
        upower = run.run_or_none(["upower", "-i", "/org/freedesktop/UPower/devices/battery_BAT0"])
        if upower:
            info = BatteryInfo()
            for line in upower.split("\n"):
                stripped = line.strip()
                if "percentage:" in stripped:
                    val = stripped.split(":")[1].strip().replace("%", "")
                    info.percent = float(val)
                elif "state:" in stripped:
                    info.plugged_in = "charging" in stripped or "fully-charged" in stripped
                elif "time to empty:" in stripped:
                    parts = stripped.split(":")[1].strip().split()
                    if parts:
                        try:
                            hours = float(parts[0])
                            info.time_remaining_min = int(hours * 60)
                        except ValueError:
                            pass
            return info if info.percent is not None else None
    elif os_type == OSType.WINDOWS:
        raw = run.run_or_none(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Battery | Select-Object EstimatedChargeRemaining,BatteryStatus | ConvertTo-Json"]
        )
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    data = data[0]
                return BatteryInfo(
                    percent=float(data.get("EstimatedChargeRemaining", 0)),
                    plugged_in=data.get("BatteryStatus", 0) == 2,
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    return None


def collect_displays(run: CommandRunner, os_type: OSType) -> List[DisplayInfo]:
    displays: List[DisplayInfo] = []

    if os_type == OSType.MACOS:
        sp = run.run_or_none(["system_profiler", "SPDisplaysDataType"])
        if sp:
            current: Optional[DisplayInfo] = None
            in_displays_section = False
            for line in sp.split("\n"):
                stripped = line.strip()
                if "Displays:" in stripped:
                    in_displays_section = True
                    continue
                if in_displays_section:
                    if "Resolution" in stripped:
                        if current is None:
                            current = DisplayInfo()
                        res = stripped.split(":", 1)[1].strip()
                        res_clean = res.split("@")[0].strip()
                        current.resolution = res_clean.replace(" ", "")
                    elif stripped.endswith(":") and not stripped.startswith("Displays"):
                        if current:
                            displays.append(current)
                        current = DisplayInfo(name=stripped.rstrip(":"))
                if "Hz" in stripped:
                    for word in stripped.split():
                        cleaned = word.replace("Hz", "").strip()
                        if cleaned.isdigit():
                            if current:
                                current.refresh_rate_hz = int(cleaned)
            if current:
                displays.append(current)

    elif os_type == OSType.LINUX:
        xrandr = run.run_or_none(["xrandr", "--query"])
        if xrandr:
            for line in xrandr.split("\n"):
                if " connected " in line:
                    parts = line.split()
                    name = parts[0]
                    disp = DisplayInfo(name=name)
                    for part in parts:
                        if "x" in part and "+" in part:
                            res = part.split("+")[0]
                            disp.resolution = res
                            break
                    displays.append(disp)
                elif "*" in line and displays:
                    parts = line.strip().split()
                    for part in parts:
                        if "*" in part:
                            hz = part.replace("*", "").replace("+", "")
                            try:
                                displays[-1].refresh_rate_hz = int(float(hz))
                            except ValueError:
                                pass

    elif os_type == OSType.WINDOWS:
        raw = run.run_or_none(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object Name,CurrentHorizontalResolution,"
             "CurrentVerticalResolution,CurrentRefreshRate | ConvertTo-Json"]
        )
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    data = [data]
                for d in data:
                    h = d.get("CurrentHorizontalResolution")
                    v = d.get("CurrentVerticalResolution")
                    res = f"{h}x{v}" if h and v else None
                    displays.append(DisplayInfo(
                        name=d.get("Name"), resolution=res,
                        refresh_rate_hz=d.get("CurrentRefreshRate"),
                    ))
            except (json.JSONDecodeError, KeyError):
                pass

    return displays


def collect_audio(run: CommandRunner, os_type: OSType) -> List[AudioInfo]:
    devices: List[AudioInfo] = []

    if os_type == OSType.MACOS:
        sp = run.run_or_none(["system_profiler", "SPAudioDataType"])
        if sp:
            current_name: Optional[str] = None
            for line in sp.split("\n"):
                stripped = line.strip()
                if stripped.endswith(":") and not stripped.startswith("Audio") and len(stripped) > 1:
                    current_name = stripped.rstrip(":")
                elif current_name:
                    if "Default Output Device: Yes" in stripped:
                        devices.append(AudioInfo(name=current_name, type="output"))
                        current_name = None
                    elif "Default Input Device: Yes" in stripped:
                        devices.append(AudioInfo(name=current_name, type="input"))
                        current_name = None

    elif os_type == OSType.LINUX:
        pactl = run.run_or_none(["pactl", "list", "sinks", "short"])
        if pactl:
            for line in pactl.split("\n"):
                parts = line.split("\t")
                if len(parts) >= 2:
                    devices.append(AudioInfo(name=parts[1], type="output"))
        pactl_src = run.run_or_none(["pactl", "list", "sources", "short"])
        if pactl_src:
            for line in pactl_src.split("\n"):
                parts = line.split("\t")
                if len(parts) >= 2 and "monitor" not in parts[1].lower():
                    devices.append(AudioInfo(name=parts[1], type="input"))
        if not devices:
            aplay = run.run_or_none(["aplay", "-l"])
            if aplay:
                for line in aplay.split("\n"):
                    if line.startswith("card "):
                        name = line.split(":")[1].strip() if ":" in line else line
                        devices.append(AudioInfo(name=name, type="output"))

    elif os_type == OSType.WINDOWS:
        raw = run.run_or_none(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_SoundDevice | Select-Object Name,StatusInfo | ConvertTo-Json"]
        )
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    data = [data]
                for d in data:
                    devices.append(AudioInfo(name=d.get("Name", "Unknown"), type="output"))
            except (json.JSONDecodeError, KeyError):
                pass

    return devices


def collect_bluetooth(run: CommandRunner, os_type: OSType) -> List[BluetoothInfo]:
    devices: List[BluetoothInfo] = []

    if os_type == OSType.MACOS:
        sp = run.run_or_none(["system_profiler", "SPBluetoothDataType"])
        if sp:
            current_name: Optional[str] = None
            current_connected = False
            current_type: Optional[str] = None
            in_devices = False
            for line in sp.split("\n"):
                stripped = line.strip()
                if "Connected:" in stripped or "Devices" in stripped:
                    in_devices = True
                    continue
                if in_devices and stripped.endswith(":") and len(stripped) > 1:
                    if current_name:
                        devices.append(BluetoothInfo(
                            name=current_name, connected=current_connected, device_type=current_type,
                        ))
                    current_name = stripped.rstrip(":")
                    current_connected = False
                    current_type = None
                elif current_name:
                    if "Connected: Yes" in stripped:
                        current_connected = True
                    elif "Minor Type:" in stripped:
                        current_type = stripped.split(":", 1)[1].strip().lower()
            if current_name:
                devices.append(BluetoothInfo(
                    name=current_name, connected=current_connected, device_type=current_type,
                ))

    elif os_type == OSType.LINUX:
        paired = run.run_or_none(["bluetoothctl", "devices"])
        if paired:
            for line in paired.split("\n"):
                parts = line.strip().split(" ", 2)
                if len(parts) >= 3 and parts[0] == "Device":
                    mac = parts[1]
                    name = parts[2]
                    info_raw = run.run_or_none(["bluetoothctl", "info", mac])
                    connected = False
                    if info_raw and "Connected: yes" in info_raw:
                        connected = True
                    devices.append(BluetoothInfo(name=name, connected=connected))

    elif os_type == OSType.WINDOWS:
        raw = run.run_or_none(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq 'Bluetooth' } | "
             "Select-Object Name,Status | ConvertTo-Json"]
        )
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    data = [data]
                for d in data:
                    name = d.get("Name", "")
                    if name and "Bluetooth" not in name:
                        devices.append(BluetoothInfo(
                            name=name, connected=d.get("Status") == "OK",
                        ))
            except (json.JSONDecodeError, KeyError):
                pass

    return devices


def collect_sensors(run: CommandRunner, os_type: OSType) -> Optional[SensorInfo]:
    temps: Dict[str, float] = {}
    fans: Dict[str, int] = {}

    if HAS_PSUTIL:
        try:
            t = psutil.sensors_temperatures()
            if t:
                for group_name, entries in t.items():
                    for entry in entries:
                        label = entry.label or group_name
                        temps[label] = entry.current
        except (AttributeError, Exception):
            pass
        try:
            f = psutil.sensors_fans()
            if f:
                for group_name, entries in f.items():
                    for entry in entries:
                        label = entry.label or group_name
                        fans[label] = entry.current
        except (AttributeError, Exception):
            pass

    if not temps and os_type == OSType.MACOS:
        raw = run.run_or_none(["osx-cpu-temp"])
        if raw:
            for line in raw.split("\n"):
                if "CPU" in line and "°C" in line:
                    val = line.split(":")[1].strip().replace("°C", "").strip()
                    try:
                        temps["CPU"] = float(val)
                    except ValueError:
                        pass

    if not temps and os_type == OSType.LINUX:
        raw = run.run_or_none(["sensors"])
        if raw:
            current_group = ""
            for line in raw.split("\n"):
                if not line.startswith(" ") and line.strip():
                    current_group = line.strip()
                elif "°C" in line and ":" in line:
                    label = line.split(":")[0].strip()
                    val_str = line.split(":")[1].strip().split("°C")[0].strip().lstrip("+")
                    try:
                        temps[f"{current_group}/{label}"] = float(val_str)
                    except ValueError:
                        pass
                elif "RPM" in line and ":" in line:
                    label = line.split(":")[0].strip()
                    val_str = line.split(":")[1].strip().split("RPM")[0].strip()
                    try:
                        fans[f"{current_group}/{label}"] = int(float(val_str))
                    except ValueError:
                        pass

    if temps or fans:
        return SensorInfo(temperatures=temps, fan_speeds=fans)
    return None


# ---------------------------------------------------------------------------
# Software collectors
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

class _Color:
    """ANSI escape code wrapper. All methods return plain text if disabled."""
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def green(self, t: str) -> str: return self._wrap("32", t)
    def yellow(self, t: str) -> str: return self._wrap("33", t)
    def red(self, t: str) -> str: return self._wrap("31", t)
    def dim(self, t: str) -> str: return self._wrap("2", t)
    def bold(self, t: str) -> str: return self._wrap("1", t)
    def cyan(self, t: str) -> str: return self._wrap("36", t)

# ---------------------------------------------------------------------------
# Terminal output formatter
# ---------------------------------------------------------------------------

def format_terminal(report: SystemReport, use_color: bool = True) -> str:
    """Format the report as a human-readable terminal string."""
    c = _Color(enabled=use_color)
    lines: List[str] = []
    W = 58  # display width

    # Header
    sep = "=" * W if not use_color else "\u2550" * W
    lines.append("")
    lines.append(c.bold(f"{'System Report':^{W}}"))
    hostname = report.os.hostname if report.os else "unknown"
    subtitle = f"{report.timestamp[:19]} | {hostname}"
    lines.append(c.dim(f"{subtitle:^{W}}"))
    lines.append(sep)

    def section(title: str) -> None:
        lines.append("")
        lines.append(c.bold(c.cyan(f" {title}")))

    def row(label: str, value: str) -> None:
        lines.append(f"  {label:<16}{value}")

    # OS & Hardware
    section("OS & Hardware")
    if report.os:
        row("OS", f"{report.os.type} {report.os.version} ({report.os.arch})")
    if report.cpu:
        cores = ""
        if report.cpu.cores_physical or report.cpu.cores_logical:
            p = report.cpu.cores_physical or "?"
            l = report.cpu.cores_logical or "?"
            freq = f" @ {report.cpu.freq_mhz:.0f} MHz" if report.cpu.freq_mhz else ""
            cores = f" -- {p} cores / {l} threads{freq}"
        row("CPU", f"{report.cpu.model or 'Unknown'}{cores}")
    if report.memory:
        mem_extra = ""
        if report.memory.type:
            mem_extra += f" {report.memory.type}"
        if report.memory.speed_mhz:
            mem_extra += f" @ {report.memory.speed_mhz} MHz"
        row("Memory", f"{report.memory.total_gb} GB{mem_extra}")
    for gpu in report.gpu:
        vram = ""
        if gpu.vram_gb:
            unified = " (unified)" if gpu.unified else ""
            vram = f" ({gpu.vram_gb} GB VRAM{unified})"
        row("GPU", f"{gpu.model or 'Unknown'}{vram}")

    # Storage
    if report.storage:
        section("Storage")
        for disk in report.storage:
            dtype = f" {disk.disk_type}" if disk.disk_type else ""
            fs = f" {disk.fs_type}" if disk.fs_type else ""
            free = f" -- {disk.free_gb} GB free" if disk.free_gb else ""
            row(disk.device[:16], f"{disk.total_gb} GB{fs}{dtype}{free}")

    # Network
    if report.network:
        section("Network")
        for iface in report.network:
            parts = []
            if iface.ipv4:
                parts.append(iface.ipv4)
            if iface.speed_mbps:
                parts.append(f"{iface.speed_mbps} Mbps")
            if iface.mac:
                parts.append(iface.mac)
            itype = f" ({iface.type})" if iface.type else ""
            status = c.green("UP") if iface.is_up else c.dim("DOWN")
            detail = " | ".join(parts)
            row(f"{iface.name}{itype}"[:16], f"{status} {detail}")

    # Battery
    if report.battery:
        section("Battery")
        plug = "Plugged in" if report.battery.plugged_in else "On battery"
        time_left = ""
        if report.battery.time_remaining_min is not None:
            h = report.battery.time_remaining_min // 60
            m = report.battery.time_remaining_min % 60
            time_left = f" | {h}h{m:02d}m remaining"
        row("Battery", f"{report.battery.percent}% | {plug}{time_left}")

    # Peripherals
    has_peripherals = report.displays or report.audio or report.bluetooth
    if has_peripherals:
        section("Peripherals")
        for d in report.displays:
            res = d.resolution or "?"
            hz = f" @ {d.refresh_rate_hz}Hz" if d.refresh_rate_hz else ""
            row("Display", f"{d.name or 'Unknown'} ({res}{hz})")
        for a in report.audio:
            row("Audio", f"{a.name} ({a.type})")
        for b in report.bluetooth:
            conn = c.green("connected") if b.connected else c.dim("paired")
            btype = f" [{b.device_type}]" if b.device_type else ""
            row("Bluetooth", f"{b.name} ({conn}){btype}")

    # Sensors
    if report.sensors:
        section("Sensors")
        for name, temp in report.sensors.temperatures.items():
            row(name[:16], f"{temp:.1f} C")
        for name, rpm in report.sensors.fan_speeds.items():
            row(name[:16], f"{rpm} RPM")

    # Python
    if report.python:
        section("Python")
        for inst in report.python.installations:
            row("Python " + inst.version, inst.path)
        if report.python.virtual_envs:
            conda_envs = [e for e in report.python.virtual_envs if e.type == "conda"]
            venvs = [e for e in report.python.virtual_envs if e.type == "venv"]
            if conda_envs:
                names = ", ".join(e.name for e in conda_envs[:6])
                extra = f" (+{len(conda_envs) - 6} more)" if len(conda_envs) > 6 else ""
                row("Conda envs", f"{names}{extra}")
            for v in venvs:
                row("Venv", v.path)
        if report.python.active_env:
            row("Active env", report.python.active_env)

    # Dev Tools
    if report.dev_tools:
        section("Dev Tools")
        tool_strs = []
        for t in report.dev_tools:
            ver = f" {t.version}" if t.version else ""
            tool_strs.append(f"{t.name}{ver}")
        for i in range(0, len(tool_strs), 4):
            chunk = tool_strs[i:i + 4]
            row("", "    ".join(f"{s:<18}" for s in chunk))

    # Packages
    if report.packages:
        section("Packages")
        for pkg in report.packages:
            row(pkg.manager, f"{pkg.count} packages")

    # Services
    if report.services:
        section("Services")
        running = [s for s in report.services if s.status == "running"]
        stopped = [s for s in report.services if s.status != "running"]
        if running:
            row("Running", ", ".join(s.name for s in running))
        if stopped:
            row("Stopped", ", ".join(s.name for s in stopped))

    # Errors
    if report.errors:
        lines.append("")
        lines.append(c.red(c.bold(" Errors")))
        for err in report.errors:
            indicator = c.red("[X]") if err.category != "shutdown" else c.dim("[-]")
            suggestion = f" -- {err.suggestion}" if err.suggestion else ""
            lines.append(f"  {indicator} {err.collector:<16}{err.message}{suggestion}")

    # Footer
    lines.append("")
    lines.append("=" * W if not use_color else "\u2501" * W)
    n_ok = 15 - len(report.errors)
    n_err = len(report.errors)
    lines.append(
        f"Completed in {report.duration_seconds:.1f}s -- "
        f"{c.green(str(n_ok) + ' succeeded')}, "
        f"{c.red(str(n_err) + ' failed') if n_err else '0 failed'}"
    )
    lines.append("=" * W if not use_color else "\u2501" * W)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON & Text formatters
# ---------------------------------------------------------------------------

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
