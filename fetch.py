import subprocess
import platform
import signal
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import sys
import time
from enum import Enum

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_requested = False
_active_processes: List[subprocess.Popen] = []


def _signal_handler(signum: int, _frame) -> None:
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    log.warning("Received %s — shutting down gracefully", sig_name)
    _shutdown_requested = True
    # Terminate any child processes still running
    for proc in _active_processes:
        try:
            proc.terminate()
        except OSError:
            pass


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ---------------------------------------------------------------------------
# Enums & data
# ---------------------------------------------------------------------------

class OSType(Enum):
    MACOS = "macOS"
    LINUX = "Linux"
    WINDOWS = "Windows"
    UNKNOWN = "Unknown"


class Architecture(Enum):
    X86_64 = "x86_64"
    ARM64 = "arm64"
    ARMV7 = "armv7l"
    I386 = "i386"
    UNKNOWN = "Unknown"


@dataclass
class SystemInfo:
    os_type: str
    os_version: Optional[str] = None
    architecture: Optional[str] = None
    cpu_model: Optional[str] = None
    cpu_cores_physical: Optional[int] = None
    cpu_cores_logical: Optional[int] = None
    gpu_model: Optional[str] = None
    total_memory_gb: Optional[float] = None
    total_vram_gb: Optional[float] = None
    total_storage_gb: Optional[float] = None
    python_versions: List[str] = field(default_factory=list)
    virtual_envs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class Benchmark(ABC):

    COMMAND_TIMEOUT = 15  # seconds per command

    def __init__(self) -> None:
        self.os_type = self.detect_os()
        self.info = SystemInfo(os_type=self.os_type.value)
        self._collect_arch()

    # -- OS / arch detection --------------------------------------------------

    @staticmethod
    def detect_os() -> OSType:
        system = platform.system().lower()
        mapping = {"darwin": OSType.MACOS, "linux": OSType.LINUX, "windows": OSType.WINDOWS}
        return mapping.get(system, OSType.UNKNOWN)

    def _collect_arch(self) -> None:
        machine = platform.machine().lower()
        arch_map = {
            "x86_64": Architecture.X86_64,
            "amd64": Architecture.X86_64,
            "arm64": Architecture.ARM64,
            "aarch64": Architecture.ARM64,
            "armv7l": Architecture.ARMV7,
            "i386": Architecture.I386,
            "i686": Architecture.I386,
        }
        arch = arch_map.get(machine, Architecture.UNKNOWN)
        self.info.architecture = arch.value
        self.info.os_version = platform.platform()

    # -- Command helpers ------------------------------------------------------

    def execute_command(self, command: List[str], timeout: Optional[int] = None) -> str:
        if _shutdown_requested:
            raise InterruptedError("Shutdown requested")
        timeout = timeout or self.COMMAND_TIMEOUT
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            _active_processes.append(proc)
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            finally:
                _active_processes.remove(proc)
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, command, stdout, stderr)
            return stdout.strip()
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            _active_processes.discard(proc) if proc in _active_processes else None
            raise RuntimeError(f"Command timed out ({timeout}s): {' '.join(command)}")
        except FileNotFoundError:
            raise RuntimeError(f"Command not found: {command[0]}")

    def _safe_collect(self, label: str, fn) -> None:
        """Run a collector; log + record errors instead of crashing."""
        if _shutdown_requested:
            return
        try:
            fn()
            log.info("Collected %s", label)
        except InterruptedError:
            log.warning("Skipped %s (shutdown)", label)
        except Exception as e:
            msg = f"{label}: {e}"
            log.warning("Failed to collect %s — %s", label, e)
            self.info.errors.append(msg)

    # -- Abstract interface ---------------------------------------------------

    @abstractmethod
    def run(self) -> None:
        pass

    def collect_results(self) -> SystemInfo:
        return self.info


# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------
class MacBenchmark(Benchmark):

    def run(self) -> None:
        self._safe_collect("CPU", self._collect_cpu)
        self._safe_collect("Memory", self._collect_memory)
        self._safe_collect("GPU", self._collect_gpu)
        self._safe_collect("Storage", self._collect_storage)
        self._safe_collect("Python", self._collect_python)

    def _collect_cpu(self) -> None:
        # Apple Silicon doesn't expose machdep.cpu.brand_string
        try:
            self.info.cpu_model = self.execute_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        except Exception:
            chip = self.execute_command(["sysctl", "-n", "hw.chip"])
            self.info.cpu_model = f"Apple {chip}"

        self.info.cpu_cores_physical = int(self.execute_command(["sysctl", "-n", "hw.physicalcpu"]))
        self.info.cpu_cores_logical = int(self.execute_command(["sysctl", "-n", "hw.logicalcpu"]))

    def _collect_memory(self) -> None:
        mem_bytes = int(self.execute_command(["sysctl", "-n", "hw.memsize"]))
        self.info.total_memory_gb = round(mem_bytes / (1024 ** 3), 1)

    def _collect_gpu(self) -> None:
        sp_output = self.execute_command(["system_profiler", "SPDisplaysDataType"])

        for line in sp_output.split("\n"):
            stripped = line.strip()
            if "Chipset Model" in stripped or "Chip Model" in stripped:
                self.info.gpu_model = stripped.split(":", 1)[1].strip()
                break

        # VRAM — discrete GPUs report this explicitly
        for line in sp_output.split("\n"):
            if "VRAM" in line and ":" in line:
                val = line.split(":")[1].strip()
                parts = val.split()
                if len(parts) >= 2:
                    num = float(parts[0])
                    if "MB" in parts[1].upper():
                        num /= 1024
                    self.info.total_vram_gb = round(num, 1)
                break

        # Apple Silicon unified memory — GPU shares total RAM
        if self.info.total_vram_gb is None and self.info.gpu_model and "Apple" in (self.info.gpu_model or ""):
            self.info.total_vram_gb = self.info.total_memory_gb

    def _collect_storage(self) -> None:
        df_output = self.execute_command(["df", "-g", "/"])
        lines = df_output.split("\n")
        if len(lines) >= 2:
            parts = lines[1].split()
            self.info.total_storage_gb = float(parts[1])

    def _collect_python(self) -> None:
        seen_paths: set = set()
        for cmd in ["python3", "python"]:
            try:
                path = self.execute_command(["which", cmd])
                real = os.path.realpath(path)
                if real in seen_paths:
                    continue
                seen_paths.add(real)
                version = self.execute_command([cmd, "--version"])
                self.info.python_versions.append(f"{version} ({path})")
            except Exception:
                pass
        self._find_envs()

    def _find_envs(self) -> None:
        seen: set = set()
        # Conda / miniforge / miniconda / anaconda
        for base in ["miniforge3", "miniconda3", "anaconda3"]:
            envs_dir = os.path.expanduser(f"~/{base}/envs")
            if not os.path.isdir(envs_dir):
                continue
            try:
                for name in sorted(os.listdir(envs_dir)):
                    full = os.path.join(envs_dir, name)
                    if os.path.isdir(full) and full not in seen:
                        seen.add(full)
                        self.info.virtual_envs.append(f"conda: {name}")
            except OSError:
                pass
        # venvs under ~/Projects and cwd
        for search_dir in {os.path.expanduser("~/Projects"), os.getcwd()}:
            if not os.path.isdir(search_dir):
                continue
            try:
                result = self.execute_command(
                    ["find", search_dir, "-maxdepth", "4", "-name", "pyvenv.cfg", "-type", "f"],
                    timeout=10,
                )
                for cfg in result.split("\n"):
                    cfg = cfg.strip()
                    if cfg:
                        venv_dir = os.path.dirname(cfg)
                        if venv_dir not in seen:
                            seen.add(venv_dir)
                            self.info.virtual_envs.append(f"venv: {venv_dir}")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Linux
# ---------------------------------------------------------------------------
class LinuxBenchmark(Benchmark):

    def run(self) -> None:
        self._safe_collect("CPU", self._collect_cpu)
        self._safe_collect("Memory", self._collect_memory)
        self._safe_collect("GPU", self._collect_gpu)
        self._safe_collect("Storage", self._collect_storage)
        self._safe_collect("Python", self._collect_python)

    def _collect_cpu(self) -> None:
        cpuinfo = self.execute_command(["cat", "/proc/cpuinfo"])
        for line in cpuinfo.split("\n"):
            if "model name" in line:
                self.info.cpu_model = line.split(":")[1].strip()
                break
        if not self.info.cpu_model:
            self.info.cpu_model = platform.processor() or "Unknown"

        try:
            cores_output = self.execute_command(["nproc", "--all"])
            self.info.cpu_cores_logical = int(cores_output)
        except Exception:
            pass
        # Physical cores from lscpu
        try:
            lscpu = self.execute_command(["lscpu"])
            for line in lscpu.split("\n"):
                if "Core(s) per socket" in line:
                    cores_per = int(line.split(":")[1].strip())
                elif "Socket(s)" in line:
                    sockets = int(line.split(":")[1].strip())
            self.info.cpu_cores_physical = cores_per * sockets
        except Exception:
            pass

    def _collect_memory(self) -> None:
        meminfo = self.execute_command(["cat", "/proc/meminfo"])
        for line in meminfo.split("\n"):
            if "MemTotal" in line:
                total_kb = int(line.split(":")[1].strip().split()[0])
                self.info.total_memory_gb = round(total_kb / (1024 ** 2), 1)
                break

    def _collect_gpu(self) -> None:
        # Try lspci first
        try:
            lspci = self.execute_command(["lspci"])
            for line in lspci.split("\n"):
                if "VGA" in line or "3D" in line or "Display" in line:
                    self.info.gpu_model = line.split(":", 2)[2].strip() if line.count(":") >= 2 else line
                    break
        except Exception:
            pass
        # nvidia-smi for NVIDIA GPUs
        try:
            name = self.execute_command(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"])
            self.info.gpu_model = name.split("\n")[0].strip()
            vram_out = self.execute_command(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            self.info.total_vram_gb = round(float(vram_out.split("\n")[0]) / 1024, 1)
        except Exception:
            pass
        # AMD ROCm
        if not self.info.total_vram_gb:
            try:
                rocm = self.execute_command(["rocm-smi", "--showmeminfo", "vram"])
                for line in rocm.split("\n"):
                    if "Total" in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == "Total":
                                mb = float(parts[i + 2])
                                self.info.total_vram_gb = round(mb / 1024, 1)
                                break
                        break
            except Exception:
                pass

    def _collect_storage(self) -> None:
        df_output = self.execute_command(["df", "--block-size=G", "/"])
        lines = df_output.split("\n")
        if len(lines) >= 2:
            parts = lines[1].split()
            self.info.total_storage_gb = float(parts[1].rstrip("G"))

    def _collect_python(self) -> None:
        seen_paths: set = set()
        for cmd in ["python3", "python"]:
            try:
                path = self.execute_command(["which", cmd])
                real = os.path.realpath(path)
                if real in seen_paths:
                    continue
                seen_paths.add(real)
                version = self.execute_command([cmd, "--version"])
                self.info.python_versions.append(f"{version} ({path})")
            except Exception:
                pass

        seen_envs: set = set()
        # Conda
        for base in ["miniconda3", "miniforge3", "anaconda3"]:
            envs_dir = os.path.expanduser(f"~/{base}/envs")
            if not os.path.isdir(envs_dir):
                continue
            try:
                for name in sorted(os.listdir(envs_dir)):
                    full = os.path.join(envs_dir, name)
                    if os.path.isdir(full) and full not in seen_envs:
                        seen_envs.add(full)
                        self.info.virtual_envs.append(f"conda: {name}")
            except OSError:
                pass
        # venvs
        for search_dir in {os.path.expanduser("~"), os.getcwd()}:
            if not os.path.isdir(search_dir):
                continue
            try:
                result = self.execute_command(
                    ["find", search_dir, "-maxdepth", "4", "-name", "pyvenv.cfg", "-type", "f"],
                    timeout=10,
                )
                for cfg in result.split("\n"):
                    cfg = cfg.strip()
                    if cfg:
                        venv_dir = os.path.dirname(cfg)
                        if venv_dir not in seen_envs:
                            seen_envs.add(venv_dir)
                            self.info.virtual_envs.append(f"venv: {venv_dir}")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
class WindowsBenchmark(Benchmark):

    def run(self) -> None:
        self._safe_collect("CPU", self._collect_cpu)
        self._safe_collect("Memory", self._collect_memory)
        self._safe_collect("GPU", self._collect_gpu)
        self._safe_collect("Storage", self._collect_storage)
        self._safe_collect("Python", self._collect_python)

    def _ps(self, script: str) -> str:
        """Run a PowerShell one-liner and return stdout."""
        return self.execute_command(["powershell", "-NoProfile", "-Command", script])

    def _collect_cpu(self) -> None:
        self.info.cpu_model = self._ps("(Get-CimInstance Win32_Processor).Name").split("\n")[0].strip()
        try:
            self.info.cpu_cores_physical = int(self._ps("(Get-CimInstance Win32_Processor).NumberOfCores"))
            self.info.cpu_cores_logical = int(self._ps("(Get-CimInstance Win32_Processor).NumberOfLogicalProcessors"))
        except (ValueError, Exception):
            pass

    def _collect_memory(self) -> None:
        raw = self._ps("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
        mem_bytes = int(raw.strip())
        self.info.total_memory_gb = round(mem_bytes / (1024 ** 3), 1)

    def _collect_gpu(self) -> None:
        self.info.gpu_model = self._ps("(Get-CimInstance Win32_VideoController).Name").split("\n")[0].strip()
        try:
            vram_bytes = int(self._ps("(Get-CimInstance Win32_VideoController).AdapterRAM").split("\n")[0])
            self.info.total_vram_gb = round(vram_bytes / (1024 ** 3), 1)
        except (ValueError, Exception):
            pass
        # AdapterRAM caps at 4 GB on some drivers — try nvidia-smi as fallback
        if self.info.total_vram_gb and self.info.total_vram_gb <= 4.0:
            try:
                nv = self.execute_command(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
                self.info.total_vram_gb = round(float(nv.split("\n")[0]) / 1024, 1)
            except Exception:
                pass

    def _collect_storage(self) -> None:
        raw = self._ps("(Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='C:'\").Size")
        size_bytes = int(raw.strip())
        self.info.total_storage_gb = round(size_bytes / (1024 ** 3), 1)

    def _collect_python(self) -> None:
        seen_paths: set = set()
        for cmd in ["python", "python3"]:
            try:
                path = self._ps(f"(Get-Command {cmd}).Source").strip()
                real = os.path.realpath(path)
                if real in seen_paths:
                    continue
                seen_paths.add(real)
                version = self.execute_command([cmd, "--version"])
                self.info.python_versions.append(f"{version} ({path})")
            except Exception:
                pass

        seen_envs: set = set()
        # Conda
        for base in ["miniconda3", "miniforge3", "Anaconda3"]:
            envs_dir = os.path.join(os.path.expanduser("~"), base, "envs")
            if not os.path.isdir(envs_dir):
                continue
            try:
                for name in sorted(os.listdir(envs_dir)):
                    full = os.path.join(envs_dir, name)
                    if os.path.isdir(full) and full not in seen_envs:
                        seen_envs.add(full)
                        self.info.virtual_envs.append(f"conda: {name}")
            except OSError:
                pass
        # venvs — walk common dirs (no `find` on Windows)
        for search_root in {os.path.expanduser("~\\Projects"), os.getcwd()}:
            if not os.path.isdir(search_root):
                continue
            try:
                for root, dirs, files in os.walk(search_root):
                    # Don't descend too deep
                    depth = root.replace(search_root, "").count(os.sep)
                    if depth >= 4:
                        dirs.clear()
                        continue
                    if "pyvenv.cfg" in files:
                        if root not in seen_envs:
                            seen_envs.add(root)
                            self.info.virtual_envs.append(f"venv: {root}")
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_results(result: SystemInfo, filepath: str = "benchmark_results.txt") -> None:
    with open(filepath, "w") as f:
        f.write("=== System Information ===\n\n")
        f.write(f"OS:              {result.os_type}\n")
        f.write(f"OS Version:      {result.os_version or 'N/A'}\n")
        f.write(f"Architecture:    {result.architecture or 'N/A'}\n")
        f.write(f"CPU:             {result.cpu_model or 'N/A'}\n")
        if result.cpu_cores_physical or result.cpu_cores_logical:
            phys = result.cpu_cores_physical or "?"
            logi = result.cpu_cores_logical or "?"
            f.write(f"CPU Cores:       {phys} physical / {logi} logical\n")
        f.write(f"GPU:             {result.gpu_model or 'N/A'}\n")
        f.write(f"Total Memory:    {result.total_memory_gb or 'N/A'} GB\n")
        vram_note = " (unified)" if (result.total_vram_gb and result.total_vram_gb == result.total_memory_gb) else ""
        f.write(f"Total VRAM:      {result.total_vram_gb or 'N/A'} GB{vram_note}\n")
        f.write(f"Total Storage:   {result.total_storage_gb or 'N/A'} GB\n")
        f.write(f"Timestamp:       {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result.timestamp))}\n")

        f.write("\n--- Python Installations ---\n")
        if result.python_versions:
            for pv in result.python_versions:
                f.write(f"  {pv}\n")
        else:
            f.write("  None found\n")

        f.write("\n--- Virtual Environments ---\n")
        if result.virtual_envs:
            for ve in result.virtual_envs:
                f.write(f"  {ve}\n")
        else:
            f.write("  None found\n")

        if result.errors:
            f.write("\n--- Collection Errors ---\n")
            for err in result.errors:
                f.write(f"  [!] {err}\n")

    log.info("Results saved to %s", filepath)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
BENCHMARKS: Dict[OSType, type] = {
    OSType.MACOS: MacBenchmark,
    OSType.LINUX: LinuxBenchmark,
    OSType.WINDOWS: WindowsBenchmark,
}


def main() -> None:
    os_type = Benchmark.detect_os()
    bench_cls = BENCHMARKS.get(os_type)

    if bench_cls is None:
        log.error("Unsupported OS: %s", os_type.value)
        sys.exit(1)

    log.info("Detected %s (%s)", os_type.value, platform.machine())
    bench = bench_cls()

    try:
        bench.run()
    except InterruptedError:
        log.warning("Collection interrupted — saving partial results")

    if _shutdown_requested:
        log.info("Saving partial results before exit")

    result = bench.collect_results()
    save_results(result)


if __name__ == "__main__":
    main()
