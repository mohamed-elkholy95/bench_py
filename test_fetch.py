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
    assert "battery" not in loaded
    assert "sensors" not in loaded


def test_detect_os_returns_valid_enum():
    """detect_os() returns a known OSType member."""
    from fetch import detect_os, OSType
    result = detect_os()
    assert isinstance(result, OSType)
    assert result != OSType.UNKNOWN


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


def test_collect_cpu_returns_cpu_info():
    from fetch import collect_cpu, CommandRunner, detect_os, CpuInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        info = collect_cpu(runner, os_type)
        assert isinstance(info, CpuInfo)
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
        for gpu in gpus:
            assert isinstance(gpu, GpuInfo)


def test_collect_storage_returns_list():
    from fetch import collect_storage, CommandRunner, detect_os, StorageInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        disks = collect_storage(runner, os_type)
        assert isinstance(disks, list)
        assert len(disks) > 0
        assert isinstance(disks[0], StorageInfo)
        assert disks[0].total_gb is not None
        assert disks[0].total_gb > 0


def test_collect_network_returns_list():
    from fetch import collect_network, CommandRunner, detect_os, NetworkInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        ifaces = collect_network(runner, os_type)
        assert isinstance(ifaces, list)
        assert len(ifaces) > 0
        for iface in ifaces:
            assert isinstance(iface, NetworkInfo)
            assert len(iface.name) > 0


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


def test_collect_python_returns_info():
    from fetch import collect_python, CommandRunner, detect_os, PythonInfo
    os_type = detect_os()
    with CommandRunner() as runner:
        info = collect_python(runner, os_type)
        assert isinstance(info, PythonInfo)
        assert len(info.installations) > 0


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
