"""Read-only identities, scoped OS lease, telemetry and event timing for H3 probes.

No model loading, queue submission, process termination or automatic retries.
The caller owns its server and decides how to stop it when the guard trips.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import time


MIB = 1024**2


def file_identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    if not path.is_file():
        raise ValueError("Identity requires a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * MIB), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError(f"File changed during identity scan: {path.name}")
    return {"path": str(path), "bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
            "sha256": digest.hexdigest()}


class SerialProbeLease:
    """An OS-held lock, not a stale-file or PID-only lock; never deletes lockfiles."""
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise RuntimeError("Another acceleration probe owns the serial GPU lease") from error
        self.handle = handle
        return self

    def __exit__(self, *args):
        handle, self.handle = self.handle, None
        if handle is not None:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


class NvmlResourceReader:
    """NVML/psutil observation only; does not create a Torch CUDA context."""
    def __enter__(self):
        import psutil
        import pynvml
        self.nvml, self.psutil = pynvml, psutil
        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() != 1:
                raise RuntimeError("Initial progressive probe is single-GPU only")
            self.device = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.uuid = str(pynvml.nvmlDeviceGetUUID(self.device))
        except BaseException:
            pynvml.nvmlShutdown()
            raise
        return self

    def sample(self):
        gpu = self.nvml.nvmlDeviceGetMemoryInfo(self.device)
        ram = self.psutil.virtual_memory()
        # Python 3.11 Windows monotonic() can have a coarse GetTickCount64 tick:
        # startup and the first worker observation may otherwise share a stamp.
        # perf_counter uses the high-resolution monotonic performance counter.
        return {"monotonic": time.perf_counter(), "clock": "perf_counter", "gpu_uuid": self.uuid,
                "gpu_total_bytes": int(gpu.total), "gpu_used_bytes": int(gpu.used), "gpu_free_bytes": int(gpu.free),
                "gpu_utilization_percent": int(self.nvml.nvmlDeviceGetUtilizationRates(self.device).gpu),
                "ram_total_bytes": int(ram.total), "ram_available_bytes": int(ram.available)}

    def __exit__(self, *args):
        self.nvml.nvmlShutdown()


@dataclass(frozen=True)
class GuardPolicy:
    startup_free_gpu_bytes: int = 12000 * MIB
    startup_free_ram_bytes: int = 16384 * MIB
    minimum_free_gpu_bytes: int = 512 * MIB
    minimum_free_ram_bytes: int = 4096 * MIB
    consecutive_low_samples: int = 3
    maximum_sample_gap_seconds: float = 3.0

    def __post_init__(self):
        for name in ("startup_free_gpu_bytes", "startup_free_ram_bytes", "minimum_free_gpu_bytes",
                     "minimum_free_ram_bytes", "consecutive_low_samples"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Invalid guard threshold: {name}")
        if self.startup_free_gpu_bytes < self.minimum_free_gpu_bytes or self.startup_free_ram_bytes < self.minimum_free_ram_bytes:
            raise ValueError("Startup reserve cannot be below runtime reserve")
        if not math.isfinite(self.maximum_sample_gap_seconds) or self.maximum_sample_gap_seconds <= 0:
            raise ValueError("Invalid telemetry gap")


class ResourceGuard:
    """Sticky failure state; caller must not clear it and retry automatically."""
    def __init__(self, policy=None):
        self.policy = policy or GuardPolicy()
        self.reason = None
        self.previous = None
        self.gpu_uuid = None
        self.low_samples = 0
        self.samples = 0
        self.minimum_gpu_free = None
        self.minimum_ram_free = None
        self.maximum_gpu_used = 0

    def observe(self, row, *, startup=False):
        if self.reason:
            return self.reason
        required = ("gpu_total_bytes", "gpu_used_bytes", "gpu_free_bytes", "ram_total_bytes", "ram_available_bytes")
        try:
            if not isinstance(row, dict) or any(type(row.get(k)) is not int or row[k] < 0 for k in required):
                raise ValueError("invalid resource fields")
            stamp = row["monotonic"]
            if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                raise ValueError("invalid clock")
            if not isinstance(row.get("gpu_uuid"), str) or not row["gpu_uuid"]:
                raise ValueError("missing GPU identity")
            if (row["gpu_total_bytes"] == 0 or row["ram_total_bytes"] == 0 or
                    max(row["gpu_free_bytes"], row["gpu_used_bytes"]) > row["gpu_total_bytes"] or
                    row["ram_available_bytes"] > row["ram_total_bytes"]):
                raise ValueError("impossible resource snapshot")
            if self.previous is not None and not 0 < stamp - self.previous <= self.policy.maximum_sample_gap_seconds:
                raise ValueError("stale, reversed or excessively delayed telemetry")
            if self.gpu_uuid is not None and self.gpu_uuid != row["gpu_uuid"]:
                raise ValueError("GPU identity changed")
        except (ValueError, KeyError, TypeError):
            self.reason = "resource_telemetry_invalid_or_stale"
            return self.reason
        self.gpu_uuid, self.previous = row["gpu_uuid"], stamp
        self.samples += 1
        free, ram = row["gpu_free_bytes"], row["ram_available_bytes"]
        self.minimum_gpu_free = free if self.minimum_gpu_free is None else min(self.minimum_gpu_free, free)
        self.minimum_ram_free = ram if self.minimum_ram_free is None else min(self.minimum_ram_free, ram)
        self.maximum_gpu_used = max(self.maximum_gpu_used, row["gpu_used_bytes"])
        if startup:
            if free < self.policy.startup_free_gpu_bytes or ram < self.policy.startup_free_ram_bytes:
                self.reason = "startup_resource_margin_insufficient"
        elif free < 128 * MIB or ram < 1024 * MIB:
            self.reason = "critical_resource_margin"
        else:
            low = free < self.policy.minimum_free_gpu_bytes or ram < self.policy.minimum_free_ram_bytes
            self.low_samples = self.low_samples + 1 if low else 0
            if self.low_samples >= self.policy.consecutive_low_samples:
                self.reason = "sustained_resource_margin_insufficient"
        return self.reason

    def report(self):
        return {"status": "failed" if self.reason or not self.samples else "observations_within_policy",
                "reason": self.reason, "samples": self.samples, "gpu_uuid": self.gpu_uuid,
                "minimum_gpu_free_bytes": self.minimum_gpu_free, "minimum_ram_available_bytes": self.minimum_ram_free,
                "maximum_observed_device_used_bytes": self.maximum_gpu_used,
                "scope": "periodic_whole_device_and_system_samples_not_exact_peak_or_process_attribution"}


def summarize_execution_events(events, prompt_id, expected_nodes, elapsed_seconds):
    """Approximate server-node intervals from received events, never kernel timings."""
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("Invalid elapsed wall time")
    intervals, seen, cached = [], set(), set()
    previous = 0.0
    active = None
    terminal = None
    for event in events:
        stamp, data = event["elapsed_seconds"], event.get("data", {})
        if not math.isfinite(stamp) or not previous <= stamp <= elapsed_seconds:
            raise ValueError("Nonmonotonic event clock")
        previous = stamp
        if data.get("prompt_id") != prompt_id:
            continue
        kind = event["type"]
        if terminal is not None and (kind in {"execution_cached", "execution_success", "execution_error", "execution_interrupted"}
                                     or kind == "executing" and data.get("node") is not None):
            raise ValueError("Execution event arrived after the terminal event")
        if kind == "execution_cached":
            cached.update(map(str, data.get("nodes", [])))
        if kind == "executing" and data.get("node") is not None:
            node_id = str(data["node"])
            if node_id in seen:
                raise ValueError("Repeated node execution needs a separate dynamic-graph timing contract")
            if active:
                intervals.append({"node": active[0], "seconds": stamp - active[1]})
            active = (node_id, stamp)
            seen.add(node_id)
        elif kind in {"execution_success", "execution_error", "execution_interrupted"}:
            if terminal is not None:
                raise ValueError("Multiple terminal events")
            terminal = kind
            if active:
                intervals.append({"node": active[0], "seconds": stamp - active[1]})
                active = None
    missing, unexpected = set(expected_nodes) - seen, seen - set(expected_nodes)
    return {"terminal": terminal, "complete_uncached_graph": terminal == "execution_success" and not cached and not missing and not unexpected,
            "graph_cached_nodes": sorted(cached), "missing_nodes": sorted(missing), "unexpected_nodes": sorted(unexpected),
            "elapsed_seconds": elapsed_seconds, "node_intervals": intervals,
            "unassigned_seconds": elapsed_seconds - sum(row["seconds"] for row in intervals),
            "timing_scope": "client_submission_through_terminal_and_history; node_intervals_include_dispatch_overhead_not_cuda_kernel_times"}
