"""Bounded read-only stage observations. No CUDA init/reset/sync, admission or retry."""

from contextvars import ContextVar
import logging
import math
import os
import sys
import threading
import time

_CURRENT = ContextVar("t8_meridian_resource_trace", default=None)


class ObservationErrors(list):
    def append(self, value):
        if value not in self and len(self) < 32:
            super().append(value)


class ResourceReader:
    """Optional telemetry; allocator reads only after another owner initialized CUDA."""

    def __init__(self):
        self.errors = ObservationErrors()
        self.process = self.psutil = self.nvml = None
        self.devices = []
        try:
            import psutil

            self.psutil = psutil
            self.process = psutil.Process(os.getpid())
        except Exception as error:
            self.errors.append("psutil:" + type(error).__name__)
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                identity = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(identity, bytes):
                    identity = identity.decode("utf8")
                self.devices.append((str(identity), handle))
        except Exception as error:
            self.errors.append("NVML:" + type(error).__name__)

    def sample(self):
        row = dict(
            stamp=time.perf_counter(), devices=[], torch_allocator=self.allocator()
        )
        if self.process is not None:
            try:
                memory = self.process.memory_info()
                ram = self.psutil.virtual_memory()
                row.update(
                    process_rss_bytes=int(memory.rss),
                    system_available_bytes=int(ram.available),
                )
                if hasattr(memory, "private"):
                    row["process_private_bytes"] = int(memory.private)
            except Exception as error:
                self.errors.append("RAM:" + type(error).__name__)
        for identity, handle in self.devices:
            try:
                memory = self.nvml.nvmlDeviceGetMemoryInfo(handle)
                row["devices"].append(
                    dict(
                        uuid=identity,
                        total_bytes=int(memory.total),
                        used_bytes=int(memory.used),
                        free_bytes=int(memory.free),
                        utilization_percent=int(
                            self.nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        ),
                    )
                )
            except Exception as error:
                self.errors.append("GPU:" + type(error).__name__)
        return row

    def allocator(self):
        # Never import Torch, initialize CUDA or reset the caller's peak counters.
        # Explicit logical-device indices avoid changing current-device state in
        # the observer thread. Counters are process-wide, not node-exclusive.
        torch = sys.modules.get("torch")
        if torch is None:
            return dict(status="torch_not_loaded", devices=[])
        try:
            cuda = torch.cuda
            if not cuda.is_initialized():
                return dict(status="CUDA_not_initialized", devices=[])
            return dict(
                status="observed",
                devices=[
                    dict(
                        logical_device=i,
                        allocated_bytes=int(cuda.memory_allocated(i)),
                        reserved_bytes=int(cuda.memory_reserved(i)),
                    )
                    for i in range(cuda.device_count())
                ],
                scope="current process allocator per visible logical device; not exclusive node allocation or reset peak",
            )
        except Exception as error:
            self.errors.append("Torch-allocator:" + type(error).__name__)
            return dict(status="unavailable", devices=[])

    def close(self):
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()  # Only this reader's NVML reference, not CUDA/model state.
            except Exception as error:
                self.errors.append("NVML-close:" + type(error).__name__)
            self.nvml = None


class ResourceTrace:
    def __init__(self, stage, *, reader_factory=ResourceReader, interval=1.0):
        if not math.isfinite(interval) or interval < 0.1 or interval > 60:
            raise ValueError("Resource observation interval must be .1..60 seconds")
        self.stage, self.reader_factory, self.interval = (
            stage,
            reader_factory,
            float(interval),
        )
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.reader = self.thread = None
        self.first = self.last = None
        self.maximum = {}
        self.minimum = {}
        self.devices = {}
        self.torch_devices = {}
        self.samples = 0
        self.errors = ObservationErrors()
        self.offloads = []

    def observe(self):
        try:
            row = self.reader.sample()
            with self.lock:
                self.first = row if self.first is None else self.first
                self.last = row
                self.samples += 1
                for device in row.get("torch_allocator", {}).get("devices", []):
                    prior = self.torch_devices.setdefault(
                        str(device["logical_device"]), {}
                    )
                    for metric in ("allocated_bytes", "reserved_bytes"):
                        key = "maximum_observed_" + metric
                        prior[key] = max(prior.get(key, device[metric]), device[metric])
                for key in (
                    "process_rss_bytes",
                    "process_private_bytes",
                    "system_available_bytes",
                ):
                    if key in row:
                        self.maximum[key] = max(
                            self.maximum.get(key, row[key]), row[key]
                        )
                        self.minimum[key] = min(
                            self.minimum.get(key, row[key]), row[key]
                        )
                for device in row.get("devices", []):
                    prior = self.devices.setdefault(
                        device["uuid"],
                        dict(
                            maximum_observed_used_bytes=0,
                            minimum_observed_free_bytes=device["free_bytes"],
                            maximum_observed_utilization_percent=0,
                        ),
                    )
                    prior["maximum_observed_used_bytes"] = max(
                        prior["maximum_observed_used_bytes"], device["used_bytes"]
                    )
                    prior["minimum_observed_free_bytes"] = min(
                        prior["minimum_observed_free_bytes"], device["free_bytes"]
                    )
                    prior["maximum_observed_utilization_percent"] = max(
                        prior["maximum_observed_utilization_percent"],
                        device["utilization_percent"],
                    )
        except Exception as error:
            self.errors.append("sample:" + type(error).__name__)

    def watch(self):
        try:
            while not self.stop.wait(self.interval):
                self.observe()
            self.observe()
        finally:
            try:
                self.reader.close()
            except Exception as error:
                self.errors.append("close:" + type(error).__name__)

    def __enter__(self):
        self.started = time.perf_counter()
        self.token = _CURRENT.set(self)
        try:
            self.reader = self.reader_factory()
            self.observe()
            self.thread = threading.Thread(
                target=self.watch, name="t8-meridian-observation", daemon=True
            )
            self.thread.start()
        except Exception as error:
            self.errors.append("start:" + type(error).__name__)
            if self.reader is not None:
                try:
                    self.reader.close()
                except Exception as close_error:
                    self.errors.append("close:" + type(close_error).__name__)
        return self

    def __exit__(self, *args):
        self.stop.set()
        if self.thread is not None and self.thread.is_alive():
            try:
                self.thread.join(timeout=1.5)
            except Exception as error:
                self.errors.append("join:" + type(error).__name__)
        self.seconds = time.perf_counter() - self.started
        _CURRENT.reset(self.token)
        return False

    def report(self):
        with self.lock:
            errors = set(self.errors + list(getattr(self.reader, "errors", [])))
            return dict(
                schema="t8.meridian.resource_observation.v1",
                stage=self.stage,
                seconds=getattr(self, "seconds", time.perf_counter() - self.started),
                interval_seconds=self.interval,
                samples=self.samples,
                before=self.first,
                after=self.last,
                maximum_observed=dict(self.maximum),
                minimum_observed=dict(self.minimum),
                devices={k: dict(v) for k, v in self.devices.items()},
                torch_allocator_devices={
                    k: dict(v) for k, v in self.torch_devices.items()
                },
                errors=sorted(errors),
                observer_still_finishing=bool(
                    self.thread is not None and self.thread.is_alive()
                ),
                owned_offloads=list(self.offloads),
                scope="periodic whole-device/system/current-Core-process observations; includes other models/processes, not node-only allocation or exact transient peak",
                controls_model_state=False,
                admission_gate=False,
                automatic_retry=False,
            )


def owned_offload(operation, name):
    """Time the caller's existing owned cleanup; preserve its result/error unchanged."""
    trace = _CURRENT.get()
    started = time.perf_counter()
    status = "failed"
    try:
        value = operation()
        status = "complete"
        return value
    finally:
        if trace is not None:
            trace.offloads.append(
                dict(
                    name=name, status=status, wall_seconds=time.perf_counter() - started
                )
            )


def note_cleanup_failure(original, cleanup_error, owner):
    """Keep the primary error on Python3.10 too; diagnostic notes are optional."""
    message = f"Owned {owner} cleanup also failed: {cleanup_error}"
    try:
        note = getattr(original, "add_note", None)
        if callable(note):
            note(message)
            return
    except Exception:
        pass  # A custom error's broken note handler must not replace the primary error.
    logging.warning("%s; original %s retained", message, type(original).__name__)
