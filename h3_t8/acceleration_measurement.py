"""Scoped wall-time accounting; never equates speed with quality acceptance.

No automatic CUDA synchronization, resource reset, model loading or filesystem
writes. The caller supplies resource observations and marks actual cache hits.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math
import time


SCHEMA = "t8.h3.acceleration.measurement.v1"
WORKLOAD_FIELDS = frozenset({"model_sha256", "conditioning_sha256", "video_vae_sha256",
                             "audio_vae_sha256", "width", "height", "frames", "fps_num",
                             "fps_den", "seed", "nfe", "cfg", "recipe_sha256", "core_revision"})


class Measurement:
    def __init__(self, workload, *, cache_state, clock=time.perf_counter):
        if not isinstance(workload, dict) or not WORKLOAD_FIELDS.issubset(workload):
            raise ValueError("A complete workload identity is required")
        if cache_state not in ("cold", "warm"):
            raise ValueError("cache_state must be cold or warm")
        for name in ("model_sha256", "conditioning_sha256", "recipe_sha256", "video_vae_sha256", "audio_vae_sha256"):
            value = workload[name]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase SHA256")
        for name in ("width", "height", "frames", "fps_num", "fps_den", "nfe"):
            value = workload[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        seed = workload["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        cfg = workload["cfg"]
        if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not math.isfinite(cfg) or cfg < 0:
            raise ValueError("cfg must be finite and nonnegative")
        if not isinstance(workload["core_revision"], str) or not workload["core_revision"]:
            raise ValueError("core_revision is required")
        self.workload = deepcopy(workload)
        self.cache_state = cache_state
        self.clock = clock
        self.started = self._now()
        self.stages = []
        self.observations = []
        self.active = False
        self.closed = False
        self.failed = False
        self.cache_hit = False

    def _now(self):
        value = float(self.clock())
        if not math.isfinite(value):
            raise ValueError("Clock must be finite")
        return value

    def _open(self):
        if self.closed:
            raise RuntimeError("Measurement is already closed")

    def record_cache_hit(self):
        self._open()
        self.cache_hit = True

    def observe_resources(self, observation):
        self._open()
        if not isinstance(observation, dict) or not observation:
            raise ValueError("Resource observation must be a nonempty mapping")
        values = {}
        for name, value in observation.items():
            if not isinstance(name, str) or not name.endswith("_bytes"):
                raise ValueError("Resource fields must explicitly use bytes")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Resource bytes must be nonnegative integers")
            values[name] = value
        self.observations.append({"elapsed_seconds": self._now() - self.started, **values})

    @contextmanager
    def stage(self, name):
        self._open()
        if self.active:
            raise RuntimeError("Nested stages would double-count elapsed time")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Stage needs a name")
        began = self._now()
        self.active = True
        status = "complete"
        try:
            yield
        except BaseException:
            self.failed = True
            status = "failed"
            raise
        finally:
            self.active = False
            duration = self._now() - began
            if duration < 0:
                self.failed = True
                raise RuntimeError("Clock moved backwards")
            self.stages.append({"name": name, "seconds": duration, "status": status})

    def finish(self):
        self._open()
        if self.active:
            raise RuntimeError("Cannot finish during a stage")
        elapsed = self._now() - self.started
        accounted = sum(stage["seconds"] for stage in self.stages)
        if elapsed <= 0 or accounted > elapsed + 1e-9:
            raise RuntimeError("Invalid total or double-counted wall time")
        self.closed = True
        return {"schema": SCHEMA, "status": "failed" if self.failed else "complete",
                "workload": deepcopy(self.workload), "cache_state": self.cache_state,
                "cache_hit": self.cache_hit, "elapsed_seconds": elapsed,
                "stage_seconds": accounted, "unassigned_seconds": max(0., elapsed - accounted),
                "stages": deepcopy(self.stages), "resource_observations": deepcopy(self.observations),
                "resource_scope": "observed_samples_not_continuous_peaks",
                "quality": "not_evaluated", "synchronization": "caller_controlled"}


def compare_equal_workload(baseline, candidate):
    """Compare wall time only. Different-model/product comparisons are separate."""
    for report in (baseline, candidate):
        if report.get("schema") != SCHEMA or report.get("status") != "complete":
            raise ValueError("Both timing reports must be complete")
        if report.get("cache_hit") is not False:
            raise ValueError("Cached graph outputs cannot prove acceleration")
        value = report.get("elapsed_seconds")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid elapsed time")
        if not isinstance(report.get("workload"), dict) or not WORKLOAD_FIELDS.issubset(report["workload"]):
            raise ValueError("Missing workload identity")
        if report.get("cache_state") not in ("cold", "warm"):
            raise ValueError("Missing cache state")
    if baseline["workload"] != candidate["workload"]:
        raise ValueError("Workload mismatch; report separately as a product comparison")
    if baseline["cache_state"] != candidate["cache_state"]:
        raise ValueError("Cannot compare cold with warm timing")
    ratio = candidate["elapsed_seconds"] / baseline["elapsed_seconds"]
    return {"scope": "wall_time_only_quality_and_memory_not_accepted",
            "saved_fraction": 1 - ratio, "speedup": 1 / ratio,
            "baseline_seconds": baseline["elapsed_seconds"],
            "candidate_seconds": candidate["elapsed_seconds"]}
