"""Bind a recorded Windows NGX LUID to CUDA/NVML UUID without creating a CUDA context.

Uses NVIDIA's documented cuDeviceGetLuid device-management API, not a device-name
guess: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html
Enumeration alone is not DLSS frame-generation qualification.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import re
import uuid

from .resources import file_identity


def cuda_device_inventory():
    if os.name != "nt":
        raise RuntimeError("This graphics LUID route is Windows-only")
    driver_path = Path(os.environ["SystemRoot"]) / "System32/nvcuda.dll"
    driver = ctypes.WinDLL(str(driver_path.resolve(strict=True)))
    signatures = {"cuInit": [ctypes.c_uint], "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDeviceGetLuid": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_int],
        "cuDeviceGetUuid": [ctypes.c_void_p, ctypes.c_int],
        "cuDeviceGetName": [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]}
    def call(name, *args):
        fn = getattr(driver, name)
        fn.argtypes, fn.restype = signatures[name], ctypes.c_int
        result = fn(*args)
        if result != 0:
            raise RuntimeError(f"CUDA device enumeration {name} returned {result}")
    call("cuInit", 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    if not 1 <= count.value <= 32:
        raise RuntimeError("Unexpected CUDA device count")
    rows = []
    for ordinal in range(count.value):
        device, mask = ctypes.c_int(), ctypes.c_uint()
        luid, unique, name = ctypes.create_string_buffer(8), ctypes.create_string_buffer(16), ctypes.create_string_buffer(256)
        call("cuDeviceGet", ctypes.byref(device), ordinal)
        call("cuDeviceGetLuid", luid, ctypes.byref(mask), device.value)
        call("cuDeviceGetUuid", unique, device.value)
        call("cuDeviceGetName", name, 256, device.value)
        rows.append({"ordinal": ordinal, "luid": hex(int.from_bytes(luid.raw, "little")), "node_mask": mask.value,
            "uuid": "GPU-" + str(uuid.UUID(bytes=unique.raw)), "name": name.value.decode("utf-8", "strict")})
    return {"devices": rows, "driver": file_identity(driver_path), "cuda_context_created": False,
        "api_scope": "cuInit and device-management queries only; no allocation/kernel/primary context"}


def bind_probe(probe, inventory, expected_uuid):
    if not probe.get("supports_native_2x_by_report"):
        raise ValueError("Recorded feature report does not support native 2x")
    matches = re.findall(r"SetGPUArch:: Gpu count = (\d+), luid: (0x[0-9a-fA-F]+)", probe["stderr_tail"])
    identities = {(int(count), int(luid, 16)) for count, luid in matches}
    if len(identities) != 1:
        raise ValueError("Missing or ambiguous actual NGX GPU LUID")
    count, luid = identities.pop()
    if count != 1:
        raise ValueError("Only the observed single-device NGX route is qualified")
    candidates = [row for row in inventory["devices"] if int(row["luid"], 16) == luid]
    if len(candidates) != 1 or candidates[0]["node_mask"] != 1 or candidates[0]["uuid"] != expected_uuid:
        raise ValueError("NGX LUID is not uniquely bound to the expected physical CUDA/NVML UUID")
    return {"status": "recorded_NGX_LUID_matches_current_CUDA_and_NVML_UUID",
        "device": candidates[0], "generation_qualified": False, "audio_qualified": False, "quality_qualified": False}
