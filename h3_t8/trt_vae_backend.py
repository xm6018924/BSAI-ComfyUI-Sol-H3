"""Scoped real TensorRT backend. No import-time TRT/CUDA work or compilation.

Only identity-bound local bundles are loaded. Comfy is asked to free GPU memory
BEFORE deserialization, then again for the engine-reported context size. The
explicit startup reserve is a conservative admission policy, NOT an estimate
derived from engine file size. Engines/contexts are not cached between leases.
"""

from contextlib import contextmanager, ExitStack
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import tempfile

from .trt_vae_build import digest_file, graph_spec, loaded_windows_libraries, validate_request, flex_output_shape, validate_quantized_manifest
from .trt_vae_contract import positive_int

GIB = 1024**3
_ACTIVE = threading.Lock()


def inspect_bundle(path, kind, shapes=None):
    """CPU metadata checks; payload is hashed once from the bytes later loaded."""
    root = Path(path).resolve(strict=True)
    engine_path, manifest_path = root / "model.engine", root / "manifest.json"
    for member in (engine_path, manifest_path):
        if member.is_symlink() or not member.is_file() or member.resolve().parent != root:
            raise ValueError("Bundle files must stay inside their selected directory")
    if manifest_path.stat().st_size > 1024**2:
        raise ValueError("Engine manifest exceeds metadata limit")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    request = manifest["source_request"]
    if validate_request(request) != manifest.get("request_sha256"):
        raise ValueError("Engine source request identity mismatch")
    validate_quantized_manifest(manifest, request)
    input_name, output_name, input_shape, output_shape = graph_spec(request)
    if kind != request.get("kind", "decoder") or (shapes is not None and not supports_shapes(request, shapes)):
        raise ValueError("Selected engine does not support all required tile shapes")
    for key, value in {"input_name": input_name, "output_name": output_name,
                       "input_shape": list(input_shape), "output_shape": list(output_shape),
                       "io_dtype": "float16", "runtime_version": request["runtime_version"],
                       "gpu_uuid": request["gpu_uuid"], "driver_version": request["driver_version"]}.items():
        if manifest.get(key) != value:
            raise ValueError(f"Engine manifest mismatch: {key}")
    size = manifest.get("engine_bytes")
    positive_int(size, "engine_bytes")
    if engine_path.stat().st_size != size or size > 16 * GIB:
        raise ValueError("Engine payload size differs or exceeds supported host buffer")
    digest = manifest.get("engine_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Missing engine payload digest")
    return root, manifest


def supports_shapes(request, shapes):
    shapes = set(map(tuple, shapes))
    if not shapes:
        return False
    if request["schema"] == "t8-trt-decoder-flex-build-v1":
        try:
            for shape in shapes:
                flex_output_shape(shape)
        except ValueError:
            return False
        return True
    return shapes == {graph_spec(request)[2]}


def select_bundle(paths, kind, shapes):
    """Validate every selected manifest, then require one exact profile match."""
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    requested = set(map(tuple, shapes))
    candidates = [inspect_bundle(path, kind) for path in paths]
    matches = [(root, manifest) for root, manifest in candidates
               if supports_shapes(manifest["source_request"], requested)]
    if len(matches) != 1:
        raise ValueError("Select exactly one matching engine per required tile profile; no padding or ambiguous fallback")
    return matches[0]


def read_payload(root, manifest):
    # Hash the SAME immutable bytes handed to deserialize, not a previous file read.
    payload = (root / "model.engine").read_bytes()
    if len(payload) != manifest["engine_bytes"] or hashlib.sha256(payload).hexdigest() != manifest["engine_sha256"]:
        raise ValueError("Engine payload identity mismatch")
    return payload


def check_loaded_modules(site):
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in ("tensorrt", "tensorrt_bindings", "tensorrt_libs"):
            location = getattr(module, "__file__", None)
            if not location or not Path(location).resolve().is_relative_to(site):
                raise RuntimeError("Another TensorRT installation is already imported; use a clean compatible process")


def check_existing_dlls(site):
    """Reject already mapped incompatible TRT DLLs BEFORE importing bindings."""
    if os.name != "nt":
        raise RuntimeError("This initial TRT backend is qualified only on Windows")
    import ctypes
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetModuleHandleW.argtypes, api.GetModuleHandleW.restype = [ctypes.c_wchar_p], ctypes.c_void_p
    api.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    api.GetModuleFileNameW.restype = ctypes.c_uint32
    for name in ("nvinfer_10.dll", "nvonnxparser_10.dll", "nvinfer_plugin_10.dll", "nvinfer_builder_resource_10.dll"):
        handle = api.GetModuleHandleW(name)
        if handle:
            buffer = ctypes.create_unicode_buffer(32768)
            count = api.GetModuleFileNameW(handle, buffer, len(buffer))
            if not count or count >= len(buffer) or not Path(buffer.value).resolve().is_relative_to(site):
                raise RuntimeError(f"Already mapped incompatible TensorRT DLL: {name}; restart with one runtime")


def load_runtime(manifest, site):
    site = Path(site).resolve(strict=True)
    request = manifest["source_request"]
    if site != Path(request["runtime_site"]).resolve():
        raise ValueError("Runtime directory differs from engine build identity")
    for path, digest in request["runtime_sources"].items():
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_relative_to(site) or digest_file(resolved) != digest:
            raise ValueError("Runtime source identity changed; rebuild/revalidate explicitly")
    check_loaded_modules(site)
    check_existing_dlls(site)
    # No installation, PATH changes, or replacing/deleting existing modules.
    entry = str(site)
    sys.path.insert(0, entry)
    try:
        trt = importlib.import_module("tensorrt")
    finally:
        if sys.path and sys.path[0] is entry:
            sys.path.pop(0)
        else:
            sys.path.remove(entry)
    check_loaded_modules(site)
    if trt.__version__ != manifest["runtime_version"]:
        raise RuntimeError("TensorRT version differs from engine build")
    mapped = loaded_windows_libraries()
    if any(not Path(mapped[name]).is_relative_to(site) for name in ("nvinfer_10.dll", "nvonnxparser_10.dll")):
        raise RuntimeError("Actual TensorRT DLL mapping differs from selected runtime")
    return trt, mapped


class ComfyMemory:
    """Original Core manages its models; raw driver free memory admits TRT."""
    def __init__(self, device):
        import torch
        import comfy.model_management as mm
        self.torch, self.mm, self.device = torch, mm, torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("TRT requires an explicit CUDA device")

    def identity(self):
        import pynvml
        pynvml.nvmlInit()
        try:
            props = self.torch.cuda.get_device_properties(self.device)
            driver = pynvml.nvmlSystemGetDriverVersion()
            return {"gpu_uuid": "GPU-" + str(props.uuid).removeprefix("GPU-"),
                    "driver_version": driver.decode() if isinstance(driver, bytes) else str(driver),
                    "compute_capability": [props.major, props.minor],
                    "torch_cuda": self.torch.version.cuda}
        finally:
            pynvml.nvmlShutdown()

    def reserve(self, amount):
        positive_int(amount, "GPU admission reserve")
        with self.torch.cuda.device(self.device):
            # Two positional arguments are common to old/current Core. Do not
            # unload unrelated apps or register a fictitious PyTorch model size.
            self.mm.free_memory(amount, self.device)
            self.mm.soft_empty_cache()
            driver_free = int(self.torch.cuda.mem_get_info(self.device)[0])
            free = min(driver_free, self.physical_free())
            if free < amount <= driver_free:
                # WDDM's CUDA budget may exceed physically free dedicated VRAM.
                # Ask Comfy to offload the extra physical deficit, not other apps.
                self.mm.free_memory(driver_free + amount - free, self.device)
                self.mm.soft_empty_cache()
                free = self.free_now()
        if free < amount:
            raise RuntimeError(f"Insufficient actual GPU free memory after Comfy offload: {free} < {amount}")
        return free

    def check(self, minimum_free):
        self.mm.throw_exception_if_processing_interrupted()
        free = self.free_now()
        if free < minimum_free:
            raise RuntimeError("TRT VAE stopped: GPU safety margin exhausted")
        return free

    def execution_context(self):
        return self.torch.cuda.device(self.device)

    def free_now(self):
        return min(int(self.torch.cuda.mem_get_info(self.device)[0]), self.physical_free())

    def physical_free(self):
        import pynvml
        pynvml.nvmlInit()
        try:
            uuid = "GPU-" + str(self.torch.cuda.get_device_properties(self.device).uuid).removeprefix("GPU-")
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            return int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
        finally:
            pynvml.nvmlShutdown()

    def release(self):
        with self.torch.cuda.device(self.device):
            self.torch.cuda.synchronize(self.device)
            gc.collect()
            self.mm.soft_empty_cache()


@contextmanager
def runtime_serial_lease():
    from .dlss_fi_backend.resources import SerialProbeLease
    with ExitStack() as stack:
        for name in ("T8-DLSS-FI-serial.lock", "T8-TRT-VAE-serial.lock"):
            stack.enter_context(SerialProbeLease(Path(tempfile.gettempdir()) / name))
        yield


class ScopedBackend:
    """Callable lease factory for H3VAEInterface; explicit trusted local bundles.

    Initial profiles each have one fixed shape. Missing T1/small tiles fail at
    preflight, not by hidden padding or a mixed native/TRT output.
    """
    def __init__(self, bundles, runtime_site, *, device="cuda:0", check=lambda: None,
                 startup_reserve_bytes=9 * GIB, safety_margin_bytes=GIB,
                 memory=None, runtime_loader=load_runtime, runner_factory=None, serial_lease=runtime_serial_lease):
        positive_int(startup_reserve_bytes, "startup reserve")
        positive_int(safety_margin_bytes, "safety margin")
        if startup_reserve_bytes <= safety_margin_bytes:
            raise ValueError("Startup reserve must exceed safety margin")
        self.bundles, self.runtime_site, self.device = dict(bundles), runtime_site, device
        self.check, self.startup_reserve, self.margin = check, startup_reserve_bytes, safety_margin_bytes
        self.memory, self.runtime_loader, self.runner_factory = memory, runtime_loader, runner_factory
        self.serial_lease = serial_lease
        self.last_report = {"status": "not_loaded"}

    @contextmanager
    def __call__(self, kind, shapes):
        if kind not in self.bundles:
            raise ValueError(f"No explicit {kind} engine bundle selected")
        root, manifest = select_bundle(self.bundles[kind], kind, shapes)
        if not _ACTIVE.acquire(blocking=False):
            raise RuntimeError("Another TRT VAE lease is active")
        memory = self.memory
        runtime = engine = context = runner = payload = logger = None
        resources = ExitStack()
        report = {"status": "loading", "kind": kind, "engine_sha256": manifest["engine_sha256"],
                  "startup_admission_bytes": self.startup_reserve,
                  "memory_scope": "Admission policy plus actual driver/context queries; not file-size VRAM estimate"}
        started = time.perf_counter()
        try:
            resources.enter_context(self.serial_lease())
            import psutil
            if psutil.virtual_memory().available < manifest["engine_bytes"] + 2 * GIB:
                raise RuntimeError("Insufficient CPU RAM for engine deserialization buffer")
            self.check()
            memory = memory or ComfyMemory(self.device)
            identity = memory.identity()
            for key in ("gpu_uuid", "driver_version", "compute_capability", "torch_cuda"):
                if str(identity[key]).lower() != str(manifest[key]).lower():
                    raise RuntimeError(f"Engine device/runtime identity changed: {key}; rebuild explicitly")
            resources.enter_context(memory.execution_context())
            report["free_before_deserialize"] = memory.reserve(self.startup_reserve)
            trt, mapped = self.runtime_loader(manifest, self.runtime_site)
            report["loaded_libraries"] = mapped
            payload = read_payload(root, manifest)
            self.check()
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(payload)
            payload = None
            if engine is None:
                raise RuntimeError("Engine deserialization failed")
            context_bytes = int(engine.device_memory_size_v2)
            if context_bytes < 0:
                raise RuntimeError("Invalid context memory requirement")
            report["context_device_bytes"] = context_bytes
            # Extra 256MiB covers the fixed tile I/O plus temporary transfer.
            report["free_before_context"] = memory.reserve(context_bytes + self.margin + 256 * 1024**2)
            self.check()
            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError("Engine context creation failed")
            from .trt_vae_engine import TileEngineRunner
            profile = "t1" if manifest["source_request"]["schema"] == "t8-trt-encoder-t1-build-v1" else "default"
            if manifest["source_request"]["schema"] == "t8-trt-decoder-flex-build-v1":
                profile = "flex"
            runner = (self.runner_factory or TileEngineRunner)(engine, context, trt, device=self.device, kind=kind, profile=profile)
            report["load_seconds"] = time.perf_counter() - started
            report["free_after_load"] = memory.check(self.margin)

            def tile(value):
                if runner is None:
                    raise RuntimeError("TRT tile callable belongs to an expired lease")
                self.check()
                memory.check(self.margin)
                # CPU assembly keeps whole-frame canvases out of scarce VRAM.
                # No CUDA tensor tied to a tile survives the scoped lease.
                result = runner(value).cpu()
                self.check()
                memory.check(self.margin)
                return result

            yield tile
            self.check()
            report.update(status="complete", calls=runner.calls)
        except BaseException as error:
            report.update(status="failed", error=str(error))
            raise
        finally:
            try:
                if runner is not None:
                    runner.close()
            except BaseException as error:
                report.update(status="cleanup_failed", cleanup_error=str(error))
                raise
            finally:
                runner = context = engine = runtime = payload = logger = None
                try:
                    if memory is not None:
                        memory.release()
                        report["free_after_release"] = memory.free_now()
                except BaseException as error:
                    report.update(status="cleanup_failed", cleanup_error=str(error))
                    raise
                finally:
                    report["lease_seconds"] = time.perf_counter() - started
                    self.last_report = report
                    try:
                        resources.close()
                    finally:
                        _ACTIVE.release()
