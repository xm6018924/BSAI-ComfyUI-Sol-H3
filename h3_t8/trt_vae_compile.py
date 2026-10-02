"""Explicit serial guarded Windows TRT build. Default: decoder, preflight only."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import uuid

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from dlss_fi_backend.process import IsolatedTaskError, run_isolated  # noqa: E402
from dlss_fi_backend.resources import GuardPolicy, NvmlResourceReader, ResourceGuard, SerialProbeLease  # noqa: E402
from trt_vae_build import (  # noqa: E402
    INPUT_SHAPE, MODEL_SHA, TRT_VERSION, WEIGHTS_SHA, ENCODER_INPUT_SHAPE, ENCODER_MODEL_SHA, digest_file, publish_bundle,
    validate_request, verify_sources, write_new_json, ENCODER_T1_INPUT_SHAPE, ENCODER_T1_MODEL_SHA, DECODER_FLEX_MODEL_SHA,
    DECODER_W4_MODEL_SHA,
)


def execute_owned(request, run_dir, destination, *, check, cancel=None, timeout=1800, worker=None):
    """Testable controller stage. Caller holds leases and has checked resources."""
    run_dir, destination = Path(run_dir), Path(destination)
    worker = Path(worker) if worker is not None else Path(__file__).with_name("trt_vae_compile_worker.py")
    validate_request(request)
    cancel = cancel if cancel is not None else threading.Event()
    receipt = {"status": "not_started", "controller_pid": os.getpid()}
    try:
        check()
        isolated = run_isolated(worker, ["--request", str(run_dir / "request.json")],
                                timeout=timeout, cancel=cancel, check=check)
        receipt["isolated"] = isolated
        check()
        if cancel.is_set():
            raise InterruptedError("Build cancelled before publication")
        # Source scans can take seconds; no worker/GPU remains active here.
        verify_sources(request)
        if cancel.is_set() or (run_dir / "cancel.request").exists():
            raise InterruptedError("Build cancelled before publication")
        manifest = publish_bundle(request["staging_path"], destination, request)
        receipt.update(status="compiled_not_execution_qualified", bundle=str(destination),
                       engine_sha256=manifest["engine_sha256"])
    except IsolatedTaskError as error:
        receipt.update(status="worker_failed", isolated=error.receipt)
        raise
    except BaseException as error:
        receipt.update(status="controller_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_new_json(run_dir / "terminal.json", receipt)
    return receipt


def main(argv=None, check_interrupt=lambda: None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-site", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--kind", choices=("decoder", "decoder-flex", "decoder-w4a16", "encoder", "encoder-t1"), default="decoder")
    parser.add_argument("--encoder-norm-precision", choices=("default", "fp32_norm_affine"), default="default")
    args = parser.parse_args(argv)
    check_interrupt()
    if not 1 <= args.timeout <= 3600:
        raise ValueError("Build deadline must be 1..3600 seconds")
    run_dir = args.run_dir.resolve()
    if not args.run_dir.is_absolute() or run_dir.exists() or run_dir == run_dir.parent:
        raise ValueError("Use a new explicitly addressed absolute build-log directory")
    runtime = args.runtime_site.resolve(strict=True)
    model = args.onnx.resolve(strict=True)
    encoder = args.kind in ("encoder", "encoder-t1")
    single_image = args.kind == "encoder-t1"
    flex = args.kind == "decoder-flex"
    w4 = args.kind == "decoder-w4a16"
    if args.kind != "encoder" and args.encoder_norm_precision != "default":
        raise ValueError("This precision policy is only audited for the T17 encoder")
    weights = None if encoder else model.with_suffix(".onnx.data")
    model_sha = ENCODER_T1_MODEL_SHA if single_image else (ENCODER_MODEL_SHA if encoder else MODEL_SHA)
    input_shape = ENCODER_T1_INPUT_SHAPE if single_image else (ENCODER_INPUT_SHAPE if encoder else INPUT_SHAPE)
    schema = "t8-trt-encoder-t1-build-v1" if single_image else ("t8-trt-encoder-build-v1" if encoder else "t8-trt-build-v1")
    if flex:
        model_sha, schema = DECODER_FLEX_MODEL_SHA, "t8-trt-decoder-flex-build-v1"
        weights = model.with_name("minimax_h3_vae_decoder.onnx.data")
    if w4:
        model_sha, schema, weights = DECODER_W4_MODEL_SHA, "t8-trt-decoder-w4a16-build-v1", None
    if digest_file(model) != model_sha or (weights is not None and digest_file(weights) != WEIGHTS_SHA):
        raise ValueError("Model sources differ from the verified fixed assets")
    runtime_sources = {str(path.resolve()): digest_file(path) for path in runtime.rglob("*")
                       if path.is_file() and path.suffix in (".py", ".pyd", ".dll")}
    if not runtime_sources:
        raise ValueError("Empty runtime")
    worker_sources = {str(path.resolve()): digest_file(path) for path in (
        Path(__file__), Path(__file__).with_name("trt_vae_compile_worker.py"), PROJECT / "trt_vae_build.py",
        PROJECT / "dlss_fi_backend/process.py", PROJECT / "dlss_fi_backend/resources.py")}
    engines = model.parent / "engines"
    if shutil.disk_usage(model.parent).free < 20 * 1024**3:
        raise RuntimeError("At least 20GiB free disk required for engine build and evidence")
    run_dir.mkdir(parents=True, exist_ok=False)
    policy = GuardPolicy(startup_free_gpu_bytes=12000 * 1024**2,
                         startup_free_ram_bytes=24 * 1024**3,
                         minimum_free_gpu_bytes=1024 * 1024**2,
                         minimum_free_ram_bytes=8 * 1024**3)
    try:
        with ExitStack() as stack:
            # Same H3 research lease plus public FI lease, then TRT cross-copy lease.
            for path in (Path(tempfile.gettempdir()) / "T8-DLSS-FI-serial.lock",
                         Path(tempfile.gettempdir()) / "T8-TRT-VAE-serial.lock"):
                stack.enter_context(SerialProbeLease(path))
            reader = stack.enter_context(NvmlResourceReader())
            guard = ResourceGuard(policy)
            first = reader.sample()
            reason = guard.observe(first, startup=True)
            preflight = {"status": "ready" if reason is None else "not_ready", "reason": reason,
                         "policy": asdict(policy), "resource_snapshot": first, "execute_requested": args.execute}
            write_new_json(run_dir / "preflight.json", preflight)
            if reason:
                raise RuntimeError(reason)
            if not args.execute:
                print(json.dumps(preflight, indent=2))
                return preflight
            engines.mkdir(exist_ok=True)
            unique = uuid.uuid4().hex
            staging = engines / f".building-{unique}"
            staging.mkdir(exist_ok=False)
            prefix = "encoder-t1-fp16" if single_image else ("encoder-fp16" if encoder else "fp16")
            if flex:
                prefix = "decoder-flex-fp16"
            if w4:
                prefix = "decoder-w4a16"
            destination = engines / f"{prefix}-{unique}"
            request = {"schema": schema, "precision": "w4a16" if w4 else "fp16", "runtime_version": TRT_VERSION,
                       "runtime_site": str(runtime), "runtime_sources": runtime_sources,
                       "worker_sources": worker_sources, "model_path": str(model), "weights_path": str(weights) if weights else None,
                       "model_sha256": model_sha, "weights_sha256": None if weights is None else WEIGHTS_SHA,
                       "gpu_uuid": first["gpu_uuid"], "device_index": 0, "input_shape": list(input_shape),
                       "driver_version": str(reader.nvml.nvmlSystemGetDriverVersion()),
                       "workspace_bytes": 4 * 1024**3, "staging_path": str(staging)}
            if encoder:
                request["kind"] = "encoder"
                request["encoder_norm_precision"] = args.encoder_norm_precision
            elif flex or w4:
                request["kind"] = "decoder"
            validate_request(request)
            write_new_json(run_dir / "request.json", request)
            snapshot = run_dir / "source-snapshot"
            snapshot.mkdir()
            for path, digest in worker_sources.items():
                target = snapshot / (digest + "-" + Path(path).name)
                shutil.copyfile(path, target)
                if digest_file(target) != digest:
                    raise RuntimeError("Build source snapshot identity mismatch")
            write_new_json(run_dir / "controller.json", {"pid": os.getpid(), "destination": str(destination),
                                                        "cancel_file": str(run_dir / "cancel.request")})
            cancel = threading.Event()
            last = time.perf_counter()
            with (run_dir / "resources.jsonl").open("x", encoding="utf8") as resource_log:
                resource_log.write(json.dumps(first) + "\n")

                def check():
                    nonlocal last
                    check_interrupt()
                    if (run_dir / "cancel.request").exists():
                        cancel.set()
                        raise InterruptedError("User requested build cancellation")
                    now = time.perf_counter()
                    if now - last < .25:
                        return
                    row = reader.sample()
                    reason = guard.observe(row)
                    resource_log.write(json.dumps(row) + "\n")
                    resource_log.flush()
                    last = now
                    if reason:
                        raise RuntimeError(reason)

                try:
                    result = execute_owned(request, run_dir, destination, check=check,
                                           cancel=cancel, timeout=args.timeout)
                finally:
                    write_new_json(run_dir / "resource-summary.json", guard.report())
            print(json.dumps(result, indent=2))
    except BaseException as error:
        if not (run_dir / "terminal.json").exists():
            write_new_json(run_dir / "terminal.json", {"status": "preflight_or_launch_failed",
                           "controller_pid": os.getpid(), "error": f"{type(error).__name__}: {error}"})
        raise
    return result


if __name__ == "__main__":
    main()
