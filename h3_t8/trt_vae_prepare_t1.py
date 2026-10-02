"""Explicit single-frame export in an owned, guarded, serial GPU worker."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, write_new_json, ENCODER_T1_MODEL_SHA  # noqa: E402
from dlss_fi_backend.process import run_isolated, IsolatedTaskError  # noqa: E402
from dlss_fi_backend.resources import GuardPolicy, NvmlResourceReader, ResourceGuard, SerialProbeLease  # noqa: E402


def publish_graph(source,destination):
    """New output only. Atomic on the supported Windows filesystem."""
    import os
    import uuid
    if digest_file(source) != ENCODER_T1_MODEL_SHA:
        raise ValueError("Unpinned single-frame graph")
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        raise ValueError("Graph destination appeared; preserving it")
    staging = destination.with_name("."+destination.name+"."+uuid.uuid4().hex+".partial")
    with Path(source).open("rb") as src, staging.open("xb") as out:
        shutil.copyfileobj(src,out)
        out.flush()
        os.fsync(out.fileno())
    if digest_file(staging) != ENCODER_T1_MODEL_SHA:
        raise ValueError("Graph copy hash differs; private partial retained")
    # Windows rename fails, rather than replacing an existing destination.
    if os.name != "nt":
        raise ValueError("Atomic no-replace publication is only implemented for Windows")
    staging.rename(destination)


def main(argv=None, check_interrupt=lambda:None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-vae", "core-directory", "output", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    check_interrupt()
    if sys.platform != "win32":
        raise ValueError("This guarded export currently supports Windows only")
    core = args.core_directory.resolve(strict=True)
    destination = args.output.resolve()
    if not args.output.is_absolute() or destination.suffix != ".onnx":
        raise ValueError("Use an explicitly named absolute .onnx destination")
    if destination.exists():
        if digest_file(destination) != ENCODER_T1_MODEL_SHA:
            raise ValueError("Existing graph differs; not overwriting it")
        return {"status":"existing_pinned_t1_graph_reused_no_export","output":str(destination)}
    root = args.run_dir.resolve()
    if root.exists() or not args.run_dir.is_absolute() or root == root.parent or shutil.disk_usage(core).free < 3 * 1024**3:
        raise ValueError("Need new absolute run directory and3GiB disk")
    if digest_file(args.native_vae) != "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522":
        raise ValueError("Unexpected native VAE identity")
    worker = Path(__file__).with_name("trt_vae_prepare_t1_worker.py")
    paths = [Path(__file__), worker, args.native_vae,
             PROJECT / "trt_vae_build.py", PROJECT / "dlss_fi_backend/process.py", PROJECT / "dlss_fi_backend/resources.py"]
    paths += [core / name for name in ("comfy/ldm/minimax/vae.py", "comfy/ops.py", "comfy/model_management.py")]
    sources = {str(path.resolve()): digest_file(path) for path in paths}
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root / "source-snapshot"
    snapshot.mkdir()
    for path, sha in sources.items():
        if Path(path).suffix == ".py":
            shutil.copyfile(path, snapshot / (sha + "-" + Path(path).name))
    terminal = {"status": "not_started"}
    try:
        with ExitStack() as stack:
            for path in (Path(tempfile.gettempdir()) / "T8-DLSS-FI-serial.lock", Path(tempfile.gettempdir()) / "T8-TRT-VAE-serial.lock"):
                stack.enter_context(SerialProbeLease(path))
            reader = stack.enter_context(NvmlResourceReader())
            guard = ResourceGuard(GuardPolicy(minimum_free_gpu_bytes=1024**3, minimum_free_ram_bytes=8*1024**3))
            first = reader.sample()
            reason = guard.observe(first, startup=True)
            write_new_json(root / "preflight.json", {"snapshot": first, "reason": reason})
            if reason:
                raise RuntimeError(reason)
            request = {"native_vae": str(args.native_vae.resolve()), "core_directory":str(core),
                       "sources": sources, "gpu_uuid": first["gpu_uuid"]}
            write_new_json(root / "request.json", request)
            if not args.execute:
                terminal = {"status": "preflight_only"}
                return
            cancel = threading.Event()
            last = time.perf_counter()
            with (root / "resources.jsonl").open("x", encoding="utf8") as log:
                log.write(json.dumps(first) + "\n")
                def check():
                    nonlocal last
                    check_interrupt()
                    if (root / "cancel.request").exists():
                        cancel.set()
                        raise InterruptedError("T1 export cancelled")
                    if time.perf_counter() - last < .25:
                        return
                    row = reader.sample()
                    last = time.perf_counter()
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if error := guard.observe(row):
                        raise RuntimeError(error)
                try:
                    receipt = run_isolated(worker, ["--request", str(root / "request.json")], timeout=600, cancel=cancel, check=check)
                    terminal = {"status": "export_worker_complete_not_trt_qualified", "isolated": receipt}
                    check_interrupt()
                    graph = root/"encoder-t1.onnx"
                    if digest_file(graph) != ENCODER_T1_MODEL_SHA:
                        raise ValueError("Exported graph differs from pinned T1. Keep evidence; do not compile or substitute.")
                    for path,digest in sources.items():
                        if digest_file(path) != digest:
                            raise ValueError("Source changed during export")
                    publish_graph(graph,destination)
                    terminal.update(status="pinned_t1_export_published_not_runtime_qualified",output=str(destination),
                                    graph_sha256=ENCODER_T1_MODEL_SHA)
                finally:
                    write_new_json(root / "resource-summary.json", guard.report())
    except IsolatedTaskError as error:
        terminal = {"status": "worker_failed", "isolated": error.receipt}
        raise
    except BaseException as error:
        terminal = {"status": "controller_failed", "error": str(error)}
        raise
    finally:
        write_new_json(root / "terminal.json", terminal)
    print(json.dumps(terminal, indent=2))
    return terminal


if __name__ == "__main__":
    main()
