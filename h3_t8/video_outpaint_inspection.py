"""Bounded, cancellable subprocess boundary around unchanged media validators."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import psutil


SCHEMA = "t8.outpaint.isolated_media_contract/v1"
MAX_JSON_BYTES = 8 * 1024**2
MAX_LOG_BYTES = 1024**2
WORKER_PATH = Path(__file__).with_name("video_outpaint_inspection_worker.py")


def isolated_media_contract(action, path, *, parameters=None, interrupt_check=None, timeout=1800):
    if action not in {"inspect", "validate_final"} or timeout <= 0:
        raise ValueError("invalid isolated media operation or timeout")
    import comfy.cli_args
    core_directory = Path(comfy.cli_args.__file__).resolve().parents[1]
    request = {"schema": SCHEMA, "action": action, "path": str(Path(path).resolve(strict=True)),
               "core_directory": str(core_directory),
               "parameters": parameters or {}}
    blob = json.dumps(request, sort_keys=True).encode()
    if len(blob) > MAX_JSON_BYTES:
        raise ValueError("media request metadata exceeds protocol bound")
    expected = hashlib.sha256(blob).hexdigest()
    process = None
    children = {}
    with tempfile.TemporaryDirectory(prefix="t8-outpaint-inspect-") as directory:
        request_path = Path(directory) / "request.json"
        request_path.write_bytes(blob)
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            try:
                if interrupt_check:
                    interrupt_check()
                process = subprocess.Popen([sys.executable, "-I", str(WORKER_PATH), str(request_path)],
                    stdin=subprocess.DEVNULL, stdout=output, stderr=errors, shell=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
                deadline = time.monotonic() + timeout
                while process.poll() is None:
                    if interrupt_check:
                        interrupt_check()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("isolated outpaint media inspection timed out")
                    if os.fstat(output.fileno()).st_size > MAX_JSON_BYTES or os.fstat(errors.fileno()).st_size > MAX_LOG_BYTES:
                        raise RuntimeError("isolated media worker exceeded response/log metadata bounds")
                    try:
                        for child in psutil.Process(process.pid).children(recursive=True):
                            children[(child.pid, child.create_time())] = child
                    except psutil.NoSuchProcess:
                        pass
                    try:
                        process.wait(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        pass
                errors.seek(0)
                detail = errors.read(MAX_LOG_BYTES + 1).decode("utf-8", errors="replace")
                if process.returncode:
                    raise RuntimeError(f"isolated outpaint media worker exited {process.returncode}: {detail[-4000:]}")
                output.seek(0)
                response_blob = output.read(MAX_JSON_BYTES + 1)
                if len(response_blob) > MAX_JSON_BYTES or len(detail.encode()) > MAX_LOG_BYTES:
                    raise RuntimeError("isolated media worker exceeded response/log metadata bounds")
                try:
                    response = json.loads(response_blob)
                except (UnicodeDecodeError, ValueError) as error:
                    raise RuntimeError(f"isolated media worker returned invalid JSON: {detail[-2000:]}") from error
                if not isinstance(response, dict) or response.get("schema") != SCHEMA or response.get("request_sha256") != expected:
                    raise RuntimeError("isolated media response identity mismatch")
                if response.get("ok") is not True:
                    exception = {"ValueError": ValueError, "FileNotFoundError": FileNotFoundError,
                                 "TimeoutError": TimeoutError}.get(response.get("error_type"), RuntimeError)
                    raise exception("isolated outpaint media inspection: " + str(response.get("error", "unknown error"))
                                    + ("\n" + detail[-4000:] if detail else ""))
                if not isinstance(response.get("result"), dict):
                    raise RuntimeError("isolated media response has no result object")
                return response["result"]
            finally:
                from .video_outpaint_compose import _stop_owned
                _stop_owned(process)
                # Retain child identities seen while the worker was alive, also
                # handling an abrupt worker exit after spawning FFprobe/FFmpeg.
                for child in children.values():
                    try:
                        if child.is_running():
                            child.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(list(children.values()), timeout=5)
