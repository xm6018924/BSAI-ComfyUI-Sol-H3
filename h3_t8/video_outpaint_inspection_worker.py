"""CPU-only child for existing read-only media contracts; never imports node registration."""
from __future__ import annotations

import contextlib
from fractions import Fraction
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback
import types


SCHEMA = "t8.outpaint.isolated_media_contract/v1"
MAX_JSON_BYTES = 8 * 1024**2


def dispatch(request):
    root = Path(__file__).resolve().parent
    # The trusted parent supplies its actual Core location. Do not infer it from
    # checkout depth: isolated worktrees and extracted packages may live anywhere.
    core_root = Path(request["core_directory"]).resolve(strict=True)
    if not (core_root / "comfy/cli_args.py").is_file():
        raise ValueError("invalid ComfyUI Core directory")
    sys.path.insert(0, str(core_root))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    name = "t8_outpaint_media_worker"
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    original = importlib.import_module(name + ".dlss_nr_advanced")
    path = Path(request["path"]).resolve(strict=True)
    if request["action"] == "inspect":
        from comfy_api.input_impl import VideoFromFile
        checked_path, info = original._file_source_contract(VideoFromFile(str(path)))
        return {"path": str(checked_path), "info": {**info, "rate": str(info["rate"]), "time_base": str(info["time_base"])}}
    if request["action"] == "validate_final":
        kwargs = dict(request["parameters"])
        kwargs["rate"] = Fraction(kwargs["rate"])
        return original._validate_final_file(path, **kwargs)
    raise ValueError("unknown isolated media operation")


def main(request_path):
    path = Path(request_path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("media request metadata exceeds protocol bound")
    blob = path.read_bytes()
    identity = hashlib.sha256(blob).hexdigest()
    try:
        request = json.loads(blob)
        if request.get("schema") != SCHEMA:
            raise ValueError("invalid media request schema")
        # Import warnings and third-party progress never mix with JSON protocol.
        with contextlib.redirect_stdout(sys.stderr):
            result = dispatch(request)
        response = {"schema": SCHEMA, "request_sha256": identity, "ok": True, "result": result}
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        response = {"schema": SCHEMA, "request_sha256": identity, "ok": False,
                    "error_type": type(error).__name__, "error": str(error)}
    print(json.dumps(response), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: inspection_worker.py REQUEST_JSON")
    main(sys.argv[1])
