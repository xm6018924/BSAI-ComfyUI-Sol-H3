"""Immutable, content-addressed candidate images and explicit selection receipts.

No pickle, model weights or arbitrary file paths are serialized. Reload receives
the independently prepared source providers and checks them against the saved
execution. It never chooses the newest candidate and never invokes a sampler.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .long_video_delivery import _atomic_write_bytes, _manifest_lock
from .video_outpaint_candidates import _receipt, _validate_candidate, select_candidate
from .video_outpaint_media import validate_outpaint_source
from .video_outpaint_plan import canonical
from .video_outpaint_source_runtime import outpaint_gpu_lease
from .video_outpaint_window_store import OutpaintWindowStore
from .video_outpaint_pixel_receipt import validate_source_mode


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("archive id must be a lowercase SHA256, not a path")
    return value


def _read(path, limit):
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("candidate archive record exceeds its size limit")
    return raw


def _read_record(root, kind, digest):
    raw = _read(root / f"{kind}-{_digest(digest)}.json", 1024*1024)
    record = json.loads(raw)
    if not isinstance(record, dict) or record.get("sha256") != digest:
        raise ValueError("candidate archive identity mismatch")
    body = {k: v for k, v in record.items() if k != "sha256"}
    if record != _receipt(body):
        raise ValueError("candidate archive integrity mismatch")
    return record


def _publish(root, filename, raw):
    path = root / filename
    with _manifest_lock(root):
        if path.exists():
            if path.is_symlink() or _read(path, len(raw)) != raw:
                raise ValueError("existing candidate archive asset differs; refusing overwrite")
        else:
            _atomic_write_bytes(path, raw)


def validate_archived_preview(handle):
    report = handle["preview_report"]
    body = {k: v for k, v in report.items() if k != "sha256"}
    if (report != _receipt(body) or report.get("schema") != "t8.h3.video_outpaint.candidate_first_frame/v1"
            or report.get("candidate_sha256") != handle["candidate"]["sha256"]
            or report.get("plan_sha256") != handle["prepared"]["plan"]["plan_sha256"]
            or report.get("source_sha256") != handle["prepared"]["inspection"]["sha256"]
            or report.get("video_vae_sha256") != handle["prepared"]["source"].identity["video_vae_sha256"]):
        raise ValueError("candidate preview identity or integrity mismatch")
    mode = validate_source_mode(report.get("source_mode", "preserve_source"))
    if (report.get("source_exact_before_encoding") is not (mode == "preserve_source")
            or report.get("source_reconstructed", False) is not (mode == "joint_decode")):
        raise ValueError("candidate preview source mode contradicts its pixel evidence")
    _validate_candidate(handle["candidate"], handle["windows"])
    return report


def _verify_prepared(prepared, store):
    validate_outpaint_source(prepared["inspection"], prepared["plan"])
    if (prepared["plan"] != store.plan or prepared["source"].plan != store.plan
            or prepared["source"].position() is not None
            or hashlib.sha256(prepared["source"].path.read_bytes()).hexdigest() != store.identity["source_cache_sha256"]
            or prepared["conditioning"].verify() != store.identity["conditioning_sha256"]
            or prepared["audio"].verify() != store.identity["audio_source_sha256"]):
        raise ValueError("saved candidate does not match the prepared source/text/audio")


def save_candidate_archive(handle, image, *, interrupt_check=None):
    with outpaint_gpu_lease():
        if interrupt_check:
            interrupt_check()
        report = validate_archived_preview(handle)
        _verify_prepared(handle["prepared"], handle["windows"])
        output = handle["prepared"]["plan"]["output"]
        expected = (1, output["height"], output["width"], 3)
        if (not isinstance(image, torch.Tensor) or tuple(image.shape) != expected
                or image.device.type != "cpu" or not image.is_floating_point()
                or not torch.isfinite(image).all() or torch.any((image < 0) | (image > 1))):
            raise ValueError("candidate archive requires its one normalized CPU RGB image")
        rgb = (image * 255).round().to(torch.uint8)[0].contiguous().numpy()
        if hashlib.sha256(rgb.tobytes()).hexdigest() != report["rgb8_sha256"]:
            raise ValueError("candidate archive image differs from its preview")
        buffer = BytesIO()
        Image.fromarray(rgb).save(buffer, format="PNG")
        png = buffer.getvalue()
        image_sha = hashlib.sha256(png).hexdigest()
        record = _receipt(json.loads(canonical({"schema": "t8.h3.video_outpaint.candidate_archive/v1",
            "candidate": handle["candidate"], "preview_report": report, "settings": handle["settings"],
            "png_sha256": image_sha})))
        _validate_settings(record["settings"], handle["windows"].identity)
        root = handle["windows"].root
        if interrupt_check:
            interrupt_check()
        _publish(root, f"preview-{image_sha}.png", png)
        _publish(root, f"candidate-{record['sha256']}.json", canonical(record).encode())
        return record["sha256"]


def _validate_settings(settings, identity):
    if (not isinstance(settings, dict) or set(settings) != {"seed", "steps", "noise_algorithm"}
            or any(settings[key] != identity[key] for key in settings)
            or any(type(settings[key]) is not int for key in ("seed", "steps"))):
        raise ValueError("candidate archive settings differ from the execution identity")


def _load_candidate(prepared, root, archive_id):
    root = Path(root).resolve()
    record = _read_record(root, "candidate", archive_id)
    if (set(record) != {"schema", "candidate", "preview_report", "settings", "png_sha256", "sha256"}
            or record["schema"] != "t8.h3.video_outpaint.candidate_archive/v1"):
        raise ValueError("unsupported candidate archive record")
    if not (root / "outpaint_windows.json").is_file():
        raise FileNotFoundError("saved candidate sampling manifest is missing; no replacement will be created")
    identity = record["candidate"]["identity"]
    store = OutpaintWindowStore(root, prepared["plan"], execution_identity={
        k: v for k, v in identity.items() if k not in {"plan_sha256", "implementation_sha256"}})
    if store.identity != identity:
        raise ValueError("saved candidate implementation identity mismatch")
    _verify_prepared(prepared, store)
    _validate_settings(record["settings"], identity)
    handle = {"prepared": prepared, "windows": store, "candidate": record["candidate"],
              "preview_report": record["preview_report"], "settings": record["settings"],
              "cache_root": root, "archive_id": archive_id}
    report = validate_archived_preview(handle)
    output = store.plan["output"]
    width, height = output["width"], output["height"]
    if (report.get("width"), report.get("height")) != (width, height):
        raise ValueError("saved preview geometry differs from its plan")
    image_sha = _digest(record["png_sha256"])
    png = _read(root / f"preview-{image_sha}.png", width*height*4 + 1024*1024)
    if hashlib.sha256(png).hexdigest() != image_sha:
        raise ValueError("saved candidate PNG hash mismatch")
    with Image.open(BytesIO(png)) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (width, height):
            raise ValueError("saved candidate PNG format or geometry mismatch")
        rgb = np.array(image)
    if hashlib.sha256(rgb.tobytes()).hexdigest() != report["rgb8_sha256"]:
        raise ValueError("saved candidate RGB hash mismatch")
    return handle, torch.from_numpy(rgb).float().unsqueeze(0)/255


def load_candidate_archive(prepared, root, archive_id, *, interrupt_check=None):
    with outpaint_gpu_lease():
        if interrupt_check:
            interrupt_check()
        result = _load_candidate(prepared, root, archive_id)
        if interrupt_check:
            interrupt_check()
        return result


def save_selection_archive(selected, *, interrupt_check=None):
    with outpaint_gpu_lease():
        if interrupt_check:
            interrupt_check()
        handle, _ = _load_candidate(selected["prepared"], selected["cache_root"], selected["archive_id"])
        if (selected["selection"] != select_candidate(handle["candidate"], handle["windows"])
                or selected["preview_report"] != handle["preview_report"]
                or selected["settings"] != handle["settings"]):
            raise ValueError("selection no longer matches the saved candidate preview")
        record = _receipt({"schema": "t8.h3.video_outpaint.selection_archive/v1",
            "candidate_archive_id": selected["archive_id"], "selection": selected["selection"]})
        if interrupt_check:
            interrupt_check()
        _publish(handle["windows"].root, f"selection-{record['sha256']}.json", canonical(record).encode())
        return record["sha256"]


def load_selection_archive(prepared, root, selection_id, *, interrupt_check=None):
    with outpaint_gpu_lease():
        if interrupt_check:
            interrupt_check()
        record = _read_record(Path(root).resolve(), "selection", selection_id)
        if (set(record) != {"schema", "candidate_archive_id", "selection", "sha256"}
                or record["schema"] != "t8.h3.video_outpaint.selection_archive/v1"):
            raise ValueError("unsupported saved selection")
        handle, image = _load_candidate(prepared, root, record["candidate_archive_id"])
        if record["selection"] != select_candidate(handle["candidate"], handle["windows"]):
            raise ValueError("saved selection disagrees with its actual candidate")
        if interrupt_check:
            interrupt_check()
        return {**handle, "selection": record["selection"], "selection_id": selection_id}, image
