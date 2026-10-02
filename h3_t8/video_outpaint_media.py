"""Strict file identity and source-audio publication for isolated outpainting.

Reuses existing read-only media validators, not the DLSS runtime or its binaries.
No source image batch is materialized; the inherited CFR validator retains only
O(frame count) timestamp metadata. RGB generation/encoding belongs to the caller.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from .dlss_nr_advanced import _sha256_file
from .skin_finish_p1 import _file_video_source_path, _strict_validate_encoded_video
from .video_outpaint_plan import canonical, validate_outpaint_plan
from .video_outpaint_delivery import delivery_report_path, publish_with_delivery_report
from .video_outpaint_inspection import isolated_media_contract


SCHEMA = "t8.h3.video_outpaint.source_file/v1"
BOUNDED_MULTITHREAD_DECODE_THREADS = 4
BOUNDED_MULTITHREAD_DECODE_ATTEMPTS = 3


def _strict_validate_bounded_multithread_video(
    path, *, attempts=BOUNDED_MULTITHREAD_DECODE_ATTEMPTS,
    threads=BOUNDED_MULTITHREAD_DECODE_THREADS,
):
    """Catch streams that pass one-thread decode but fail a bounded parallel decoder."""
    attempts = int(attempts)
    threads = int(threads)
    if attempts < 1 or threads < 2:
        raise ValueError("bounded multithread decode needs positive attempts and at least two threads")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for strict outpaint video validation")
    for attempt in range(1, attempts + 1):
        checked = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", str(threads),
             "-xerror", "-err_detect", "explode", "-nostdin",
             "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300, check=False,
        )
        diagnostics = (checked.stderr or "").strip()
        if checked.returncode or diagnostics:
            raise RuntimeError(
                f"bounded {threads}-thread H.264 decode failed on attempt {attempt}/{attempts}:\n"
                + diagnostics[-4000:]
            )
    return {"bounded_multithread_decode_threads": threads,
            "bounded_multithread_decode_attempts": attempts,
            "bounded_multithread_decode_passed": True}


def _file_source_contract(source_video, *, interrupt_check=None):
    path = _file_video_source_path(source_video)
    response = isolated_media_contract("inspect", path, interrupt_check=interrupt_check)
    checked_path = Path(response["path"])
    if checked_path != path:
        raise ValueError("isolated source inspection returned a different path")
    info = dict(response["info"])
    info["rate"] = Fraction(info["rate"])
    info["time_base"] = Fraction(info["time_base"])
    return checked_path, info


def _validate_final_file(path, *, interrupt_check=None, **parameters):
    parameters["rate"] = str(parameters["rate"])
    return isolated_media_contract("validate_final", path, parameters=parameters, interrupt_check=interrupt_check)


def _packet_copy(candidate, source, temporary, interrupt_check=None):
    # Same libav demux semantics as validation. FFmpeg7 rounded a final AAC
    # packet from171 to192 samples where installed PyAV/libav62 preserves171.
    # Native mux is isolated so a library crash cannot take down ComfyUI.
    from .video_outpaint_compose import _stop_owned
    worker = Path(__file__).with_name("video_outpaint_packet_mux.py")
    process = None
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.Popen([sys.executable, "-I", str(worker), str(candidate), str(source), str(temporary)],
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.monotonic()+300
            while process.poll() is None:
                if interrupt_check:
                    interrupt_check()
                if time.monotonic() > deadline:
                    raise TimeoutError("isolated outpaint packet mux timed out")
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            stdout.seek(0)
            stderr.seek(0)
            error = stderr.read(4000).decode("utf-8", errors="replace")
            if process.returncode or error.strip():
                raise RuntimeError(f"isolated outpaint mux failed ({process.returncode}): {error}")
            return json.loads(stdout.read(16384))
        finally:
            _stop_owned(process)


def _seal(data):
    return {**data, "inspection_sha256": hashlib.sha256(canonical(data).encode()).hexdigest()}


def inspect_outpaint_source(source_video, *, interrupt_check=None):
    """Decode-check a file-backed, untrimmed SDR CFR source and bind actual bytes."""
    path = _file_video_source_path(source_video)
    initial_sha = _sha256_file(path)
    if interrupt_check is None:
        from comfy.model_management import throw_exception_if_processing_interrupted
        interrupt_check = throw_exception_if_processing_interrupted
    checked_path, info = _file_source_contract(source_video, interrupt_check=interrupt_check)
    if info["rate"] != 24:
        raise ValueError("current outpaint file route requires 24fps CFR; no implicit retiming")
    if path != checked_path or initial_sha != _sha256_file(path):
        raise ValueError("source file changed during inspection")
    return _seal({
        "schema": SCHEMA, "path": str(path), "sha256": initial_sha, "bytes": path.stat().st_size,
        "width": info["width"], "height": info["height"], "frames": info["frame_count"], "fps": "24/1",
        "cfr": info["cfr"], "sdr": info["sdr"], "audio_packets": info["audio_packets"],
        "audio_pcm": info["audio_pcm"], "full_image_batch_materialized": False,
        "scope": "input_file_contract_only_not_generation_acceptance",
    })


def validate_outpaint_source(inspection, plan):
    data = dict(inspection)
    digest = data.pop("inspection_sha256", None)
    if data.get("schema") != SCHEMA or _seal(data)["inspection_sha256"] != digest:
        raise ValueError("source inspection integrity mismatch")
    checked = validate_outpaint_plan(plan)
    if any(data.get(key) != value for key, value in checked["source"].items()):
        raise ValueError("source inspection does not match the outpaint plan")
    path = Path(data["path"]).resolve(strict=True)
    if not path.is_file() or path.stat().st_size != data["bytes"] or _sha256_file(path) != data["sha256"]:
        raise ValueError("source file bytes changed since inspection")
    return path, checked


def _retain_validation_failure(target, candidate, muxed, *, stage, error, plan, candidate_sha_at_entry=None):
    """Keep failed media separate from deliverable names for actual diagnosis.

    Hard links retain the exact already-created private files without duplicating
    their data. If that filesystem cannot link them, report the retention failure;
    never hide the original validation error or publish a corrupt final file.
    """
    try:
        root = Path(tempfile.mkdtemp(prefix=".outpaint-failure-", dir=target.parent))
        files = {}
        for label, path in (("video_only", candidate), ("muxed", muxed)):
            if path is None or not path.is_file():
                continue
            kept = root / f"{label}.mp4"
            os.link(path, kept)
            files[label] = {"file": kept.name, "bytes": kept.stat().st_size, "sha256": _sha256_file(kept)}
        report = {"schema": "t8.h3.video_outpaint.validation_failure/v1", "status": "failed_not_deliverable",
            "stage": stage, "error": str(error), "plan_sha256": plan["plan_sha256"],
            "source_sha256": plan["source"]["sha256"], "files": files, "final_video_published": False}
        report["candidate_sha256_before_media_inspection"] = candidate_sha_at_entry
        (root / "diagnostic.json").write_text(canonical(report), encoding="utf-8")
        return str(root)
    except OSError as retention_error:
        return f"retention failed: {retention_error}"


def finalize_outpaint_file(video_only, source_inspection, plan, output_path, *, interrupt_check=None, pixel_receipt=None,
                          delivery_metadata=None, source_mode="preserve_source"):
    """Validate video, copy the original audio packets, then publish without overwrite.

    Does not claim source-pixel equality after lossy encoding. The compositor
    supplies its pre-encode receipt; a persistent report is linked before video.
    """
    from comfy_api.input_impl import VideoFromFile
    from .video_outpaint_pixel_receipt import validate_pixel_receipt, validate_source_mode

    mode = validate_source_mode(source_mode)
    if mode == "joint_decode" and pixel_receipt is None:
        raise ValueError("joint decode requires an explicit reconstructed-source pixel receipt")

    def checkpoint():
        if interrupt_check is not None:
            interrupt_check()

    checkpoint()
    source_path, checked = validate_outpaint_source(source_inspection, plan)
    candidate = Path(video_only).resolve(strict=True)
    target = Path(output_path).resolve()
    if target == source_path or target == candidate or target.exists() or delivery_report_path(target).exists():
        raise FileExistsError("outpaint output must be a new path, never source or candidate")
    if target.suffix.lower() != ".mp4":
        raise ValueError("current packet-copy finalizer requires an MP4 output")
    initial_candidate_sha = _sha256_file(candidate)
    if pixel_receipt is not None:
        validate_pixel_receipt(pixel_receipt, plan=checked, source_sha256=source_inspection["sha256"],
                               candidate_sha256=initial_candidate_sha, source_mode=mode)
    _, info = _file_source_contract(VideoFromFile(str(candidate)), interrupt_check=checkpoint)
    if _sha256_file(candidate) != initial_candidate_sha:
        target.parent.mkdir(parents=True, exist_ok=True)
        diagnostic = _retain_validation_failure(target, candidate, None, stage="candidate_changed_during_media_inspection",
            error="encoded candidate bytes changed during read-only media inspection", plan=checked,
            candidate_sha_at_entry=initial_candidate_sha)
        raise ValueError(f"encoded candidate changed during media inspection; Diagnostic: {diagnostic}")
    output = checked["output"]
    if (info["width"], info["height"], info["frame_count"], info["rate"]) != (
        output["width"], output["height"], output["frames"], Fraction(output["fps"])
    ):
        raise ValueError("encoded outpaint geometry/frame count/rate differs from plan")
    if info["audio_packets"]:
        raise ValueError("outpaint candidate must be video-only; generated audio is not deliverable")
    source_cfr = source_inspection["cfr"]
    first_time = Fraction(info["cfr"]["first_pts"]) * Fraction(info["cfr"]["time_base"])
    source_time = Fraction(source_cfr["first_pts"]) * Fraction(source_cfr["time_base"])
    if first_time != source_time:
        raise ValueError("encoded outpaint shifts the original video start timestamp")
    checkpoint()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        # A decodable PyAV CFR inspection alone is not the same as strict H.264
        # validation. Test the encoder result separately before adding audio so
        # a later failure can be attributed to the correct media boundary.
        _strict_validate_encoded_video(candidate)
        _strict_validate_bounded_multithread_video(candidate)
    except RuntimeError as error:
        diagnostic = _retain_validation_failure(target, candidate, None, stage="encoded_video_before_mux",
                                                error=error, plan=checked, candidate_sha_at_entry=initial_candidate_sha)
        raise RuntimeError(f"Outpaint encoded video failed strict validation: {error}\nDiagnostic: {diagnostic}") from error
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.stem}.outpaint-", suffix=".mp4", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        mux_report = _packet_copy(candidate, source_path, temporary, interrupt_check=checkpoint)
        try:
            evidence = _validate_final_file(
                temporary, frame_count=output["frames"], width=output["width"], height=output["height"],
                rate=Fraction(output["fps"]), source_audio_packets=source_inspection["audio_packets"],
                source_audio_pcm=source_inspection["audio_pcm"],
                interrupt_check=checkpoint,
            )
            evidence = {**evidence, **_strict_validate_bounded_multithread_video(temporary)}
        except (RuntimeError, ValueError) as error:
            diagnostic = _retain_validation_failure(target, candidate, temporary, stage="muxed_video_and_audio",
                                                    error=error, plan=checked, candidate_sha_at_entry=initial_candidate_sha)
            raise RuntimeError(f"Outpaint muxed media failed strict validation: {error}\nDiagnostic: {diagnostic}") from error
        final_cfr = evidence["cfr"]
        if Fraction(final_cfr["first_pts"]) * Fraction(final_cfr["time_base"]) != source_time:
            raise RuntimeError("final mux shifted the source video timestamp")
        validate_outpaint_source(source_inspection, checked)
        if _sha256_file(candidate) != initial_candidate_sha:
            raise ValueError("encoded candidate changed during finalization")
        checkpoint()
        final_sha = _sha256_file(temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        report = {
            "schema": "t8.h3.video_outpaint.final_file/v1", "path": str(target), "sha256": final_sha,
            "plan_sha256": checked["plan_sha256"], "source_sha256": source_inspection["sha256"],
            "candidate_sha256": initial_candidate_sha, "media": evidence, "atomic_no_replace_publish": True,
            "audio_regenerated": False, "padding_delivered": False, "perceptual_acceptance": False,
            "pre_encode_pixel_evidence_verified": pixel_receipt is not None,
            "pixel_receipt_sha256": pixel_receipt.get("receipt_sha256") if pixel_receipt is not None else None,
            "pixel_receipt": pixel_receipt,
            "source_mode": mode,
            "source_reconstructed": mode == "joint_decode",
            "source_exact_before_encoding": bool(pixel_receipt is not None and mode == "preserve_source"),
            "packet_mux": mux_report,
        }
        metadata = dict(delivery_metadata or {})
        if set(metadata) & set(report):
            raise ValueError("composition metadata cannot replace final validation evidence")
        report = publish_with_delivery_report(temporary, target, {**report, **metadata}, interrupt_check=checkpoint)
    finally:
        temporary.unlink(missing_ok=True)
    return VideoFromFile(str(target)), report
