"""Bounded global-latent decode, exact source paste-back and isolated file encode."""
from __future__ import annotations

from contextlib import ExitStack
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import torch

from .video_outpaint_finish import finish_outpaint_frames, geometry_settings
from .video_outpaint_decode import iter_decode_outpaint_shot
from .video_outpaint_delivery import delivery_report_path
from .video_outpaint_media import finalize_outpaint_file, validate_outpaint_source, _sha256_file
from .video_outpaint_plan import canonical, _integer
from .video_outpaint_prepare import sampling_to_output_canvas, SequentialOutpaintFrameReader
from .video_outpaint_source_runtime import loaded_video_vae_identity, outpaint_gpu_lease
from .video_outpaint_rgb_capture import EncoderRGBCapture
from .video_outpaint_pixel_receipt import validate_source_mode


OUTPAINT_VIDEO_ENCODER_POLICY = (
    "ffmpeg_rawvideo_pipe_libx264_all_intra_decoder_safe_threads1_v1"
)
OUTPAINT_X264_PARAMS = (
    "threads=1:lookahead_threads=1:sliced_threads=0:ref=1:bframes=0:"
    "keyint=1:min-keyint=1:scenecut=0:cabac=0"
)


def _outpaint_encoder_output_args(*, pixel_format, crf, frame_count):
    """Return a decoder-stable H.264 contract for even and odd canvases."""
    if pixel_format not in {"yuv420p", "yuv444p"}:
        raise ValueError("unsupported outpaint encoder pixel format")
    _integer(crf, "crf", 0, 51)
    _integer(frame_count, "frame_count", 1)
    # Baseline is the broadly playable contract for normal even yuv420p output.
    # H.264 Baseline cannot represent yuv444p, so exact odd geometry uses High
    # 4:4:4 Predictive while retaining the same all-intra decoder-safe structure.
    profile = "baseline" if pixel_format == "yuv420p" else "high444"
    return [
        "-frames:v", str(frame_count), "-c:v", "libx264", "-threads", "1",
        "-x264-params", OUTPAINT_X264_PARAMS, "-profile:v", profile,
        "-preset", "medium", "-crf", str(crf), "-pix_fmt", pixel_format,
    ], profile


def _stop_owned(process):
    if process is None:
        return
    # PATH may resolve to a launcher (e.g. Chocolatey), so EOF/reap must cover its
    # real encoder child too. Never search/kill all ffmpeg processes on the host.
    import psutil
    descendants = []
    try:
        owner = psutil.Process(process.pid)
        descendants = owner.children(recursive=True)
    except psutil.NoSuchProcess:
        owner = None
    if process.stdin is not None and not process.stdin.closed:
        try:
            process.stdin.close()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if owner is not None:
            try:
                descendants.extend(owner.children(recursive=True))
            except psutil.NoSuchProcess:
                pass
        for child in descendants:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.kill()
        process.wait(timeout=5)
    _, alive = psutil.wait_procs(descendants, timeout=1)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=5)


def compose_sampled_outpaint(
    *, vae, inspection, source_store, window_store, output_path,
    color_match=True, color_settings=None, crf=18, interrupt_check=None, progress=None,
    capture_rgb_path=None, expected_first_frame_sha256=None,
    geometry_align=False, alignment_band_pixels=64, alignment_max_displacement=8.0,
    source_mode="preserve_source",
):
    """Compose a new file; never invokes the H3 diffusion model or changes old videos.

    Even dimensions use broadly playable H.264/yuv420p. Odd exact dimensions use
    H.264/yuv444p, explicitly reported as requiring player compatibility review;
    no silent crop, resize or extra border is added to make an encoder happy.
    """
    def checkpoint():
        if interrupt_check:
            interrupt_check()

    _integer(crf, "crf", 0, 51)
    if expected_first_frame_sha256 is not None and (
        not isinstance(expected_first_frame_sha256, str) or len(expected_first_frame_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_first_frame_sha256)
    ):
        raise ValueError("expected first-frame SHA256 must be a lowercase hex digest")
    if not isinstance(color_match, bool):
        raise ValueError("color_match must be boolean")
    settings = dict(color_settings or {})
    mode = validate_source_mode(source_mode)
    preserve_source = mode == "preserve_source"
    geometry = geometry_settings(geometry_align, alignment_band_pixels, alignment_max_displacement)
    if not set(settings) <= {"strength", "strip_pixels", "fade_pixels", "temporal_alpha", "max_offset"}:
        raise ValueError("unsupported outpaint color setting")
    plan = window_store.plan
    source, checked = validate_outpaint_source(inspection, plan)
    if source_store.plan != plan or source_store.position() is not None:
        raise ValueError("composition needs the complete matching source cache")
    if (hashlib.sha256(source_store.path.read_bytes()).hexdigest() != window_store.identity["source_cache_sha256"]
            or window_store.snapshot()["status"] != "sampled"):
        raise ValueError("composition requires matching completed sampled windows")
    target = Path(output_path).resolve()
    if target.exists() or delivery_report_path(target).exists() or target == source:
        raise FileExistsError("composition output must be a new file, never the source")
    if target.suffix.lower() != ".mp4":
        raise ValueError("outpaint composition currently requires MP4")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for outpaint composition")
    output = checked["output"]
    width, height = output["width"], output["height"]
    pixel_format = "yuv444p" if width % 2 or height % 2 else "yuv420p"
    encoder_args, encoder_profile = _outpaint_encoder_output_args(
        pixel_format=pixel_format, crf=crf, frame_count=checked["source"]["frames"]
    )
    first_time = Fraction(inspection["cfr"]["first_pts"]) * Fraction(inspection["cfr"]["time_base"])
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.stem}.video-only-", suffix=".mp4", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    process = None
    try:
        with outpaint_gpu_lease(), tempfile.TemporaryFile(mode="w+b") as diagnostics, ExitStack() as stack:
            checkpoint()
            identity = loaded_video_vae_identity(vae, interrupt_check=interrupt_check)
            from .video_outpaint_source_runtime import source_vae_identity_matches
            from .patch_stack_policy import model_identity_matches
            if not source_vae_identity_matches(source_store, identity):
                raise ValueError("loaded decode VAE differs from the source preparation VAE")
            window_sha = hashlib.sha256(window_store.path.read_bytes()).hexdigest()
            command = [ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
                       "-video_size", f"{width}x{height}", "-framerate", "24", "-i", "pipe:0", "-an",
                       *encoder_args,
                       "-vf", "scale=in_range=full:out_range=limited:out_color_matrix=bt709", "-color_range", "tv",
                       "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                       "-output_ts_offset", f"{float(first_time):.12f}", "-movflags", "+faststart", "-f", "mp4",
                       str(temporary)]
            capture = None
            if capture_rgb_path is not None:
                capture = stack.enter_context(EncoderRGBCapture(capture_rgb_path, width=width, height=height,
                    frames=checked["source"]["frames"], provenance={"plan_sha256": checked["plan_sha256"],
                        "source_sha256": inspection["sha256"], "window_manifest_sha256": window_sha,
                        "video_vae_sha256": identity["sha256"], "encoder_command": command,
                        "color_match": color_match, "color_settings": settings, "geometry_settings": geometry,
                        "source_mode": mode}))
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=diagnostics,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), shell=False, bufsize=0)
            source_hash, pasted_hash = hashlib.sha256(), hashlib.sha256()
            color_state = None
            color_report = None
            delivered = 0
            source_abs_error = source_changed_values = source_values = 0
            rect = output["source_rect"]
            with SequentialOutpaintFrameReader(source, checked, interrupt_check=interrupt_check) as reader:
                for shot_index in range(len(checked["shots"])):
                    for start, pixels, _ in iter_decode_outpaint_shot(
                        vae, lambda a, b: window_store.read_video_range(shot_index, a, b), checked,
                        shot_index=shot_index, interrupt_check=interrupt_check,
                    ):
                        checkpoint()
                        if start != delivered:
                            raise ValueError("decode delivery has a gap or duplicate frame")
                        original = reader(start, start+pixels.shape[0])
                        candidate = sampling_to_output_canvas(pixels, checked)
                        composed, color_state, preservation = finish_outpaint_frames(
                            original.float()/255, candidate, checked, start_frame=start,
                            state=color_state, enabled=color_match, source_mode=mode, **settings, **geometry,
                        )
                        if preservation["source_exact_before_encoding"] is not preserve_source:
                            raise ValueError("compositor did not preserve the original source pixels")
                        if color_report is None:
                            color_report = {
                                "algorithm": preservation["algorithm"],
                                "color_match_applied": preservation["color_match_applied"],
                                "settings": preservation["settings"],
                                "chunks": 0,
                                "shot_resets": 0,
                                "source_exact_before_encoding": preserve_source,
                                "source_reconstructed": not preserve_source,
                                "geometry_status_counts": {},
                            }
                        elif (
                            color_report["algorithm"] != preservation["algorithm"]
                            or color_report["color_match_applied"] != preservation["color_match_applied"]
                            or color_report["settings"] != preservation["settings"]
                        ):
                            raise ValueError("boundary color behavior changed during composition")
                        color_report["chunks"] += 1
                        color_report["shot_resets"] += preservation["shot_resets"]
                        color_report["state_sha256"] = preservation["state_sha256"]
                        alignment = preservation.get("geometry_alignment", {})
                        for frame in alignment.get("frames", []):
                            status = frame["status"]
                            counts = color_report["geometry_status_counts"]
                            counts[status] = counts.get(status, 0) + 1
                        for local in range(composed.shape[0]):
                            checkpoint()
                            encoded_rgb = (composed[local].clamp(0, 1)*255).round().to(torch.uint8)
                            patch = encoded_rgb[rect[1]:rect[3], rect[0]:rect[2]].contiguous()
                            if preserve_source and not torch.equal(patch, original[local]):
                                raise ValueError("RGB8 conversion changed the source region before encoding")
                            difference = (patch.to(torch.int16)-original[local].to(torch.int16)).abs()
                            source_abs_error += int(difference.sum())
                            source_changed_values += int(torch.count_nonzero(difference))
                            source_values += difference.numel()
                            source_hash.update(original[local].contiguous().numpy().tobytes())
                            pasted_hash.update(patch.numpy().tobytes())
                            payload = memoryview(encoded_rgb.contiguous().numpy().tobytes())
                            if delivered == 0 and expected_first_frame_sha256 is not None:
                                if hashlib.sha256(payload).hexdigest() != expected_first_frame_sha256:
                                    raise ValueError("composed first-frame RGB differs from the selected candidate preview")
                            if capture is not None:
                                capture.write_frame(payload)
                            while payload:
                                checkpoint()
                                written = process.stdin.write(payload)
                                if not written:
                                    raise BrokenPipeError("outpaint encoder stopped accepting frame bytes")
                                payload = payload[written:]
                            delivered += 1
                        if progress:
                            progress({"stage": "decode_compose_encode", "frames_written": delivered,
                                      "total_frames": checked["source"]["frames"]})
            if delivered != checked["source"]["frames"]:
                raise ValueError("composition frame count differs from the original source")
            if color_report is None or color_report["shot_resets"] != len(checked["shots"]):
                raise ValueError("boundary color state did not reset exactly once per shot")
            process.stdin.close()
            deadline = time.monotonic()+300
            while process.poll() is None:
                checkpoint()
                if time.monotonic() >= deadline:
                    raise TimeoutError("outpaint FFmpeg encode did not finish")
                time.sleep(0.05)
            diagnostics.seek(0)
            error = diagnostics.read(4000).decode("utf-8", errors="replace")
            if process.returncode or error.strip():
                raise RuntimeError("isolated outpaint encode failed: " + error)
            if capture is not None:
                capture.record["candidate_sha256_at_encoder_exit"] = _sha256_file(temporary)
            validate_outpaint_source(inspection, checked)
            if not model_identity_matches(identity, loaded_video_vae_identity(vae)):
                raise ValueError("decode VAE changed while composing the output")
            if hashlib.sha256(window_store.path.read_bytes()).hexdigest() != window_sha:
                raise ValueError("sampling checkpoints changed while composing the output")
            candidate_sha = _sha256_file(temporary)
            if capture is not None:
                capture.record["candidate_sha256_before_finalizer"] = candidate_sha
            receipt = {"schema": "t8.h3.video_outpaint.pre_encode_pixels/v1", "plan_sha256": checked["plan_sha256"],
                       "source_sha256": inspection["sha256"], "candidate_sha256": candidate_sha,
                       "window_manifest_sha256": window_sha, "frame_count": delivered, "width": width, "height": height,
                       "source_rgb_sha256": source_hash.hexdigest(), "pasted_rgb_sha256": pasted_hash.hexdigest(),
                       "source_exact_before_encoding": preserve_source, "lossy_encoded_equality_claimed": False}
            if not preserve_source:
                receipt.update(schema="t8.h3.video_outpaint.reconstructed_pixels/v1",
                    source_mode=mode, source_reconstructed=True,
                    reconstructed_source_rgb_sha256=receipt.pop("pasted_rgb_sha256"))
            receipt["receipt_sha256"] = hashlib.sha256(canonical(receipt).encode()).hexdigest()
            video, report = finalize_outpaint_file(temporary, inspection, checked, target,
                interrupt_check=interrupt_check, pixel_receipt=receipt, source_mode=mode, delivery_metadata={
                    "color_match_enabled": color_match if preserve_source else False,
                    "color_match_requested": color_match, "color_settings": settings, "encoder_crf": crf,
                    "source_postprocessing_bypassed": not preserve_source,
                    "source_region_rgb8_mae": source_abs_error/source_values,
                    "source_region_changed_value_fraction": source_changed_values/source_values,
                    "geometry_settings": geometry,
                    "color_match_report": color_report,
                    "encoder_pixel_format": pixel_format, "odd_size_player_review_required": pixel_format == "yuv444p",
                    "encoder_profile": encoder_profile, "encoder_x264_params": OUTPAINT_X264_PARAMS,
                    "encoder_policy": OUTPAINT_VIDEO_ENCODER_POLICY,
                    "vae_decode_max_tokens": 7, "full_rgb_video_materialized": False,
                    "plan": checked, "execution_identity": window_store.identity,
                    "source_cache_identity": source_store.identity,
                    **({"selected_first_frame_rgb8_sha256": expected_first_frame_sha256,
                        "selected_first_frame_verified_before_encoding": True}
                       if expected_first_frame_sha256 is not None else {}),
                    "composition_implementation_sha256": {name: _sha256_file(Path(__file__).with_name(name))
                        for name in ("video_outpaint_compose.py", "video_outpaint_media.py", "video_outpaint_packet_mux.py", "video_outpaint_delivery.py",
                                     "video_outpaint_color.py", "video_outpaint_finish.py", "video_outpaint_alignment.py", "video_outpaint_pixel_receipt.py",
                                     "video_outpaint_decode.py", "video_outpaint_rgb_capture.py",
                                     "video_outpaint_inspection.py", "video_outpaint_inspection_worker.py")}})
            return video, report
    finally:
        _stop_owned(process)
        temporary.unlink(missing_ok=True)
