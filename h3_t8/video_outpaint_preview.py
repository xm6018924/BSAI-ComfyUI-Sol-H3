"""Bounded source-frame / geometry preview, never a generated outpaint candidate.

The model/output coordinates come from the verified Plan, not another aspect
calculator. FFmpeg decodes one selected frame in its own CPU process; no VAE,
diffusion weights or complete IMAGE batch are loaded by this component.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
import time

from .video_outpaint_plan import validate_outpaint_plan


def preview_geometry(plan, frame_index=0, max_edge=768):
    checked = validate_outpaint_plan(plan)
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or not 0 <= frame_index < checked["source"]["frames"]:
        raise ValueError("preview frame_index must be an existing zero-based source frame")
    if isinstance(max_edge, bool) or not isinstance(max_edge, int) or not 64 <= max_edge <= 1536:
        raise ValueError("preview max_edge must be an integer from 64 to 1536; this only limits the thumbnail")
    out = checked["output"]
    scale = min(Fraction(1), Fraction(max_edge, max(out["width"], out["height"])))
    width, height = max(1, math.ceil(out["width"] * scale)), max(1, math.ceil(out["height"] * scale))
    x0, y0, x1, y1 = out["source_rect"]
    rect = [math.floor(x0 * scale), math.floor(y0 * scale), math.ceil(x1 * scale), math.ceil(y1 * scale)]
    shot_index = next(i for i, shot in enumerate(checked["shots"]) if shot["start"] <= frame_index < shot["stop"])
    sampling = checked["sampling"]
    return {
        "schema": "t8.h3.video_outpaint.geometry_preview/v1",
        "scope": "source_frame_and_geometry_only_not_generated_or_accepted",
        "plan_sha256": checked["plan_sha256"], "source_sha256": checked["source"]["sha256"],
        "frame_index": frame_index, "time_seconds": str(Fraction(frame_index, 24)), "shot_index": shot_index,
        "output_width": out["width"], "output_height": out["height"], "source_rect_output": list(out["source_rect"]),
        "margins_left_top_right_bottom": list(out["margins"]),
        "sampling_width": sampling["width"], "sampling_height": sampling["height"],
        "sampling_megapixels": sampling["pixels"] / 1_000_000,
        "thumbnail_width": width, "thumbnail_height": height, "thumbnail_scale": str(scale),
        "source_rect_thumbnail": rect,
        "thumbnail_rounding": "canvas and source bounds round outward by less than one display pixel per edge",
        "display_only_resizing": True, "source_modified": False, "model_called": False,
        "full_image_batch_materialized": False, "generated_reference_selected": False,
        "source_thumbnail_subpixel": (x1 - x0) * scale < 1 or (y1 - y0) * scale < 1,
    }


def _decode_thumbnail(path, frame_index, width, height, *, interrupt_check=None, timeout=120):
    from .video_outpaint_compose import _stop_owned

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is needed for a source-frame geometry preview")
    command = [ffmpeg, "-v", "error", "-nostdin", "-noautorotate", "-threads", "1", "-i", str(path),
        "-map", "0:v:0", "-vf", f"select=eq(n\\,{frame_index}),scale={width}:{height}:flags=lanczos",
        "-frames:v", "1", "-fps_mode", "passthrough", "-an", "-sn", "-dn", "-threads", "1",
        "-filter_threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    expected = width * height * 3
    process = None
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            if interrupt_check:
                interrupt_check()
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if interrupt_check:
                    interrupt_check()
                if os.fstat(stdout.fileno()).st_size > expected:
                    raise RuntimeError("preview decoder exceeded the single-thumbnail byte count")
                if os.fstat(stderr.fileno()).st_size > 1024 * 1024:
                    raise RuntimeError("preview decoder error log exceeded 1MiB")
                if time.monotonic() > deadline:
                    raise TimeoutError("source-frame geometry preview timed out")
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            stderr.seek(0)
            error = stderr.read(4000).decode("utf-8", errors="replace")
            if process.returncode or error.strip():
                raise RuntimeError(f"source-frame preview decoder failed ({process.returncode}): {error}")
            if os.fstat(stdout.fileno()).st_size != expected:
                raise RuntimeError("preview decoder did not return exactly one selected RGB thumbnail")
            stdout.seek(0)
            return stdout.read(expected)
        finally:
            _stop_owned(process)


def _draw_preview(rgb, geometry):
    from PIL import Image, ImageDraw

    width, height = geometry["thumbnail_width"], geometry["thumbnail_height"]
    rect = geometry["source_rect_thumbnail"]
    tw, th = rect[2] - rect[0], rect[3] - rect[1]
    if len(rgb) != tw * th * 3:
        raise ValueError("thumbnail bytes do not match the mapped source bounds")
    # The footer is outside the actual canvas; labels never cover the source.
    canvas = Image.new("RGB", (max(480, width), height + 104), (17, 23, 33))
    draw = ImageDraw.Draw(canvas)
    for y in range(0, height, 16):
        for x in range(0, width, 16):
            color = (37, 70, 94) if (x // 16 + y // 16) % 2 else (51, 91, 119)
            draw.rectangle((x, y, min(width, x + 16) - 1, min(height, y + 16) - 1), fill=color)
    thumbnail = Image.frombytes("RGB", (tw, th), rgb)
    canvas.paste(thumbnail, (rect[0], rect[1]))
    lines = [
        "GEOMETRY PREVIEW - NOT AI-GENERATED",
        "PHOTO = original frame | BLUE CHECKERS = area to generate",
        f"Output {geometry['output_width']}x{geometry['output_height']} | frame {geometry['frame_index']} (zero based)",
        f"Model {geometry['sampling_width']}x{geometry['sampling_height']} = {geometry['sampling_megapixels']:.6f} MP",
        f"Margins L/T/R/B: {geometry['margins_left_top_right_bottom']} | shot {geometry['shot_index'] + 1}",
        "Display thumbnail only. Source video and audio are unchanged.",
    ]
    for i, line in enumerate(lines):
        draw.text((6, height + 5 + i * 16), line, fill=(220, 232, 242))
    return canvas


def render_source_geometry_preview(inspection, plan, frame_index=0, max_edge=768, *, interrupt_check=None):
    from .video_outpaint_media import validate_outpaint_source

    geometry = preview_geometry(plan, frame_index, max_edge)
    if interrupt_check:
        interrupt_check()
    path, checked = validate_outpaint_source(inspection, plan)
    rect = geometry["source_rect_thumbnail"]
    try:
        rgb = _decode_thumbnail(path, frame_index, rect[2] - rect[0], rect[3] - rect[1], interrupt_check=interrupt_check)
        if interrupt_check:
            interrupt_check()
        image = _draw_preview(rgb, geometry)
    finally:
        # Also check after a failed or cancelled read. No preview/candidate is
        # returned for source bytes changed since Plan; no cache is modified.
        validate_outpaint_source(inspection, checked)
    report = {**geometry, "preview_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
              "preview_image_width": image.width, "preview_image_height": image.height,
              "decoder": "isolated CPU FFmpeg, sequential frame index selection, one RGB thumbnail",
              "preview_is_pixel_preservation_evidence": False}
    return image, report
