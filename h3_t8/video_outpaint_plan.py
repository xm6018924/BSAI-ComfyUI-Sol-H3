"""Pure geometry and shot-local temporal contracts for the isolated outpaint route.

No model imports, registration, source crop, media I/O or GPU allocation.
"""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from typing import Mapping


SCHEMA = "t8.h3.video_outpaint.plan/v1"
ASPECTS = ("source", "9:16", "16:9", "1:1", "4:3", "3:4", "9:12", "custom")
WINDOW_FRAMES = (39, 56, 73, 90, 107, 124, 158, 192)


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value, label, minimum=0, maximum=1_000_000):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _finite(value, label, low, high):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be in [{low}, {high}]")
    return value


def _align_frames(count):
    return max(5, count) + (5 - max(5, count)) % 17


def _video_t(count):
    return (count - 5) // 17 * 5 + 2


def _sampling_canvas(width, height, megapixels):
    padded_w = math.ceil(width / 32) * 32
    padded_h = math.ceil(height / 32) * 32
    budget = math.floor(megapixels * 1_000_000) if megapixels else padded_w * padded_h
    if budget < 32 * 32:
        raise ValueError("sampling budget cannot fit one H3 spatial block")
    scale = min(1.0, math.sqrt(budget / (padded_w * padded_h)))
    model_w = max(32, math.floor(padded_w * scale / 32) * 32)
    model_h = max(32, math.floor(padded_h * scale / 32) * 32)
    while model_w * model_h > budget:
        if model_w / padded_w >= model_h / padded_h and model_w > 32:
            model_w -= 32
        elif model_h > 32:
            model_h -= 32
        elif model_w > 32:
            model_w -= 32
        else:
            raise ValueError("sampling budget cannot fit the requested canvas")
    # One continuous scale for both axes; unused model space is padding, not stretching.
    scale = min(model_w / padded_w, model_h / padded_h)
    offset_x = (model_w - padded_w * scale) / 2
    offset_y = (model_h - padded_h * scale) / 2
    return model_w, model_h, scale, offset_x, offset_y, budget


def _align_source_axis(model_extent, output_extent, source_start, source_extent, scale, preferred):
    """Use available model-canvas slack to align source edges to latent cells.

    Exact delivery geometry can be odd while H3 still needs a 16-pixel latent
    ownership boundary.  Moving the *sampling* canvas inside its padded extent
    avoids locking a partial generated strip as source context.  The delivered
    source rectangle remains unchanged and is pasted back exactly later.
    """
    slack = max(0.0, model_extent - output_extent * scale)

    def clamp(value):
        return min(slack, max(0.0, value))

    candidates = {clamp(preferred), 0.0, slack}
    for position in (source_start * scale, (source_start + source_extent) * scale):
        first = math.floor((position - slack) / 16) - 1
        last = math.ceil((position + slack) / 16) + 1
        for cell in range(first, last + 1):
            candidates.add(clamp(cell * 16 - position))

    def distance_to_cell(value):
        return abs(value - round(value / 16) * 16)

    def score(offset):
        start = offset + source_start * scale
        stop = offset + (source_start + source_extent) * scale
        return (
            max(distance_to_cell(start), distance_to_cell(stop)),
            distance_to_cell(start) + distance_to_cell(stop),
            abs(offset - preferred),
            offset,
        )

    return min(candidates, key=score)


def _shot_windows(start, stop, window_frames):
    count = stop - start
    aligned = _align_frames(count)
    window = min(aligned, window_frames)
    final = aligned - window
    starts = sorted({0, final, *range(0, final + 1, 34)})
    global_audio_t = round(aligned / 24 * 40)
    window_audio_t = round(window / 24 * 40)
    committed_frames = committed_video = committed_audio = 0
    windows = []
    for frame_start in starts:
        video_start = frame_start // 17 * 5
        audio_start = global_audio_t - window_audio_t if frame_start == final else round(frame_start / 24 * 40)
        video_stop = video_start + _video_t(window)
        audio_stop = audio_start + window_audio_t
        if video_start > committed_video or audio_start > committed_audio:
            raise ValueError("window schedule has a latent gap")
        windows.append({
            "source_start": start + frame_start,
            "render_frames": window,
            "source_frames": max(0, min(count - frame_start, window)),
            "video_start": video_start,
            "audio_start": audio_start,
            "context_video_latents": max(0, committed_video - video_start),
            "context_audio_latents": max(0, committed_audio - audio_start),
            "deliver_start": start + min(committed_frames, count),
            "deliver_stop": start + min(frame_start + window, count),
        })
        committed_frames = frame_start + window
        committed_video, committed_audio = video_stop, audio_stop
    return {"start": start, "stop": stop, "aligned_frames": aligned, "windows": windows}


def build_outpaint_plan(
    *, source_sha256: str, width: int, height: int, frame_count: int,
    source_fps: str = "24", aspect: str = "9:16",
    left: int = 0, top: int = 0, right: int = 0, bottom: int = 0,
    anchor_x: float = 0.5, anchor_y: float = 0.5,
    generation_megapixels: float = 0.5, window_frames: int = 73,
    cut_frames: tuple[int, ...] = (),
) -> dict:
    """Plan exact output pixels; custom margins are literal, never silently rounded.

    Source identity is supplied by the media inspector and must be rechecked by the executor.
    This planner currently accepts 24fps CFR only; no implicit frame-rate conversion.
    """
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in source_sha256):
        raise ValueError("source_sha256 must be a SHA-256 hex digest")
    width = _integer(width, "width", 1, 32768)
    height = _integer(height, "height", 1, 32768)
    frame_count = _integer(frame_count, "frame_count", 1)
    try:
        fps = Fraction(str(source_fps))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("source_fps must be a valid rate") from error
    if fps != 24:
        raise ValueError("current outpaint timeline requires explicit 24fps CFR input; no silent retiming")
    if aspect not in ASPECTS:
        raise ValueError("unsupported target aspect")
    margins = [_integer(v, n, 0, 32768) for v, n in zip((left, top, right, bottom), ("left", "top", "right", "bottom"))]
    anchor_x = _finite(anchor_x, "anchor_x", 0, 1)
    anchor_y = _finite(anchor_y, "anchor_y", 0, 1)
    megapixels = _finite(generation_megapixels, "generation_megapixels", 0, 64)
    _integer(window_frames, "window_frames", 1, 192)
    if window_frames not in WINDOW_FRAMES:
        raise ValueError("window_frames must be a supported 17k+5 overlapping window")
    if aspect == "custom":
        left, top, right, bottom = margins
        output_w, output_h = width + left + right, height + top + bottom
    else:
        if any(margins):
            raise ValueError("explicit margins require custom aspect; refusing to ignore supplied values")
        ratio = Fraction(width, height) if aspect == "source" else Fraction(aspect.replace(":", "/"))
        unit = max(math.ceil(width / ratio.numerator), math.ceil(height / ratio.denominator))
        output_w, output_h = ratio.numerator * unit, ratio.denominator * unit
        left = math.floor((output_w - width) * anchor_x)
        top = math.floor((output_h - height) * anchor_y)
        right, bottom = output_w - width - left, output_h - height - top
    if output_w * output_h > 64_000_000:
        raise ValueError("output exceeds the explicit 64MP planning limit")
    mw, mh, scale, ox, oy, budget = _sampling_canvas(output_w, output_h, megapixels)
    # The output rectangle is literal, but the padded H3 canvas may have a few
    # unused pixels.  Spend that slack on a latent-aligned source boundary.  For
    # example, a requested 736x607 output with a 95px top margin is sampled as a
    # 736x608 canvas with a +1px Y offset, so the source starts at model row 96
    # instead of contaminating the 80..95 generated latent band.
    ox = _align_source_axis(mw, output_w, left, width, scale, ox)
    oy = _align_source_axis(mh, output_h, top, height, scale, oy)
    rect = [left, top, left + width, top + height]
    model_rect = [ox + left * scale, oy + top * scale, ox + (left + width) * scale, oy + (top + height) * scale]
    lock_box = [max(0, math.floor(model_rect[0] / 16)), max(0, math.floor(model_rect[1] / 16)),
                min(mw // 16, math.ceil(model_rect[2] / 16)), min(mh // 16, math.ceil(model_rect[3] / 16))]
    cuts = [_integer(c, "cut frame", 1, frame_count - 1) for c in cut_frames]
    if cuts != sorted(set(cuts)):
        raise ValueError("cut_frames must be strictly increasing and unique")
    bounds = [0, *cuts, frame_count]
    plan = {
        "schema": SCHEMA,
        "request": {"source_sha256": source_sha256.lower(), "width": width, "height": height,
                    "frame_count": frame_count, "source_fps": "24/1", "aspect": aspect,
                    "left": margins[0], "top": margins[1], "right": margins[2], "bottom": margins[3],
                    "anchor_x": anchor_x, "anchor_y": anchor_y, "generation_megapixels": megapixels,
                    "window_frames": window_frames, "cut_frames": cuts},
        "source": {"sha256": source_sha256.lower(), "width": width, "height": height, "frames": frame_count, "fps": "24/1"},
        "output": {"width": output_w, "height": output_h, "frames": frame_count, "fps": "24/1",
                   "source_rect": rect, "margins": [left, top, right, bottom], "source_resized": False,
                   "source_cropped": False, "has_outpaint": any((left, top, right, bottom))},
        "sampling": {"width": mw, "height": mh, "pixels": mw * mh, "budget_pixels": budget,
                     "isotropic_scale": scale, "offset": [ox, oy], "source_rect": model_rect,
                     "source_lock_latent_box": lock_box, "window_frames": window_frames},
        "policies": {"padding_delivered": False, "audio": "preserve_source", "source_pixels": "exact_before_encoding",
                     "parallel_gpu_jobs": 1, "color_match": "expanded_region_only", "color_match_enabled": True,
                     "automatic_accept": False, "vram_safe": None},
        "shots": [_shot_windows(a, b, window_frames) for a, b in zip(bounds, bounds[1:])],
    }
    plan["plan_sha256"] = hashlib.sha256(canonical(plan).encode()).hexdigest()
    return plan


def validate_outpaint_plan(plan: Mapping) -> dict:
    """Reject stale/tampered plans before a downstream component consumes coordinates."""
    data = json.loads(canonical(dict(plan)))
    digest = data.pop("plan_sha256", None)
    if data.get("schema") != SCHEMA or digest != hashlib.sha256(canonical(data).encode()).hexdigest():
        raise ValueError("outpaint plan schema or integrity hash mismatch")
    data["plan_sha256"] = digest
    try:
        rebuilt = build_outpaint_plan(**data["request"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("outpaint plan contains invalid request parameters") from error
    if rebuilt != data:
        raise ValueError("outpaint plan coordinates do not match its request")
    return rebuilt
