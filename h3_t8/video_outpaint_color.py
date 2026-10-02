"""Shot-local, chunk-invariant boundary tint correction for outpaint delivery.

Estimate paired RGB offsets on the *same* inner source strips, not unrelated
generated scenery. Fade the correction into the expanded area only. This is an
optional finishing step, not a geometry repair or a perceptual quality guarantee.
"""
from __future__ import annotations

import hashlib

import torch

from .video_outpaint_composite import composite_outpaint_frames
from .video_outpaint_plan import canonical, _finite, _integer, validate_outpaint_plan


SCHEMA = "t8.h3.video_outpaint.boundary_color/v1"


def _digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _weights(plan, fade_pixels, device):
    output = plan["output"]
    x0, y0, x1, y1 = output["source_rect"]
    x = torch.arange(output["width"], device=device, dtype=torch.float32)[None, :]
    y = torch.arange(output["height"], device=device, dtype=torch.float32)[:, None]
    distances = (x0 - x, y0 - y, x - x1 + 1, y - y1 + 1)
    return torch.stack([
        torch.where(d > 0, (1 - (d - 1) / fade_pixels).clamp(0, 1), 0)
        .expand(output["height"], output["width"])
        for d in distances
    ], dim=-1)


def _offsets(source_rgb, candidate_rgb, rect, strip_pixels, limit):
    x0, y0, x1, y1 = rect
    difference = source_rgb - candidate_rgb[y0:y1, x0:x1]
    strips = (difference[:, :strip_pixels], difference[:strip_pixels],
              difference[:, -strip_pixels:], difference[-strip_pixels:])
    return torch.stack([strip.mean(dim=(0, 1)) for strip in strips]).clamp(-limit, limit)


def color_match_outpaint_frames(
    source, candidate, plan, *, start_frame=0, state=None, enabled=True,
    strength=1.0, strip_pixels=16, fade_pixels=128, temporal_alpha=0.25, max_offset=0.15,
):
    """Return composited frames, a JSON-safe continuation state and mechanical report.

    State is bound to source/geometry/settings and the next absolute frame. Start
    without state only at a shot boundary; never leak a previous shot's tint. RGB
    accepts normalized floats or uint8; all auxiliary channels are left untouched.
    Only one frame of correction workspace is created in addition to the chunk.
    """
    checked = validate_outpaint_plan(plan)
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be boolean")
    settings = {"enabled": enabled, "strength": _finite(strength, "strength", 0, 1),
                "strip_pixels": _integer(strip_pixels, "strip_pixels", 1, 256),
                "fade_pixels": _integer(fade_pixels, "fade_pixels", 1, 8192),
                "temporal_alpha": _finite(temporal_alpha, "temporal_alpha", 0.001, 1),
                "max_offset": _finite(max_offset, "max_offset", 0, 0.5)}
    result, preservation = composite_outpaint_frames(source, candidate, checked, start_frame=start_frame)
    if source.dtype != torch.uint8 and not source.is_floating_point():
        raise ValueError("boundary color accepts normalized floating RGB or uint8")
    if source.is_floating_point() and any(torch.any((rgb < 0) | (rgb > 1)) for rgb in (source[..., :3], candidate[..., :3])):
        raise ValueError("floating RGB must be normalized to [0,1]")
    binding = _digest({"plan_sha256": checked["plan_sha256"], "settings": settings})
    shot_starts = {shot["start"] for shot in checked["shots"]}
    previous = None
    if state is not None:
        if not isinstance(state, dict):
            raise ValueError("invalid boundary color state")
        data = dict(state)
        digest = data.pop("state_sha256", None)
        if (set(data) != {"schema", "binding", "next_frame", "offsets"}
                or digest != _digest(data) or data.get("schema") != SCHEMA
                or data.get("binding") != binding or data.get("next_frame") != start_frame):
            raise ValueError("boundary color state integrity/source/settings/sequence mismatch")
        previous = torch.tensor(data["offsets"], device=source.device, dtype=torch.float32)
        if previous.shape != (4, 3) or not torch.isfinite(previous).all() or torch.any(previous.abs() > settings["max_offset"] + 1e-6):
            raise ValueError("boundary color state has invalid offsets")
    elif start_frame not in shot_starts:
        raise ValueError("boundary color continuation requires state unless starting a new shot")
    active = enabled and settings["strength"] > 0 and checked["output"]["has_outpaint"]
    weights = _weights(checked, settings["fade_pixels"], source.device) if active else None
    resets = 0
    normalizer = 255.0 if source.dtype == torch.uint8 else 1.0
    for local in range(source.shape[0]):
        if start_frame + local in shot_starts:
            previous = None
            resets += 1
        if active:
            measured = _offsets(source[local, ..., :3].float() / normalizer,
                                candidate[local, ..., :3].float() / normalizer,
                                checked["output"]["source_rect"], settings["strip_pixels"], settings["max_offset"])
            previous = measured if previous is None else torch.lerp(previous, measured, settings["temporal_alpha"])
            correction = (weights @ previous) / weights.sum(dim=-1, keepdim=True).clamp_min(1)
            rgb = (candidate[local, ..., :3].float() / normalizer + correction * settings["strength"]).clamp(0, 1)
            converted = (rgb * 255).round().to(source.dtype) if source.dtype == torch.uint8 else rgb.to(source.dtype)
            # Even distant generated pixels with zero weight keep their original bits.
            changed = weights.sum(dim=-1, keepdim=True) > 0
            result[local, ..., :3] = torch.where(changed, converted, result[local, ..., :3])
        else:
            previous = torch.zeros((4, 3), device=source.device, dtype=torch.float32)
    continuation = {"schema": SCHEMA, "binding": binding,
                    "next_frame": start_frame + source.shape[0], "offsets": previous.tolist()}
    continuation["state_sha256"] = _digest(continuation)
    x0, y0, x1, y1 = checked["output"]["source_rect"]
    report = {**preservation, "color_match_applied": bool(active), "settings": settings,
              "source_exact_before_encoding": torch.equal(result[:, y0:y1, x0:x1], source),
              "state_sha256": continuation["state_sha256"], "shot_resets": resets,
              "algorithm": "paired_inner_strip_rgb_offset_ema_outer_fade",
              "perceptual_acceptance": False}
    return result, continuation, report
