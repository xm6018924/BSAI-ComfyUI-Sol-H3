"""Pixel ownership for outpainting; no sampling, color transform or audio generation."""
from __future__ import annotations

import torch

from .video_outpaint_plan import validate_outpaint_plan


def composite_outpaint_frames(source: torch.Tensor, candidate: torch.Tensor, plan: dict, *, start_frame: int = 0):
    """Return one bounded chunk with exact source RGB/auxiliary channels restored.

    Timestamps/audio are handled by the file finalizer, not inferred from tensor length.
    Neither input tensor is mutated. A candidate must already be mapped to output space.
    """
    checked = validate_outpaint_plan(plan)
    metadata = checked["source"]
    output = checked["output"]
    if isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 0:
        raise ValueError("start_frame must be a nonnegative integer")
    if not isinstance(source, torch.Tensor) or source.ndim != 4 or source.shape[-1] < 3:
        raise ValueError("source must be IMAGE [N,H,W,C>=3]")
    count, height, width, channels = source.shape
    if count < 1 or start_frame + count > metadata["frames"]:
        raise ValueError("source chunk is empty or exceeds the original frame range")
    if (width, height) != (metadata["width"], metadata["height"]):
        raise ValueError("source chunk geometry differs from the source-bound plan")
    if not isinstance(candidate, torch.Tensor) or tuple(candidate.shape) != (count, output["height"], output["width"], channels):
        raise ValueError("candidate must match the chunk count, output geometry and source channels")
    if source.dtype != candidate.dtype or source.device != candidate.device:
        raise ValueError("source/candidate dtype and device must match without implicit conversion")
    if not torch.isfinite(source).all() or not torch.isfinite(candidate).all():
        raise ValueError("source/candidate contain non-finite pixels")
    x0, y0, x1, y1 = output["source_rect"]
    result = candidate.clone()
    result[:, y0:y1, x0:x1, :] = source
    return result, {
        "schema": "t8.h3.video_outpaint.pixel_preservation/v1",
        "plan_sha256": checked["plan_sha256"],
        "start_frame": start_frame,
        "stop_frame": start_frame + count,
        "source_exact_before_encoding": torch.equal(result[:, y0:y1, x0:x1, :], source),
        "source_resized": False,
        "input_mutated": False,
        "lossy_encoded_pixel_equality_claimed": False,
        "audio_processed": False,
        "color_match_applied": False,
    }
