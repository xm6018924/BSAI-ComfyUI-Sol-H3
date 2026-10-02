"""Opt-in, bounded local color stabilization; optical flow never warps output pixels."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch


MODE = "bounded_motion_color_exp"
SCHEMA = "t8.motion_local_color/v1"


def _require_cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Motion Color EXP requires OpenCV; other Color Match modes do not") from error
    if not callable(getattr(cv2, "calcOpticalFlowFarneback", None)):
        raise RuntimeError("OpenCV does not provide the required Farneback optical flow API")
    return cv2


def runtime_identity():
    cv2 = _require_cv2()
    return {"schema": SCHEMA, "opencv_version": cv2.__version__,
            "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()}


def _opponent(rgb):
    y = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
    return np.stack((y, rgb[..., 2] - y, rgb[..., 0] - y), axis=-1)


def _rgb_offset(value):
    y, cb, cr = np.moveaxis(value, -1, 0)
    r, b = y + cr, y + cb
    g = (y - .2126 * r - .0722 * b) / .7152
    return np.stack((r, g, b), axis=-1).astype(np.float32)


def _align_neighbor(cv2, target, neighbor, target_low, neighbor_low):
    """Target-to-neighbor sampling with forward/backward and structure confidence."""
    gray_a = np.rint(np.clip(_opponent(target)[..., 0], 0, 1) * 255).astype(np.uint8)
    gray_b = np.rint(np.clip(_opponent(neighbor)[..., 0], 0, 1) * 255).astype(np.uint8)
    arguments = (None, .5, 3, 21, 3, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    forward = cv2.calcOpticalFlowFarneback(gray_a, gray_b, *arguments)
    backward = cv2.calcOpticalFlowFarneback(gray_b, gray_a, *arguments)
    h, w = gray_a.shape
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = x + forward[..., 0], y + forward[..., 1]
    warped_back = cv2.remap(backward, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    fb = np.linalg.norm(forward + warped_back, axis=-1)
    magnitude = np.linalg.norm(forward, axis=-1)
    inside = (mx >= 1) & (mx <= w - 2) & (my >= 1) & (my <= h - 2)
    hi_a = _opponent(target - target_low)[..., 0]
    hi_b = _opponent(neighbor - neighbor_low)[..., 0]
    warped_hi = cv2.remap(hi_b, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    structure_error = np.abs(hi_a - warped_hi)
    confidence = inside.astype(np.float32) * np.clip(1 - fb / (1.25 + .05 * magnitude), 0, 1)
    confidence *= np.clip(1 - structure_error / .06, 0, 1)
    valid = confidence >= .35
    warped = cv2.remap(neighbor_low, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return _opponent(warped), valid, confidence


def stabilize_local_motion_color(frames, reference_frames, *, affected_frames=12,
                                 work_max_dimension=256, maximum_rgb_offset=.008,
                                 maximum_luma_offset=.006, maximum_chroma_offset=.004):
    """Correct bracketed local outliers only; retain the source global RGB mean.

    Neighbor images estimate a smooth offset field, never supply output geometry,
    texture or pixels. All estimates use immutable source frames, not corrected history.
    """
    settings = (maximum_rgb_offset, maximum_luma_offset, maximum_chroma_offset)
    if any(not math.isfinite(float(x)) or not 0 <= float(x) <= .02 for x in settings):
        raise ValueError("Motion Color offsets must be finite and in0..0.02")
    if not 1 <= int(affected_frames) <= 24 or not 32 <= int(work_max_dimension) <= 512:
        raise ValueError("Motion Color window must be1..24 and work dimension32..512")
    for tensor in (frames, reference_frames):
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or tensor.shape[-1] < 3
                or tensor.shape[0] < 1 or not torch.is_floating_point(tensor)
                or not bool(torch.isfinite(tensor).all())
                or float(tensor[..., :3].min()) < 0 or float(tensor[..., :3].max()) > 1):
            raise ValueError("Motion Color requires finite SDR NHWC floating-point RGB frames")
    if frames.shape[1:3] != reference_frames.shape[1:3]:
        raise ValueError("Motion Color reference canvas mismatch")
    cv2 = _require_cv2()
    count = min(int(affected_frames), int(frames.shape[0]))
    ref_count = min(5, int(reference_frames.shape[0]))
    source_count = min(count + 2, int(frames.shape[0]))
    h, w = map(int, frames.shape[1:3])
    scale = min(1., float(work_max_dimension) / max(h, w))
    size = (max(8, round(w * scale)), max(8, round(h * scale)))
    joined = torch.cat((reference_frames[-ref_count:, ..., :3].detach().cpu().float(),
                        frames[:source_count, ..., :3].detach().cpu().float())).numpy()
    small = [cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) for rgb in joined]
    low = [cv2.GaussianBlur(rgb, (0, 0), 3.) for rgb in small]
    report = {"schema": SCHEMA, "enabled": True, "applied": False,
              "runtime": runtime_identity(), "frame_count": count,
              "reference_frame_count": ref_count, "work_size": list(size),
              "maximum_rgb_offset": float(maximum_rgb_offset),
              "maximum_luma_offset": float(maximum_luma_offset),
              "maximum_chroma_offset": float(maximum_chroma_offset),
              "maximum_applied_rgb_delta": 0., "maximum_global_rgb_mean_delta": 0.,
              "maximum_applied_luma_delta": 0., "maximum_applied_chroma_delta": 0.,
              "geometry_warped": False, "audio_touched": False, "latent_touched": False,
              "method": "bracketed_motion_confident_local_opponent_median_zero_global_mean_cosine_fade",
              "frames": []}
    # A genuine large cut is not color-corrected, even if flow happens to find matches.
    cut = float(np.abs(small[ref_count].mean((0, 1)) - small[ref_count - 1].mean((0, 1))).max())
    if cut >= .20 or maximum_rgb_offset == 0:
        report["status"] = "abstain_large_cut_or_zero_strength"
        return frames, report
    output = None
    for index in range(count):
        center = ref_count + index
        current = _opponent(low[center])
        values, masks, confidences, positions = [], [], [], []
        for other in range(max(0, center - 2), min(len(small), center + 3)):
            if other == center:
                continue
            aligned, valid, confidence = _align_neighbor(
                cv2, small[center], small[other], low[center], low[other])
            values.append(aligned)
            masks.append(valid)
            confidences.append(confidence)
            positions.append(other)
        if not values or not any(x > center for x in positions):
            report["frames"].append({"index": index, "status": "abstain_no_future_support"})
            continue
        neighbors = np.stack(values)
        valid = np.stack(masks)
        confidence = np.stack(confidences).mean(0)
        stack = np.concatenate((current[None], np.where(valid[..., None], neighbors, np.nan)))
        target = np.nanmedian(stack, axis=0)
        difference = target - current
        # Require both past and future evidence of the same signed color outlier.
        evidence = valid[..., None] & ((neighbors - current) * difference > .0015 ** 2)
        before = np.any(evidence[np.array(positions) < center], axis=0)
        after = np.any(evidence[np.array(positions) > center], axis=0)
        support = before & after & (np.sum(valid, axis=0)[..., None] >= 2)
        bounded = np.clip(difference, [-maximum_luma_offset, -maximum_chroma_offset,
                                      -maximum_chroma_offset],
                          [maximum_luma_offset, maximum_chroma_offset, maximum_chroma_offset])
        field = _rgb_offset(bounded * support) * confidence[..., None]
        field = cv2.GaussianBlur(field, (0, 0), 2.)
        # Preserve intentional uniform lighting; only redistribute local offsets.
        weight = ((np.sum(valid, axis=0) >= 2) & (confidence >= .35)).astype(np.float32)
        weight *= confidence
        field *= weight[..., None]
        field -= field.sum((0, 1), keepdims=True) / max(float(weight.sum()), 1e-6) * weight[..., None]
        field = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)
        full_weight = cv2.resize(weight, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        field -= field.sum((0, 1), keepdims=True) / max(float(full_weight.sum()), 1e-6) * full_weight
        opponent = _opponent(field)
        limiter = max(1., float(np.abs(field).max()) / max(maximum_rgb_offset, 1e-12),
                      float(np.abs(opponent[..., 0]).max()) / max(maximum_luma_offset, 1e-12),
                      float(np.abs(opponent[..., 1:]).max()) / max(maximum_chroma_offset, 1e-12))
        field /= limiter  # One scale retains smoothness and zero mean, unlike per-pixel clipping.
        fade = .5 * (1 + math.cos(math.pi * index / max(1, count - 1)))
        delta = torch.from_numpy(np.ascontiguousarray(field * fade)).to(frames.device)
        original = frames[index, ..., :3].float()
        # A common per-pixel gamut factor retains the bounded luma/chroma direction.
        allowed = torch.where(delta > 0, (1 - original) / delta.abs().clamp_min(1e-12),
                              original / delta.abs().clamp_min(1e-12))
        delta *= allowed.min(-1).values.clamp(0, 1)[..., None]
        corrected = (original + delta).clamp(0, 1).to(frames.dtype)
        actual = corrected.float() - original
        maximum = float(actual.abs().max())
        report["frames"].append({"index": index, "valid_fraction": float(valid.mean()),
                                 "outlier_fraction": float(support.mean()), "fade": fade,
                                 "maximum_rgb_delta": maximum})
        if maximum <= 1e-7:
            continue
        if output is None:
            output = frames.clone()
        output[index, ..., :3] = corrected
        report["maximum_applied_rgb_delta"] = max(report["maximum_applied_rgb_delta"], maximum)
        report["maximum_global_rgb_mean_delta"] = max(
            report["maximum_global_rgb_mean_delta"], float(actual.mean((0, 1)).abs().max()))
        actual_opponent = _opponent(actual.detach().cpu().numpy())
        report['maximum_applied_luma_delta'] = max(
            report['maximum_applied_luma_delta'], float(np.abs(actual_opponent[..., 0]).max()))
        report['maximum_applied_chroma_delta'] = max(
            report['maximum_applied_chroma_delta'], float(np.abs(actual_opponent[..., 1:]).max()))
    report.update(applied=output is not None, status="applied" if output is not None else "no_confident_local_outlier")
    return frames if output is None else output, report
