"""Source RGB choreography windows, independent of generated AV continuity.

This is a T8 implementation of the editable-reference role, not a pose-control
or upstream DanceTransfer sampler port. Source timestamps are CFR frame-start
timestamps; callers must normalize VFR media before constructing this input.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import torch

from .core import FPS, align_frame_count_down, sorted_autogrow_items, validate_audio
from .video_outpaint_identity import value_identity


DANCE_MOTION_TYPE = "H3_T8_DANCE_MOTION"


def dance_audio_at_source_start(audio, start_seconds):
    """Align original PCM to the motion offset, without resampling or remixing."""
    if audio is None:
        return None, {"status": "no_source_audio"}
    waveform, rate = validate_audio(audio, "Dance source_audio")
    if rate <= 0 or not waveform.is_floating_point():
        raise ValueError("Dance source_audio needs a positive sample rate and floating PCM")
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("Dance audio start must be finite and nonnegative")
    start_sample = round(Fraction(str(start_seconds)) * rate)
    if start_sample >= waveform.shape[-1]:
        raise ValueError("Dance source_audio ends before the chosen source start")
    result = audio if start_sample == 0 else {
        "waveform": waveform[..., start_sample:].contiguous(), "sample_rate": rate}
    return result, {"status": "original_pcm_aligned", "start_sample": start_sample,
        "sample_rate": rate, "sample_count": int(waveform.shape[-1]) - start_sample,
        "source_start_seconds": start_seconds, "resampled": False}


@dataclass(frozen=True)
class DanceMotionSource:
    frames: torch.Tensor
    fps_numerator: int = 24
    fps_denominator: int = 1
    start_seconds: float = 0.0

    def __post_init__(self):
        if (not isinstance(self.frames, torch.Tensor) or self.frames.ndim != 4
                or self.frames.shape[0] < 5 or self.frames.shape[-1] != 3
                or min(self.frames.shape[1:3]) < 1
                or not self.frames.is_floating_point()):
            raise ValueError("Dance source requires a floating RGB IMAGE batch [frames,height,width,3]")
        if self.frames.device.type != "cpu" or not self.frames.is_contiguous():
            raise ValueError("Keep the Dance source IMAGE batch contiguous on CPU; only segment windows enter VAE")
        if (type(self.fps_numerator) is not int or type(self.fps_denominator) is not int
                or not 1 <= self.fps_numerator <= 240000
                or not 1 <= self.fps_denominator <= 10000
                or not 1 <= self.fps <= 240):
            raise ValueError("Dance source FPS must be a positive rational rate between 1 and 240")
        if not math.isfinite(self.start_seconds) or self.start_seconds < 0:
            raise ValueError("Dance source start_seconds must be finite and nonnegative")

    @property
    def fps(self):
        return Fraction(self.fps_numerator, self.fps_denominator)

    def contract(self, interrupt_check=None):
        return {"schema": "t8.dance.source_rgb/v1", "role": "editable_reference",
                "frames": value_identity(self.frames, interrupt_check=interrupt_check),
                "fps_numerator": self.fps.numerator, "fps_denominator": self.fps.denominator,
                "start_seconds": self.start_seconds, "output_fps": FPS,
                "resample": "frame_start_zero_order_hold", "hidden_tail": "hold_last_only_after_delivery"}

    def window_indices(self, plan):
        timeline_start = round(float(plan.timeline_start_seconds) * FPS)
        if abs(float(plan.timeline_start_seconds) * FPS - timeline_start) > 1e-6:
            raise ValueError("Dance segment timeline must lie on the H3 24fps grid")
        start = timeline_start - int(plan.context_frames)
        delivered_end = timeline_start + int(plan.final_frame_count)
        render_end = start + int(plan.render_frames)
        if start < 0 or delivered_end > render_end or int(plan.render_frames) < 5:
            raise ValueError("Dance segment has an invalid source/continuity window")
        source_start = Fraction(str(self.start_seconds))
        indices = []
        hidden_hold = 0
        for frame in range(start, render_end):
            timestamp = source_start + Fraction(frame, FPS)
            index = math.floor(timestamp * self.fps)
            if index >= len(self.frames):
                if not plan.is_final_segment or frame < delivered_end:
                    raise ValueError("Dance source ends before delivered motion; do not loop or pad missing choreography")
                index = len(self.frames) - 1
                hidden_hold += 1
            indices.append(index)
        return indices, {"status": "source_rgb_window", "role": "editable_reference",
            "render_start_frame": start, "render_end_frame_exclusive": render_end,
            "delivered_start_frame": timeline_start, "delivered_end_frame_exclusive": delivered_end,
            "generated_context_frames": int(plan.context_frames),
            "source_indices": indices, "hidden_tail_hold_frames": hidden_hold,
            "reference_rgb_frames": len(indices),
            "reference_aligned_frames": align_frame_count_down(len(indices)),
            "source_fps": [self.fps.numerator, self.fps.denominator],
            "source_start_seconds": self.start_seconds}

    def references(self, plan, existing=None):
        entries = sorted_autogrow_items(existing)
        if len(entries) >= 3:
            raise ValueError("Dance motion needs one reference-video slot; at most two other videos may be connected")
        # Preserve the order and ordinals of user references and their soundtracks.
        ordinal = max((number for number, _value in entries), default=-1) + 1
        refs = dict(existing or {})
        key = f"ref_video_{ordinal}"
        while key in refs:
            ordinal += 1
            key = f"ref_video_{ordinal}"
        indices, report = self.window_indices(plan)
        refs[key] = self.frames.index_select(0, torch.tensor(indices, dtype=torch.long))
        report["input_ordinal"] = ordinal
        report["prompt_video_number"] = len(entries) + 1
        return refs, report


def prepare_dance_motion(source, segments, reference_video_policy, ref_videos, ref_video_audios,
                         interrupt_check=None):
    """Validate *all* delivery windows before a first sample/cache mutation."""
    if type(source) is not DanceMotionSource:
        raise ValueError("source_motion must come from the Dance Motion Source node")
    source.__post_init__()
    entries = sorted_autogrow_items(ref_videos)
    if len(entries) >= 3:
        raise ValueError("Dance motion needs one free reference-video slot")
    # Never let a formerly orphan soundtrack silently attach to the new source.
    video_ordinals = {number for number, _ in entries}
    if any(number not in video_ordinals for number, _ in sorted_autogrow_items(ref_video_audios)):
        raise ValueError("Reference-video audio must have its own existing video; Dance music goes to final_audio")
    windows = []
    for segment in segments:
        _indices, report = source.window_indices(segment.plan)
        count = report["reference_rgb_frames"]
        if reference_video_policy == "official_2_to_15s" and not 48 <= count <= 360:
            raise ValueError("Dance source window is outside the selected official 48–360 reference-frame policy")
        windows.append(report)
    return {"source": source.contract(interrupt_check), "windows": windows}
