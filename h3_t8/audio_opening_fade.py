"""Opt-in PCM-only opening mute/fade. No sampling or global audio changes."""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import json
import math

import torch


def _fraction(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number or rational")
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError) as error:
        raise ValueError(f"{name} must be a finite number or rational") from error
    return result


def _ceil(value: Fraction):
    return -(-value.numerator // value.denominator)


def mute_fade_audio(audio, enabled=True, mute_first_frames=1, fade_in_ms=10.0, fps="24"):
    report = {"schema": "t8.audio.opening_mute_fade.v1", "status": "bypass",
              "video_changed": False, "time_shift_seconds": 0,
              "limitation": "PCM contract only; lossy encoding can change decoded samples outside the envelope."}
    if not enabled:
        return audio, json.dumps(report, ensure_ascii=False)
    if isinstance(mute_first_frames, bool) or not isinstance(mute_first_frames, int) or mute_first_frames < 0:
        raise ValueError("mute_first_frames must be a nonnegative integer")
    fade_ms = _fraction(fade_in_ms, "fade_in_ms")
    if fade_ms < 0:
        raise ValueError("fade_in_ms must be nonnegative")
    if mute_first_frames == 0 and fade_ms == 0:
        return audio, json.dumps(report, ensure_ascii=False)
    rate = _fraction(fps, "fps")
    if rate <= 0:
        raise ValueError("fps must be positive; fractional rates such as 30000/1001 are accepted")
    if not isinstance(audio, Mapping):
        raise ValueError("audio must be AUDIO with waveform and sample_rate")
    waveform, sample_rate = audio.get("waveform"), audio.get("sample_rate")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or min(waveform.shape[:2]) < 1:
        raise ValueError("waveform must be floating PCM [batch,channels,samples]")
    if not waveform.is_floating_point() or not bool(torch.isfinite(waveform).all()):
        raise ValueError("waveform must be finite floating PCM")
    samples = waveform.shape[-1]
    mute = min(_ceil(mute_first_frames * sample_rate / rate), samples)
    requested_fade = _ceil(fade_ms * sample_rate / 1000)
    fade = min(requested_fade, samples - mute)
    output = waveform.clone()
    output[..., :mute] = 0
    if fade:
        # A one-sample envelope is zero: continuity at the mute boundary wins.
        phase = torch.linspace(0, math.pi, fade, device=waveform.device, dtype=torch.float64)
        gain = ((1 - phase.cos()) / 2).to(dtype=waveform.dtype)
        output[..., mute:mute + fade] *= gain
    result = dict(audio)
    result["waveform"] = output
    report.update(status="processed", fps_numerator=rate.numerator, fps_denominator=rate.denominator,
                  sample_rate=sample_rate, total_samples=samples,
                  mute_samples=mute, fade_samples=fade, requested_fade_samples=requested_fade,
                  affected_samples=mute + fade, total_samples_unchanged=True,
                  envelope="half_cosine_inclusive_endpoints; single_sample_zero",
                  mute_end_seconds=str(Fraction(mute, sample_rate)),
                  fade_end_seconds=str(Fraction(mute + fade, sample_rate)),
                  warning="Long mute/fade settings can suppress the first spoken word; this is not a cause diagnosis.")
    return result, json.dumps(report, ensure_ascii=False, indent=2)


def mute_fade_video_components(video, enabled=True, mute_first_frames=1, fade_in_ms=10.0):
    """Exact frame-tensor pass-through for native CFR components, not VFR normalization."""
    if not enabled or (mute_first_frames == 0 and _fraction(fade_in_ms, "fade_in_ms") == 0):
        return video, json.dumps({"status": "bypass", "video_changed": False})
    from comfy_api.latest import InputImpl, Types
    if not isinstance(video, InputImpl.VideoFromComponents):
        from .opening_fade_file import mute_fade_file_video
        return mute_fade_file_video(video, mute_first_frames, fade_in_ms)
    components = video.get_components()
    if components.audio is None:
        return video, json.dumps({"status": "no_audio_passthrough", "video_changed": False})
    audio, report = mute_fade_audio(components.audio, enabled, mute_first_frames, fade_in_ms, components.frame_rate)
    replacement = Types.VideoComponents(images=components.images, frame_rate=components.frame_rate,
                                       audio=audio, metadata=components.metadata, alpha=components.alpha)
    return InputImpl.VideoFromComponents(replacement, bit_depth=video.get_bit_depth(),
                                        color_space=video.get_color_space()), report
