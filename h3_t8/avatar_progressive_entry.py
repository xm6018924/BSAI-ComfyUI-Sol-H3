"""Independent source-recording entry into our existing initialized-AV executor.

No external Avatar source is copied. No new checkpoint/tiling/voice-clone model.
"""
from __future__ import annotations

import json

import torch


@torch.inference_mode()
def sample_avatar_progressive(*, av_latent, callback=None, **settings):
    from .core import nested_av_parts
    from .progressive_masking import normalize_av_masks
    from .progressive_sampling_runtime import sample_progressive_h3
    video, source_audio = nested_av_parts(av_latent)
    if not all(bool(torch.isfinite(t).all()) for t in (video, source_audio)):
        raise ValueError("Avatar requires finite encoded source AV")
    raw_mask = av_latent.get("noise_mask")
    if not getattr(raw_mask, "is_nested", False) or len(raw_mask.unbind()) != 2:
        raise ValueError("Connect Audio Conditioning lock_source: explicit video/audio masks are required")
    masks = normalize_av_masks(raw_mask, video, source_audio)
    audio_mask = masks.unbind()[1]
    if bool(torch.count_nonzero(audio_mask)):
        raise ValueError("Avatar source-recording route requires audio mask=0 throughout; "
                         "native generated/voice-reference audio belongs to the existing sampler route")
    if "input_mode" in settings or "continuation" in settings:
        raise ValueError("This independent phase1 entry does not accept continuation or input_mode overrides")
    source_before = source_audio.detach().clone()
    output, text = sample_progressive_h3(av_latent=av_latent, callback=callback,
                                        input_mode="initialized_av_exp", **settings)
    if not torch.equal(source_audio, source_before):
        raise RuntimeError("Avatar source anchor was modified upstream during execution")
    report = json.loads(text)
    actual_audio = nested_av_parts(output)[1]
    report["avatar"] = dict(schema="t8.avatar.progressive_entry.v1", source_recording=True,
                            audio_mask_zero=True, clean_anchor_separate_from_high_restart=True,
                            source_audio_latent_max_abs_difference=float((actual_audio - source_before).abs().max()),
                            source_input_unchanged=True, delivered_original_audio_requires_explicit_mux=True,
                            voice_cloning=False, spatial_tiling=False,
                            trained_model_quality_qualified=False, long_video_qualified=False)
    return output, json.dumps(report, ensure_ascii=False, allow_nan=False)
