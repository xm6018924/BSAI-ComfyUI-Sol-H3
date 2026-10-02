"""Standard Comfy H3 LATENT boundary for the existing learned H3→LTX adapter.

No weights, imports of upstream code, CUDA initialization, resampling, additional
upscaler, audio conversion or implicit frame cropping. A separately owned adapter
performs the learned mapping; this module makes its temporal contract explicit.
"""
from fractions import Fraction
import math

import torch


FRAME_POLICIES = ("exact", "pad_to_ltx_grid", "crop_to_ltx_grid")


def timeline(source_frames, fps, frame_policy):
    if type(source_frames) is not int or source_frames < 5 or (source_frames - 5) % 17:
        raise ValueError("Expected an explicit native H3 17n+5 frame count, not a guessed latent duration")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps != 24:
        raise ValueError("This adapter recipe requires source CFR24; no implicit FPS conversion")
    if frame_policy not in FRAME_POLICIES:
        raise ValueError("Unknown LTX temporal policy")
    padded = ((source_frames - 1 + 7) // 8) * 8 + 1
    cropped = ((source_frames - 1) // 8) * 8 + 1
    if frame_policy == "exact" and padded != source_frames:
        raise ValueError("Source does not fit LTX 8n+1; explicitly choose pad or crop and adjust delivery audio")
    if frame_policy == "crop_to_ltx_grid" and cropped < 9:
        raise ValueError("Cropping this short source would remove nearly the entire clip")
    output = cropped if frame_policy == "crop_to_ltx_grid" else padded
    return dict(source_frames=source_frames, output_frames=output,
        adapter_decode_frames=padded, source_fps=24, output_fps=24,
        source_duration=[source_frames, 24], output_duration=[output, 24],
        duration_delta=str(Fraction(output - source_frames, 24)),
        h3_latent_frames=((source_frames - 5) // 17) * 5 + 2,
        adapter_latent_frames=(padded - 1) // 8 + 1,
        output_latent_frames=(output - 1) // 8 + 1,
        temporal_policy=frame_policy, audio_duration_adjustment_required=output != source_frames)


def extract_video(latent, *, expected_frames, reference_prefix_latents=0):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("Expected a standard Comfy LATENT with samples")
    if type(reference_prefix_latents) is not int or reference_prefix_latents < 0:
        raise ValueError("Reference prefix must be an explicit nonnegative latent-frame count")
    samples = latent["samples"]
    audio = None
    if getattr(samples, "is_nested", False):
        parts = tuple(samples.unbind())
        if len(parts) != 2:
            raise ValueError("H3 joint AV must contain exactly video then audio")
        video, audio = parts
        if not isinstance(audio, torch.Tensor) or audio.ndim != 4 or tuple(audio.shape[1:3]) != (32, 2):
            raise ValueError("H3 audio branch must be B,32,2,T; never reinterpret it as LTX audio")
    else:
        video = samples
    if (not isinstance(video, torch.Tensor) or video.is_meta or video.ndim != 5
            or video.shape[1] != 24 or video.shape[0] < 1 or not video.is_floating_point()):
        raise ValueError("Expected real floating H3 video B,24,T,H/16,W/16")
    if audio is not None and audio.shape[0] != video.shape[0]:
        raise ValueError("H3 video/audio batch mismatch")
    if video.shape[2] != expected_frames + reference_prefix_latents:
        raise ValueError("Source frame count and exact reference prefix do not match H3 temporal shape")
    if min(video.shape[-2:]) < 2 or any(size % 2 for size in video.shape[-2:]):
        raise ValueError("Existing spatial canvas must align to32 pixels; do not stretch or add another upscaler")
    if not torch.isfinite(video).all():
        raise ValueError("H3 video contains nonfinite values")
    return video[:, :, reference_prefix_latents:], audio


def convert_video_latent(latent, *, adapter, source_frames, fps=24, frame_policy="exact",
                         reference_prefix_latents=0, normalization="comfy_normalized",
                         check_cancel=lambda: None):
    check_cancel()
    if normalization not in ("comfy_normalized", "raw_h3"):
        raise ValueError("Explicit H3 normalization required")
    geometry = timeline(source_frames, fps, frame_policy)
    video, audio = extract_video(latent, expected_frames=geometry["h3_latent_frames"],
                                reference_prefix_latents=reference_prefix_latents)
    height, width = int(video.shape[-2] * 16), int(video.shape[-1] * 16)
    check_cancel()
    with torch.inference_mode():
        output = adapter.convert(video.clone(), pixel_frames=source_frames,
            pixel_height=height, pixel_width=width,
            input_normalization="normalized" if normalization == "comfy_normalized" else "raw")
    check_cancel()
    expected = (video.shape[0], 128, geometry["adapter_latent_frames"], height // 32, width // 32)
    if (not isinstance(output, torch.Tensor) or output.is_meta or tuple(output.shape) != expected
            or not output.is_floating_point() or not torch.isfinite(output).all()):
        raise ValueError("Learned adapter output violates LTX shape/dtype/finite contract")
    output = output[:, :, :geometry["output_latent_frames"]].contiguous()
    report = dict(schema="t8.h3_ltx.standard_latent.v1", **geometry,
        width=width, height=height, source_normalization=normalization,
        output_normalization="normalized_ltx_video", reference_prefix_latents=reference_prefix_latents,
        source_video_shape=list(video.shape), output_video_shape=list(output.shape),
        source_audio_shape=list(audio.shape) if audio is not None else None,
        audio_policy="H3 audio is NOT converted; source LATENT returned unchanged for separate audio handling",
        mask_policy="H3 noise masks/reference metadata are not transferred to LTX",
        additional_upscaler_calls=0, human_qualified=False)
    # Never attach H3 masks, reference conditions or AV metadata to LTX video.
    return {"samples": output}, latent, report
