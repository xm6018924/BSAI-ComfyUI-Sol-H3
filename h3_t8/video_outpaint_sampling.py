"""Single-window H3 sampler boundary for a resumable outpaint coordinator.

The caller supplies one encoded window and its deterministic noise slice. No full
video is allocated here and no global model unload/patch is performed. File-backed
source encoding, noise identity and durable commits belong to the coordinator.
"""
from __future__ import annotations

import torch
import comfy.nested_tensor
from comfy.ldm.minimax.model import FRAME_PER_TOKEN

from .video_outpaint_plan import validate_outpaint_plan


SEAM_CONTEXT_INSET_LATENTS = 2


def _tensor(value, shape, label):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{label} must have shape {tuple(shape)}")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{label} must contain finite floating-point values")
    return value


def _condition(conditioning, video_context, audio_context):
    if video_context is None:
        return conditioning
    result = []
    for cross, metadata in conditioning:
        options = dict(metadata)
        options["minimax_keyframes"] = [
            *options.get("minimax_keyframes", []),
            {"resolved_frame_index": 0, "latent": video_context, "audio_latent": audio_context},
        ]
        result.append([cross, options])
    return result


def _observed_video_tokens(frame_count, latent_t):
    """Only complete source-supported tokens are fixed; padding is not observed."""
    observed_frames = 0
    for token in range(latent_t):
        span = FRAME_PER_TOKEN[token % len(FRAME_PER_TOKEN)]
        if observed_frames+span > frame_count:
            return token
        observed_frames += span
    return latent_t


def _seam_safe_source_lock_box(plan):
    """Keep the complete source rectangle locked, including its boundary cells.

    Releasing an inner strip changes source geometry; exact RGB pasteback later
    cannot restore continuity with the generated exterior. Retain this private
    helper name for existing diagnostic imports, not as an adjustable inset.
    """
    return list(plan["sampling"]["source_lock_latent_box"])


def _source_noise_weights(plan, *, device, mode="context_then_project"):
    """Sampling context weights; committed source ownership is separate below."""
    x0, y0, x1, y1 = plan["sampling"]["source_lock_latent_box"]
    weights = torch.zeros((y1-y0, x1-x0), device=device, dtype=torch.float32)
    if mode not in {"strict", "context_then_project"}:
        raise ValueError("unknown source edge mode")
    if mode == "strict":
        return weights
    left, top, right, bottom = (bool(v) for v in plan["output"]["margins"])
    inset = SEAM_CONTEXT_INSET_LATENTS
    if x1-x0 <= inset*(left+right) or y1-y0 <= inset*(top+bottom):
        return weights
    xx = torch.arange(x1-x0,device=device)[None,:]
    yy = torch.arange(y1-y0,device=device)[:,None]
    for distance, active in ((xx,left),(yy,top),(x1-x0-1-xx,right),(y1-y0-1-yy,bottom)):
        if active:
            weights = torch.maximum(weights, ((inset-distance).float()/(inset+1)).clamp(0,1))
    return weights


def sample_outpaint_window(
    *, model, conditioning, plan, shot_index, window_index,
    video_latent, audio_latent, video_noise, audio_noise,
    context=None, audio_noise_mask=None, seed=20260808, steps=20, sampler_name="res_multistep", scheduler="simple",
    sample_function=None, source_edge_mode="context_then_project",
):
    checked = validate_outpaint_plan(plan)
    if isinstance(shot_index, bool) or not isinstance(shot_index, int) or not 0 <= shot_index < len(checked["shots"]):
        raise ValueError("invalid shot index")
    shot = checked["shots"][shot_index]
    if isinstance(window_index, bool) or not isinstance(window_index, int) or not 0 <= window_index < len(shot["windows"]):
        raise ValueError("invalid window index")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
        raise ValueError("steps must be an integer in [1, 100]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be a uint64 integer")
    window = shot["windows"][window_index]
    sampling = checked["sampling"]
    video_t = (window["render_frames"] - 5) // 17 * 5 + 2
    audio_t = round(window["render_frames"] / 24 * 40)
    video_shape = (1, 24, video_t, sampling["height"] // 16, sampling["width"] // 16)
    audio_shape = (1, 32, 2, audio_t)
    video = _tensor(video_latent, video_shape, "video_latent").clone()
    audio = _tensor(audio_latent, audio_shape, "audio_latent").clone()
    _tensor(video_noise, video_shape, "video_noise")
    _tensor(audio_noise, audio_shape, "audio_noise")
    if any(t.device != video.device for t in (audio, video_noise, audio_noise)):
        raise ValueError("window latent and noise devices must match")
    audio_mask = torch.zeros((1, 1, 2, audio_t), dtype=torch.float32, device=audio.device)
    if audio_noise_mask is not None:
        audio_mask = _tensor(audio_noise_mask, audio_mask.shape, "audio_noise_mask").clone()
        if audio_mask.device != audio.device or torch.any((audio_mask != 0) & (audio_mask != 1)):
            raise ValueError("audio noise mask must be binary and on the latent device")
    overlap = window["context_video_latents"]
    audio_overlap = window["context_audio_latents"]
    x0, y0, x1, y1 = sampling["source_lock_latent_box"]
    if overlap:
        if not isinstance(context, dict) or any(context.get(key) != value for key, value in {
            "plan_sha256": checked["plan_sha256"], "shot_index": shot_index,
            "target_window_index": window_index,
        }.items()):
            raise ValueError("context is missing or bound to another source/shot/window")
        previous = _tensor(context.get("video_tail"), (1, 24, overlap, *video_shape[-2:]), "context video")
        previous_audio = _tensor(context.get("audio_tail"), (1, 32, 2, audio_overlap), "context audio")
        if previous.device != video.device or previous.dtype != video.dtype:
            raise ValueError("context video dtype/device differs from this window")
        if previous_audio.device != audio.device or previous_audio.dtype != audio.dtype:
            raise ValueError("context audio dtype/device differs from this window")
        if not torch.equal(previous[:, :, :, y0:y1, x0:x1], video[:, :, :overlap, y0:y1, x0:x1]):
            raise ValueError("source video context disagrees across the window boundary")
        observed = (audio_mask[..., :audio_overlap] == 0).expand_as(previous_audio)
        if not torch.equal(previous_audio[observed], audio[..., :audio_overlap][observed]):
            raise ValueError("source audio context disagrees across the window boundary")
        video[:, :, :overlap] = previous
        audio[..., :audio_overlap] = previous_audio
        audio_mask[..., :audio_overlap] = 0
    elif context is not None:
        raise ValueError("a new shot must not inherit context from a previous shot")
    mask = torch.ones((1, 1, video_t, *video_shape[-2:]), dtype=torch.float32, device=video.device)
    global_t = (shot["aligned_frames"]-5)//17*5+2
    observed_stop = _observed_video_tokens(shot["stop"]-shot["start"], global_t)
    local_observed = max(0, min(video_t, observed_stop-window["video_start"]))
    sx0, sy0, sx1, sy1 = sampling["source_lock_latent_box"]
    mask[:, :, :local_observed, sy0:sy1, sx0:sx1] = _source_noise_weights(
        checked, device=video.device, mode=source_edge_mode
    )
    mask[:, :, :overlap] = 0
    if checked["output"]["has_outpaint"] and not torch.count_nonzero(mask):
        raise ValueError("sampling canvas leaves no generated latent cells; increase the MP budget")
    window_conditioning = _condition(conditioning, video[:, :, :overlap].clone() if overlap else None,
                                     audio[..., :audio_overlap].clone() if overlap else None)
    if sample_function is None:
        from comfy.sample import sample as native_sample
        sample_function = native_sample
    latent = comfy.nested_tensor.NestedTensor((video.clone(), audio.clone()))
    noise = comfy.nested_tensor.NestedTensor((video_noise.clone(), audio_noise.clone()))
    sampled = sample_function(
        model, noise, steps, 1.0, sampler_name, scheduler,
        window_conditioning, window_conditioning, latent,
        denoise=1.0, noise_mask=comfy.nested_tensor.NestedTensor((mask.clone(), audio_mask.clone())), seed=seed,
    )
    if not isinstance(sampled, comfy.nested_tensor.NestedTensor):
        raise ValueError("H3 sampler did not return joint AV latents")
    result_video, result_audio = sampled.unbind()
    _tensor(result_video, video_shape, "sampled video")
    _tensor(result_audio, audio_shape, "sampled audio")
    # Reinforce ownership at the API boundary even if a backend writes locked cells.
    if result_video.dtype != video.dtype or result_video.device != video.device:
        raise ValueError("sampler changed the video dtype/device")
    if result_audio.dtype != audio.dtype or result_audio.device != audio.device:
        raise ValueError("sampler changed the audio dtype/device")
    # An edge band is temporary sampling context, never owned source content.
    # Project the full observed source back BEFORE committing/decoding and BEFORE
    # deriving next-window context. This is not just a final RGB pasteback.
    generated_mask = mask.to(torch.bool)
    generated_mask[:, :, :local_observed, sy0:sy1, sx0:sx1] = False
    result_video = torch.where(generated_mask, result_video, video)
    result_audio = torch.where(audio_mask.to(torch.bool), result_audio, audio)
    next_context = None
    if window_index + 1 < len(shot["windows"]):
        next_window = shot["windows"][window_index + 1]
        next_context = {
            "plan_sha256": checked["plan_sha256"], "shot_index": shot_index,
            "target_window_index": window_index + 1,
            "video_tail": result_video[:, :, next_window["video_start"] - window["video_start"]:].clone(),
            "audio_tail": result_audio[..., next_window["audio_start"] - window["audio_start"]:].clone(),
        }
    return {"samples": comfy.nested_tensor.NestedTensor((result_video, result_audio))}, next_context, {
        "schema": "t8.h3.video_outpaint.window_sample/v1", "plan_sha256": checked["plan_sha256"],
        "shot_index": shot_index, "window_index": window_index,
        "source_and_context_latents_restored": True, "source_audio_latent_preserved": True,
        "sampled_audio_discarded": not bool(torch.count_nonzero(audio_mask)), "global_model_unload_called": False,
        "audio_output_policy": "original_file_track_only",
        "generated_audio_context_cells": int(torch.count_nonzero(audio_mask)),
        "generated_latent_cells": int(torch.count_nonzero(mask)),
        "source_lock_latent_box": [x0, y0, x1, y1],
        "source_edge_context_regenerated": bool(torch.count_nonzero(mask[:, :, :local_observed, sy0:sy1, sx0:sx1])),
        "source_edge_latents_projected_before_decode": True,
        "seam_context_inset_latents": SEAM_CONTEXT_INSET_LATENTS if source_edge_mode != "strict" else 0,
        "seam_context_policy": ("source_guided_ramp_then_full_source_projection_v4"
                                if source_edge_mode != "strict" else "full_observed_source_latent_lock_v3"),
        "final_source_pixels_restored_by_compositor": True,
        "deliver_start": window["deliver_start"], "deliver_stop": window["deliver_stop"],
        "backend": "comfy.sample.sample" if getattr(sample_function, "__module__", None) == "comfy.sample" else "injected_callable",
    }
