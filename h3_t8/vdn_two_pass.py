"""VDN-specific high-resolution restart; no LightX2V schedule or extra LoRA."""

import json

import torch

from .core import nested_av_parts
from .vdn_h3_advanced import ATTACHMENT_KEY, STAGES, setup_vdn_execution


def setup_vdn_refine(model, av_latent, first_pass_latent, refine_steps=4):
    receipt = getattr(model, "get_attachment", lambda _: None)(ATTACHMENT_KEY)
    stage = receipt.get("stage") if isinstance(receipt, dict) else None
    if stage not in STAGES or receipt.get("status") != "configured":
        raise ValueError("VDN refine requires the OpenVDN Composer MODEL")
    base_steps = int(STAGES[stage]["steps"])
    refine_steps = int(refine_steps)
    if not 1 <= refine_steps < base_steps:
        raise ValueError(f"refine_steps must be between 1 and {base_steps - 1} for {stage}")
    video, audio = nested_av_parts(av_latent)
    first_video, first_audio = nested_av_parts(first_pass_latent)
    if tuple(video.shape[:3]) != tuple(first_video.shape[:3]) or tuple(audio.shape) != tuple(first_audio.shape):
        raise ValueError("VDN two-pass handoff must preserve batch, channels, frame count and audio length")
    if any(high < low for high, low in zip(video.shape[-2:], first_video.shape[-2:])):
        raise ValueError("VDN refinement canvas must not shrink the first-pass latent")
    if not all(bool(torch.isfinite(value).all()) for value in (video, audio, first_video, first_audio)):
        raise ValueError("VDN two-pass input contains NaN or Inf")
    planned, sampler, full_sigmas, execution_json = setup_vdn_execution(model, av_latent)
    # Restart on the model's own trained-stage grid. This is an EXP profile,
    # not a claim that DMD was trained for learned-upscaled input distributions.
    sigmas = full_sigmas[-(refine_steps + 1):].clone()
    if not bool(torch.all(sigmas[:-1] > sigmas[1:])) or float(sigmas[-1]) != 0:
        raise RuntimeError("VDN refine schedule must descend strictly to zero")
    report = json.loads(execution_json)
    report.update({
        "status": "experimental_refine_planned",
        "profile": "complete_first_pass_then_native_stage_tail_restart",
        "first_pass_nfe": base_steps,
        "refine_nfe": refine_steps,
        "total_nfe": base_steps + refine_steps,
        "steps": refine_steps,
        "nfe": refine_steps,
        "sigma_count": int(sigmas.numel()),
        "refine_sigmas": sigmas.tolist(),
        "first_video_shape": list(first_video.shape),
        "high_video_shape": list(video.shape),
        "audio_shape": list(audio.shape),
        "audio_matches_first_pass": torch.equal(audio, first_audio),
        "fresh_restart_noise_required": True,
        "additional_turbo_lora": False,
        "quality_validation": "pending_real_generation_and_human_review",
    })
    return planned, sampler, sigmas, json.dumps(report, ensure_ascii=False, indent=2)
