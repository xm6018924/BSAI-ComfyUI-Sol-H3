"""Bounded global-trajectory H3 decode with native *pre-clamp* temporal blending.

Do not decode each sampling window independently or blend already-clamped RGB:
both differ from the native global decoder at temporal boundaries.
"""
from __future__ import annotations

import torch

from .video_outpaint_plan import _integer, validate_outpaint_plan


def _raw_decode_chunk(vae, normalized):
    import comfy.model_management as management

    vae.throw_exception_if_invalid()
    with torch.no_grad(), management.cuda_device_context(vae.device):
        vae.prepare_decode(tuple(normalized.shape))
        stage = vae.first_stage_model
        z = normalized.to(device=vae.device, dtype=vae.vae_dtype)
        mean = stage.latents_mean.view(1, -1, 1, 1, 1).to(z)
        std = stage.latents_std.view(1, -1, 1, 1, 1).to(z)
        raw = stage._adaptive_decode(z * std + mean)
        expected = (1, 3, 28, normalized.shape[-2]*16, normalized.shape[-1]*16)
        if tuple(raw.shape) != expected or not raw.is_floating_point() or not torch.isfinite(raw).all():
            raise ValueError("H3 raw decoder violated its seven-token/28-frame contract")
        return raw.detach().to("cpu").contiguous()


def iter_decode_outpaint_shot(vae, read_latents, plan, *, shot_index=0, interrupt_check=None):
    """Yield (absolute start frame, model-canvas RGB chunk, mechanical report).

    At most seven normalized latent tokens and one 28-frame raw decode enter a
    model call; only five raw overlap frames survive into the next call. Source
    padding is clipped from delivery. Caller owns the shared execution lease.
    """
    checked = validate_outpaint_plan(plan)
    _integer(shot_index, "shot_index", 0, len(checked["shots"])-1)
    stage = getattr(vae, "first_stage_model", None)
    expected_contract = {"clip_length": 17, "tokens_chunk_size": 5, "token_drop": 3,
                         "token_overlap": 2, "frame_pre_padding": 3, "frame_overlap": 5,
                         "vae_ratio": 16, "vae_ratio_t": 4}
    if any(getattr(stage, key, None) != value for key, value in expected_contract.items()):
        raise ValueError("bounded decode requires the native H3 17/5 temporal overlap contract")
    shot = checked["shots"][shot_index]
    count = shot["stop"]-shot["start"]
    total_tokens = (shot["aligned_frames"]-5)//17*5+2
    chunk_count = max(1, (total_tokens+3)//5-1)
    overlap = None
    delivered = 0
    for index in range(chunk_count):
        if interrupt_check:
            interrupt_check()
        start, stop = index*5, min(index*5+7, total_tokens)
        normalized = read_latents(start, stop)
        expected = (1, 24, stop-start, checked["sampling"]["height"]//16, checked["sampling"]["width"]//16)
        if not isinstance(normalized, torch.Tensor) or tuple(normalized.shape) != expected or not torch.isfinite(normalized).all():
            raise ValueError("sampled latent reader returned invalid global trajectory data")
        if stop-start < 7:
            normalized = torch.cat((normalized, normalized[:, :, -1:].expand(-1, -1, 7-(stop-start), -1, -1)), dim=2)
        raw = _raw_decode_chunk(vae, normalized)
        current = raw[:, :, 3:20]
        if overlap is not None:
            current = stage.blend(overlap, current, 5, dim=-3)
        overlap = raw[:, :, 23:28].clone()
        emit = min(17, count-delivered)
        if emit > 0:
            pixels = stage._finalize_pixels(current[:, :, :emit])[0].movedim(0, -1).contiguous()
            report = {"plan_sha256": checked["plan_sha256"], "shot_index": shot_index,
                      "decode_chunk": index, "latent_start": start, "latent_stop": stop,
                      "raw_blend_before_clamp": True, "padding_delivered": False}
            yield shot["start"]+delivered, pixels, report
            delivered += emit
        if index == chunk_count-1 and delivered < count:
            emit = min(5, count-delivered)
            pixels = stage._finalize_pixels(overlap[:, :, :emit])[0].movedim(0, -1).contiguous()
            yield shot["start"]+delivered, pixels, {**report, "final_overlap_tail": True}
            delivered += emit
    if delivered != count:
        raise ValueError("bounded H3 decode did not deliver the exact original shot length")
