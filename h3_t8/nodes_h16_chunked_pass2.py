"""Optional H16-3 time-chunked PASS 2 audio refinement adapter.

This is a small, native ComfyUI wrapper around the formal T8 v4 masked
low-sigma executor.  It keeps the released executor as the source of truth and
only adds the missing optional audio delivery policy described in Issue #18:
capture each completed chunk's refined audio, place it on the absolute H3
audio timeline, and crossfade only chunk overlaps.  The default remains the
safe first-pass audio passthrough.
"""
from __future__ import annotations

import json
import logging

import torch
from comfy_api.latest import io

from . import chunked_two_pass_upscale_advanced as _chunked


log = logging.getLogger(__name__)
_VAE_DOWNSAMPLE = 16
_TEMPORAL_STRATEGIES = ("guarded_overlap_exp", "full_clip_safe")
_AUDIO_OUTPUTS = ("refined_exp", "preserve_first_pass")
_DEFAULT_UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"


def _nested_av_parts(latent):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not getattr(samples, "is_nested", False):
        raise ValueError(
            "H16-3 PASS2 expects a native MiniMax H3 nested video/audio LATENT"
        )
    parts = getattr(samples, "tensors", None)
    if parts is None:
        parts = samples.unbind()
    parts = tuple(parts)
    if len(parts) != 2:
        raise ValueError("H16-3 PASS2 expects exactly video and audio tensors")
    video, audio = parts
    if video.ndim != 5 or tuple(video.shape[:2]) != (1, 24):
        raise ValueError(f"unexpected H3 video latent shape: {tuple(video.shape)}")
    if audio.ndim != 4 or tuple(audio.shape[:3]) != (1, 32, 2):
        raise ValueError(f"unexpected H3 audio latent shape: {tuple(audio.shape)}")
    if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
        raise ValueError("H16-3 PASS2 received NaN or Inf in the AV latent")
    return video, audio


def _converted_to_raw(converted):
    raw = []
    for item in converted:
        if isinstance(item, dict):
            metadata = dict(item)
            tensor = metadata.pop("cross_attn", None)
            raw.append([tensor, metadata])
        else:
            raw.append(list(item[:2]))
    return raw


def _raw_conditioning_from_guider(guider):
    conditions = (
        getattr(guider, "original_conds", None)
        or getattr(guider, "conds", None)
        or {}
    )
    positive = conditions.get("positive")
    if not positive:
        raise ValueError("guider does not contain positive conditioning")
    negative = conditions.get("negative")
    model = getattr(guider, "model_patcher", None)
    if model is None:
        raise ValueError("guider does not expose the required ModelPatcher")
    return (
        _converted_to_raw(positive),
        _converted_to_raw(negative) if negative else None,
        float(getattr(guider, "cfg", 1.0) or 1.0),
        model,
    )


def _executor_segments(video, plan):
    frame_count = _chunked.frames_for_tokens(int(video.shape[2]))
    if plan.get("temporal_strategy", "full_clip_safe") == "full_clip_safe":
        return [(0, 0, int(video.shape[2]), frame_count)]
    segments, _ = _chunked.compute_temporal_segments(
        int(video.shape[2]),
        int(plan["temporal_chunk_frames"]),
        int(plan["temporal_overlap_frames"]),
    )
    return segments


def _energy_gate(audio: torch.Tensor, kernel: int = 64) -> torch.Tensor:
    """Return a smooth active-audio gate; quiet tails stay on first-pass audio."""
    mono = audio.abs().mean(dim=(0, 1))
    length = int(mono.shape[-1])
    size = max(1, min(int(kernel), length))
    padding = size // 2
    energy = torch.nn.functional.avg_pool1d(
        mono.view(1, 1, -1), size, stride=1, padding=padding
    ).view(-1)[..., :length]
    ordered, _ = torch.sort(energy)
    floor = float(ordered[int(0.2 * length)])
    peak = float(ordered[int(0.9 * length)])
    if peak <= floor * 1.5:
        return torch.ones_like(energy)
    gate = ((energy - floor * 2.0) / max(1e-6, peak - floor * 2.0)).clamp(0.0, 1.0)
    return torch.nn.functional.avg_pool1d(
        gate.view(1, 1, -1), size, stride=1, padding=padding
    ).view(-1)[..., :length]


def _merge_audio_segments(input_audio, segments, refined_chunks, frame_rescale):
    if len(segments) != len(refined_chunks):
        raise RuntimeError(
            f"captured refined audio chunks ({len(refined_chunks)}) do not match "
            f"temporal windows ({len(segments)})"
        )
    total = int(input_audio.shape[-1])
    placed = torch.zeros_like(input_audio)
    weight = torch.zeros((1, 1, 1, total), dtype=torch.float32, device=input_audio.device)
    previous_end = None
    placed_count = 0
    for segment, refined in zip(segments, refined_chunks, strict=True):
        _start_token, start_frame, _end_token, end_frame = segment
        start = int(round(start_frame * frame_rescale))
        end = min(total, int(round(end_frame * frame_rescale)))
        if end <= start:
            continue
        refined = refined.to(device=input_audio.device, dtype=input_audio.dtype)
        span = input_audio[..., start:end]
        if tuple(refined.shape) != tuple(span.shape):
            raise RuntimeError(
                f"refined audio shape {tuple(refined.shape)} != input span {tuple(span.shape)}"
            )
        if previous_end is None or start >= previous_end:
            placed[..., start:end] = refined
            weight[..., start:end] = 1.0
        else:
            overlap = min(previous_end - start, end - start)
            blend = torch.linspace(
                0.0, 1.0, overlap, device=input_audio.device, dtype=input_audio.dtype
            ).view(1, 1, 1, overlap)
            placed[..., start : start + overlap] = (
                placed[..., start : start + overlap] * (1.0 - blend)
                + refined[..., :overlap] * blend
            )
            weight[..., start : start + overlap] = torch.maximum(
                weight[..., start : start + overlap], blend
            )
            placed[..., start + overlap : end] = refined[..., overlap:]
            weight[..., start + overlap : end] = 1.0
        previous_end = end
        placed_count += 1
    if placed_count == 0:
        raise RuntimeError("no refined audio chunk was placed")
    gate = _energy_gate(input_audio).to(input_audio).view(1, 1, 1, -1)
    effective = gate * weight.to(gate)
    return input_audio * (1.0 - effective) + placed * effective, placed_count


def _run_refined_audio(model, positive, latent, noise, sampler, sigmas, plan, negative, cfg):
    video, audio = _nested_av_parts(latent)
    original_sample_piece = _chunked.sample_piece
    captured = []

    def capture(piece, *args, **kwargs):
        result = original_sample_piece(piece, *args, **kwargs)
        if getattr(result, "is_nested", False) and len(result.tensors) == 2:
            captured.append(result.tensors[1].detach().clone())
        return result

    _chunked.sample_piece = capture
    try:
        output, report_json = _chunked.execute_chunked_two_pass_upscale(
            model, positive, latent, noise, sampler, sigmas, plan,
            negative=negative, cfg=cfg,
        )
    finally:
        _chunked.sample_piece = original_sample_piece

    try:
        segments = _executor_segments(video, plan)
        refined_audio, count = _merge_audio_segments(
            audio, segments, captured, getattr(_chunked, "FRAME_RESCALE", 5.0 / 3.0)
        )
        samples = output.get("samples")
        if not getattr(samples, "is_nested", False) or len(samples.tensors) != 2:
            raise RuntimeError("chunked executor did not return a nested AV latent")
        output = dict(output)
        output["samples"] = _chunked.comfy.nested_tensor.NestedTensor(
            (samples.tensors[0], refined_audio)
        )
        report = json.loads(report_json)
        report.update({"audio_output": "refined_exp", "refined_audio_chunks": count,
                       "audio_merge": "absolute_frame_rescale_crossfade_energy_gate"})
        report_json = json.dumps(report, ensure_ascii=False, sort_keys=True)
        log.info(
            "H16-3 refined audio merge complete: chunks=%d, "
            "mode=absolute_frame_rescale_crossfade_energy_gate",
            count,
        )
        return output, report_json
    except Exception as error:  # pragma: no cover - exercised by GPU/worker faults
        log.warning("H16-3 refined audio merge failed; preserving first-pass audio: %s", error)
        return output, report_json


class DeciiaChunkedPass2Sampler(io.ComfyNode):
    """Drop-in H3 v4 PASS 2 sampler with opt-in refined audio delivery."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DeciiaChunkedPass2Sampler",
            display_name="H16-3 分块 PASS2 · 音频精修（可选）",
            category="T8/MiniMax H3/Audio/Experimental",
            description=(
                "Official T8 v4 full-frame temporal PASS 2 wrapper. "
                "preserve_first_pass is the safe default; refined_exp explicitly "
                "publishes captured second-pass audio with absolute timeline mapping, "
                "overlap crossfade and quiet-tail protection. Any merge failure falls "
                "back to first-pass audio without dropping the video. Decode output, "
                "not denoised_output."
            ),
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
                io.Combo.Input("temporal_strategy", options=list(_TEMPORAL_STRATEGIES), default="guarded_overlap_exp"),
                io.Int.Input("temporal_chunk_frames", default=34, min=17, max=3600, step=17),
                io.Int.Input("temporal_overlap_frames", default=17, min=0, max=1700, step=17),
                io.Float.Input("anchor_strength", default=0.999, min=0.0, max=1.0, step=0.001),
                io.Combo.Input("audio_output", options=list(_AUDIO_OUTPUTS), default="preserve_first_pass"),
            ],
            outputs=[
                io.Latent.Output("output"),
                io.Latent.Output("denoised_output", tooltip="Placeholder equal to output; decode output."),
            ],
        )

    @classmethod
    def execute(
        cls, noise, guider, sampler, sigmas, latent_image,
        temporal_strategy="guarded_overlap_exp", temporal_chunk_frames=34,
        temporal_overlap_frames=17, anchor_strength=0.999,
        audio_output="preserve_first_pass",
    ):
        video, _audio = _nested_av_parts(latent_image)
        positive, negative, cfg, model = _raw_conditioning_from_guider(guider)
        target_width = int(video.shape[-1]) * _VAE_DOWNSAMPLE
        target_height = int(video.shape[-2]) * _VAE_DOWNSAMPLE
        plan, _ = _chunked.build_chunked_two_pass_masked_low_sigma_plan(
            model_name=_DEFAULT_UPSCALER,
            target_width=target_width, target_height=target_height,
            temporal_chunk_frames=int(temporal_chunk_frames),
            temporal_overlap_frames=int(temporal_overlap_frames),
            anchor_strength=float(anchor_strength),
            tile_width=target_width, tile_height=target_height,
            spatial_overlap=0, spatial_fade=0, minimum_tile_size=256,
            overlap_blend="smoothstep", precision="fp16", release_policy="offload_after",
            spatial_strategy="full_frame_safe", temporal_strategy=temporal_strategy,
            second_pass_audio_policy="joint_av_preserve_input",
            video_mask_policy="inherit_if_present_else_generate_all",
        )
        if audio_output == "refined_exp":
            output, _report = _run_refined_audio(
                model, positive, latent_image, noise, sampler, sigmas, plan,
                negative, cfg,
            )
        elif audio_output == "preserve_first_pass":
            output, _report = _chunked.execute_chunked_two_pass_upscale(
                model, positive, latent_image, noise, sampler, sigmas, plan,
                negative=negative, cfg=cfg,
            )
        else:
            raise ValueError(f"unknown audio_output: {audio_output!r}")
        return io.NodeOutput(output, output)


H16_CHUNKED_PASS2_NODE_CLASSES = [DeciiaChunkedPass2Sampler]
