"""Authenticated spatial text router for native MiniMax H3 outpaint.

Only target-video queries receive a spatial mask. Text, references, audio and
condition rows retain native attention. The explicit mask is allocated one
bounded query chunk at a time; no dense S-by-S mask is created.
"""
from __future__ import annotations

from .patch_stack_policy import warn_patch_stack

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
import comfy.patcher_extension
from comfy.ldm.modules import attention as attention_module

from .video_outpaint_regional_conditioning import (
    OutpaintRegionalConditioningProvider,
    REGIONAL_PAYLOAD_KEY,
    regional_claim,
    validate_regional_binding,
)


REGIONAL_WRAPPER_KEY = "t8_h3_outpaint_regional"
REGIONAL_RUNTIME_KEY = "t8_h3_outpaint_regional_runtime"
REGIONAL_PATCH_VERSION = 1


def _active_wrapper_groups(model):
    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    return {
        str(key): list(value)
        for key, value in getattr(model, "wrappers", {}).get(wrapper_type, {}).items()
        if bool(value)
    }


def _runtime_route(layout, binding, shot, device):
    shot_binding = binding["shots"][shot]
    if tuple(layout.signature)[0] != int(shot_binding["text_len"]):
        raise RuntimeError("regional MODEL and CONDITIONING text lengths differ")
    segments = list(layout.segments)
    video = [item for item in segments if item[2] == "video"]
    audio = [item for item in segments if item[2] == "audio"]
    if len(video) != 1 or len(audio) != 1 or video[0][1] != int(layout.seq_len) or audio[0][1] != video[0][0]:
        raise RuntimeError("regional routing requires native H3 target audio/video tail segments")
    _, latent_t, latent_h, latent_w, _ = tuple(layout.signature)
    rows, columns = latent_h // 2, latent_w // 2
    grid = binding["sampling_grid"]
    if (rows != int(grid["rows"]) or columns != int(grid["columns"])
            or int(video[0][1] - video[0][0]) != latent_t * rows * columns):
        raise RuntimeError("regional routing sampling grid differs from the runtime H3 layout")
    regions = []
    for item in shot_binding["regions"]:
        allowed = torch.zeros(rows * columns, dtype=torch.bool, device=device)
        indices = torch.tensor(item["spatial_rows"], dtype=torch.long, device=device)
        if indices.numel() == 0 or int(indices.min()) < 0 or int(indices.max()) >= allowed.numel():
            raise RuntimeError("regional binding contains invalid spatial rows")
        allowed[indices] = True
        regions.append({
            "text_key_start": int(item["text_key_start"]),
            "text_key_end": int(item["text_key_end"]),
            "allowed": allowed,
        })
    return {
        "binding_sha256": binding["binding_sha256"],
        "seq_len": int(layout.seq_len),
        "video_start": int(video[0][0]),
        "video_end": int(video[0][1]),
        "frame_rows": rows * columns,
        "regions": regions,
    }


def make_outpaint_regional_bias(relative_query_rows, seq_len, frame_rows, regions, *, dtype):
    if relative_query_rows.ndim != 1 or relative_query_rows.dtype != torch.long:
        raise ValueError("regional query rows must be a one-dimensional long tensor")
    bias = torch.zeros((relative_query_rows.numel(), int(seq_len)),
                       dtype=dtype, device=relative_query_rows.device)
    spatial = torch.remainder(relative_query_rows, int(frame_rows))
    blocked_value = torch.finfo(dtype).min
    for item in regions:
        start, end = int(item["text_key_start"]), int(item["text_key_end"])
        if not 0 <= start < end <= int(seq_len):
            raise RuntimeError("regional text-key span is outside the packed sequence")
        allowed = item["allowed"]
        if allowed.device != relative_query_rows.device or allowed.dtype != torch.bool:
            raise RuntimeError("regional spatial mask has the wrong device or dtype")
        blocked = ~allowed[spatial]
        if bool(blocked.any()):
            bias[blocked, start:end] = blocked_value
    return bias


def route_outpaint_regional_attention(q, k, v, heads, mask=None, attn_precision=None,
                                      skip_reshape=False, skip_output_reshape=False,
                                      transformer_options=None, *, query_chunk_rows, backend=None, **kwargs):
    transformer_options = transformer_options or {}
    route = transformer_options.get(REGIONAL_RUNTIME_KEY)
    delegate_kwargs = dict(kwargs)
    delegate_kwargs["_inside_attn_wrapper"] = True
    delegate = backend.attention if backend is not None else attention_module.optimized_attention
    if route is None or q.shape[-2] != int(route["seq_len"]):
        return delegate(
            q, k, v, heads, mask=mask, attn_precision=attn_precision,
            skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
            transformer_options=transformer_options, **delegate_kwargs)
    if mask is not None:
        warn_patch_stack("regional outpaint combines the supplied mask with its spatial bias; composition unverified")
    if not skip_reshape or skip_output_reshape:
        raise RuntimeError("regional outpaint routing received an unsupported H3 attention layout")
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != heads:
        raise RuntimeError("regional outpaint routing requires H3 batch-one packed attention")

    outputs = []
    video_start, video_end = int(route["video_start"]), int(route["video_end"])
    if video_start:
        outputs.append(delegate(
            q[:, :, :video_start], k, v, heads, mask=_mask_query_slice(mask, 0, video_start), attn_precision=attn_precision,
            skip_reshape=True, skip_output_reshape=False,
            transformer_options=transformer_options, **delegate_kwargs))
    for start in range(0, video_end - video_start, int(query_chunk_rows)):
        end = min(start + int(query_chunk_rows), video_end - video_start)
        relative = torch.arange(start, end, dtype=torch.long, device=q.device)
        bias = make_outpaint_regional_bias(relative, route["seq_len"], route["frame_rows"],
                                            route["regions"], dtype=q.dtype)
        selected_mask = _mask_query_slice(mask, video_start + start, video_start + end)
        if selected_mask is not None:
            bias = (bias.masked_fill(~selected_mask, float('-inf')) if selected_mask.dtype == torch.bool
                    else bias + selected_mask)
        biased_delegate = backend.attention if backend is not None else attention_module.attention_pytorch
        outputs.append(biased_delegate(
            q[:, :, video_start + start:video_start + end], k, v, heads,
            mask=bias, attn_precision=attn_precision, skip_reshape=True,
            skip_output_reshape=False, transformer_options=transformer_options,
            **delegate_kwargs))
    if video_end < int(route["seq_len"]):
        outputs.append(delegate(
            q[:, :, video_end:int(route["seq_len"])], k, v, heads,
            mask=_mask_query_slice(mask, video_end, int(route["seq_len"])),
            attn_precision=attn_precision, skip_reshape=True, skip_output_reshape=False,
            transformer_options=transformer_options, **delegate_kwargs))
    return torch.cat(outputs, dim=1)


def _mask_query_slice(mask, start, end):
    if mask is None or mask.ndim < 2 or mask.shape[-2] == 1:
        return mask
    return mask[..., start:end, :]


class _RegionalDiffusionWrapper:
    def __init__(self, binding, query_chunk_rows):
        self.binding = binding
        self.binding_sha256 = binding["binding_sha256"]
        self.query_chunk_rows = int(query_chunk_rows)

    def __call__(self, executor, x, timestep, context, transformer_options=None, **kwargs):
        transformer_options = transformer_options if transformer_options is not None else {}
        if len(executor.wrappers) != 1:
            warn_patch_stack('another diffusion wrapper was added after regional outpaint binding')
        override = transformer_options.get("optimized_attention_override")
        if getattr(override, "_t8_outpaint_regional_binding_sha256", None) != self.binding_sha256:
            warn_patch_stack('regional outpaint attention owner was replaced after binding')
        replacements = transformer_options.get("patches_replace", {})
        if isinstance(replacements, Mapping) and any(bool(value) for value in replacements.values()):
            warn_patch_stack('regional outpaint routing refuses runtime block/attention replacements')
        supplied = kwargs.pop(REGIONAL_PAYLOAD_KEY, None)
        matching = [shot for shot in range(len(self.binding["shots"]))
                    if supplied == regional_claim(self.binding_sha256, shot)]
        if len(matching) != 1:
            raise RuntimeError("regional outpaint MODEL and CONDITIONING are not an authenticated pair")
        payload = kwargs.get("minimax_payload")
        if not isinstance(payload, Mapping) or payload.get("layout") is None:
            raise RuntimeError("regional outpaint could not find the native H3 packed layout")
        if REGIONAL_RUNTIME_KEY in transformer_options:
            raise RuntimeError("nested regional outpaint runtime state was refused")
        transformer_options[REGIONAL_RUNTIME_KEY] = _runtime_route(
            payload["layout"], self.binding, matching[0], x[0].device)
        try:
            return executor(x, timestep, context, transformer_options, **kwargs)
        finally:
            transformer_options.pop(REGIONAL_RUNTIME_KEY, None)


class _RegionalAttentionRouter:
    def __init__(self, binding_sha256, query_chunk_rows, backend=None):
        self.binding_sha256 = binding_sha256
        self.query_chunk_rows = int(query_chunk_rows)
        self.backend = backend

    def __call__(self, *args, **kwargs):
        return route_outpaint_regional_attention(
            *args, query_chunk_rows=self.query_chunk_rows, backend=self.backend, **kwargs)


def regional_model_contract(model):
    """Return a verified regional composition, ``None`` when wholly absent."""
    attachment = model.get_attachment(REGIONAL_WRAPPER_KEY) if hasattr(model, "get_attachment") else None
    groups = _active_wrapper_groups(model)
    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    other_active_wrappers = {
        str(kind): {str(key): list(value) for key, value in keyed.items() if bool(value)}
        for kind, keyed in getattr(model, "wrappers", {}).items()
        if kind != wrapper_type and isinstance(keyed, dict)
        and any(bool(value) for value in keyed.values())
    }
    transformer = getattr(model, "model_options", {}).get("transformer_options", {})
    override = transformer.get("optimized_attention_override") if isinstance(transformer, dict) else None
    has_any = attachment is not None or REGIONAL_WRAPPER_KEY in groups or getattr(
        override, "_t8_outpaint_regional_binding_sha256", None) is not None
    if not has_any:
        return None
    if not isinstance(attachment, dict) or set(attachment) != {
        "patch_version", "binding", "binding_sha256", "query_chunk_rows",
        "conditioning_manifest_sha256", "adapter_sha256",
    }:
        raise RuntimeError("regional outpaint MODEL attachment is missing or has unknown fields")
    if attachment["patch_version"] != REGIONAL_PATCH_VERSION:
        raise RuntimeError("regional outpaint MODEL patch version is unsupported")
    binding = validate_regional_binding(attachment["binding"])
    binding_sha = binding["binding_sha256"]
    if attachment["binding_sha256"] != binding_sha:
        raise RuntimeError("regional outpaint MODEL attachment hash differs from its binding")
    if set(groups) != {REGIONAL_WRAPPER_KEY} or len(groups[REGIONAL_WRAPPER_KEY]) != 1:
        warn_patch_stack('regional outpaint requires exactly one regional diffusion wrapper')
    if other_active_wrappers:
        warn_patch_stack('regional outpaint found unsupported non-diffusion MODEL wrappers')
    if REGIONAL_WRAPPER_KEY not in groups or len(groups[REGIONAL_WRAPPER_KEY]) != 1:
        raise RuntimeError("regional outpaint's own diffusion wrapper is missing or duplicated")
    wrapper = groups[REGIONAL_WRAPPER_KEY][0]
    if (type(wrapper) is not _RegionalDiffusionWrapper or set(vars(wrapper)) != {
            "binding", "binding_sha256", "query_chunk_rows"}
            or wrapper.binding != binding or wrapper.binding_sha256 != binding_sha
            or wrapper.query_chunk_rows != attachment["query_chunk_rows"]):
        raise RuntimeError("regional outpaint diffusion wrapper differs from its authenticated attachment")
    router = getattr(override, "_t8_outpaint_regional_router", None)
    if router is None:
        if override is not None and not callable(override):
            raise TypeError("regional outpaint attention override must be callable")
        warn_patch_stack("regional outpaint has a later user-selected attention override; regional routing may be bypassed")
    elif (type(router) is not _RegionalAttentionRouter or set(vars(router)) != {
            "binding_sha256", "query_chunk_rows", "backend"}
            or router.binding_sha256 != binding_sha
            or router.query_chunk_rows != attachment["query_chunk_rows"]
            or getattr(override, "_t8_outpaint_regional_binding_sha256", None) != binding_sha):
        raise RuntimeError("regional outpaint optimized-attention owner differs from its binding")
    if not 32 <= int(attachment["query_chunk_rows"]) <= 2048:
        raise RuntimeError("regional outpaint query chunk size is outside its audited range")
    expected_adapter = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if attachment["adapter_sha256"] != expected_adapter:
        raise RuntimeError("regional outpaint adapter implementation changed")
    result = {
        "kind": "h3_spatial_text_routing",
        "patch_version": REGIONAL_PATCH_VERSION,
        "binding_sha256": binding_sha,
        "guidance_sha256": binding["guidance_sha256"],
        "conditioning_manifest_sha256": attachment["conditioning_manifest_sha256"],
        "query_chunk_rows": attachment["query_chunk_rows"],
        "region_count": sum(len(shot["regions"]) for shot in binding["shots"]),
        "adapter_sha256": expected_adapter,
    }
    if router is None or router.backend is not None:
        result.update(portable_cache_reuse=False, composition_verified=False)
    return result


def patch_outpaint_regional_model(model, conditioning, query_chunk_rows=256):
    from comfy.model_base import MiniMaxH3

    if not isinstance(conditioning, OutpaintRegionalConditioningProvider):
        raise ValueError("regional MODEL composer requires a prepared regional conditioning provider")
    if not hasattr(model, "clone") or not isinstance(getattr(model, "model", None), MiniMaxH3):
        raise ValueError("regional outpaint requires a native MiniMax H3 MODEL")
    if isinstance(query_chunk_rows, bool) or not isinstance(query_chunk_rows, int) or not 32 <= query_chunk_rows <= 2048:
        raise ValueError("query_chunk_rows must be an integer in [32, 2048]")
    if regional_model_contract(model) is not None:
        raise ValueError("MODEL already has regional outpaint routing")
    for name in ("patches", "weight_wrapper_patches", "additional_models", "callbacks",
                 "injections", "hook_patches", "forced_hooks", "current_hooks"):
        if getattr(model, name, None):
            warn_patch_stack(f'regional outpaint does not cover MODEL {name}')
    if _active_wrapper_groups(model):
        warn_patch_stack('regional outpaint must own the only diffusion-model wrapper')
    from .video_outpaint_model_patches import inspect_outpaint_model_patches_advisory
    inspect_outpaint_model_patches_advisory(model)
    binding = validate_regional_binding(conditioning.binding, conditioning.plan)
    conditioning_sha = conditioning.verify()
    patched = model.clone()
    wrapper = _RegionalDiffusionWrapper(binding, query_chunk_rows)
    prior = patched.model_options.get("transformer_options", {}).get("optimized_attention_override")
    backend = None
    if prior is not None:
        from .relay_sol_backend import capture_composed_backend, UserSelectedBackend
        from .h3_core_compat import plain_attention_backend
        from .progressive_attention import _PlainDelegate
        if not callable(prior):
            raise TypeError("regional outpaint attention override must be callable")
        plain = plain_attention_backend(prior)
        backend = _PlainDelegate(prior, plain) if plain is not None else capture_composed_backend(prior)
        if backend is None:
            backend = UserSelectedBackend(prior)
        warn_patch_stack("regional outpaint preserves the user's prior attention delegate including biased calls")
    router = _RegionalAttentionRouter(binding["binding_sha256"], query_chunk_rows, backend)
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        REGIONAL_WRAPPER_KEY,
        wrapper,
    )
    patched.set_model_optimized_attention(router)
    override = patched.model_options["transformer_options"]["optimized_attention_override"]
    override._t8_outpaint_regional_router = router
    override._t8_outpaint_regional_binding_sha256 = binding["binding_sha256"]
    patched.set_attachments(REGIONAL_WRAPPER_KEY, {
        "patch_version": REGIONAL_PATCH_VERSION,
        "binding": binding,
        "binding_sha256": binding["binding_sha256"],
        "query_chunk_rows": query_chunk_rows,
        "conditioning_manifest_sha256": conditioning_sha,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    return patched, regional_model_contract(patched)
