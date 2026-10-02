"""Source-bound FastH3 V2 recipe. Never changes the released four-step route."""
from __future__ import annotations

from .patch_stack_policy import warn_patch_stack

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
import math
from types import SimpleNamespace

import torch
from torch import nn

import comfy.patcher_extension as extension
import comfy.samplers

from . import sampling
from .h3_core_compat import plain_attention_backend

SCHEMA = "t8.fasth3_v2.recipe.v1"
KEY = "t8_fasth3_v2_owner_v1"
RUN_KEY = "t8_fasth3_v2_runtime_v1"
RUNG_STEPS = (999, 874, 749, 624, 500, 375, 250, 125)
VIDEO_SHIFT, AUDIO_SHIFT = 10.0, 3.0
PROFILES = ("trained_vsa_exp", "dense_compat_exp", "official_comfy_template_exp")
MODEL_REVISION = "0de92ab26fcb74ee47596332d93e55d20cddfd45"
MODEL_FILE = "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"
MODEL_BYTES = 22128378696
MODEL_SHA256 = "0922785978dc9bfe1adf27d8b291b0ca763f9f165f882e6cb297c72fbb6deda8"


def dmd_sigmas(shift=VIDEO_SHIFT):
    base = torch.tensor([v / 1000.0 for v in RUNG_STEPS] + [0.0], dtype=torch.float32)
    return sampling.shift_sigma(base, float(shift))


def _core_sparse():
    from comfy_extras import nodes_sparse_attention as sparse
    required = ("SparseAttnPatch", "install_override", "h3_eligible", "h3_sparse_attention")
    if not all(callable(getattr(sparse, name, None)) for name in required):
        raise RuntimeError("FastH3 V2 requires native H3 gate-aware chunked VSA support")
    params = inspect.signature(sparse.ck.sol_attn_chunked).parameters
    if not {"coarse_gate", "block_len", "sink_blocks"}.issubset(params):
        raise RuntimeError("FastH3 V2 requires Kitchen chunked VSA gate/block/sink interfaces")
    return sparse


def _gates(model):
    from comfy.ldm.minimax.model import MiniMaxH3Model
    diffusion = model.get_model_object("diffusion_model")
    if not isinstance(diffusion, MiniMaxH3Model):
        raise ValueError("FastH3 V2 requires native MiniMaxH3Model, not an OpenVDN architecture")
    if not diffusion.blocks:
        raise ValueError("FastH3 V2 has no live H3 blocks")
    for i, block in enumerate(diffusion.blocks):
        attn = block.attn
        gate = getattr(attn, "to_gate_compress", None)
        weight = getattr(gate, "weight", None)
        expected = (attn.heads * attn.head_dim, attn.qkv_proj.weight.shape[1])
        if not callable(gate) or weight is None or tuple(weight.shape) != expected:
            raise ValueError(f"FastH3 V2 missing/incompatible learned gate at block {i}")
        if attn.head_dim != 128:
            raise ValueError("FastH3 V2 VSA requires head_dim128")
    return diffusion


def _noise_scale(model):
    guider = getattr(model, "inner_model", None)
    base = getattr(guider, "inner_model", None)
    return float(getattr(getattr(base, "model_sampling", None), "noise_scale", 1.0))


def sample_v2_euler(model, x, sigmas, extra_args=None, callback=None, disable=None,
                    *, video_values, packed_values, stage_start=0, stage_end=8):
    expected = dmd_sigmas()[stage_start:stage_end + 1].to(sigmas.device)
    if not 0 <= stage_start < stage_end <= 8 or sigmas.shape != expected.shape or not torch.allclose(
        sigmas.float(), expected, rtol=0.0, atol=1e-6
    ):
        raise ValueError("FastH3 V2 needs its exact DMD rungs, not a uniform or reset schedule")
    if not sampling.model_uses_raw_audio_velocity(getattr(model, "inner_model", None)):
        # The setup checks the real MODEL. Direct callable CPU/reference models
        # intentionally have no guider; their contract is raw AV velocity.
        if getattr(model, "inner_model", None) is not None and not callable(
            getattr(getattr(model.inner_model, "inner_model", None), "audio_scale", None)
        ):
            raise RuntimeError("FastH3 V2 requires the current raw audio velocity protocol")
    if stage_start:
        return sampling.sample_minimax_h3_dual_clock_euler(
            model, x, sigmas, extra_args, callback, disable, video_values=video_values,
            packed_values=packed_values, shift_video=VIDEO_SHIFT, shift_audio=AUDIO_SHIFT,
            audio_velocity_is_raw=True,
        )
    noise = getattr(model, "noise", None)
    if not isinstance(noise, torch.Tensor) or noise.shape != x.shape:
        raise ValueError("FastH3 V2 first window requires explicitly injected target Gaussian noise")
    raw = noise.to(device=x.device, dtype=torch.float32) * _noise_scale(model)
    mask = (extra_args or {}).get("denoise_mask")
    if mask is None:
        x = raw
    else:
        if mask.shape != x.shape:
            raise ValueError("FastH3 V2 mask must match the packed AV tensor")
        # Only fully generated rows use the T2AV raw-noise recipe. Fractional
        # conditioning is explicitly EXP and keeps native initialization/masks.
        x = torch.where(mask >= 1.0 - 1e-7, raw, x.float())

    def call(current, sigma, **kwargs):
        return model(current, sigma, **kwargs)

    # Do not expose KSAMPLER.noise on this callable: old partial-audio rebasing
    # must NOT run a second time on the raw-noise first-window contract.
    return sampling.sample_minimax_h3_dual_clock_euler(
        call, x, sigmas, extra_args, callback, disable, video_values=video_values,
        packed_values=packed_values, shift_video=VIDEO_SHIFT, shift_audio=AUDIO_SHIFT,
        audio_velocity_is_raw=True,
    )


class _HeadProjection:
    def __init__(self, source, heads, dim, start, end, gate=False):
        self.source, self.heads, self.dim = source, heads, dim
        self.start, self.end, self.gate = start, end, gate

    def __call__(self, x):
        value = self.source(x)
        if self.gate:
            return value.view(-1, self.heads, self.dim)[:, self.start:self.end].reshape(
                value.shape[0], -1)
        return torch.cat(tuple(v.view(-1, self.heads, self.dim)[:, self.start:self.end].reshape(
            value.shape[0], -1) for v in value.chunk(3, dim=-1)), dim=-1)


class _V2Runtime:
    def __init__(self, profile, sparse, patch, head_chunks=1):
        self.profile, self.sparse, self.patch = profile, sparse, patch
        self.head_chunks = head_chunks
        self.counts, self.dense_reasons, self.steps = Counter(), Counter(), {}
        self.override, self.dit, self.guard, self.prepare, self.cleanup = None, {}, None, None, None
        self.previous_override = None
        self.previous_dit = {}
        self.dense_sol_backend = None
        self._dense_sol_frozen = None
        self._dense_sol_kernel = None
        self._dense_sol_fallback = None
        self._dense_sol_attention = None
        self._dense_sol_type = None
        self._dense_sol_report = None
        self._dense_sol_fallback_execution = None
        self.frozen_config = self._config()
        self.closed = True

    def _config(self):
        p = self.patch
        return (self.profile, self.head_chunks, None if p is None else (
            p.tau, p.topk_ratio, p.vsa, p.sigma_start, p.sigma_end, p.min_tokens,
            frozenset(p.dense_blocks), p.sink_conditioning, p.extra_tokens))

    def validate_options(self, options):
        if self._config() != self.frozen_config:
            raise RuntimeError("FastH3 V2 frozen recipe parameters were changed")
        if options.get("optimized_attention_override") is not self.override:
            warn_patch_stack('FastH3 V2 attention owner was replaced; no silent Dense fallback')
        dit = options.get("patches_replace", {}).get("dit", {})
        if set(dit) != set(self.dit) or any(dit[k] is not v for k, v in self.dit.items()):
            warn_patch_stack('FastH3 V2 DiT/VSA producer owner was replaced')
        backend = self.dense_sol_backend
        if backend is not None:
            # Authenticate executable objects before calling report(), which
            # can otherwise execute a replaced nested fallback method.
            if (type(backend) is not self._dense_sol_type
                    or backend.kernel is not self._dense_sol_kernel
                    or backend.fallback is not self._dense_sol_fallback
                    or type(backend).attention is not self._dense_sol_attention
                    or type(backend).report is not self._dense_sol_report
                    or any(name in vars(backend) for name in ("attention", "report"))
                    or self._fallback_execution(backend.fallback) != self._dense_sol_fallback_execution
                    or (backend.fallback is not None and any(name in vars(backend.fallback)
                        for name in ("attention", "_selected_attention", "report")))):
                raise RuntimeError("FastH3 V2 protected Sol backend was replaced")
            if self.dense_sol_contract() != self._dense_sol_frozen:
                raise RuntimeError("FastH3 V2 protected Sol backend was replaced")

    @staticmethod
    def _fallback_execution(fallback):
        if fallback is None:
            return None
        return (type(fallback), fallback.kernel, fallback.masked_kernel,
                type(fallback).attention, type(fallback)._selected_attention, type(fallback).report)

    def dense_sol_contract(self):
        """Inert cache/config identity, excluding actual-call counters and last layout."""
        if self.dense_sol_backend is None:
            return None
        report = deepcopy(self.dense_sol_backend.report())
        report.pop("completed_calls", None)
        report["h3_exact_prefix"].pop("last_block_range", None)
        if isinstance(report.get("fallback"), dict):
            report["fallback"].pop("completed_calls", None)
        return report

    def snapshot(self):
        return {"schema": SCHEMA, "profile": self.profile, "counts": dict(self.counts),
                "dense_reasons": dict(self.dense_reasons), "steps": self.steps,
                "head_chunks": self.head_chunks, "actual_vsa_dispatched": self.counts["vsa"] > 0,
                "attention_dispatch_observed": self.profile != "dense_compat_exp" or self.dense_sol_backend is not None,
                "dense_sol_backend": self.dense_sol_backend.report() if self.dense_sol_backend is not None else None,
                "dense_profile_note": "Preserved backend; actual backend dispatch requires separate instrumentation"
                    if self.profile == "dense_compat_exp" and self.dense_sol_backend is None else None,
                "quality_accepted": False, "performance_guarantee": False}

    def block_patch(self, block, index):
        def attention(h, rope_freqs=None, transformer_options=None):
            options = transformer_options or {}
            attn = block.attn
            if getattr(attn, "to_gate_compress", None) is None:
                raise RuntimeError(f"FastH3 V2 learned gate disappeared at block {index}")
            if self.head_chunks == 1:
                out = self.sparse.h3_sparse_attention(attn, h, rope_freqs, options, self.patch, index)
            else:
                outputs = []
                width = math.ceil(attn.heads / self.head_chunks)
                for start in range(0, attn.heads, width):
                    end = min(start + width, attn.heads)
                    view = SimpleNamespace(heads=end-start, head_dim=attn.head_dim,
                        q_norm=attn.q_norm, k_norm=attn.k_norm, out_proj=nn.Identity(),
                        qkv_proj=_HeadProjection(attn.qkv_proj, attn.heads, attn.head_dim, start, end),
                        to_gate_compress=_HeadProjection(attn.to_gate_compress, attn.heads,
                                                       attn.head_dim, start, end, gate=True))
                    outputs.append(self.sparse.h3_sparse_attention(
                        view, h, rope_freqs, options, self.patch, (index, start, end)))
                out = attn.out_proj(torch.cat(outputs, dim=-1))
            self.counts["vsa"] += 1
            return out

        def patch(args, extra):
            options = args["transformer_options"]
            self.validate_options(options)
            sigma = float(options.get("sigmas", [float("nan")])[0])
            step = self.steps.setdefault(str(sigma), {"vsa": 0, "dense": 0})
            if self.sparse.h3_eligible(block.attn, args["img"], args["rope_freqs"], options, self.patch, index):
                step["vsa"] += 1
                return extra["original_block"]({**args, "attention": attention})
            step["dense"] += 1
            self.counts["dense"] += 1
            reason = self.patch.dense_reason(options, args["img"].shape[0], index)
            reason = reason or "CUDA/BF16/head_dim/rope/kernel/layout eligibility"
            self.dense_reasons[reason] += 1
            return extra["original_block"](args)
        return patch


@dataclass(frozen=True)
class _V2Receipt:
    profile: str
    runtime: _V2Runtime
    schema: str = SCHEMA

    def on_model_patcher_clone(self):
        return self


def capture_fast_h3_v2_owner(model):
    getter = getattr(model, "get_attachment", None)
    receipt = getter(KEY) if callable(getter) else getattr(model, "attachments", {}).get(KEY)
    if receipt is None:
        return None
    if not isinstance(receipt, _V2Receipt) or receipt.schema != SCHEMA or receipt.profile not in PROFILES:
        raise RuntimeError("FastH3 V2 receipt is foreign or malformed")
    options = model.model_options["transformer_options"]
    if options.get(RUN_KEY) is not receipt.runtime or receipt.profile != receipt.runtime.profile:
        raise RuntimeError("FastH3 V2 runtime token was replaced")
    receipt.runtime.validate_options(options)
    if receipt.profile != "dense_compat_exp" or receipt.runtime.dense_sol_backend is not None:
        if model.get_wrappers("diffusion_model", KEY) != [receipt.runtime.guard]:
            raise RuntimeError("FastH3 V2 wrapper owner was replaced")
        for role, function in ((extension.CallbacksMP.ON_PREPARE_STATE, receipt.runtime.prepare),
                               (extension.CallbacksMP.ON_CLEANUP, receipt.runtime.cleanup)):
            if model.callbacks.get(role, {}).get(KEY) != [function]:
                raise RuntimeError("FastH3 V2 callback owner was replaced")
    if receipt.runtime.dense_sol_backend is not None:
        from .relay_sol_backend import capture_sol_relay_backend
        from .vdn_attention_compat import _factory_closure
        runtime = receipt.runtime
        for function, name in ((runtime.override, "sol_override"), (runtime.guard, "guard"),
                               (runtime.prepare, "prepare"), (runtime.cleanup, "cleanup")):
            state = _factory_closure(function, _install_runtime, name)
            if state is None or state.get("runtime") is not runtime:
                raise RuntimeError("FastH3 V2 protected Sol execution factory was replaced")
        backend = capture_sol_relay_backend(receipt.runtime.previous_override)
        if (backend is None or backend.kernel is not receipt.runtime._dense_sol_kernel
                or backend.config != receipt.runtime.dense_sol_backend.config
                or (backend.fallback is None) != (runtime._dense_sol_fallback is None)
                or (backend.fallback is not None and (
                    type(backend.fallback) is not type(runtime._dense_sol_fallback)
                    or backend.fallback.kernel is not runtime._dense_sol_fallback.kernel
                    or (backend.fallback.masked_kernel is None) != (runtime._dense_sol_fallback.masked_kernel is None)))):
            raise RuntimeError("FastH3 V2 original Sol source/configuration was replaced")
    return receipt


def _install_runtime(model, profile, min_tokens):
    from .h3_memory_advanced import inspect_t8_memory_composition
    from .prompt_relay_advanced import PROMPT_RELAY_WRAPPER_KEY, prompt_relay_model_contract
    allowed_wrappers = ()
    if model.get_wrappers("diffusion_model", PROMPT_RELAY_WRAPPER_KEY):
        if profile != "dense_compat_exp":
            warn_patch_stack('FastH3 V2 Relay requires explicit dense_compat_exp; native VSA cannot express query-varying timeline bias')
        prompt_relay_model_contract(model)  # Authenticate, never trust a public marker.
        allowed_wrappers = (PROMPT_RELAY_WRAPPER_KEY,)
    memory = inspect_t8_memory_composition(model, allowed_wrapper_keys=allowed_wrappers) or {}
    sparse = _core_sparse() if profile != "dense_compat_exp" else None
    patch = None if sparse is None else sparse.SparseAttnPatch(
        tau=1.0, topk_ratio=0.2 if profile == "trained_vsa_exp" else 0.1, vsa=True,
        sigma_start=1.0 if profile == "trained_vsa_exp" else float(
            model.get_model_object("model_sampling").percent_to_sigma(0.2)),
        sigma_end=0.0, min_tokens=int(min_tokens), dense_blocks=set(),
        sink_conditioning="exact_kv_and_rows", extra_tokens=0, verbose=False)
    runtime = _V2Runtime(profile, sparse, patch, memory.get("head_chunks", 1))
    m = model.clone()
    options = m.model_options["transformer_options"]
    options[RUN_KEY] = runtime
    runtime.override = options.get("optimized_attention_override")
    runtime.dit = dict(options.get("patches_replace", {}).get("dit", {}))
    runtime.previous_dit = dict(runtime.dit)
    if sparse is not None:
        override = options.get("optimized_attention_override")
        if override is not None and plain_attention_backend(override) is None:
            warn_patch_stack('FastH3 V2: another attention algorithm owns this MODEL branch')
        if options.get("patches_replace", {}).get("dit"):
            warn_patch_stack('FastH3 V2: another DiT producer owns this MODEL branch')
        sparse.install_override(patch, options)
        runtime.override = options["optimized_attention_override"]
        from .patch_stack_policy import compose_dit_hook
        for i, block in enumerate(_gates(m).blocks):
            current = runtime.block_patch(block, i)
            previous = options.get("patches_replace", {}).get("dit", {}).get(("double_block", i))
            m.set_model_patch_replace(compose_dit_hook(previous, current, "FastH3 V2"), "dit", "double_block", i)
        # Native ModelPatcher replaces transformer_options using copy-on-write.
        # Bind the final dictionary, not the pre-install local reference.
        options = m.model_options["transformer_options"]
        runtime.dit = dict(options["patches_replace"]["dit"])

    elif not allowed_wrappers:
        # The external generic Sol selector sparsifies packed audio/text too.
        # Use only its authenticated kernel/settings with the existing H3
        # exact-Q and exact-KV prefix adapter; do not normalize decoded audio.
        # Relay already owns an explicit bias-preserving adapter and is left
        # unchanged, as are ordinary PyTorch/KJ/unknown Dense overrides.
        from .relay_sol_backend import capture_sol_relay_backend
        backend = capture_sol_relay_backend(runtime.override)
        if backend is not None:
            runtime.previous_override = runtime.override
            runtime.dense_sol_backend = backend
            runtime._dense_sol_frozen = runtime.dense_sol_contract()
            runtime._dense_sol_kernel = backend.kernel
            runtime._dense_sol_fallback = backend.fallback
            runtime._dense_sol_attention = type(backend).attention
            runtime._dense_sol_type = type(backend)
            runtime._dense_sol_report = type(backend).report
            runtime._dense_sol_fallback_execution = runtime._fallback_execution(backend.fallback)

            def sol_override(func, q, k, v, heads, *args, **kwargs):
                if args:
                    raise ValueError("FastH3 V2 protected Sol requires named attention options")
                runtime.validate_options(kwargs.get("transformer_options") or {})
                return runtime.dense_sol_backend.attention(q, k, v, heads, **kwargs)

            options["optimized_attention_override"] = sol_override
            runtime.override = sol_override

    if sparse is not None or runtime.dense_sol_backend is not None:
        def guard(executor, *args, **kwargs):
            live = kwargs.get("transformer_options", args[3] if len(args) >= 4 else None)
            if not isinstance(live, dict):
                raise RuntimeError("FastH3 V2 runtime transformer_options missing")
            runtime.validate_options(live)
            return executor(*args, **kwargs)

        def prepare(patcher, timestep, model_options):
            runtime.validate_options(model_options["transformer_options"])
            if runtime.closed:
                runtime.counts.clear()
                runtime.dense_reasons.clear()
                runtime.steps.clear()
                if patch is not None:
                    patch.reset()
                if runtime.dense_sol_backend is not None:
                    runtime.dense_sol_backend.counters.clear()
                    runtime.dense_sol_backend.last_exact_prefix = None
                    if runtime.dense_sol_backend.fallback is not None:
                        runtime.dense_sol_backend.fallback.counters.clear()
                runtime.closed = False

        def cleanup(patcher):
            if patch is not None:
                patch.reset()  # Keep counters after cleanup for the downstream audit.
            runtime.closed = True

        runtime.guard, runtime.prepare, runtime.cleanup = guard, prepare, cleanup
        m.add_wrapper_with_key("diffusion_model", KEY, guard)
        m.add_callback_with_key(extension.CallbacksMP.ON_PREPARE_STATE, KEY, prepare)
        m.add_callback_with_key(extension.CallbacksMP.ON_CLEANUP, KEY, cleanup)
    m.set_attachments(KEY, _V2Receipt(profile, runtime))
    return m


def refresh_fast_h3_v2_memory(model):
    receipt = capture_fast_h3_v2_owner(model)
    if receipt is None:
        return model
    runtime = receipt.runtime
    live = model.model_options["transformer_options"]
    dit = live.get("patches_replace", {}).get("dit", {})
    if (live.get("optimized_attention_override") is not runtime.override
            or set(dit) != set(runtime.dit)
            or any(dit[key] is not function for key, function in runtime.dit.items())
            or (receipt.profile != "dense_compat_exp" and runtime.previous_dit)):
        # The clean native path can be rebound to new head groups. An opaque
        # composed/replaced producer cannot be removed and reconstructed safely:
        # retain its exact executable owners instead of silently discarding it.
        warn_patch_stack("FastH3 V2 memory refresh bypassed to preserve user-selected owners; "
                         "VSA head grouping is not rebound on this branch")
        return model
    m = model.clone()
    options = m.model_options["transformer_options"]
    if receipt.profile != "dense_compat_exp":
        options.pop("optimized_attention_override", None)
        # Restore the original, authenticated plain backend if one existed.
        cells = dict(zip(runtime.override.__code__.co_freevars, runtime.override.__closure__ or ()))
        previous = cells.get("previous")
        if previous is not None and previous.cell_contents is not None:
            options["optimized_attention_override"] = previous.cell_contents
        replaces = dict(options.get("patches_replace", {}))
        replaces.pop("dit", None)
        options["patches_replace"] = replaces
    elif runtime.dense_sol_backend is not None:
        options["optimized_attention_override"] = runtime.previous_override
    for attr in ("wrappers", "callbacks"):
        container = getattr(m, attr)
        setattr(m, attr, {role: {key: value for key, value in values.items() if key != KEY}
                          for role, values in container.items()})
    m.remove_attachments(KEY)
    options.pop(RUN_KEY, None)
    return _install_runtime(m, receipt.profile, 12288 if runtime.patch is None else runtime.patch.min_tokens)


def build_fast_h3_v2_setup(model, av_latent, profile="trained_vsa_exp", min_tokens=12288):
    if profile not in PROFILES or not 0 <= int(min_tokens) <= 1048576:
        raise ValueError("Unknown FastH3 V2 profile or invalid min_tokens")
    _gates(model)
    if capture_fast_h3_v2_owner(model) is not None:
        raise ValueError("FastH3 V2 is already installed on this branch")
    if not sampling.model_uses_raw_audio_velocity(model):
        raise RuntimeError("FastH3 V2 requires native raw audio velocity support")
    m, _, _ = sampling.setup_dual_clock_sampling(model, av_latent, 8, VIDEO_SHIFT, AUDIO_SHIFT)
    m = _install_runtime(m, profile, min_tokens)
    video, audio = sampling.nested_av_parts(av_latent)
    video_values = math.prod(video.shape[1:])
    packed_values = video_values + math.prod(audio.shape[1:])
    if profile == "official_comfy_template_exp":
        m, sampler, sigmas = sampling.setup_dual_clock_sampling(
            m, av_latent, 8, VIDEO_SHIFT, AUDIO_SHIFT, sampler_name="res_multistep", scheduler="simple")
    else:
        sampler = comfy.samplers.KSAMPLER(sample_v2_euler, extra_options={
            "video_values": video_values, "packed_values": packed_values})
        sigmas = dmd_sigmas()
    report = {"schema": SCHEMA, "profile": profile, "nfe": 8,
        "video_shift": VIDEO_SHIFT, "audio_shift": AUDIO_SHIFT,
        "rungs": list(RUNG_STEPS) if profile != "official_comfy_template_exp" else "Core simple8",
        "trained_scope": "T2AV only; FL2VA/Ref2VA and split 4+upscale+4 are EXP",
        "vsa_required": profile != "dense_compat_exp", "min_tokens": int(min_tokens),
        "model_revision": MODEL_REVISION, "model_file_sha256": MODEL_SHA256,
        "model_validation": "live H3 and gate structure; content provenance must be separately audited",
        "dense_sol_protection": capture_fast_h3_v2_owner(m).runtime.dense_sol_contract(),
        "warning": "Template differs from trained recipe. VSA eligibility failures are counted; "
                   "Dense profile preserves its backend and needs separate dispatch instrumentation. No quality/16GB/speed guarantee.",
        "source_model_unchanged": True}
    return m, sampler, sigmas, json.dumps(report, ensure_ascii=False, indent=2)
