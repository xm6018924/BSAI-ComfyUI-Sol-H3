"""Compose audited KJ H3 memory forwards with Relay without bypassing its owner.

Projection/RMSNorm/RoPE and early input release remain in the installed KJ
forward. Head grouping moves *inside* the attention delegate, so Relay and EAV
see one full-head block call. No global KJ/Sage function is modified.
"""
from __future__ import annotations

from .patch_stack_policy import advisory_inspection

import ast
from collections import Counter
import hashlib
import inspect
from pathlib import Path
import sys
from types import FunctionType, MethodType

import torch
import comfy.model_management as mm
import comfy.quant_ops
from comfy.ldm.modules import attention as core_attention

from .relay_kj_backend import KJRelayBackend, _audited_mask_kernel, _ast_hash, _codes, _live_code_matches
from .vdn_attention_compat import _factory_closure
from .video_outpaint_model_patches import _verified_module


_SAGE_FORWARD_AST = "c24f4c87d63e25321ed2587e1241e10aee3f7965c15b905ece132d629fd1573a"
MEMORY_TOKEN_KEY = "t8_relay_memory_owner_token"
PROGRESSIVE_MEMORY_RUNTIME_KEY = 't8_progressive_memory_runtime'


def _verify_sage_forward(function):
    if not isinstance(function, FunctionType):
        raise ValueError("KJ H3 memory forward must be a Python function")
    module = sys.modules.get(function.__module__)
    if module is None or function.__globals__ is not vars(module):
        raise ValueError("KJ H3 memory forward is detached from its module")
    path = inspect.getsourcefile(function)
    payload = Path(path).read_bytes()
    definitions = [node for node in ast.parse(payload).body
                   if isinstance(node, ast.FunctionDef) and node.name == "minimax_sageattn_forward"]
    if (len(definitions) != 1 or _ast_hash(definitions[0]) != _SAGE_FORWARD_AST
            or not _live_code_matches(function, _codes(compile(payload, path, "exec", dont_inherit=True)), module)
            or module.mm is not mm or module._ck is not comfy.quant_ops.ck or module.torch is not torch):
        raise ValueError("KJ H3 memory Sage implementation changed; compatibility audit required")
    return hashlib.sha256(payload).hexdigest()


@advisory_inspection
def inspect_memory_composition(model):
    """Authenticate complete KJ sets, including our own composed methods."""
    patches = {key: value for key, value in getattr(model, "object_patches", {}).items()
               if key.startswith("diffusion_model.blocks.")}
    if not patches:
        return None
    blocks = list(model.model.diffusion_model.blocks)
    if not blocks:
        raise ValueError("KJ memory composition requires H3 blocks")
    attention_present = "diffusion_model.blocks.0.attn.forward" in patches
    block_present = "diffusion_model.blocks.0.forward" in patches
    ffn_present = "diffusion_model.blocks.0.mlp.forward" in patches
    expected = {}
    raw_forwards = {}
    adapted_backend = None
    kind = None
    sources = set()
    ffn_settings = None
    lowmem_module = None
    for index, block in enumerate(blocks):
        prefix = f"diffusion_model.blocks.{index}"
        roles = []
        if attention_present:
            roles.append((prefix + ".attn.forward", block.attn, "attention"))
        if block_present:
            roles.append((prefix + ".forward", block, "block"))
        if ffn_present:
            roles.append((prefix + ".mlp.forward", block.mlp, "ffn"))
        for path, owner, role in roles:
            method = patches.get(path)
            if not isinstance(method, MethodType) or method.__self__ is not owner:
                raise ValueError("KJ memory requires complete patches bound to the matching H3 blocks")
            expected[path] = method
            original = method
            composed = _factory_closure(method.__func__, _make_memory_forward, "forward")
            if composed is not None:
                original = composed.get("original")
                backend = composed.get("backend")
                if (role != "attention" or not isinstance(original, MethodType)
                        or original.__self__ is not owner or type(backend) is not HeadGroupedBackend):
                    raise ValueError("Invalid composed KJ memory owner")
                if adapted_backend is not None and backend is not adapted_backend:
                    raise ValueError("KJ memory blocks refer to different attention owners")
                adapted_backend = backend
            name = original.__func__.__name__
            if role == "attention" and name == "minimax_sageattn_forward":
                digest = _verify_sage_forward(original.__func__)
                observed_kind = "kj_memory_sage"
            else:
                module, check, contract = _verified_module(original.__func__)
                lowmem_module = module
                expected_name = {"attention": "minimax_attn_lowmem_forward",
                                 "block": "minimax_block_lowmem_forward", "ffn": "wrapped_forward"}[role]
                check(original.__func__, expected_name)
                digest = contract["source_sha256"]
                observed_kind = "kj_low_vram"
                if role == "ffn":
                    cells = inspect.getclosurevars(original.__func__).nonlocals
                    patch = cells.get("self")
                    if set(cells) != {"self"} or type(patch) is not module.MiniMaxFFNChunkPatch:
                        raise ValueError("Unknown KJ FFN closure")
                    config = (patch.num_chunks, patch.seq_threshold)
                    if (any(type(value) is not int for value in config)
                            or not 2 <= config[0] <= 64 or not 256 <= config[1] <= 262144
                            or config[1] % 256 or (ffn_settings is not None and config != ffn_settings)):
                        raise ValueError("KJ FFN settings are invalid or inconsistent between blocks")
                    ffn_settings = config
            sources.add(digest)
            if role == "attention":
                if kind is not None and kind != observed_kind:
                    raise ValueError("KJ attention implementations differ between blocks")
                kind = observed_kind
                raw_forwards[path] = original
    if set(expected) != set(patches):
        raise ValueError("Unknown or partial H3 block patches cannot be composed with Relay")
    options = model.model_options.get("transformer_options", {})
    groups = options.get("minimax_head_chunks", 1)
    if type(groups) is not int or not 1 <= groups <= 56:
        raise ValueError("KJ head_chunks must be an integer in 1..56")
    if "sol_take_forward" in options and (lowmem_module is None
            or options["sol_take_forward"] is not lowmem_module.minimax_attn_lowmem_forward):
        raise ValueError("KJ low-memory Sol compose forward is not authentic")
    if adapted_backend is not None:
        if adapted_backend.head_chunks != groups:
            raise ValueError("KJ head grouping changed after Relay binding")
        for path in raw_forwards:
            state = _factory_closure(expected[path].__func__, _make_memory_forward, "forward")
            if state is None or state.get("backend") is not adapted_backend:
                raise ValueError("Only part of the KJ memory set has been composed")
    return {"kind": kind or "kj_ffn_only", "head_chunks": groups,
            "ffn_settings": ffn_settings, "source_sha256s": sorted(sources),
            "methods": expected, "raw_forwards": raw_forwards, "backend": adapted_backend}


class _NativeDelegate:
    def __init__(self):
        self.counters = Counter()

    def attention(self, q, k, v, heads, *, mask=None, **kwargs):
        target = core_attention.optimized_attention if mask is None else core_attention.attention_pytorch
        output = target(q, k, v, heads, mask=mask, **{**kwargs, "_inside_attn_wrapper": True})
        self.counters["native_optimized" if mask is None else "pytorch:bias"] += 1
        return output

    def report(self):
        return {"kind": "native_global_delegate", "completed_calls": dict(self.counters),
                "kernel_proof": "native global wrapper may fallback; not a Sage execution count"}


class HeadGroupedBackend:
    def __init__(self, delegate, head_chunks, contract):
        self.delegate = delegate
        self.head_chunks = head_chunks
        self.runtime_token = object()
        self.expected_methods = {}
        self.contract = {key: contract[key] for key in ("kind", "head_chunks", "ffn_settings", "source_sha256s")}

    def report(self):
        return {**self.delegate.report(), "memory_composition": self.contract,
                "head_grouping": "inside_delegate; full-head Relay/EAV once per block",
                "memory_kernel_policy": "attention uses the reported delegate; KJ projections retained; no KJ in-place K centering"}

    def attention(self, q, k, v, heads, *, mask=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        if not skip_reshape or q.ndim != 4 or q.shape[1] != heads or k.shape[1] != heads or v.shape[1] != heads:
            raise RuntimeError("KJ H3 grouped Relay requires full HND tensors with matching heads")
        groups = min(self.head_chunks, heads)
        dim = q.shape[-1]
        output = torch.empty((q.shape[0], q.shape[2], heads * dim), dtype=v.dtype, device=v.device)
        start = 0
        for group in range(groups):
            end = start + heads // groups + (group < heads % groups)
            group_mask = mask
            if mask is not None and mask.ndim == 4 and mask.shape[1] == heads:
                group_mask = mask[:, start:end]
            part = self.delegate.attention(q[:, start:end], k[:, start:end], v[:, start:end], end - start,
                                           mask=group_mask, skip_reshape=True, skip_output_reshape=False, **kwargs)
            output[:, :, start * dim:end * dim] = part
            start = end
        if skip_output_reshape:
            return output.reshape(q.shape[0], q.shape[2], heads, dim).transpose(1, 2)
        return output


def bind_memory_runtime(backend, route):
    if type(backend) is HeadGroupedBackend:
        for path, expected in backend.expected_methods.items():
            current = getattr(expected.__self__, "forward", None)
            if (not isinstance(current, MethodType) or current.__self__ is not expected.__self__
                    or current.__func__ is not expected.__func__):
                from .patch_stack_policy import warn_patch_stack
                warn_patch_stack(f"KJ memory forward has a later user-selected owner: {path}")
        route[MEMORY_TOKEN_KEY] = backend.runtime_token


def _make_memory_forward(original, backend):
    def forward(self, x, rope_freqs=None, transformer_options=None):
        from .prompt_relay_advanced import PROMPT_RELAY_RUNTIME_KEY
        from .enhance_a_video_advanced import EAV_RUNTIME_KEY
        options = transformer_options or {}
        route = options.get(PROMPT_RELAY_RUNTIME_KEY,
                            options.get(EAV_RUNTIME_KEY, options.get(PROGRESSIVE_MEMORY_RUNTIME_KEY, {})))
        if route.get(MEMORY_TOKEN_KEY) is not backend.runtime_token:
            raise RuntimeError("KJ memory forward requires its paired Relay runtime owner")
        local_options = {**options, "minimax_head_chunks": 1}

        def routed_nhd(qkv, dtype):
            q, k, v = qkv
            qkv.clear()
            heads = q.shape[2]
            out = core_attention.optimized_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                    heads, skip_reshape=True, transformer_options=options)
            return out.reshape(q.shape[0], q.shape[1], heads, q.shape[3]).to(dtype)

        def routed_hnd(*args, **kwargs):
            return core_attention.optimized_attention(*args, **{**kwargs, "transformer_options": options})

        function = original.__func__
        local_globals = {**function.__globals__, "_sageattn_int8_fp8_nhd": routed_nhd,
                         "optimized_attention": routed_hnd}
        scoped = FunctionType(function.__code__, local_globals, function.__name__, function.__defaults__, function.__closure__)
        return scoped(self, x, rope_freqs=rope_freqs, transformer_options=local_options)
    return MethodType(forward, original.__self__)


def adapt_memory_for_relay(model, selected_backend, *, allow_existing=False):
    contract = inspect_memory_composition(model)
    if contract is None or not contract["raw_forwards"]:
        return model, selected_backend
    if contract["backend"] is not None:
        if allow_existing and selected_backend is None:
            return model, contract["backend"]
        from .patch_stack_policy import warn_patch_stack
        warn_patch_stack("KJ memory was already bound; retaining its existing runtime delegate")
        return model, contract["backend"]
    delegate = selected_backend
    if delegate is None and contract["kind"] == "kj_memory_sage":
        package = sys.modules.get("sageattention")
        kernel = getattr(package, "sageattn", None)
        if not callable(kernel):
            raise RuntimeError("KJ H3 memory Sage requires its already-installed sageattention package")
        masked = None
        candidate = getattr(package, "sageattn_qk_int8_pv_fp16_triton", None)
        if _audited_mask_kernel(candidate):
            from .scoped_sage_triton import build_scoped_mask_kernel
            masked = build_scoped_mask_kernel(candidate)
        delegate = KJRelayBackend("auto", kernel, contract["source_sha256s"][0], masked is not None, masked)
    backend = HeadGroupedBackend(delegate or _NativeDelegate(), contract["head_chunks"], contract)
    patched = model.clone()
    for path, original in contract["raw_forwards"].items():
        patched.add_object_patch(path, _make_memory_forward(original, backend))
    backend.expected_methods = {path: patched.object_patches[path] for path in contract["methods"]}
    return patched, backend
