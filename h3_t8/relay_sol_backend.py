"""Scoped adapter for the audited ComfyUI-sol-attn selector (not SolAttn_triton).

Relay splits queries and adds a float bias. Sol's equal-shape unmasked kernel
cannot implement that equation. Those calls retain a declared dense/Sage
delegate, never lose the bias, and never count as Sol execution.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import inspect
import math
from pathlib import Path
import sys
from types import FunctionType

import torch
from comfy.ldm.modules import attention as core_attention

from .relay_kj_backend import _ast_hash, _cells, _codes, _live_code_matches, capture_kj_relay_backend

_DEFINITIONS = {
    "_make_override": "5c016846ef21b5f909b4a9b4625264a7d01b0465693b4a420628125798f06a35",
    "SolAttentionPatch": "cb935d383664695210bcab6764114f34b241681f508b424f7a121606d35caf06",
}
_KERNEL_FILES = {
    "fwd.py": "4244f2fad835ab7e1de79d8d9bbeebc4b1c64f12b83aac3be8b027275c848a16",
    "preprocess.py": "9c93016927885b9cf39ade0e642c93ccddc48476577e90df12dd8f39ae563141",
    "quant.py": "8fec174fd49e667dbe8b1fdc899de22213444bd497614316bea2ae3f7322db95",
}
_ARCHES = {(8, 6), (8, 9), (9, 0), (10, 0), (12, 0), (12, 1)}


def h3_exact_prefix(layout, tokens):
    """Protect packed text/reference/audio queries and keys, including mixed tiles.

    Use Core's actual layout, never infer token boundaries from tensor length.
    Unsupported/missing layouts must use the declared dense delegate.
    """
    segments = getattr(layout, 'segments', None)
    if not segments or getattr(layout, 'seq_len', None) != tokens:
        return None
    end = 0
    kinds = []
    for segment in segments:
        if not isinstance(segment, (tuple, list)) or len(segment) != 3:
            return None
        start, stop, kind = segment
        if (type(start) is not int or type(stop) is not int or start != end
                or stop < start or stop > tokens
                or kind not in {'text', 'cond', 'ref_img', 'cond_audio', 'ref_audio', 'audio', 'video'}):
            return None
        end = stop
        kinds.append(kind)
    if (end != tokens or kinds[0] != 'text' or kinds[-2:] != ['audio', 'video']
            or kinds.count('video') != 1 or kinds.count('audio') != 1):
        return None
    return (0, (segments[-1][0] + 63) // 64)


def _verify_kernel(kernel, directory):
    if not isinstance(kernel, FunctionType):
        return False
    path = Path(inspect.getsourcefile(kernel)).resolve()
    if path != (directory / "sol_kernel/fwd.py").resolve():
        return False
    for name, digest in _KERNEL_FILES.items():
        if hashlib.sha256((path.parent / name).read_bytes()).hexdigest() != digest:
            return False
    module = sys.modules.get(kernel.__module__)
    return (module is not None and getattr(module, "sol_attn", None) is kernel
            and getattr(module, "torch", None) is torch
            and _live_code_matches(kernel, _codes(compile(path.read_bytes(), str(path), "exec", dont_inherit=True)), module))


class SolRelayBackend:
    def __init__(self, kernel, config, source_sha256, fallback=None):
        self.kernel = kernel
        self.config = dict(config)
        self.source_sha256 = source_sha256
        self.fallback = fallback
        self.counters = Counter()

    def report(self):
        return {"kind": "audited_sol_attn_selector", "configuration": self.config,
                "source_sha256": self.source_sha256, "kernel_source_sha256s": dict(_KERNEL_FILES),
                "completed_calls": dict(self.counters),
                "h3_exact_prefix": {"policy": "packed_nonvideo_q_and_kv_exact_v1",
                    "last_block_range": getattr(self, 'last_exact_prefix', None)},
                "fallback": self.fallback.report() if self.fallback else "pytorch_sdpa",
                "relay_policy": "float bias or query/KV mismatch cannot use this Sol kernel",
                "error_policy": "kernel errors and cancellation propagate; no hidden retry",
                "measurement_scope": "actual completed kernel calls, not sparse-block counts or speed"}

    def attention(self, q, k, v, heads, *, mask=None, skip_reshape=False,
                  skip_output_reshape=False, **kwargs):
        reason = None
        options = kwargs.get('transformer_options') or {}
        exact_prefix = h3_exact_prefix(options.get('minimax_h3_layout'), q.shape[2]) if q.ndim == 4 else None
        if mask is not None:
            reason = "relay_or_other_mask"
        elif not skip_reshape or q.ndim != 4 or q.shape[1] != heads:
            reason = "not_hnd"
        elif q.shape != k.shape or q.shape != v.shape:
            reason = "query_kv_shape_mismatch"
        elif q.shape[-1] != 128:
            reason = "head_dimension"
        elif any(value.dtype != torch.bfloat16 for value in (q, k, v)):
            reason = "not_bfloat16"
        elif q.device.type != "cuda" or k.device != q.device or v.device != q.device:
            reason = "not_shared_cuda"
        elif any(value.requires_grad for value in (q, k, v)):
            reason = "autograd"
        elif kwargs.get("low_precision_attention", True) is False or kwargs.get("enable_gqa", False):
            reason = "precision_or_gqa"
        elif q.shape[2] < self.config["min_tokens"]:
            reason = "below_min_tokens"
        elif torch.cuda.get_device_capability(q.device) not in _ARCHES:
            reason = "unsupported_architecture"
        elif exact_prefix is None:
            reason = "missing_or_unsupported_h3_layout"
        if reason:
            common = dict(mask=mask, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
            if self.fallback is not None:
                result = self.fallback.attention(q, k, v, heads, **common)
            else:
                result = core_attention.attention_pytorch(q, k, v, heads,
                    **{**common, "_inside_attn_wrapper": True})
            self.counters["fallback:" + reason] += 1
            return result
        # Deliberately bypass upstream's broad catch/strict=False retry. A failed
        # kernel must not become an apparently successful fast generation.
        result = self.kernel(*(value.transpose(1, 2).contiguous() for value in (q, k, v)),
                             scale=kwargs.get("scale"), tau=self.config["tau"],
                             thresh_type=self.config["thresh_type"],
                             int8_qk=self.config["int8_qk"], int8_pv=self.config["int8_pv"],
                             sink_blocks=exact_prefix, sink_q=exact_prefix)
        self.counters["sol:completed"] += 1
        self.last_exact_prefix = exact_prefix
        return result.transpose(1, 2) if skip_output_reshape else result.reshape(q.shape[0], q.shape[2], -1)


def capture_sol_relay_backend(override):
    if not isinstance(override, FunctionType):
        return None
    cells = _cells(override)
    if set(cells) != {"tau", "min_tokens", "strict", "fallback_override", "thresh_type", "int8_qk", "int8_pv", "dispatch_log"}:
        return None
    try:
        module = sys.modules.get(override.__module__)
        if module is None or override.__globals__ is not vars(module) or module.torch is not torch:
            return None
        path = Path(inspect.getsourcefile(override))
        payload = path.read_bytes()
        definitions = {node.name: node for node in ast.parse(payload).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _DEFINITIONS}
        if set(definitions) != set(_DEFINITIONS) or any(_ast_hash(node) != _DEFINITIONS[name] for name, node in definitions.items()):
            return None
        if not _live_code_matches(override, _codes(compile(payload, str(path), "exec", dont_inherit=True)), module):
            return None
        config = {key: cells[key] for key in ("tau", "min_tokens", "strict", "thresh_type", "int8_qk", "int8_pv")}
        if (type(config["tau"]) is not float or not math.isfinite(config["tau"]) or not 0 <= config["tau"] <= 4
                or type(config["min_tokens"]) is not int or not 256 <= config["min_tokens"] <= 131072
                or any(type(config[key]) is not bool for key in ("strict", "int8_qk", "int8_pv"))
                or config["thresh_type"] not in ("diag", "exact")
                or not _verify_kernel(module.sol_attn, path.parent)):
            return None
        previous = cells["fallback_override"]
        fallback = capture_kj_relay_backend(previous)
        if previous is not None and fallback is None:
            return None  # Do not invoke or silently discard unknown chained patches.
        return SolRelayBackend(module.sol_attn, config, hashlib.sha256(payload).hexdigest(), fallback)
    except (OSError, TypeError, ValueError, SyntaxError, AttributeError):
        return None


def capture_composed_backend(override):
    backend = capture_kj_relay_backend(override) or capture_sol_relay_backend(override)
    if backend is not None or override is None:
        return backend
    from .h3_core_compat import plain_attention_backend
    if plain_attention_backend(override) is not None:
        return None
    if not callable(override):
        raise TypeError("optimized_attention_override must be callable")
    from .patch_stack_policy import warn_patch_stack
    warn_patch_stack("Unrecognized attention override retained as a user-selected delegate")
    return UserSelectedBackend(override)


class UserSelectedBackend:
    """Use Core's override protocol as-is; do not retry or remove Relay bias."""
    def __init__(self, override):
        self.override = override
        self.counters = Counter()

    def attention(self, q, k, v, heads, **kwargs):
        output = self.override(core_attention.optimized_attention, q, k, v, heads, **kwargs)
        self.counters["delegate:completed"] += 1
        return output

    def report(self):
        return {"kind": "user_selected_unverified_delegate",
                "portable_cache_reuse": False,
                "completed_calls": dict(self.counters),
                "kernel_verified": False}
