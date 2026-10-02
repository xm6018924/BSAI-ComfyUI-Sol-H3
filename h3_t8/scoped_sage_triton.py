"""Use installed Sage's audited JIT with scoped launch settings, no global patch.

The audited implementation accumulates logits in log2 units but adds its float
mask verbatim. Relay/SDPA biases use natural-log units. The adapter converts the
bias, not the model's Q/K quantization or attention algorithm. The same JIT is
launched with two pipeline stages to fit Ada's per-block shared memory limit.
No upstream source, weights, binary, or runtime is redistributed here.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import sys
from types import CodeType, FunctionType

import torch


_AUDITED_AST = {
    "forward": "61679e4fd6062f16a89d913935eb98f9d0308cd986f9ab32408c8d0049f46f7e",
    "_attn_fwd": "8721d631449c82153f689cdaf66e3996a6f3df9d710a4aac309d6e2b8fc382fa",
    "_attn_fwd_inner": "68000c7d6c9dd59e3f9037e1e8b8132c0554fd1836bf904cba5fdfcad9c0a74f",
}


def _codes(code):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            yield from _codes(constant)


def _clone_function(function, globals_copy):
    cloned = FunctionType(function.__code__, globals_copy, function.__name__,
                          function.__defaults__, function.__closure__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    return cloned


def _natural_bias_to_log2(mask, dtype):
    if mask.dtype == torch.bool:
        return mask
    return (mask.float() * 1.4426950408889634).to(dtype)


class _TwoStageLaunch:
    def __init__(self, jit):
        self.jit = jit

    def __getitem__(self, grid):
        launch = self.jit[grid]

        def scoped_launch(*args, **kwargs):
            # All strides, quantization tiles and numerical arguments stay as
            # supplied by the original forward. Only compiler pipelining changes.
            return launch(*args, **{**kwargs, "num_stages": 2})
        return scoped_launch


def build_scoped_mask_kernel(kernel):
    """Called only after authenticating the outer Sage kernel's live code."""
    try:
        forward = kernel.__globals__["attn_false"]
        module = sys.modules.get(forward.__module__)
        if not isinstance(forward, FunctionType) or module is None or forward.__globals__ is not vars(module):
            return None
        path = inspect.getsourcefile(forward)
        payload = Path(path).read_bytes()
        definitions = {node.name: node for node in ast.parse(payload).body
                       if isinstance(node, ast.FunctionDef) and node.name in _AUDITED_AST}
        if set(definitions) != set(_AUDITED_AST):
            return None
        compiled = tuple(_codes(compile(payload, path, "exec", dont_inherit=True)))
        for name, node in definitions.items():
            if hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() != _AUDITED_AST[name]:
                return None
            live = getattr(module, name, None)
            live = live if isinstance(live, FunctionType) else getattr(live, "fn", None)
            if (not isinstance(live, FunctionType) or live.__globals__ is not vars(module)
                    or not any(live.__code__ == code for code in compiled)):
                return None
        forward_globals = {**vars(module), "_attn_fwd": _TwoStageLaunch(module._attn_fwd)}
        scoped_forward = _clone_function(forward, forward_globals)
        scoped_kernel = _clone_function(kernel, {**kernel.__globals__, "attn_false": scoped_forward})

        def masked_kernel(q, k, v, *, attn_mask, **kwargs):
            if attn_mask is None:
                raise ValueError("scoped mask kernel requires an explicit mask")
            attn_mask = _natural_bias_to_log2(attn_mask, q.dtype)
            return scoped_kernel(q, k, v, attn_mask=attn_mask, **kwargs)
        return masked_kernel
    except (OSError, TypeError, ValueError, KeyError, AttributeError, SyntaxError):
        return None
