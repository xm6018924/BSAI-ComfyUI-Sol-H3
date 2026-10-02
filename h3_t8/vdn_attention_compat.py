"""Recognize the official sparse node and give VDN ownership on its MODEL branch."""

import sys
import types
from pathlib import Path

import comfy.model_patcher


def _factory_closure(function, factory, name):
    code = getattr(factory, "__code__", None)
    if code is None:
        return None
    expected = [item for item in code.co_consts
                if isinstance(item, types.CodeType) and item.co_name == name]
    if not any(getattr(function, "__code__", None) is item for item in expected):
        return None
    if function.__globals__ is not factory.__globals__:
        return None
    try:
        return {name: cell.cell_contents for name, cell in
                zip(function.__code__.co_freevars, function.__closure__ or ())}
    except ValueError:
        return None


def native_sparse_state(function, kind):
    # Comfy's extension loader uses a file-derived module name, whereas a plain
    # Python import uses comfy_extras.nodes_sparse_attention. Resolve the actual
    # function owner and verify the Core file, not just an import alias.
    core = sys.modules.get(getattr(function, "__module__", ""))
    if core is None:
        return None
    source = getattr(core, "__file__", None)
    expected = Path(comfy.model_patcher.__file__).resolve().parent.parent / "comfy_extras/nodes_sparse_attention.py"
    if source is None or Path(source).resolve() != expected:
        return None
    factory_name, closure_name = {
        "override": ("make_attention_override", "override"),
        "block": ("make_h3_block_patch", "block_patch"),
        "attention": ("make_h3_block_patch", "attention"),
        "callback": ("apply_block_sparse_attention", "<lambda>"),
    }[kind]
    values = _factory_closure(function, getattr(core, factory_name, None), closure_name)
    if values is None or type(values.get("patch")) is not getattr(core, "SparseAttnPatch", None):
        return None
    return values


def without_native_sparse(options):
    """Copy only modified containers; leave unknown patches for normal conflict checks."""
    result = options.copy()
    removed = 0
    override = options.get("optimized_attention_override")
    while (state := native_sparse_state(override, "override")) is not None:
        override = state.get("previous")
        removed += 1
    if override is None:
        result.pop("optimized_attention_override", None)
    else:
        result["optimized_attention_override"] = override
    replacements = options.get("patches_replace", {})
    dit = replacements.get("dit", {})
    kept = {}
    for key, hook in dit.items():
        if native_sparse_state(hook, "block") is not None:
            removed += 1
        else:
            kept[key] = hook
    if kept != dit:
        result["patches_replace"] = {**replacements, "dit": kept}
    return result, removed


def prepare_vdn_attention_model(model):
    # Retain explicit user choices. The inspection-only normalizer remains for
    # authenticated identity adapters; it must not erase execution owners.
    _, found = without_native_sparse(model.model_options.get("transformer_options", {}))
    if found:
        from .patch_stack_policy import warn_patch_stack
        warn_patch_stack("Existing Core sparse override/DiT callbacks retained; the later algorithm may be bypassed")
    return model, 0
