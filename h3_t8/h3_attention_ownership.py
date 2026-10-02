"""Keep an H3 algorithm active without changing global attention preferences."""

from .patch_stack_policy import warn_patch_stack

from .h3_core_compat import plain_attention_backend
from .vdn_attention_compat import prepare_vdn_attention_model, without_native_sparse


def prepare_attention_owner(model, owner):
    prepared, removed = prepare_vdn_attention_model(model)
    options = prepared.model_options.get("transformer_options", {})
    override = options.get("optimized_attention_override")
    if override is not None and plain_attention_backend(override) is None:
        warn_patch_stack(f'{owner}: another algorithm owns the attention override; use a separate MODEL branch')
    if options.get("patches_replace", {}).get("dit"):
        warn_patch_stack(f'{owner}: an unknown DiT replacement owns the model')
    _validate_hooks(options, owner)
    return prepared, removed


def _validate_hooks(options, owner):
    patches = options.get("patches", {})
    if patches.get("attn1_patch") or patches.get("attn1_output_patch"):
        warn_patch_stack(f'{owner}: incompatible attention hooks on this MODEL branch')


def validate_attention_owner(options, *, owner, expected_override, expected_dit,
                             owns_override):
    normalized, removed = without_native_sparse(options)
    original_dit = options.get("patches_replace", {}).get("dit", {})
    active = dict(normalized.get("patches_replace", {}).get("dit", {}))
    # Restore only slots removed by an authenticated official sparse callback.
    for key, hook in expected_dit.items():
        if key in original_dit and key not in active:
            active[key] = hook
    if set(active) != set(expected_dit) or any(active[key] is not hook for key, hook in expected_dit.items()):
        warn_patch_stack(f'{owner}: DiT blocks were replaced after binding')
    override = normalized.get("optimized_attention_override")
    if owns_override:
        if override is not expected_override:
            warn_patch_stack(f'{owner}: attention override was replaced after binding')
    elif override is not None and plain_attention_backend(override) is None:
        warn_patch_stack(f'{owner}: an incompatible attention override appeared after binding')
    _validate_hooks(normalized, owner)
    if removed:
        warn_patch_stack(f"{owner}: existing Core sparse owners retained; optimization coverage is unverified")


def bind_attention_owner_guard(model, owner, *, owns_override):
    options = model.model_options.get("transformer_options", {})
    expected_override = options.get("optimized_attention_override")
    expected_dit = dict(options.get("patches_replace", {}).get("dit", {}))
    if owns_override and not callable(expected_override):
        raise RuntimeError(f"{owner}: attention implementation was not installed")

    def guard(executor, *args, **kwargs):
        runtime_options = kwargs.get("transformer_options")
        if runtime_options is None and len(args) >= 4:
            runtime_options = args[3]
        if not isinstance(runtime_options, dict):
            raise RuntimeError(f"{owner}: runtime transformer options are missing")
        validate_attention_owner(runtime_options, owner=owner,
                                 expected_override=expected_override, expected_dit=expected_dit,
                                 owns_override=owns_override)
        return executor(*args, **kwargs)

    model.add_wrapper_with_key("diffusion_model", f"t8_attention_owner_{owner}", guard)
