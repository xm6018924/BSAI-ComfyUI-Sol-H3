"""Rebind authenticated Relay inputs to each progressive spatial layout.

Text token spans and temporal event coordinates are unchanged. Only the native
packed spatial layout and its paired conditioning hash change between stages.
No global attention function or original MODEL/CONDITIONING is modified.
"""

from collections import Counter
import copy

import comfy.conds
from comfy.patcher_extension import WrappersMP

from . import prompt_relay_advanced as relay
from .relay_kj_memory import HeadGroupedBackend, inspect_memory_composition
from .h3_core_compat import plain_attention_backend
from .progressive_attention import _PlainDelegate


def _fresh_delegate(backend):
    if backend is None:
        return None
    if type(backend) is HeadGroupedBackend:
        # Raw memory methods will be rebound to a fresh owner by Relay.
        return _fresh_delegate(backend.delegate)
    result = copy.copy(backend)
    for name in ('counters', 'calls'):
        if hasattr(result, name):
            setattr(result, name, Counter())
    if hasattr(result, 'fallback'):
        result.fallback = _fresh_delegate(result.fallback)
    if hasattr(result, 'masked_delegate'):
        result.masked_delegate = _fresh_delegate(result.masked_delegate)
    return result


def detach_relay_input(model):
    """Authenticate before removing our own owner on a disposable clone."""
    if model.get_attachment(relay.PROMPT_RELAY_WRAPPER_KEY) is None:
        return model, None
    contract = relay.prompt_relay_model_contract(model)
    memory = inspect_memory_composition(model)
    base = model.clone()
    base.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, relay.PROMPT_RELAY_WRAPPER_KEY)
    # Core removal leaves an empty wrapper-type dictionary. Remove only that
    # empty container, so content identity does not mistake it for a live owner.
    if not base.wrappers.get(WrappersMP.DIFFUSION_MODEL):
        base.wrappers.pop(WrappersMP.DIFFUSION_MODEL, None)
    base.remove_attachments(relay.PROMPT_RELAY_WRAPPER_KEY)
    base.model_options['transformer_options'].pop('optimized_attention_override', None)
    if memory:
        for path, original in memory['raw_forwards'].items():
            base.add_object_patch(path, original)
    return base, contract


def strip_paired_conditioning(conditioning, contract, *, required):
    output = []
    for embedding, metadata in conditioning:
        updated = dict(metadata)
        binding = updated.pop(relay.PROMPT_RELAY_BINDING_KEY, None)
        conds = dict(updated.get('model_conds', {}))
        payload = conds.pop(relay.PROMPT_RELAY_PAYLOAD_KEY, None)
        if binding is None and payload is None and not required:
            output.append([embedding, updated])
            continue
        if (binding != contract['binding'] or type(payload) is not comfy.conds.CONDConstant
                or payload.cond != contract['binding_hash']):
            raise ValueError('Progressive Relay MODEL and CONDITIONING are not paired')
        if conds:
            updated['model_conds'] = conds
        else:
            updated.pop('model_conds', None)
        output.append([embedding, updated])
    return output


def _stage_binding(conditioning, contract, plan, *, low):
    embedding, metadata = conditioning[0]
    binding = contract['binding']
    if int(embedding.shape[1]) != int(binding['text_len']) or binding['task'] != plan.task:
        raise ValueError('Progressive Relay text/task does not match the native stage')
    keyframes = metadata.get('minimax_keyframes', [])
    frames = 5 + (plan.video_shape[2] - 2) // 5 * 17
    layout = relay.build_packed_layout(
        binding['text_len'], plan.video_shape[2],
        (plan.low_height if low else plan.target_height) // 16,
        (plan.low_width if low else plan.target_width) // 16,
        plan.audio_shape[-1], keyframes=keyframes, refs=[], frame_count=frames)
    if not low and relay._layout_contract(layout) != binding['layout_contract']:
        raise ValueError('Progressive Relay input target layout does not match the latent')
    return relay._bind_layout_contract(binding, layout, resolved_task=plan.task, keyframes=keyframes, refs=[])


def prepare_relay_stage(base, contract, positive, negative, plan, *, low, backend_contract=None):
    binding = _stage_binding(positive, contract, plan, low=low)
    return install_relay_stage(base, binding, positive, negative, contract['query_chunk_rows'],
                               backend_contract=backend_contract)


def install_relay_stage(base, binding, positive, negative, query_chunk_rows, *,
                        backend_contract=None, allowed_extra_conds_versions=(), inspection_model=None):
    """Install an already layout-bound stage; callers own geometry validation."""
    # Existing Core guards require weight-free inspection, not the removal of
    # actual LoRA descriptors from the model subsequently used for sampling.
    check = (inspection_model if inspection_model is not None else base).clone()
    check.patches = {}
    hashes = relay._assert_core_contract(check,
        allowed_live_extra_conds_patch_versions=allowed_extra_conds_versions)
    backend = _fresh_delegate((backend_contract or {}).get('attention_backend'))
    source_override = (backend_contract or {}).get('source_plain_override')
    if source_override is None:
        source_override = base.model_options.get('transformer_options', {}).get('optimized_attention_override')
    plain = plain_attention_backend(source_override) if source_override is not None else None
    if backend is None and plain is not None:
        backend = _PlainDelegate(source_override, plain)
    counts = {'forward': 0, 'routed_attention': 0}

    def observe(kind):
        counts[kind] += 1

    model, _ = relay._install_prompt_relay_model(
        base, binding, query_chunk_rows, hashes, attention_backend=backend,
        execution_observer=observe)
    paired = []
    for conditioning in (positive, negative):
        marked = [[embedding, {**metadata, relay.PROMPT_RELAY_BINDING_KEY: binding}]
                  for embedding, metadata in conditioning]
        paired.append(relay._attach_binding_model_cond(marked, binding['binding_hash']))
    active = relay.prompt_relay_model_contract(model)
    report = {'binding_hash': binding['binding_hash'], 'plan_hash': binding['plan_hash'],
              'layout': binding['layout_contract'], 'query_route': binding['query_route'],
              'events': binding['events'], 'time_policy': 'unchanged_native_event_coordinates',
              'completed_calls': counts,
              'scope': 'bound_layout; application_verified_by_runtime_owner_not_media_quality'}
    return model, paired[0], paired[1], active['attention_backend'], report
