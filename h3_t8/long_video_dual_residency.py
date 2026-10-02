"""Release only this job's finished-stage residency through Core's own API."""
import gc

import comfy.model_management as mm


def _complete_native_dynamic_stage():
    # An internal loop does not return to execution.py between stages. Mirror
    # its completed-node cleanup, after sampling and residency release, so cast
    # buffers, dirty mmap pages and prefetch references cannot span the full film.
    # This clears transient process-local Core state, not other loaded models.
    import comfy.memory_management as memory
    if not getattr(memory, 'aimdo_enabled', False):
        return 'aimdo_disabled'
    import comfy.model_prefetch as prefetch
    import comfy_aimdo.model_vbar as model_vbar
    callbacks = (getattr(prefetch, 'cleanup_prefetch_queues', None),
                 getattr(mm, 'reset_cast_buffers', None),
                 getattr(model_vbar, 'vbars_reset_watermark_limits', None))
    if not all(callable(callback) for callback in callbacks):
        raise RuntimeError('Dynamic Core lacks its completed-node cleanup API')
    for callback in callbacks:
        callback()
    return 'completed'


def release_stage_residency(*components):
    patchers = []
    for component in components:
        patcher = getattr(component, 'patcher', component)
        if getattr(patcher, 'model', None) is None or not hasattr(patcher, 'load_device'):
            raise ValueError('Stage residency requires an actual Core model patcher')
        patchers.append(patcher)
    target_ids = {id(patcher.model) for patcher in patchers}
    loaded = list(mm.current_loaded_models)
    # LoadedModel.model is a weak reference in Core. Resolve it once and retain
    # the live patcher for this release transaction; do not dereference it again
    # after free_memory has detached/removed its residency entry.
    resolved = [(entry, entry.model) for entry in loaded]
    target_pairs = [(entry, patcher) for entry, patcher in resolved
                    if patcher is not None and id(patcher.model) in target_ids]
    targets = [entry for entry, _ in target_pairs]
    target_entry_ids = {id(entry) for entry in targets}
    keep = [entry for entry in loaded if id(entry) not in target_entry_ids]
    for device in {entry.device for entry in targets}:
        # Passing the non-target LoadedModel objects preserves every unrelated
        # cached model, unlike unload_all_models or an unqualified free_memory.
        mm.free_memory(1 << 60, device, keep_loaded=keep)
    freed_ram = 0
    seen = set()
    for _, patcher in target_pairs:
        if id(patcher.model) in seen:
            continue
        seen.add(id(patcher.model))
        release = getattr(patcher, 'partially_unload_ram', None)
        if callable(release):
            freed_ram += int(release(1 << 60) or 0)
    boundary = 'not_dynamic_or_not_loaded'
    if any(callable(getattr(patcher, 'is_dynamic', None)) and patcher.is_dynamic()
           for _, patcher in target_pairs):
        boundary = _complete_native_dynamic_stage()
    gc.collect()
    return {'scope': 'finished_stage_models_only', 'unloaded_entries': len(targets),
            'stale_entries_skipped': sum(patcher is None for _, patcher in resolved),
            'core_reported_ram_freed_bytes': freed_ram, 'unrelated_entries_preserved': len(keep),
            'native_stage_boundary': boundary}
