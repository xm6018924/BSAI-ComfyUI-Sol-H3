"""Independent TST MODEL owner, composed AFTER native backend/Relay/EAV.

One native Euler invocation owns one full-schedule interval. No global hooks,
cached tensor diagnostics or SelfLift source are used.
"""

import copy
import json
import logging
import math
import threading

from .patch_stack_policy import warn_patch_stack

import torch
import comfy.model_base
import comfy.samplers
from comfy.patcher_extension import WrappersMP

from .tst_runtime import TSTQueryRuntime, TST_RUNTIME_KEY
from .progressive_attention import prepare_progressive_attention
from .progressive_sampling_runtime import validate_native_model
from .vdn_attention_compat import _factory_closure


TST_MODEL_KEY = 't8_h3_tst_model_v1'


def _copy_structure(value):
    if isinstance(value, dict):
        return {key: _copy_structure(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_structure(item) for item in value)
    return value


def _same_structure(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_structure(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same_structure(a, b) for a, b in zip(left, right))
    return left == right if type(left) in (str, int, float, bool, type(None)) else left is right


def _restore_own_changes(current, installed, original):
    """Inverse only unchanged TST additions; never erase later user selections."""
    if _same_structure(current, installed):
        return _copy_structure(original)
    if isinstance(current, dict) and isinstance(installed, dict) and isinstance(original, dict):
        result = {}
        for key, value in current.items():
            if key not in original and key in installed and _same_structure(value, installed[key]):
                continue
            result[key] = (_restore_own_changes(value, installed[key], original[key])
                           if key in installed and key in original else _copy_structure(value))
        return result
    return _copy_structure(current)


def _prepare_source(model):
    """Authenticate exact existing factories before adding an outer owner."""
    from . import prompt_relay_advanced as relay
    from . import enhance_a_video_advanced as eav
    if type(model.model) is not comfy.model_base.MiniMaxH3:
        raise ValueError('TST requires current native H3, not VDN or another architecture')
    from .progressive_sampling_runtime import _empty_option
    if any(not _empty_option(value) for key, value in model.model_options.items() if key != 'transformer_options'):
        warn_patch_stack('TST retains existing sampler/model function overrides')
    for name in ('callbacks', 'injections', 'hook_patches', 'additional_models'):
        if any(bool(value) for value in getattr(model, name, {}).values()):
            warn_patch_stack(f'TST retains MODEL {name}')
    groups = {kind: {key: value for key, value in keys.items() if value}
              for kind, keys in model.wrappers.items() if any(keys.values())}
    if not groups:
        validate_native_model(model, comfy.samplers.ksampler('euler'))
        return prepare_progressive_attention(model, tst_enabled=True)[0]
    if set(groups) != {WrappersMP.DIFFUSION_MODEL} or len(groups[WrappersMP.DIFFUSION_MODEL]) != 1:
        warn_patch_stack('TST retains multiple attention/wrapper owners')
        known = {relay.PROMPT_RELAY_WRAPPER_KEY, eav.EAV_WRAPPER_KEY, eav.EAV_PROMPT_RELAY_WRAPPER_KEY}
        if any(key in known for keys in groups.values() for key in keys):
            return model.clone()
        return prepare_progressive_attention(model, tst_enabled=True)[0]
    key, wrappers = next(iter(groups[WrappersMP.DIFFUSION_MODEL].items()))
    if len(wrappers) != 1:
        warn_patch_stack('TST retains stacked diffusion wrappers')
    if key == relay.PROMPT_RELAY_WRAPPER_KEY:
        relay.prompt_relay_model_contract(model)
    else:
        factories = {eav.EAV_WRAPPER_KEY: (eav.build_eav_model, '_diffusion_wrapper', 'expected_override'),
            eav.EAV_PROMPT_RELAY_WRAPPER_KEY: (eav.build_eav_prompt_relay_model, '_combined_wrapper', 'installed')}
        if key not in factories:
            warn_patch_stack('TST retains an unrecognized diffusion owner')
            return prepare_progressive_attention(model, tst_enabled=True)[0]
        factory, name, override_name = factories[key]
        owner = _factory_closure(wrappers[0], factory, name)
        override = model.model_options.get('transformer_options', {}).get('optimized_attention_override')
        if owner is None or not callable(owner.get(override_name)):
            raise ValueError('TST received an unauthenticated EAV attention owner')
        if owner.get(override_name) is not override:
            warn_patch_stack('TST retains a later user-selected EAV attention override')
        if owner.get('stg_contract') is not None:
            warn_patch_stack('TST with STG skipped blocks has an unverified layer clock')
    return model.clone()


class BoundTSTModel:
    def __init__(self, sigmas, mode, tau, workspace):
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1:
            raise ValueError('TST needs explicit full SIGMAS, not an inferred step count')
        # The numerical runtime is also the authoritative configuration validator.
        values = tuple(sigmas.detach().cpu().tolist())
        probe = TSTQueryRuntime(values, stage_start=0, stage_end=len(values)-1,
            layer_count=1, frames=2, spatial_tokens=1, mode=mode, tau=tau, max_workspace_mib=workspace)
        self.spec = dict(full_sigmas=probe.config['full_sigmas'], mode=mode, tau=float(tau), workspace=workspace)
        self._original = copy.deepcopy(self.spec)
        self.lock = threading.Lock()
        self.active = None
        self.last_report = None
        self.installed = None

    def verify(self, model=None):
        if self.spec != self._original:
            raise RuntimeError('TST MODEL configuration changed after binding')
        if model is not None:
            observed = (model.model_options, model.wrappers, model.object_patches)
            if not _same_structure(observed, self.installed):
                warn_patch_stack('TST MODEL owner/backend changed after binding')

    def interval(self, sigmas):
        values = tuple(sigmas.detach().cpu().tolist())
        full = self.spec['full_sigmas']
        matches = [start for start in range(len(full)-len(values)+1)
                   if all(math.isclose(a, b, abs_tol=2e-6, rel_tol=2e-6)
                          for a, b in zip(full[start:start+len(values)], values))]
        if len(values) < 2 or len(matches) != 1:
            raise ValueError('TST sampler sigmas are not one interval of the bound full schedule')
        return matches[0], matches[0]+len(values)-1


def build_tst_model(model, sigmas, *, mode='disabled', tau=.2, max_workspace_mib=256):
    if mode == 'disabled':
        return model, dict(mode='disabled', identity=True, status='not_installed')
    if model.get_attachment(TST_MODEL_KEY) is not None:
        raise ValueError('TST is already installed; do not stack TST nodes')
    state = BoundTSTModel(sigmas, mode, tau, max_workspace_mib)
    patched = _prepare_source(model)
    original = model.clone()
    block_count = len(model.model.diffusion_model.blocks)

    def sample_owner(executor, model_wrap, sigmas, extra_args, callback, noise,
                     latent_image=None, denoise_mask=None, disable_pbar=False):
        state.verify(model_wrap.model_patcher)
        if len(executor.wrappers) != 1 or executor.wrappers[0] is not sample_owner:
            warn_patch_stack('TST retains another sampler wrapper after binding')
        # Validate the actual sampler passed by Core, not the node's display name.
        import comfy.k_diffusion.sampling
        sampler = executor.class_obj
        if (type(sampler) is not comfy.samplers.KSAMPLER or
                sampler.sampler_function is not comfy.k_diffusion.sampling.sample_euler or
                sampler.inpaint_options or any(k != 's_churn' or v != 0 for k, v in sampler.extra_options.items())):
            raise ValueError('Standalone TST currently requires native Euler without custom options')
        if model_wrap.cfg != 1.:
            raise ValueError('TST requires CFG1 for its single native forward clock')
        start, end = state.interval(sigmas)
        if not state.lock.acquire(blocking=False):
            raise RuntimeError('TST MODEL is already executing; use a separately configured branch')
        state.active = dict(start=start, end=end, runtime=None, thread=threading.get_ident())
        state.last_report = None
        try:
            result = executor(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
            state.verify(patched)
            runtime = state.active['runtime']
            if runtime is None:
                raise RuntimeError('TST sampler completed without a native H3 forward')
            state.last_report = runtime.snapshot(require_complete=True)
            logging.info('[T8 TST] %s', json.dumps(state.last_report, allow_nan=False))
            return result
        except BaseException as exc:
            state.last_report = dict(completed=False, status='aborted', error_type=type(exc).__name__)
            raise
        finally:
            state.active = None
            state.lock.release()

    def apply_owner(executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
        state.verify(patched)
        if len(executor.wrappers) != 1 or executor.wrappers[0] is not apply_owner:
            warn_patch_stack('TST retains another apply_model wrapper after binding')
        active = state.active
        if active is None or active['thread'] != threading.get_ident():
            raise RuntimeError('TST MODEL forward requires its active sampling owner')
        shapes = kwargs.get('latent_shapes')
        if (not isinstance(shapes, (list, tuple)) or len(shapes) != 2 or len(shapes[0]) != 5 or
                shapes[0][0] != 1 or shapes[0][-1] % 2 or shapes[0][-2] % 2):
            raise RuntimeError('TST requires the native batch1 AV geometry')
        if not isinstance(t, torch.Tensor) or t.numel() != 1 or not bool(torch.isfinite(t).all()):
            raise RuntimeError('TST requires a single finite native video sigma')
        options = transformer_options or {}
        expected_override = patched.model_options['transformer_options'].get('optimized_attention_override')
        if options.get('optimized_attention_override') is not expected_override:
            warn_patch_stack('TST actual attention backend changed after binding')
        if TST_RUNTIME_KEY in options:
            raise RuntimeError('Nested TST query owner refused')
        frames, h, w = shapes[0][2:]
        if active['runtime'] is None:
            active['runtime'] = TSTQueryRuntime(state.spec['full_sigmas'], stage_start=active['start'],
                stage_end=active['end'], layer_count=block_count, frames=frames, spatial_tokens=(h//2)*(w//2),
                mode=state.spec['mode'], tau=state.spec['tau'], max_workspace_mib=state.spec['workspace'])
        runtime = active['runtime']
        if (frames, (h//2)*(w//2)) != (runtime.config['frames'], runtime.config['spatial_tokens']):
            raise RuntimeError('TST geometry changed during a single sampler invocation')
        with runtime.forward(float(t.detach().cpu().item())):
            return executor(x, t, c_concat, c_crossattn, control, {**options, TST_RUNTIME_KEY: runtime}, **kwargs)

    patched.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, TST_MODEL_KEY, sample_owner)
    patched.add_wrapper_with_key(WrappersMP.APPLY_MODEL, TST_MODEL_KEY, apply_owner)
    patched.set_attachments(TST_MODEL_KEY, state)
    state.installed = _copy_structure((patched.model_options, patched.wrappers, patched.object_patches))
    # Save immutable pre-composition structure for an authenticated progressive
    # adapter; this is not an alternate set of weights or a sampled state.
    state.original = original
    state.original_structure = _copy_structure((original.model_options, original.wrappers, original.object_patches))
    return patched, dict(config=copy.deepcopy(state.spec), status='configured_not_executed',
        diagnostics='Per-invocation completed/aborted report is logged after sampling',
        quality_qualified=False)


def detach_tst_model(model):
    """Remove only a verified TST composition, preserving the live LoRA chain."""
    state = model.get_attachment(TST_MODEL_KEY)
    if state is None:
        return model, None
    if type(state) is not BoundTSTModel:
        raise ValueError('Unknown TST MODEL attachment')
    state.verify(model)
    for kind, name in ((WrappersMP.SAMPLER_SAMPLE, 'sample_owner'), (WrappersMP.APPLY_MODEL, 'apply_owner')):
        owners = model.get_wrappers(kind, TST_MODEL_KEY)
        closure = _factory_closure(owners[0], build_tst_model, name) if len(owners) == 1 else None
        if closure is None or closure.get('state') is not state:
            raise ValueError('TST MODEL wrapper does not match its factory-bound state')
    original = state.original
    if not _same_structure((original.model_options, original.wrappers, original.object_patches), state.original_structure):
        warn_patch_stack('TST original composition changed after binding; portable identity unverified')
    result = model.clone()
    current = model.model_options, model.wrappers, model.object_patches
    restored = tuple(_restore_own_changes(value, installed, before)
                     for value, installed, before in zip(current, state.installed, state.original_structure))
    result.model_options, result.wrappers, result.object_patches = restored
    result.attachments = {k: v for k, v in model.attachments.items() if k != TST_MODEL_KEY}
    return result, copy.deepcopy(state.spec)
