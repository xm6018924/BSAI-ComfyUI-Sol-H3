"""Compose authenticated installed backends on disposable progressive MODELs."""

from .patch_stack_policy import warn_patch_stack

from collections import Counter
import copy
import hashlib
import inspect
from pathlib import Path
import sys

from comfy.ldm.modules import attention as core_attention
from comfy.patcher_extension import WrappersMP

from .h3_core_compat import plain_attention_backend, set_h3_attention_backend
from .relay_kj_memory import (
    PROGRESSIVE_MEMORY_RUNTIME_KEY, adapt_memory_for_relay,
    bind_memory_runtime, inspect_memory_composition,
)
from .relay_sol_backend import capture_composed_backend


def backend_snapshot(backend):
    return copy.deepcopy(backend.report()) if backend is not None else {'kind': 'unchanged_core'}


def backend_phase_report(backend, before):
    """Subtract only execution counters; never subtract configuration numbers."""
    after = backend_snapshot(backend)

    def delta(current, previous):
        for key, value in current.items():
            if key == 'completed_calls':
                prior = previous.get(key, {})
                current[key] = {name: count - prior.get(name, 0) for name, count in value.items()}
                if any(count < 0 for count in current[key].values()):
                    raise RuntimeError('Attention counters reset during a progressive stage')
            elif isinstance(value, dict):
                delta(value, previous.get(key, {}) if isinstance(previous.get(key), dict) else {})

    delta(after, before)
    after['counter_scope'] = 'this_phase_completed_calls_only'
    return after


class _PlainDelegate:
    def __init__(self, override, name):
        self.override = override
        self.name = name
        self.calls = Counter()
        self.masked_delegate = None

    def _masked_sage_delegate(self):
        if self.masked_delegate is None:
            from .relay_kj_backend import KJRelayBackend, _audited_mask_kernel
            from .scoped_sage_triton import build_scoped_mask_kernel
            package = sys.modules.get('sageattention')
            candidate = getattr(package, 'sageattn_qk_int8_pv_fp16_triton', None)
            kernel = build_scoped_mask_kernel(candidate) if _audited_mask_kernel(candidate) else None
            digest = (hashlib.sha256(Path(inspect.getsourcefile(candidate)).read_bytes()).hexdigest()
                      if kernel is not None else None)
            # Reuse the already-audited Sage bias/stride/dtype adapter, not the
            # KJ node or a changed global Core selector. Only biased calls use
            # this delegate; unsupported layouts retain explicit SDPA reasons.
            self.masked_delegate = KJRelayBackend('auto', getattr(core_attention, 'sageattn', None),
                                                   digest, kernel is not None, kernel)
        return self.masked_delegate

    def attention(self, q, k, v, heads, **kwargs):
        if (self.name == 'sage' and kwargs.get('mask') is not None and q.device.type == 'cuda'
                and not core_attention.SAGE_ATTENTION_SUPPORTS_MASK):
            # Core's automatic Sage entry cannot consume Relay bias on this
            # installed version. Keep the bias and use the authenticated scoped
            # Triton route; do not count Core's silent SDPA path as Sage.
            result = self._masked_sage_delegate().attention(q, k, v, heads, **kwargs)
        elif self.override is None:
            result = core_attention.optimized_attention(q, k, v, heads,
                                                       **{**kwargs, '_inside_attn_wrapper': True})
        else:
            result = self.override(core_attention.optimized_attention, q, k, v, heads, **kwargs)
        self.calls['delegate:completed'] += 1
        return result

    def report(self):
        return {'kind': 'core_plain_delegate', 'requested_backend': self.name,
                'completed_calls': dict(self.calls),
                **({'scoped_masked_sage': {**self.masked_delegate.report(),
                     'kind': 'audited_sage_bias_delegate_for_core_selector'}} if self.masked_delegate is not None else {}),
                'scope': 'Core delegate calls; underlying kernel fallback is not independently counted'}


def inspect_progressive_attention(model):
    memory = inspect_memory_composition(model)
    options = model.model_options.get('transformer_options', {})
    override = options.get('optimized_attention_override')
    plain = plain_attention_backend(override) if override is not None else None
    backend = capture_composed_backend(override) if override is not None and plain is None else None
    if override is not None and plain is None and backend is None:
        from .relay_sol_backend import UserSelectedBackend
        warn_patch_stack('Unknown progressive attention backend retained as a user-selected delegate')
        backend = UserSelectedBackend(override)
    return memory, backend, plain


def prepare_progressive_attention(model, *, tst_enabled=False):
    memory, selected, plain = inspect_progressive_attention(model)
    if memory is None and selected is None and not tst_enabled:
        return model.clone(), None
    override = model.model_options.get('transformer_options', {}).get('optimized_attention_override')
    if selected is None and plain is not None:
        selected = _PlainDelegate(override, plain)
    prepared, backend = adapt_memory_for_relay(model, selected)
    prepared = prepared.clone()
    if backend is None and not tst_enabled:
        # FFN-only KJ changes keep their selected Core backend.
        return prepared, None
    if backend is None:
        backend = _PlainDelegate(override, plain or 'automatic')
    if tst_enabled:
        from .tst_runtime import transform_owned_queries

        def attention(q, k, v, heads, **kwargs):
            q = transform_owned_queries(q, k, heads, kwargs.get('transformer_options') or {},
                skip_reshape=kwargs.get('skip_reshape', False),
                skip_output_reshape=kwargs.get('skip_output_reshape', False))
            return backend.attention(q, k, v, heads, **kwargs)
    else:
        attention = backend.attention
    set_h3_attention_backend(prepared, attention)
    expected = prepared.model_options['transformer_options']['optimized_attention_override']

    def guard(executor, x, timestep, context, transformer_options, **kwargs):
        if transformer_options.get('optimized_attention_override') is not expected:
            warn_patch_stack('Progressive attention owner changed after binding')
        route = {}
        bind_memory_runtime(backend, route)
        options = {**transformer_options, PROGRESSIVE_MEMORY_RUNTIME_KEY: route}
        return executor(x, timestep, context, options, **kwargs)

    prepared.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, 't8_progressive_attention_owner', guard)
    return prepared, backend
