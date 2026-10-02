"""Pinned upstream streaming algorithm with explicit singleton host residency.

No global distributed initialization or fake process group. Scope only the two
factories imported by runtime.py; all phase/noise/teacher/commit logic remains
upstream. Use only in an isolated single-threaded worker under a serial lease.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from taomate_host_cache import HostResidentCleanCache
from taomate_local_session import make_local_session
from taomate_local_transport import validate_local
from taomate_weight_offload import offload_transformer_blocks
from taomate_request_state import OwnedRequestState


def cuda_sdpa_kernel(q, k, v, *, softmax_scale, causal, num_splits):
    if causal or num_splits != 1:
        raise ValueError('Only upstream noncausal single-split attention is supported')
    if any(x.device.type != 'cuda' or x.dtype != torch.bfloat16 or x.ndim != 4 for x in (q, k, v)):
        raise ValueError('Streaming kernel requires CUDA BF16 [B,S,H,D] inputs')
    from torch.nn.attention import sdpa_kernel, SDPBackend
    # Never allocate a full quadratic score tensor via the math fallback.
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        value = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
            v.transpose(1, 2), dropout_p=0.0, is_causal=False, scale=softmax_scale)
    return value.transpose(1, 2)


@contextmanager
def runtime_backend_scope(*, kernel, backend_name):
    from taomate_h3.streaming import runtime, cache, session, attention_hook
    original_cache, original_session = runtime.CleanAVKVCache, runtime.H3StreamingSession
    if original_cache is not cache.CleanAVKVCache or original_session is not session.H3StreamingSession:
        raise RuntimeError('Streaming factories already patched; cannot overwrite another owner')
    def host_factory(contract):
        return HostResidentCleanCache(cache, contract)
    def session_factory(plan, *, cache):
        if not isinstance(cache, HostResidentCleanCache):
            raise TypeError('Local streaming requires its host-resident clean cache')
        return make_local_session(session, attention_hook, plan, cache=cache,
            kernel=kernel, backend_name=backend_name)
    runtime.CleanAVKVCache, runtime.H3StreamingSession = host_factory, session_factory
    try:
        yield
    finally:
        runtime.CleanAVKVCache, runtime.H3StreamingSession = original_cache, original_session


def make_local_runtime(base10_teacher, *, minimum_free_bytes=2*1024**3, interrupt=lambda: None):
    from taomate_h3.streaming.runtime import H3StreamingRuntime
    class LocalRuntime(H3StreamingRuntime):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.request_owner = OwnedRequestState()

        def _release_owned_state(self):
            super().release_retained_state()
            self._dit_cuda_events.clear()

        def release_retained_state(self):
            self.request_owner.close(self._release_owned_state)

        def run(self, **kwargs):
            validate_local(kwargs['model'].parallel_context)
            if torch.is_grad_enabled():
                raise RuntimeError('Local streaming requires inference/no-grad')
            def invoke():
                handles = []
                try:
                    # Cooperative cancellation between owned block forwards;
                    # kernels already in flight complete before offload cleanup.
                    for block in kwargs['model'].blocks:
                        handles.append(block.register_forward_pre_hook(lambda _m, _a: interrupt()))
                    with runtime_backend_scope(kernel=cuda_sdpa_kernel, backend_name='torch_CUDA_SDPA_not_FA3'), \
                            offload_transformer_blocks(kwargs['model'], 'cuda:0', minimum_free_bytes=minimum_free_bytes) as receipt:
                        output = super(LocalRuntime, self).run(**kwargs)
                    self.last_weight_receipt = receipt
                    return output
                finally:
                    for handle in handles:
                        handle.remove()
            return self.request_owner.execute(self, kwargs['model'], invoke, self._release_owned_state, interrupt)

        def dit_timing_receipt(self):
            # A single process needs no max all_reduce, and has no ProcessGroup.
            torch.cuda.synchronize()
            return dict(noisy_calls=self._dit_noisy_calls, clean_kv_calls=self._dit_clean_calls,
                total_calls=self._dit_noisy_calls+self._dit_clean_calls,
                cuda_seconds=sum(float(a.elapsed_time(b)) for a, b in self._dit_cuda_events)/1000,
                scope='single_GPU_CUDA_events_including_weight_transfer_not_end_to_end')
    return LocalRuntime(base10_teacher=base10_teacher)
