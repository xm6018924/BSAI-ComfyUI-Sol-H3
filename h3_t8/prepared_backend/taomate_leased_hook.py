"""Isolated adapter for pinned Tao hook, with explicit kernel identity and leases.

Reuses upstream packing/conditioning/clean-commit logic; does not install global
patches or pretend an alternate dense kernel is the upstream FA3 implementation.
"""


def make_leased_hook(upstream, cache, *, kernel, backend_name):
    if not callable(kernel) or not isinstance(backend_name, str) or not backend_name.strip():
        raise ValueError('An explicit dense kernel and nonempty backend name are required')

    class LeasedStreamingHook(upstream.H3StreamingAttentionHook):
        def __init__(self):
            # Upstream constructor imports FA3 unconditionally. Set the same state
            # explicitly for this separately identified backend, not by faking FA3.
            self.cache = cache
            self._flash_attention = kernel
            self.kernel_calls = self.query_tokens = self.key_tokens = 0
            self.mode = upstream.HookMode.IDLE
            self._live_documents = None
            self._live_document_rows = {}

        def __call__(self, **kwargs):
            if not self.active:
                raise RuntimeError('streaming attention hook was called while inactive')
            with self.cache.layer(kwargs['layer_name'], kwargs['query'].device):
                return super().__call__(**kwargs)

        def receipt(self):
            return {**super().receipt(), 'backend': backend_name,
                    'cache_residency': 'CPU; one layer borrowed on compute device',
                    'cache_transfer_bytes': self.cache.transferred_bytes,
                    'peak_borrowed_layer_bytes': self.cache.peak_layer_bytes}

    return LeasedStreamingHook()
