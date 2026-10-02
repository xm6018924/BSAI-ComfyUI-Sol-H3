"""Explicit kernel/host-cache session using unchanged upstream transaction methods."""
from taomate_leased_hook import make_leased_hook


def make_local_session(upstream_session, upstream_hook, plan, *, cache, kernel, backend_name):
    class LocalStreamingSession(upstream_session.H3StreamingSession):
        def __init__(self):
            if not isinstance(plan, upstream_session.StreamPlan):
                raise TypeError('plan must be an upstream StreamPlan')
            self.plan = plan
            self.cache = cache
            self._block_index_base = self.cache.committed_blocks
            self.hook = make_leased_hook(upstream_hook, cache, kernel=kernel, backend_name=backend_name)
            self._installed = False
            self._noisy_blocks = set()

    return LocalStreamingSession()
