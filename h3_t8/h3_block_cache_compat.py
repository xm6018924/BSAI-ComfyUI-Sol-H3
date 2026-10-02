"""Scoped call adaptation for an external cache's legacy H3 finalization path.

No external source, MODEL object or module forward is patched. The cache wrapper
still performs its own hit handling, prefetch cleanup and output unpacking.
"""

from .h3_core_compat import call_h3_final_layer


class _FinalizationView:
    def __init__(self, model, timestep, options):
        self._model = model
        self._timestep = timestep
        self._options = options

    def __getattr__(self, name):
        return getattr(self._model, name)

    def final_layer(self, x, t_emb, video_seg, audio_seg, *args, **kwargs):
        layer = self._model.final_layer
        if args or kwargs:
            # A newer external wrapper already supplies the Core interface.
            return layer(x, t_emb, video_seg, audio_seg, *args, **kwargs)
        options = self._options
        shifts = (
            float(options.get("minimax_h3_sigma_shift_video", self._model.sigma_shift_video)),
            float(options.get("minimax_h3_sigma_shift_audio", self._model.sigma_shift_audio)),
        )
        sigma = (self._timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        return call_h3_final_layer(
            layer, x, t_emb, video_seg, audio_seg, sigma=sigma,
            sample_sigmas=options.get("sample_sigmas"), shifts=shifts,
        )


class _ExecutorView:
    def __init__(self, executor, timestep, options):
        self._executor = executor
        self.class_obj = _FinalizationView(executor.class_obj, timestep, options)

    def __getattr__(self, name):
        return getattr(self._executor, name)

    def __call__(self, *args, **kwargs):
        return self._executor(*args, **kwargs)


def cache_finalization_executor(executor, timestep, options):
    if not hasattr(executor.class_obj, "final_layer"):
        # Non-H3 executors cannot reach the external H3 finalization path.
        return executor
    return _ExecutorView(executor, timestep, options)
