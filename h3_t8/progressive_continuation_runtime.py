"""Internal continuation entry; public workflows/chain resume are not exposed.

Owns only authenticated prepared inputs and the existing long-video payload
patch. Progressive sampling retains its original video lift and AV clocks.
"""

import json

import torch
from comfy.nested_tensor import NestedTensor

from .long_video import patch_long_video_model
from .long_video_dual_identity import content_identity
from .progressive_continuation import ProgressiveContinuationSource, _sha


def _input_identity(value):
    if type(value) is NestedTensor:
        return {'native_av': [_input_identity(part) for part in value.unbind()]}
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized or value.device.type == 'meta':
            raise ValueError('Continuation input requires materialized dense tensors')
        # Native motion guides are channel-strided one-frame views of a tail.
        # Hash their logical values without altering the original shared views.
        return content_identity(value.detach().contiguous())
    if isinstance(value, (list, tuple)):
        return {'type': type(value).__name__, 'items': [_input_identity(part) for part in value]}
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: _input_identity(part) for key, part in sorted(value.items())}
    return content_identity(value)


class PreparedProgressiveContinuation:
    def __init__(self, source, low, high, preparation, relay=None):
        if type(source) is not ProgressiveContinuationSource:
            raise ValueError('A verified progressive accepted-parent source is required')
        source.revalidate()
        self.source, self.low, self.high = source, low, high
        self.preparation = json.loads(json.dumps(preparation))
        self.relay = json.loads(json.dumps(relay)) if relay is not None else None
        self.identity = self._identity()

    def _identity(self):
        # Include actual conditioned tokens/guide tensors/masks/source AV, not
        # an object repr or descriptive task label. Caller job identity remains
        # responsible for models/LoRA/VAE/CLIP/code and durable resume.
        return _sha(_input_identity({'low': [self.low[0], self.low[1]],
            'high': [self.high[0], self.high[1]], 'source': self.source.binding, 'relay': self.relay,
            'delivery': self.delivery_description()}))

    def delivery_description(self):
        return dict(conditioned_prompt=self.high[3], media_map=self.high[4],
                    conditioning=json.loads(self.high[5]), mux_audio_identity=_input_identity(self.high[2]))

    def verify(self, positive=None, av_latent=None, plan=None):
        self.source.revalidate()
        if self._identity() != self.identity:
            raise ValueError('Progressive continuation prepared inputs changed')
        if positive is not None and positive is not self.high[0]:
            raise ValueError('Continuation positive must be the prepared HIGH conditioning')
        if av_latent is not None and av_latent is not self.high[1]:
            raise ValueError('Continuation latent must be the prepared HIGH source')
        if plan is not None:
            request = self.source.binding['request']
            if (plan.low_width, plan.low_height, plan.target_width, plan.target_height) != (
                    request['low_width'], request['low_height'], request['width'], request['height']):
                raise ValueError('Progressive continuation canvas differs from prepared source')
            lv, la = self.low[1]['samples'].unbind()
            hv, ha = self.high[1]['samples'].unbind()
            if (tuple(lv.shape) != (*plan.video_shape[:-2], plan.low_height // 16, plan.low_width // 16)
                    or tuple(hv.shape) != tuple(plan.video_shape) or tuple(ha.shape) != tuple(plan.audio_shape)
                    or not torch.equal(la, ha)):
                raise ValueError('Continuation phase AV shapes or source audio differ')
        return {'source_sha256': self.source.sha256, 'prepared_inputs_sha256': self.identity,
                'preparation': self.preparation, 'resume_execution_qualified': False,
                'prepared_delivery': self.delivery_description(),
                'quality_qualified': False}

    def phase_models(self, low_model, high_model):
        # Native/backends are validated before this call. Install the existing
        # known owner ourselves; do not accept arbitrary incoming extra_conds.
        return patch_long_video_model(low_model), patch_long_video_model(high_model)


def sample_progressive_continuation(source, model, sampler, sigmas, *, clip, video_vae, audio_vae,
                                    prompt=None, length, upscaler_model, seed, low_evaluations=4,
                                    low_scale=.5, model_hires=None, condition_options=None,
                                    precision='fp16', reserve_vram_mib=1024, callback=None,
                                    eav_mode='disabled', eav_tau=4., eav_start_video_progress=0.,
                                    eav_end_video_progress=1., eav_max_workspace_mib=32, eav_g_hard_limit=1.5,
                                    prompt_relay_plan=None, query_chunk_rows=256, accepted_end_frame=None,
                                    checkpoint=None, producers=None,
                                    tst_mode='disabled', tst_tau=.2, tst_max_workspace_mib=256):
    """Run one continuation window; durable stage cache/loop remain pending."""
    from .progressive_sampling_runtime import sample_progressive_h3, validate_native_model
    if type(source) is not ProgressiveContinuationSource:
        raise ValueError('Use capture_continuation_source before preparing a continuation')
    if producers is not None:
        from .progressive_producers import verify_producers
        verify_producers(producers, clip=clip, video_vae=video_vae, audio_vae=audio_vae)
    validate_native_model(model, sampler)
    if model_hires is not None:
        validate_native_model(model_hires, sampler)
    projected = None
    if prompt_relay_plan is not None:
        from .progressive_continuation_relay import project_for_source, bind_prepared_relay
        projected = project_for_source(source, prompt_relay_plan, length, accepted_end_frame)
        if prompt is not None and prompt != projected['compiled_prompt']:
            raise ValueError('Continuation prompt must match the global Relay compiled prompt')
        prompt = projected['compiled_prompt']
    elif accepted_end_frame is not None:
        raise ValueError('accepted_end_frame requires a global Relay plan')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Continuation requires a prompt or global Relay plan')
    low, high, preparation = source.prepare_conditions(clip=clip, video_vae=video_vae, audio_vae=audio_vae,
        prompt=prompt, length=length, **(condition_options or {}))
    relay = bind_prepared_relay(clip, projected, low, high, query_chunk_rows) if projected is not None else None
    if producers is not None:
        verify_producers(producers, clip=clip, video_vae=video_vae, audio_vae=audio_vae)
    prepared = PreparedProgressiveContinuation(source, low, high, preparation, relay=relay)
    return sample_progressive_h3(model, high[0], high[0], high[1], sampler, sigmas,
        upscaler_model=upscaler_model, seed=seed, cfg=1., low_evaluations=low_evaluations,
        low_scale=low_scale, model_hires=model_hires, precision=precision,
        reserve_vram_mib=reserve_vram_mib, callback=callback, input_mode='initialized_av_exp',
        continuation=prepared, eav_mode=eav_mode, eav_tau=eav_tau, checkpoint=checkpoint, producers=producers,
        eav_start_video_progress=eav_start_video_progress, eav_end_video_progress=eav_end_video_progress,
        eav_max_workspace_mib=eav_max_workspace_mib, eav_g_hard_limit=eav_g_hard_limit,
        tst_mode=tst_mode, tst_tau=tst_tau, tst_max_workspace_mib=tst_max_workspace_mib)
