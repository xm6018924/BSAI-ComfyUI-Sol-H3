"""First-window preparation for the bound progressive job, not a new sampler.

Reuse the native long-video builder and exact global Relay projection. Only
the schema marker we just produced is removed; arbitrary reference/payload
metadata is not admitted by this adapter. No encoder or sampler is patched.
"""

import inspect
import json

import torch

from .long_video import (
    LONG_VIDEO_CONDITIONING_KEY, LONG_VIDEO_SCHEMA, _empty_context,
    build_long_video_conditioning,
)
from . import prompt_relay_advanced as relay
from .progressive_relay import install_relay_stage
from .prompt_relay_long_video_advanced import project_prompt_relay_plan_to_long_video_window


def prepare_first_segment(*, clip, video_vae, audio_vae, chain_id, segment,
                          width, height, condition_options=None, prompt_relay_plan=None):
    if segment.index != 0 or segment.plan.timeline_start_seconds != 0:
        raise ValueError('First preparation requires the planned segment0')
    options = {} if condition_options is None else dict(condition_options)
    owned = {'clip', 'video_vae', 'audio_vae', 'context', 'segment_index', 'context_frames',
             'prompt', 'width', 'height', 'length', 'return_details'}
    allowed = set(inspect.signature(build_long_video_conditioning).parameters) - owned
    if set(options) - allowed:
        raise ValueError('First-segment options cannot override job identity or geometry')
    projected = None
    prompt = segment.prompt
    if prompt_relay_plan is not None:
        projected, prompt, _ = project_prompt_relay_plan_to_long_video_window(
            prompt_relay_plan, 0, segment.plan.render_frames, 0,
            segment.plan.timeline_start_seconds, segment.plan.timeline_end_seconds)
    options.setdefault('context_audio', 'video_and_audio')
    result = build_long_video_conditioning(clip, video_vae, audio_vae,
        context=_empty_context(chain_id, 0), segment_index=0, context_frames=0,
        prompt=prompt, width=width, height=height, length=segment.plan.render_frames,
        return_details=True, **options)
    conditioning, latent, mux_audio, conditioned_prompt, media_map, report_text, details = result
    report = json.loads(report_text)
    if (details['context_active'] is not False or report['context_active'] is not False
            or report['segment_index'] != 0 or report['schema'] != LONG_VIDEO_SCHEMA):
        raise ValueError('First-segment builder returned an unexpected context owner')
    if details['resolved_task'] not in ('t2va', 'i2va') or details['refs']:
        raise ValueError('Progressive first window currently requires T2VA or one first-frame I2VA without references')
    clean = []
    for embedding, metadata in conditioning:
        metadata = dict(metadata)
        if metadata.pop(LONG_VIDEO_CONDITIONING_KEY, None) != LONG_VIDEO_SCHEMA:
            raise ValueError('First-segment conditioning lacks its native builder schema')
        clean.append([embedding, metadata])
    # Runtime still applies its exact native keyframe, shape, tag and metadata
    # guards. Removing this one marker is not a general whitelist bypass.
    return (clean, latent, mux_audio, conditioned_prompt, media_map, report_text, details), projected


def bind_first_relay(model, result, projected, clip, query_chunk_rows):
    if type(query_chunk_rows) is not int or not 32 <= query_chunk_rows <= 2048:
        raise ValueError('First-segment Relay query_chunk_rows must be32..2048')
    positive, latent, _, prompt, _, _, details = result
    if projected is None:
        return model, positive, positive, None
    binding = relay.build_prompt_relay_binding(clip, projected, prompt, positive, details['tokens'])
    if binding['query_route'] == 'joint_av_exp' and details['audio_mode'] == 'lock_source':
        raise ValueError('First-segment joint_av_exp cannot route locked source audio; use video_only_paper')
    # Preserve the existing intentional single/no-event no-patch semantics.
    if len(binding['events']) <= 1:
        return model, positive, positive, dict(status='passthrough',
            global_plan_hash=projected['global_plan_hash'], projected_plan_hash=projected['plan_hash'])
    video, audio = latent['samples'].unbind()
    layout = relay.build_packed_layout(binding['text_len'], *video.shape[2:], audio.shape[-1],
        keyframes=details['keyframes'], refs=details['refs'], frame_count=details['frame_count'])
    binding = relay._bind_layout_contract(binding, layout, resolved_task=details['resolved_task'],
        keyframes=details['keyframes'], refs=details['refs'])
    # Public workflows place independent TST before Long. Authenticate and
    # unwrap ONLY that owned composition before binding the first Relay plan,
    # then restore TST as the outer owner. The sampler already performs this
    # ordering for prepared continuation stages; do not loosen Relay's guard
    # or silently discard TST/its original backend and live LoRA descriptors.
    from .tst_model import build_tst_model, detach_tst_model
    base, tst_spec = detach_tst_model(model)
    model, positive, negative, _, report = install_relay_stage(
        base, binding, positive, positive, query_chunk_rows)
    if tst_spec is not None:
        model, _ = build_tst_model(model,
            torch.tensor(tst_spec['full_sigmas'], dtype=torch.float64),
            mode=tst_spec['mode'], tau=tst_spec['tau'], max_workspace_mib=tst_spec['workspace'])
    report.update(global_plan_hash=projected['global_plan_hash'],
                  projection=projected['long_video_projection'])
    return model, positive, negative, report
