"""Global Relay projection on authenticated native continuation conditions.

Only the spatial grid differs between stages. Existing long-video projection,
token binding and Core motion-layout repair supply every temporal coordinate.
No model/VAE work or mutation of the caller's global plan occurs here.
"""

import json

from . import prompt_relay_advanced as relay
from .conditioning import build_packed_layout
from .long_video import LONG_VIDEO_PATCH_VERSION, repair_long_video_layout
from .prompt_relay_long_video_advanced import (
    PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY, project_prompt_relay_plan_to_long_video_window,
)
from .progressive_relay import install_relay_stage


def project_for_source(source, global_plan, length, accepted_end_frame=None):
    binding = source.revalidate()
    request = binding['request']
    start = binding['accepted']['timeline_end_frame']
    end = (min(int(global_plan['frame_count']), start + length - request['context_frames'])
           if accepted_end_frame is None else accepted_end_frame)
    if type(start) is not int or type(end) is not int:
        raise ValueError('Continuation Relay requires exact integer timeline frames')
    return project_prompt_relay_plan_to_long_video_window(global_plan,
        request['segment_index'], length, request['context_frames'], start / 24, end / 24)[0]


def bind_prepared_relay(clip, projected, low, high, query_chunk_rows):
    if type(query_chunk_rows) is not int or not 32 <= query_chunk_rows <= 2048:
        raise ValueError('Continuation Relay query_chunk_rows must be32..2048')
    bindings = {}
    for phase, result in (('low', low), ('high', high)):
        conditioning, latent, _, prompt, _, _, details = result
        binding = relay.build_prompt_relay_binding(clip, projected, prompt, conditioning, details['tokens'])
        if binding['query_route'] == 'joint_av_exp' and details['audio_mode'] == 'lock_source':
            raise ValueError('Continuation joint_av_exp cannot route locked source audio; use video_only_paper')
        if not details['resolved_task'].lower().endswith('-motion'):
            raise ValueError('Continuation Relay requires native motion conditioning')
        video, audio = latent['samples'].unbind()
        layout = build_packed_layout(binding['text_len'], *video.shape[2:], audio.shape[-1],
            keyframes=details['keyframes'], refs=details['refs'], frame_count=details['frame_count'])
        layout = repair_long_video_layout(layout, list(details['keyframes']), list(details['refs']),
                                          details['frame_count'])
        bindings[phase] = relay._bind_layout_contract(binding, layout,
            resolved_task=details['resolved_task'], keyframes=details['keyframes'], refs=details['refs'])
    if bindings['low']['events'] != bindings['high']['events']:
        raise ValueError('Continuation stages disagree on Relay token/event coordinates')
    return {'bindings': bindings, 'projected_plan': projected, 'query_chunk_rows': query_chunk_rows}


def prepare_continuation_relay_stage(base, prepared, *, low):
    prepared.verify()
    phase = 'low' if low else 'high'
    positive = (prepared.low if low else prepared.high)[0]
    contract = prepared.relay
    from .enhance_a_video_advanced import _assert_long_video_contract
    request = prepared.source.binding['request']
    _assert_long_video_contract(base, segment_index=request['segment_index'],
                               context_frames=request['context_frames'])
    # The raw model was validated before our own motion owner was installed.
    # Inspect the underlying native Relay contract separately, as the existing
    # long-video builder does. Never remove the motion patch from execution.
    inspection = base.clone()
    inspection.object_patches.pop('extra_conds')
    model, positive, negative, backend, report = install_relay_stage(base,
        contract['bindings'][phase], positive, positive, contract['query_chunk_rows'],
        allowed_extra_conds_versions=(LONG_VIDEO_PATCH_VERSION,), inspection_model=inspection)
    projected = contract['projected_plan']
    attachment = {'schema': 1, 'global_plan_hash': projected['global_plan_hash'],
        'projected_plan_hash': projected['plan_hash'], 'binding_hash': report['binding_hash'],
        'segment_index': prepared.source.binding['request']['segment_index'],
        'accepted_source_sha256': prepared.source.sha256}
    model.set_attachments(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY, attachment)
    report.update(global_plan_hash=projected['global_plan_hash'],
        projection=json.loads(json.dumps(projected['long_video_projection'])),
        accepted_source_sha256=prepared.source.sha256)
    return model, positive, negative, backend, report
