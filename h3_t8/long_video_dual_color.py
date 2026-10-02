"""Retry-safe RGB-only color correction against the actual accepted predecessor."""
from collections import deque
import json
import torch

from .long_video import LONG_VIDEO_SCHEMA
from .long_video_delivery import _resolve_inside, _sha256_file
from .long_video_color_match_advanced import process_long_video_color_match


COLOR_MATCH_MODES = ('bounded_spatial_v2', 'bounded_spatial_temporal_exp', 'bounded_motion_color_exp')


def correct_dual_segment_color(frames, root, chain_id, segment_index, parent_candidate_id,
                               enabled=True, mode='bounded_spatial_v2'):
    if mode not in COLOR_MATCH_MODES:
        raise ValueError('Unknown dual Color Match mode')
    if not enabled or segment_index == 0:
        return frames, {'status': 'disabled' if not enabled else 'first_segment_identity',
            'enabled': bool(enabled), 'mode': mode,
            'audio_touched': False, 'latent_touched': False}
    import av
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf8'))
    if manifest.get('chain_id') != chain_id:
        raise ValueError('Color Match predecessor belongs to another chain')
    previous = manifest['segments'][segment_index - 1]
    if previous['index'] != segment_index - 1 or previous['candidate_id'] != parent_candidate_id:
        raise ValueError('Color Match predecessor candidate identity mismatch')
    path = _resolve_inside(root, root/previous['video_path'])
    if _sha256_file(path) != previous['video_sha256']:
        raise ValueError('Color Match accepted predecessor checksum changed')
    tail = deque(maxlen=5)
    count = 0
    with av.open(str(path)) as container:
        container.streams.video[0].thread_count = 2
        for frame in container.decode(video=0):
            tail.append(torch.from_numpy(frame.to_ndarray(format='rgb24')).float()/255)
            count += 1
    if count != previous['frame_count'] or not tail:
        raise ValueError('Color Match accepted predecessor frame count changed')
    if _sha256_file(path) != previous['video_sha256']:
        raise ValueError('Color Match predecessor changed during decode')
    context = {'schema': LONG_VIDEO_SCHEMA, 'empty': False, 'metadata': {
        'chain_id': chain_id, 'source_segment_index': segment_index-1,
        'target_segment_index': segment_index}}
    reference = torch.stack(list(tail))
    output, _, report = process_long_video_color_match(frames, context, chain_id, segment_index,
        temporal_stabilization=mode != 'bounded_spatial_v2',
        _reference_frames=reference, _persist_state=False)
    payload = json.loads(report)
    if mode == 'bounded_motion_color_exp' and payload['status'] != 'ABSTAIN_SCENE_CUT_OR_LARGE_COLOR_JUMP':
        from .long_video_motion_color import stabilize_local_motion_color
        output, payload['local_motion_stabilization'] = stabilize_local_motion_color(output, reference)
        payload['maximum_total_rgb_delta'] = float((output[..., :3].float() - frames[..., :3].float()).abs().max())
        if payload['local_motion_stabilization']['applied']:
            payload.update(applied=True, status='COLOR_MATCH_MOTION_LOCAL_APPLIED')
    payload.update(mode=mode, predecessor_candidate_id=parent_candidate_id,
        predecessor_video_sha256=previous['video_sha256'], state_policy='derive_from_verified_accepted_video_no_sidecar')
    return output, payload
