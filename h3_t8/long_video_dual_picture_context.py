"""Ground LOW continuation in the accepted movie; leave high AV unchanged.

RGB24 -> existing Lanczos resize -> native VAE encode matches the accepted
2026-09-13 probe. Read docs/DUAL_MODEL_SEAM_FIX_20260913.md before changing it.
"""
from collections import deque
import hashlib
import json

import torch

from .long_video_delivery import _resolve_inside, _sha256_file

NAME = 'accepted_picture_low_context_v1'
LEGACY = 'independent_low_x0'
TAIL_FRAMES = 39


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def accepted_source(root, parent, segment_index, chain_id):
    """Validate the immediate selected predecessor before any cache lookup."""
    if segment_index <= 0 or not parent:
        raise ValueError('Accepted picture context requires an immediate predecessor')
    root = root.resolve(strict=True)
    descriptor = _resolve_inside(root, root / 'candidates' / f'segment_{segment_index - 1:05d}'
                                 / parent / 'candidate.json')
    info = json.loads(descriptor.read_text(encoding='utf-8'))
    if (info['candidate_id'] != parent or info['chain_id'] != chain_id
            or info['index'] != segment_index - 1):
        raise ValueError('Not the accepted immediate predecessor')
    media = _resolve_inside(root, root / info['video_path'])
    sha = _sha256_file(media)
    if sha != info['video_sha256']:
        raise ValueError('Accepted predecessor media changed')
    count = info['frame_count']
    if not isinstance(count, int) or count < TAIL_FRAMES:
        raise ValueError('Incomplete predecessor picture tail')
    return media, {'name': NAME, 'source_media_sha256': sha,
                   'source_frame_interval': [count - TAIL_FRAMES, count],
                   'source_segment_index': segment_index - 1}


def decode_tail(media, source):
    import av
    tail, count = deque(maxlen=TAIL_FRAMES), 0
    with av.open(str(media)) as container:
        container.streams.video[0].thread_count = 2
        if container.streams.video[0].average_rate != 24:
            raise ValueError('Unexpected predecessor frame rate')
        for frame in container.decode(video=0):
            tail.append(torch.from_numpy(frame.to_ndarray(format='rgb24')))
            count += 1
    if count != source['source_frame_interval'][1] or len(tail) != TAIL_FRAMES:
        raise ValueError('Incomplete predecessor picture tail')
    if _sha256_file(media) != source['source_media_sha256']:
        raise ValueError('Accepted predecessor media changed during decode')
    return torch.stack(list(tail)).float() / 255


def reencode_tail(context, frames, vae, resize, width, height):
    if len(frames) != TAIL_FRAMES or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError('The source must be exactly39 RGB frames')
    old = context['video_tail']
    with torch.inference_mode():
        encoded = vae.encode(resize(frames, width, height)).to(old).contiguous()
    if encoded.shape != old.shape or not torch.isfinite(encoded).all():
        raise ValueError('Re-encoded accepted context has invalid latent geometry/values')
    provenance = {'name': NAME, 'source': 'accepted_candidate_decoded_last39_frames',
                  'old_low_video_sha256': tensor_sha(old),
                  'encoded_video_sha256': tensor_sha(encoded),
                  'encoded_shape': list(encoded.shape),
                  'audio_tensor_preserved': True, 'additional_sampling_nfe': 0}
    metadata = {**context['metadata'], 'video_sha256': tensor_sha(encoded),
                'video_shape': json.dumps(list(encoded.shape)), 'video_dtype': str(encoded.dtype),
                'accepted_picture_context': provenance}
    return {**context, 'video_tail': encoded, 'metadata': metadata}, provenance


def prepare_context(context, media, source, vae, width, height):
    from .core import resize_image
    frames = decode_tail(media, source)
    result, report = reencode_tail(context, frames, vae, resize_image, width, height)
    return result, {**report, **source}
