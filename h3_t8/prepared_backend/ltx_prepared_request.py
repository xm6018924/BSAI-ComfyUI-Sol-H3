"""Reusable prepared-AV boundary; no weights, CUDA, normalization or resampling.

The upstream adapter/audio encoder remain separate prerequisites. This contract
does not turn an arbitrary H3 latent or arbitrary text into LTX conditioning.
"""
from pathlib import Path
import hashlib


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def geometry(values):
    from ltx_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
    if not isinstance(values, dict) or set(values) != {'frames', 'width', 'height', 'fps'}:
        raise ValueError('Provide exact prepared frame count, width, height and fps')
    if any(type(value) is not int for value in values.values()):
        raise ValueError('Prepared geometry must use exact integer values')
    frames, width, height, fps = (values[k] for k in ('frames', 'width', 'height', 'fps'))
    if fps != 24 or not 9 <= frames <= 192 or (frames - 1) % 8:
        raise ValueError('Use native 8n+1 frames, CFR24, no more than8s; no implicit padding/crop')
    if min(width, height) < 32 or width % 32 or height % 32:
        raise ValueError('Prepared LTX spatial grid requires multiples of32')
    pixel = VideoPixelShape(1, frames, height, width, fps)
    video = VideoLatentShape.from_pixel_shape(pixel)
    audio = AudioLatentShape.from_video_pixel_shape(pixel)
    if video.token_count() > 20480:
        raise ValueError('Prepared job exceeds the tested maximum video-token envelope')
    return pixel, video, audio


def validate_tensors(inputs, values):
    import torch
    _, video, audio = geometry(values)
    for name, shape in [('video', video.to_torch_shape()), ('audio', audio.to_torch_shape())]:
        tensor = inputs.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or tensor.is_meta:
            raise ValueError('Prepared AV must be real CPU tensors')
        if tensor.dtype != torch.bfloat16 or tensor.shape != shape or not torch.isfinite(tensor).all():
            raise ValueError(f'Prepared {name} shape/dtype/finite mismatch; never reshape or pad to bypass')
    return inputs


def load_prepared(request):
    import torch
    from safetensors import safe_open
    from runtime.prompt_cache import load_cache
    if request.get('schema') != 't8-ltx-prepared-refinement-v1':
        raise ValueError('Unsupported prepared refinement request')
    if request.get('normalization') != 'normalized_ltx_av' or request.get('reference_prefix_frames') != 0:
        raise ValueError('Only normalized LTX AV without appended reference prefixes is supported')
    if type(request.get('seed')) is not int or not 0 <= request['seed'] < 2**64:
        raise ValueError('Seed must be an unsigned64 integer, not bool')
    geometry(request['geometry'])
    for key in ('inputs', 'text_cache'):
        path = Path(request[key]).resolve(strict=True)
        if sha(path) != request[key + '_sha256']:
            raise ValueError(f'Prepared {key} identity changed')
    with safe_open(request['inputs'], framework='pt', device='cpu') as source:
        metadata = source.metadata() or {}
        for field, key in [('pixel_frames', 'frames'), ('width', 'width'), ('height', 'height'), ('fps', 'fps')]:
            if metadata.get(field) != str(request['geometry'][key]):
                raise ValueError('Prepared file metadata and requested timeline disagree')
    # Only load the two consumed tensors, not stored H3 intermediates/PCM.
        inputs = {name: source.get_tensor(name) for name in ('video', 'audio')}
    validate_tensors(inputs, request['geometry'])
    cache = load_cache(request['text_cache'], prompt=request['prompt'], torch_module=torch)
    return inputs, cache
