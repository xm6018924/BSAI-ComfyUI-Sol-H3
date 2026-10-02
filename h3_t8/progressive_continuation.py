"""Accepted-parent sources for progressive continuation (internal, not a runner).

The caller holds the existing exclusive chain loop lock and supplies its verified
execution-contract SHA (models, LoRA contents/order, VAE, prompts and code). This
module adds actual accepted media/context identities; it does not authenticate a
user-supplied job label or qualify sampling, resume execution, or media quality.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import torch

from . import long_video_dual_picture_context as picture
from . import long_video_delivery as delivery
from .long_video import CONTEXT_FRAME_STEPS, _validate_context, sanitize_chain_id
from .core import AUDIO_LATENT_FPS, FPS


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _digest(value, name):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name} must be a verified lowercase SHA256')


def _checked_context(root, entry, chain_id, segment_index, context_frames,
                     width, height, job_sha256):
    value = entry.get('context_path')
    if not isinstance(value, str) or not value:
        raise ValueError('Accepted predecessor has no completed AV context')
    path = delivery._resolve_inside(root, value)
    expected = entry.get('context_sha256')
    _digest(expected, 'Accepted context SHA256')
    if delivery._sha256_file(path) != expected:
        raise ValueError('Accepted context file changed')
    context, _ = delivery._load_accepted_context_file(path, chain_id, segment_index - 1, segment_index)
    _validate_context(context, segment_index, context_frames, width, height)
    metadata = context['metadata']
    if metadata['sampling_summary'] != job_sha256:
        raise ValueError('Accepted context belongs to another execution contract')
    overhang = metadata['audio_overhang']
    if not math.isfinite(overhang) or not 0 <= overhang < 1:
        raise ValueError('Accepted context has an invalid audio overhang')
    if metadata['max_context_frames'] != picture.TAIL_FRAMES:
        raise ValueError('Accepted-picture preparation requires the full39-frame tail capacity')
    for name in ('video_tail', 'audio_tail'):
        tensor = context[name]
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError('Accepted context must contain finite floating AV tensors')
    if context['audio_tail'].shape[-1] < round(context_frames / FPS * AUDIO_LATENT_FPS):
        raise ValueError('Accepted context has insufficient completed audio tail')
    if delivery._sha256_file(path) != expected:
        raise ValueError('Accepted context changed during load')
    return context, path


@dataclass(frozen=True)
class ProgressiveContinuationSource:
    """Immutable descriptor; every prepare/cache-use boundary must revalidate it."""

    root: Path
    binding_json: str

    @property
    def binding(self):
        # Never expose shared mutable dictionaries or live context tensors.
        return json.loads(self.binding_json)

    @property
    def sha256(self):
        return hashlib.sha256(self.binding_json.encode()).hexdigest()

    def revalidate(self):
        binding = self.binding
        current = capture_continuation_source(self.root, **binding['request'])
        if current.binding_json != self.binding_json:
            raise ValueError('Progressive accepted-parent identity changed; do not reuse stage cache')
        return self.binding

    def prepare_contexts(self, video_vae):
        """RGB24 -> existing resize -> native VAE; no LOW x0 or latent bridge."""
        binding = self.revalidate()
        request = binding['request']
        high, _ = _checked_context(self.root, binding['accepted'], request['chain_id'],
            request['segment_index'], request['context_frames'], request['width'],
            request['height'], request['job_sha256'])
        high['metadata'].update(accepted_candidate_id=request['parent_candidate_id'],
                                manifest_revision=request['parent_revision'])
        audio_before = picture.tensor_sha(high['audio_tail'])
        high_before = picture.tensor_sha(high['video_tail'])
        # Only a geometry/dtype scaffold for the existing encoder adapter. It
        # must never reach a sampler; the actual LOW source is decoded RGB.
        shape = (*high['video_tail'].shape[:-2], request['low_height'] // 16, request['low_width'] // 16)
        scaffold = {**high, 'video_tail': high['video_tail'].new_zeros(shape),
                    'metadata': dict(high['metadata'])}
        media = delivery._resolve_inside(self.root, binding['accepted']['video_path'])
        low, report = picture.prepare_context(scaffold, media, binding['picture'], video_vae,
                                               request['low_width'], request['low_height'])
        if (low['audio_tail'] is not high['audio_tail']
                or picture.tensor_sha(high['audio_tail']) != audio_before
                or picture.tensor_sha(high['video_tail']) != high_before):
            raise RuntimeError('LOW picture preparation mutated the completed HIGH/AV source')
        self.revalidate()
        return low, high, {'source_sha256': self.sha256, 'picture': report,
            'context_frames': request['context_frames'],
            'context_steps': CONTEXT_FRAME_STEPS[request['context_frames']],
            'tail_capacity_frames': picture.TAIL_FRAMES,
            'low_source': 'accepted_rgb24_resize_vae', 'high_source': 'accepted_completed_av_context',
            'audio_tensor_shared_unchanged': True, 'additional_sampling_nfe': 0,
            'sampling_and_resume_qualified': False}

    def prepare_conditions(self, *, clip, video_vae, audio_vae, prompt, length,
                           context_audio='video_and_audio', **options):
        """Build real LOW/HIGH native conditioning; no sampling or MODEL patch.

        The eventual stage composer must authenticate and install the long-video
        payload/Relay/EAV owners. These outputs are deliberately NOT accepted by
        today's public progressive sampler's plain-T2VA/I2VA whitelist.
        """
        from .long_video import build_long_video_conditioning
        from .long_video_dual_model_runner import lock_high_video_prefix
        allowed = {'task_type', 'audio_mode', 'audio_denoise_strength', 'add_source_as_reference',
            'prompt_primary_audio_ordinal', 'strict_prompt_tags', 'ref_image_size',
            'reference_video_policy', 'drive_audio', 'final_audio', 'first_frame', 'last_frame',
            'ref_images', 'ref_videos', 'ref_video_audios', 'ref_audios', 'first_frame_reuse',
            'persistent_identity_image', 'persistent_identity_strategy', 'persistent_identity_interval'}
        if set(options) - allowed:
            raise ValueError('Continuation condition options cannot override source identity or geometry')
        _integer(length, 'length')
        request = self.binding['request']
        if length <= request['context_frames']:
            raise ValueError('Continuation must leave newly generated frames')
        low, high, preparation = self.prepare_contexts(video_vae)
        arguments = dict(clip=clip, video_vae=video_vae, audio_vae=audio_vae,
            segment_index=request['segment_index'], context_frames=request['context_frames'],
            context_audio=context_audio, prompt=prompt, length=length, return_details=True, **options)
        low_result = build_long_video_conditioning(context=low, width=request['low_width'],
                                                  height=request['low_height'], **arguments)
        high_result = build_long_video_conditioning(context=high, width=request['width'],
                                                   height=request['height'], **arguments)
        high_result = list(high_result)
        high_result[1], mask_report = lock_high_video_prefix(high_result[1], high,
            chain_id=request['chain_id'], segment_index=request['segment_index'],
            context_frames=request['context_frames'])
        self.revalidate()
        return low_result, tuple(high_result), {**preparation, 'high_prefix': mask_report,
            'stage_conditioning': 'existing_native_long_video_builder',
            'long_video_model_owner_installed': False}


def capture_continuation_source(root, *, chain_id, segment_index, parent_candidate_id,
                                parent_revision, job_sha256, context_frames,
                                width, height, low_width, low_height):
    """Read selected manifest + actual files before looking up any stage cache.

    Deliberately fail closed on a missing/corrupt primary manifest. Recovery of
    the durable chain itself remains the existing delivery layer's job.
    """
    root = Path(root).resolve(strict=True)
    if type(chain_id) is not str or sanitize_chain_id(chain_id) != chain_id:
        raise ValueError('Use the exact normalized chain_id')
    if (type(parent_candidate_id) is not str
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', parent_candidate_id) is None):
        raise ValueError('Invalid parent_candidate_id')
    for name, value in (('segment_index', segment_index), ('parent_revision', parent_revision),
                        ('context_frames', context_frames)):
        _integer(value, name)
    if context_frames not in CONTEXT_FRAME_STEPS:
        raise ValueError('Continuation context_frames must be5/22/39')
    _digest(job_sha256, 'Execution contract')
    for name, value in (('width', width), ('height', height), ('low_width', low_width), ('low_height', low_height)):
        _integer(value, name, 32)
        if value % 32:
            raise ValueError('Progressive canvases must align to32 pixels')
    if low_width >= width or low_height >= height:
        raise ValueError('LOW canvas must be smaller than HIGH in both dimensions')
    manifest_path = delivery._resolve_inside(root, delivery.MANIFEST_NAME)
    raw_manifest = manifest_path.read_bytes()
    manifest = delivery._validate_manifest(json.loads(raw_manifest), chain_id)
    if type(manifest['revision']) is not int or manifest['revision'] != parent_revision:
        raise ValueError('Selected manifest revision changed')
    if len(manifest['segments']) < segment_index:
        raise ValueError('Missing accepted immediate predecessor')
    entry = manifest['segments'][segment_index - 1]
    if entry['candidate_id'] != parent_candidate_id or entry.get('is_final_segment'):
        raise ValueError('Selected predecessor changed or is already final')
    if entry['sampling_summary'] != job_sha256:
        raise ValueError('Accepted predecessor belongs to another execution contract')
    if (entry['width'], entry['height'], entry['fps']) != (width, height, 24):
        raise ValueError('Accepted predecessor canvas/fps changed')
    _, picture_source = picture.accepted_source(root, parent_candidate_id, segment_index, chain_id)
    descriptor = delivery._resolve_inside(root, root / 'candidates' / f'segment_{segment_index - 1:05d}'
                                           / parent_candidate_id / 'candidate.json')
    candidate = json.loads(descriptor.read_text(encoding='utf-8'))
    candidate_context = delivery._resolve_inside(root, candidate['context_path'])
    if (candidate.get('context_sha256') != entry.get('context_sha256')
            or delivery._sha256_file(candidate_context) != entry.get('context_sha256')):
        raise ValueError('Accepted context does not match selected candidate context')
    accepted_media = delivery._resolve_inside(root, entry['video_path'])
    if (entry['video_sha256'] != picture_source['source_media_sha256']
            or delivery._sha256_file(accepted_media) != entry['video_sha256']
            or entry['frame_count'] != picture_source['source_frame_interval'][1]):
        raise ValueError('Accepted copy does not match selected candidate media')
    high, context_path = _checked_context(root, entry, chain_id, segment_index, context_frames,
                                          width, height, job_sha256)
    if manifest_path.read_bytes() != raw_manifest:
        raise ValueError('Manifest changed during continuation source validation')
    request = dict(chain_id=chain_id, segment_index=segment_index, parent_candidate_id=parent_candidate_id,
        parent_revision=parent_revision, job_sha256=job_sha256, context_frames=context_frames,
        width=width, height=height, low_width=low_width, low_height=low_height)
    binding = {'schema': 't8.progressive.accepted_parent.v1', 'request': request,
        'manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(),
        'accepted': {'video_path': accepted_media.relative_to(root).as_posix(),
            'timeline_end_frame': entry['timeline_end_frame'],
            'video_sha256': entry['video_sha256'], 'context_path': context_path.relative_to(root).as_posix(),
            'context_sha256': entry['context_sha256']}, 'picture': picture_source,
        'high_video_sha256': picture.tensor_sha(high['video_tail']),
        'completed_audio_sha256': picture.tensor_sha(high['audio_tail']),
        'context_metadata_sha256': _sha(high['metadata'])}
    return ProgressiveContinuationSource(root, _json(binding))
