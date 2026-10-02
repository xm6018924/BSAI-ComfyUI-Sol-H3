"""Opt-in standard partial4 -> learned upscale -> joint AV refine4.

Legacy video-only chunk executors are intentionally not reused: they discard
sampled audio. No RGB/latent crossfade is introduced here.
"""
from __future__ import annotations

import json

import torch

import comfy.model_management
import comfy.nested_tensor

from . import chunked_two_pass_upscale_advanced as legacy
from .learned_latent_upscale_advanced import UPSTREAM_REFINE_VIDEO_SIGMAS
from .sampling import rebind_dual_clock_sampler


SCHEMA = 't8.minimax_h3.chunked_two_pass.standard_joint_4plus4.v5'
CONTRACT = 'standard_joint_4plus4_exp'


def standard_plan(plan, parity_report_json):
    # Serialized plans can be edited externally; recheck the original grid and
    # range contract without invoking the standard branch recursively.
    fields = ('model_name', 'target_width', 'target_height', 'temporal_chunk_frames',
        'temporal_overlap_frames', 'anchor_strength', 'tile_width', 'tile_height',
        'spatial_overlap', 'spatial_fade', 'minimum_tile_size', 'overlap_blend',
        'precision', 'release_policy', 'spatial_strategy')
    legacy.build_chunked_two_pass_plan(**{key: plan[key] for key in fields})
    if plan['temporal_chunk_frames'] < 17 or plan['temporal_overlap_frames'] < 0:
        raise ValueError('Temporal chunk/overlap must be positive/nonnegative')
    if plan['spatial_strategy'] != 'full_frame_safe':
        raise ValueError('Standard joint 4+4 requires full_frame_safe; no spatial tiles')
    try:
        report = json.loads(parity_report_json)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError('Connect the standard Learned Two-Pass Parity Plan report_json') from error
    if not isinstance(report, dict) or (
        report.get('node') != 'MiniMaxH3LearnedTwoPassParityPlanT8Advanced'
        or report.get('status') != 'upstream_schedule_reproduced'
        or [report.get(key) for key in ('base_steps', 'coarse_steps', 'refine_steps')] != [8, 4, 4]
    ):
        raise ValueError('Standard chunked contract requires the original base8/coarse4/refine4 parity plan')
    expected = torch.tensor(UPSTREAM_REFINE_VIDEO_SIGMAS[4], dtype=torch.float32)
    supplied = torch.as_tensor(report.get('refine_video_sigmas', []), dtype=torch.float32)
    if supplied.shape != expected.shape or not torch.allclose(supplied, expected, atol=1e-6, rtol=0):
        raise ValueError('Parity report does not contain the standard raw refine4 sigmas')
    coarse = torch.as_tensor(report.get('coarse_video_sigmas', []))
    if coarse.shape != (5,) or not bool(torch.isfinite(coarse).all()) or not bool(
        (coarse[:-1] > coarse[1:]).all()) or not 0 < float(coarse[-1]) < float(coarse[0]) <= 1:
        raise ValueError('Parity report must contain four coarse intervals')
    return {**plan, 'schema': SCHEMA, 'sampling_contract': CONTRACT,
        'parity_report': report, 'audio_policy': 'publish_refined_joint_audio',
        'upscale_policy': 'existing_learned_network_once_before_temporal_split',
        'overlap_policy': 'exact_completed_av_prefix_no_post_sampling_blend'}


def _parts(value, name):
    video, audio = legacy._nested_parts(value, name=name)
    if not isinstance(video, torch.Tensor) or not isinstance(audio, torch.Tensor):
        raise ValueError(f'{name} must contain torch video/audio tensors')
    if video.ndim != 5 or tuple(video.shape[:2]) != (1, 24) or min(video.shape[2:]) < 1:
        raise ValueError(f'{name}: expected batch1 H3 video [1,24,T,H,W]')
    if audio.ndim != 4 or tuple(audio.shape[:3]) != (1, 32, 2) or audio.shape[-1] < 1:
        raise ValueError(f'{name}: expected H3 audio [1,32,2,T]')
    if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
        raise ValueError(f'{name} contains NaN or Inf')
    return video, audio


def _audio_bounds(start_frame, end_frame, total_frames, audio_length):
    # Match the existing H3 frame->audio mapping; include final padding exactly.
    start = round(start_frame * legacy.FRAME_RESCALE)
    stop = audio_length if end_frame == total_frames else min(
        audio_length, round(end_frame * legacy.FRAME_RESCALE))
    if not 0 <= start < stop <= audio_length:
        raise ValueError('Temporal window has an empty/out-of-range audio interval')
    return start, stop


def _append_exact(previous, piece, start, dim, overlap):
    if previous is None:
        if start != 0 or overlap:
            raise ValueError('First temporal window must start at zero')
        return piece
    if start > previous.shape[dim] or overlap != min(previous.shape[dim] - start, piece.shape[dim]):
        raise ValueError('Temporal output has a gap or changed overlap coverage')
    if overlap and not torch.equal(piece.narrow(dim, 0, overlap), previous.narrow(dim, start, overlap)):
        raise RuntimeError('Sampler changed the completed read-only AV overlap')
    tail = piece.narrow(dim, overlap, piece.shape[dim] - overlap)
    return torch.cat((previous, tail), dim=dim)


def _restore_zero_mask(result, source, mask):
    """Reject real ownership drift, restore only bounded FP32 Euler roundoff.

    Even exact inpaint x0 can change a few ulps in x+(x-x0)/sigma*(-sigma).
    Editable/partial-mask values are never changed. This is not a seam blend.
    """
    source = source.to(result)
    protected = torch.broadcast_to(mask == 0, result.shape)
    delta = (result - source).abs()
    # Four Euler intervals + AV clock arithmetic: eight FP32 eps per unit
    # magnitude, checked elementwise (not a tensor-wide scale allowance).
    bound = 8 * torch.finfo(torch.float32).eps * (1 + source.abs())
    bad = protected & (delta > bound)
    if bool(bad.any()):
        raise RuntimeError('Sampler modified read-only AV beyond FP32 roundoff: '
            f'max_abs={float(delta[protected].max()):.9g}')
    stats = {'protected_values': int(protected.sum()),
        'roundoff_changed_values': int((protected & (delta != 0)).sum()),
        'max_abs_before_exact_restore': float(delta[protected].max()) if bool(protected.any()) else 0.,
        'roundoff_bound': '8*float32_eps*(1+abs(source)), elementwise'}
    return torch.where(protected, source, result), stats


def execute_standard_chunked(model, conditioning, latent, noise, sampler, sigmas,
                             plan, negative=None, cfg=1.0):
    # Re-validate the contract before any learned-network/model execution.
    checked = standard_plan(plan, json.dumps(plan.get('parity_report')))
    function = getattr(sampler, 'sampler_function', None)
    if getattr(function, '__name__', '') != 'sample_minimax_h3_dual_clock_euler':
        raise ValueError('Standard joint temporal 4+4 requires the T8 dual-clock Euler sampler')
    for clock in ('video', 'audio'):
        if getattr(function, f'_minimax_h3_shift_{clock}', None) != checked['parity_report'].get(f'shift_{clock}'):
            raise ValueError('Sampler clocks do not match the connected parity report')
    expected = torch.tensor(UPSTREAM_REFINE_VIDEO_SIGMAS[4], dtype=torch.float32)
    actual = torch.as_tensor(sigmas).detach().to(device='cpu', dtype=torch.float32)
    if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=1e-6, rtol=0):
        raise ValueError('Connect parity refine_sigmas unchanged: exactly four standard intervals')
    if not isinstance(latent, dict):
        raise ValueError('Connect pass-1 partial4 denoised_output LATENT')
    video, audio = _parts(latent.get('samples'), 'partial4 denoised input')
    if checked['schema'] != SCHEMA:
        raise ValueError('Invalid standard joint chunked plan')
    if not conditioning:
        raise ValueError('Connect HIGH conditioning with the same prompt/media/timeline')
    target = (plan['target_height'] // 16, plan['target_width'] // 16)
    for _, metadata in conditioning:
        for keyframe in metadata.get('minimax_keyframes', []):
            image = keyframe.get('latent')
            if image is not None and tuple(image.shape[-2:]) != target:
                raise ValueError('HIGH conditioning keyframes must match the target canvas')
    frames = legacy.frames_for_tokens(video.shape[2])
    segments, _ = legacy.compute_temporal_segments(video.shape[2],
        plan['temporal_chunk_frames'], plan['temporal_overlap_frames'])
    if len(segments) > 1 and plan['temporal_overlap_frames'] == 0:
        raise ValueError('Standard temporal 4+4 requires nonzero completed AV context overlap')
    # Shape/range validation happens before expensive inference.
    bounds = [_audio_bounds(sf, ef, frames, audio.shape[-1]) for _, sf, _, ef in segments]
    for key, value in latent.items():
        if key not in {'samples', 'noise_mask', 'batch_index', 'type'}:
            raise ValueError(f'Unsupported partial latent metadata for temporal slicing: {key}')
    comfy.model_management.throw_exception_if_processing_interrupted()
    enlarged, _, _, upscale_report = legacy.learned_upscale_h3_av_latent(
        latent, plan['model_name'], 'target_dimensions', 2.0, 1.0,
        plan['target_width'], plan['target_height'], 'honor_dimensions_exp', 2.0,
        plan['precision'], plan['release_policy'])
    high_video, high_audio = _parts(enlarged['samples'], 'upscaled partial4 input')
    if tuple(high_video.shape[2:]) != (video.shape[2], *target) or not torch.equal(high_audio, audio):
        raise RuntimeError('Learned upscaler changed time/audio or returned the wrong target canvas')
    global_video_noise, global_audio_noise, noise_report = legacy._build_global_target_av_noise(
        noise, latent, video, audio, plan)
    inherited_video = None
    inherited_audio = None
    if enlarged.get('noise_mask') is not None:
        inherited_video, _ = legacy._normalize_inherited_video_mask(enlarged, high_video,
            policy='inherit_required')
        _, source_audio_mask = legacy._nested_parts(enlarged['noise_mask'], name='inherited AV mask')
        try:
            inherited_audio = torch.broadcast_to(source_audio_mask, high_audio.shape)
        except RuntimeError as error:
            raise ValueError('Inherited audio mask does not match the full audio timeline') from error
        if not torch.isfinite(inherited_audio).all() or not bool(
            ((inherited_audio >= 0) & (inherited_audio <= 1)).all()):
            raise ValueError('Inherited audio mask must be finite and within [0,1]')
    published_video = published_audio = None
    reports = []
    for index, ((start, sf, stop, ef), (audio_start, audio_stop)) in enumerate(zip(segments, bounds, strict=True)):
        comfy.model_management.throw_exception_if_processing_interrupted()
        v = high_video[:, :, start:stop].clone()
        a = high_audio[..., audio_start:audio_stop].clone()
        video_overlap = 0 if published_video is None else min(published_video.shape[2] - start, v.shape[2])
        audio_overlap = 0 if published_audio is None else min(published_audio.shape[-1] - audio_start, a.shape[-1])
        if min(video_overlap, audio_overlap) < 0:
            raise ValueError('Temporal chunk plan left a gap in the AV timeline')
        vm = torch.ones((1, 1, v.shape[2], *target), dtype=v.dtype, device=v.device)
        am = torch.ones_like(a)
        if inherited_video is not None:
            vm *= inherited_video[:, :, start:stop].to(vm)
        if inherited_audio is not None:
            am *= inherited_audio[..., audio_start:audio_stop].to(am)
        if video_overlap:
            v[:, :, :video_overlap] = published_video[:, :, start:start + video_overlap]
            vm[:, :, :video_overlap] = 0
        if audio_overlap:
            a[..., :audio_overlap] = published_audio[..., audio_start:audio_start + audio_overlap]
            am[..., :audio_overlap] = 0
        piece = {'samples': comfy.nested_tensor.NestedTensor((v, a)),
            'noise_mask': comfy.nested_tensor.NestedTensor((vm, am))}
        bound_sampler = rebind_dual_clock_sampler(model, piece, sampler)
        positive = legacy.reanchor_conditioning(conditioning, sf, ef, target)
        neg = None if negative is None else legacy.reanchor_conditioning(negative, sf, ef, target)
        prepared_noise = comfy.nested_tensor.NestedTensor((
            global_video_noise[:, :, start:stop].contiguous(),
            global_audio_noise[..., audio_start:audio_stop].contiguous()))
        result = legacy.sample_piece(piece, positive, model, noise, bound_sampler,
            sigmas, neg, cfg, prepared_noise=prepared_noise)
        result_video, result_audio = _parts(result, 'joint refine4 output')
        if result_video.shape != v.shape or result_audio.shape != a.shape:
            raise RuntimeError('Refine4 sampler changed AV geometry')
        result_video, video_roundoff = _restore_zero_mask(result_video, v, vm)
        result_audio, audio_roundoff = _restore_zero_mask(result_audio, a, am)
        published_video = _append_exact(published_video, result_video, start, 2, video_overlap)
        published_audio = _append_exact(published_audio, result_audio, audio_start, -1, audio_overlap)
        reports.append({'index': index, 'video_tokens': [start, stop], 'pixel_frames': [sf, ef],
            'audio_tokens': [audio_start, audio_stop], 'refine_nfe': 4,
            'locked_video_overlap_tokens': video_overlap, 'locked_audio_overlap_tokens': audio_overlap,
            'video_read_only_roundoff': video_roundoff, 'audio_read_only_roundoff': audio_roundoff,
            'sampler_binding': 'rebound_per_upscaled_joint_av_piece', 'audio_output': 'refined_joint_audio'})
    if published_video.shape != high_video.shape or published_audio.shape != audio.shape:
        raise RuntimeError('Published refined AV does not cover the full source timeline')
    output = {key: value for key, value in latent.items() if key in {'batch_index', 'type'}}
    output['samples'] = comfy.nested_tensor.NestedTensor((published_video, published_audio))
    report = {'schema': SCHEMA, 'status': 'completed', 'sampling_contract': CONTRACT,
        'coarse_nfe_contract': 4, 'refine_nfe_per_window': 4, 'segment_count': len(segments),
        'executor_refine_model_calls_contract': 4 * len(segments),
        'total_model_calls_including_coarse_contract': 4 + 4 * len(segments),
        'per_temporal_region_schedule': 'coarse4_then_standard_refine4',
        'refine_sigmas': actual.tolist(), 'segments': reports,
        'source_frame_count': frames, 'audio_resampled': True, 'audio_preserved_by_identity': False,
        'final_audio_policy': 'publish_refined_joint_audio', 'post_sampling_blend': False,
        'learned_upscale_calls': 1, 'upscale': json.loads(upscale_report), 'global_noise': noise_report,
        'quality_qualified': False}
    return output, json.dumps(report, ensure_ascii=False, indent=2)
