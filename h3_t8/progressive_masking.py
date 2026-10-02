"""Internal native-H3 mask/clean-anchor primitives, not a public sampler route.

The solver's starting state and the inpaint wrapper's clean source are different
objects during a progressive restart. Keep Core's input injection, token-grid
masks, conditional timesteps and AV scaling; do not substitute a post-CFG pin.
No external SelfLift source or global sampler/model mutation is used here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from .patch_stack_policy import warn_patch_stack


def _finite(tensor, name):
    if (not isinstance(tensor, torch.Tensor) or tensor.is_nested or
            not tensor.is_floating_point() or not tensor.numel() or
            not bool(torch.isfinite(tensor).all())):
        raise ValueError(f'{name} must be a nonempty finite floating tensor')


def _av_parts(value, name):
    if not getattr(value, 'is_nested', False):
        raise ValueError(f'{name} must contain native nested video and audio')
    parts = value.unbind()
    if len(parts) != 2:
        raise ValueError(f'{name} requires exactly two AV streams')
    for part, prefix, ndim, label in zip(parts, ((1, 24), (1, 32, 2)), (5, 4), ('video', 'audio')):
        _finite(part, f'{name} {label}')
        if part.ndim != ndim or tuple(part.shape[:len(prefix)]) != prefix:
            raise ValueError(f'{name} has an invalid batch-1 H3 {label} shape')
    return parts


def resize_video_source(video, height, width):
    """Bilinear source resize per latent frame: no temporal interpolation."""
    _finite(video, 'video source')
    if video.ndim != 5 or tuple(video.shape[:2]) != (1, 24):
        raise ValueError('Expected batch-1 native H3 video source')
    if any(type(n) is not int or n < 1 for n in (height, width)):
        raise ValueError('Source target dimensions must be positive integers')
    # Fold T into the batch dimension, so time cannot be resampled or averaged.
    frames = video.permute(0, 2, 1, 3, 4).reshape(-1, 24, *video.shape[-2:])
    result = F.interpolate(frames.float(), size=(height, width), mode='bilinear', align_corners=False)
    return result.reshape(1, video.shape[2], 24, height, width).permute(0, 2, 1, 3, 4).to(video.dtype)


def normalize_av_masks(noise_mask, video, audio):
    """1 generates, 0 keeps source; a plain mask controls video only.

    Video accepts HW, THW, 1THW or 11THW. Time must be one (broadcast) or
    exactly T. Audio accepts 12A or 11{1,2}{1,A}; no audio-time interpolation.
    Channels are shared explicitly, matching H3 conditional-time semantics.
    Spatial masks use Core's bilinear-equivalent resize, not nearest rounding.
    """
    import comfy.nested_tensor
    import comfy.utils

    _av_parts(comfy.nested_tensor.NestedTensor((video, audio)), 'mask target')
    if noise_mask is None:
        return None
    masks = list(noise_mask.unbind()) if getattr(noise_mask, 'is_nested', False) else [noise_mask]
    if not 1 <= len(masks) <= 2:
        raise ValueError('Mask requires video and optionally audio, no extra streams')
    for mask in masks:
        _finite(mask, 'mask')
        if bool(((mask < 0) | (mask > 1)).any()):
            raise ValueError('Mask values must be in [0,1]')
    vm = masks[0]
    if vm.ndim in (2, 3):
        vm = vm.reshape(1, 1, -1, *vm.shape[-2:])
    elif vm.ndim == 4:
        vm = vm.unsqueeze(1)
    if vm.ndim == 5 and vm.shape[1] == 24 and torch.equal(vm, vm[:, :1].expand_as(vm)):
        vm = vm[:, :1]
    if vm.ndim != 5 or tuple(vm.shape[:2]) != (1, 1) or vm.shape[2] not in (1, video.shape[2]):
        raise ValueError('Video mask requires shared channels and time 1 or the exact latent time')
    # Use Core's actual operation order (trilinear), not an algebraically
    # equivalent bilinear path with different fp32 rounding. The time guard
    # above permits only unchanged time or constant-frame broadcast.
    vm = comfy.utils.reshape_mask(vm.float(), video.shape)
    if len(masks) == 1:
        am = torch.ones(audio.shape, device=audio.device, dtype=torch.float32)
    else:
        am = masks[1]
        if am.ndim == 3:
            am = am.unsqueeze(1)
        if am.ndim == 4 and am.shape[1] == 32 and torch.equal(am, am[:, :1].expand_as(am)):
            am = am[:, :1]
        if (am.ndim != 4 or tuple(am.shape[:2]) != (1, 1) or
                am.shape[2] not in (1, 2) or am.shape[3] not in (1, audio.shape[3])):
            raise ValueError('Audio mask requires shared channels and exact time or broadcast 1')
        am = am.float().expand(audio.shape).contiguous()
    return comfy.nested_tensor.NestedTensor((vm.to(video.device), am.to(audio.device)))


def sampler_with_clean_anchor(sampler, clean_source, conditioning_noise):
    """Wrap a newly scoped native Euler sampler without changing its start state.

    Inputs are raw VAE-space AV source and native unscaled conditioning noise.
    Core constructs its initial solver state first. Immediately before Euler,
    bind the independent clean anchor to the fresh KSamplerX0Inpaint instance.
    Restore its fields even on cancellation. Callers must also pass denoise_mask
    to native sampling so extra_conds retains H3's known-token time semantics.
    This internal primitive does NOT qualify progressive continuation or EAV.
    """
    import comfy.k_diffusion.sampling
    import comfy.model_base
    import comfy.samplers
    import comfy.utils

    if (type(sampler) is not comfy.samplers.KSAMPLER or
            sampler.sampler_function is not comfy.k_diffusion.sampling.sample_euler or
            sampler.inpaint_options or
            any(k != 's_churn' or v != 0 for k, v in sampler.extra_options.items())):
        raise ValueError('Clean-anchor adapter requires unmodified native Euler with no churn')
    sources = _av_parts(clean_source, 'clean source')
    noises = _av_parts(conditioning_noise, 'conditioning noise')
    shapes = [tuple(t.shape) for t in sources]
    if [tuple(t.shape) for t in noises] != shapes:
        raise ValueError('Conditioning noise must match both clean source shapes')
    # Snapshot caller inputs now; later graph operations must not change anchors.
    sources = [t.detach().clone() for t in sources]
    noises = [t.detach().clone() for t in noises]
    native_euler = sampler.sampler_function

    def anchored_euler(model_k, state, sigmas, *, extra_args, **kwargs):
        if type(model_k) is not comfy.samplers.KSamplerX0Inpaint:
            raise RuntimeError('Expected the native inpaint wrapper')
        base = model_k.inner_model.inner_model
        if type(base) is not comfy.model_base.MiniMaxH3:
            raise RuntimeError('Clean anchors require native H3 AV scaling')
        if [tuple(s) for s in base.latent_shapes or []] != shapes:
            raise RuntimeError('Clean source does not match actual stage AV shapes')
        mask = extra_args.get('denoise_mask')
        if not isinstance(mask, torch.Tensor) or mask.shape != state.shape:
            raise RuntimeError('Pass the native packed AV denoise_mask to sampling')
        if extra_args.get('model_options', {}).get('denoise_mask_function') is not None:
            warn_patch_stack('Dynamic mask callable retained; clean-anchor semantics are unverified')
        anchor = comfy.utils.pack_latents([t.to(state) for t in sources])[0]
        anchor = base.process_latent_in(anchor)
        noise = comfy.utils.pack_latents([t.to(state) for t in noises])[0]
        original_anchor, original_noise = model_k.latent_image, model_k.noise
        try:
            model_k.latent_image, model_k.noise = anchor, noise
            return native_euler(model_k, state, sigmas, extra_args=extra_args, **kwargs)
        finally:
            model_k.latent_image, model_k.noise = original_anchor, original_noise

    return comfy.samplers.KSAMPLER(anchored_euler, dict(sampler.extra_options), {})
