"""Native progressive stage pairing without loading or merging model weights."""

from __future__ import annotations

import hashlib
import json


def _architecture(model):
    network = model.model.diffusion_model
    config = model.model.model_config.unet_config
    geometry_keys = ('hidden_size', 'num_layers', 'token_refiner_num_layers',
                     'num_attention_heads', 'attention_head_dim', 'ffn_hidden_size',
                     'text_dim', 'timestep_input_dim', 'time_embed_hidden_size',
                     'time_embed_dim', 'rope_inv_freq_len', 'norm_eps', 'qk_norm_eps')
    return {
        'network_type': f'{type(network).__module__}.{type(network).__qualname__}',
        'geometry': {key: config.get(key) for key in geometry_keys},
        'parameter_shapes': {key: list(value.shape) for key, value in network.named_parameters()},
        'buffer_shapes': {key: list(value.shape) for key, value in network.named_buffers()},
    }


def stage_report(model):
    patches = getattr(model, 'patches', {})
    return {'patch_set_uuid': str(getattr(model, 'patches_uuid', '')),
            'weight_patch_targets': len(patches),
            'weight_patch_entries': sum(len(entries) for entries in patches.values())}


def validate_stage_pair(low, high, low_sampling, high_sampling):
    """One native schedule requires identical AV coordinates across the boundary.

    The models may differ in weights/LoRA and storage dtype; their geometry and
    sampler-space transformations must agree. This is not an asset hash or a
    proof of visual compatibility between unrelated trained checkpoints.
    """
    left, right = _architecture(low), _architecture(high)
    if left != right:
        raise ValueError('Progressive stage MODEL architecture mismatch')
    for name in ('shift', 'audio_shift', 'audio_scale', 'multiplier', 'noise_scale'):
        default = 1.0 if name == 'noise_scale' else None
        if getattr(low_sampling, name, default) != getattr(high_sampling, name, default):
            raise ValueError(f'Progressive stage sampling mismatch: {name}')
    low_format, high_format = (model.get_model_object('latent_format') for model in (low, high))
    for name in ('scale_factor', 'shift_factor', 'latent_channels',
                 'spacial_downscale_ratio', 'temporal_downscale_ratio'):
        if getattr(low_format, name, None) != getattr(high_format, name, None):
            raise ValueError(f'Progressive stage latent-format mismatch: {name}')
    return {'architecture_sha256': hashlib.sha256(json.dumps(left, sort_keys=True).encode()).hexdigest(),
            'sampling_coordinates': 'shared_native_av',
            'low': stage_report(low), 'high': stage_report(high),
            'qualification': 'shape_and_coordinate_contract_not_trained_quality'}
