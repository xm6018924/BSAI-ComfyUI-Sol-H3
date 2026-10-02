"""Strict decode-only adapter for Kijai's 24-channel, 16x H3 tiny 2D VAE.

Uses Comfy's no-initialization TAE primitives, not KJNodes or a generic graph
guessed from missing keys. These are independent latent-frame previews, not
the temporal TAEHV decoder or a reconstructed 24fps movie.
"""
from __future__ import annotations

import torch
from torch import nn


def build_decoder(state):
    from comfy.taesd.taesd import Block, Clamp, conv

    # The published checkpoint has a 96 -> 64 transition and four upscales.
    # Spell out the known graph so missing parameter indices cannot invent
    # arbitrary upscales, GN blocks, channel counts or executable modules.
    model = nn.Sequential(
        Clamp(), conv(24, 96), nn.ReLU(),
        Block(96, 96), Block(96, 96), Block(96, 96),
        nn.Upsample(scale_factor=2), conv(96, 96, bias=False),
        Block(96, 96), Block(96, 96), Block(96, 96),
        nn.Upsample(scale_factor=2), conv(96, 96, bias=False),
        Block(96, 64), Block(64, 64), Block(64, 64),
        nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False), Block(64, 64), conv(64, 3),
    ).eval()
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError('Expected the known H3 2D tiny decoder key layout')
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Incompatible H3 2D tiny decoder tensor: {key}')
    model.load_state_dict(state, strict=True)
    return model


def decoder_state(state):
    """Accept one complete flat decoder, optionally under one known prefix."""
    if not isinstance(state, dict) or not state:
        raise ValueError('Empty H3 2D tiny decoder checkpoint')
    for prefix in ('', 'decoder.', 'taesd_decoder.'):
        if all(isinstance(key, str) and key.startswith(prefix)
               and key[len(prefix):].split('.', 1)[0].isdigit() for key in state):
            return {key[len(prefix):]: value for key, value in state.items()}
    raise ValueError('Unknown or mixed H3 2D tiny decoder checkpoint prefixes')


@torch.inference_mode()
def decode_frames(model, value, device, dtype, check):
    if (value.ndim != 5 or value.shape[:2] != (1, 24)
            or not 1 <= value.shape[2] <= 12 or min(value.shape[-2:]) < 1
            or max(value.shape[-2:]) > 32):
        raise ValueError('H3 2D preview requires a bounded [1,24,1..12,H,W] prefix')
    images = []
    for index in range(value.shape[2]):
        check()
        # No temporal state, audio, padding, per-frame batching or sampler input
        # mutation. Keep at most one decoded image on the inference device.
        frame = model(value[:, :, index].to(device=device, dtype=dtype))[0]
        check()
        if not bool(torch.isfinite(frame).all()):
            raise ValueError('H3 2D tiny decoder returned nonfinite pixels')
        images.append(frame.movedim(0, -1).float().cpu())
        del frame
    return torch.stack(images)
