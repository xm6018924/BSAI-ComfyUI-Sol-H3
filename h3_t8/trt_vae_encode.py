"""Core-compatible H3 encoding orchestration for an explicit moments backend.

Input is Core RGB [-1,1], not ComfyUI IMAGE [0,1]. Backend returns unnormalized
48-channel moments. Small spatial tiles require matching engines; no spatial
zero padding. Single-image temporal padding is explicit, never the default.
"""

from .trt_vae_contract import positive_int, tile_axis
from .trt_vae_decode import _blend


def encode_output_shape(shape):
    if len(shape) != 5 or shape[1] != 3:
        raise ValueError("Encoder needs B,3,T,H,W")
    for dim in shape:
        positive_int(dim, "encoder dimension")
    batch, _, frames, height, width = shape
    if height % 16 or width % 16:
        raise ValueError("Encoder spatial dimensions must be aligned to 16 pixels")
    tokens = 1 if frames == 1 else ((frames + 16) // 17) * 5 - 3
    return batch, 24, tokens, height // 16, width // 16


def required_encode_tiles(shape, single_frame_mode="native"):
    encode_output_shape(shape)
    if single_frame_mode not in ("native", "causal_zero17_first"):
        raise ValueError("Unknown single-frame encoder mode")
    time = 1 if shape[2] == 1 and single_frame_mode == "native" else 17
    return tuple(sorted({(1, 3, time, h, w) for h in tile_axis(shape[3]).lengths for w in tile_axis(shape[4]).lengths}))


def _spatial_encode(source, backend, check):
    import torch
    y, x = tile_axis(source.shape[-2]), tile_axis(source.shape[-1])
    rows, dtype = [], None
    for top, height in zip(y.starts, y.lengths):
        row = []
        for left, width in zip(x.starts, x.lengths):
            check()
            value = backend(source[..., top:top + height, left:left + width].contiguous())
            expected = (1, 48, (source.shape[2] + 3) // 4, height // 16, width // 16)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
                raise ValueError("Encoder moments tile has incorrect shape")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError("Encoder moments tile must be finite floating point")
            if dtype is not None and dtype != value.dtype:
                raise ValueError("Encoder changed moments dtype")
            dtype = value.dtype
            row.append(value.cpu())
        rows.append(row)
    # Match Core's raw, unblended previous neighbors (including at corners).
    assembled = []
    for i, row in enumerate(rows):
        pieces = []
        for j, value in enumerate(row):
            if i:
                value = _blend(rows[i - 1][j], value, y.overlaps[i - 1] // 16, -2)
            if j:
                value = _blend(row[j - 1], value, x.overlaps[j - 1] // 16, -1)
            if i < len(rows) - 1:
                value = value[..., :-y.overlaps[i] // 16, :]
            if j < len(row) - 1:
                value = value[..., :, :-x.overlaps[j] // 16]
            pieces.append(value)
        assembled.append(torch.cat(pieces, dim=-1))
    return torch.cat(assembled, dim=-2)


def encode_rgb(source, backend, *, latents_mean, latents_std, max_output_bytes,
               single_frame_mode="native", check=lambda: None):
    """Encode independently supplied raw moments tiles; fail without partial output.

    Budget bounds the final CPU latent tensor only, not full process/GPU memory.
    Production callers must preflight required_encode_tiles against their engine.
    """
    import torch
    shape = encode_output_shape(source.shape)
    required_encode_tiles(source.shape, single_frame_mode)
    positive_int(max_output_bytes, "max_output_bytes")
    if not source.is_floating_point() or not bool(torch.isfinite(source).all()) or source.min() < -1 or source.max() > 1:
        raise ValueError("Encoder input must be finite RGB in [-1,1]")
    size = 4
    for dim in shape:
        size *= dim
    if size > max_output_bytes:
        raise ValueError("Encoded CPU latent exceeds explicit output budget")
    mean, std = [torch.as_tensor(v, dtype=torch.float32, device="cpu") for v in (latents_mean, latents_std)]
    if mean.shape != (24,) or std.shape != (24,) or not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()) or not bool((std > 0).all()):
        raise ValueError("Invalid latent normalization constants")
    mean, std = mean.view(1, 24, 1, 1, 1), std.view(1, 24, 1, 1, 1)
    pixel_mean = source.new_tensor((.485, .456, .406)).view(1, 3, 1, 1, 1)
    pixel_std = source.new_tensor((.229, .224, .225)).view(1, 3, 1, 1, 1)
    output = torch.empty(shape, dtype=torch.float32, device="cpu")
    for batch in range(shape[0]):
        parts = []
        for start in range(0, source.shape[2], 17):
            check()
            clip = source[batch:batch + 1, :, start:start + 17]
            if source.shape[2] != 1 and clip.shape[2] < 17:
                clip = torch.cat((clip, clip[:, :, -1:].expand(-1, -1, 17 - clip.shape[2], -1, -1)), dim=2)
            normalized = ((clip + 1) * .5 - pixel_mean) / pixel_std
            if source.shape[2] == 1 and single_frame_mode == "causal_zero17_first":
                normalized = torch.nn.functional.pad(normalized, (0, 0, 0, 0, 0, 16))
            moments = _spatial_encode(normalized, backend, check)
            if source.shape[2] == 1:
                moments = moments[:, :, :1] if single_frame_mode == "causal_zero17_first" else moments[:, :, -1:]
            parts.append(moments)
        moments = torch.cat(parts, dim=2)
        if source.shape[2] != 1:
            moments = moments[:, :, :-3]
        value = (moments[:, :24].float() - mean) / std
        if tuple(value.shape) != (1,) + shape[1:] or not bool(torch.isfinite(value).all()):
            raise ValueError("Final encoded latent shape or values invalid")
        output[batch:batch + 1].copy_(value)
    check()
    return output
