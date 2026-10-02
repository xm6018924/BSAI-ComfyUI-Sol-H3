"""H3 tile/time orchestration for an explicitly supplied decoder backend.

This is not a public VAE loader yet. The backend receives an *unnormalized*
single-batch latent tile, and must return raw RGB [1,3,4*T,16*H,16*W].
No fixed-profile padding is allowed: T=1 and small spatial tiles require an
engine profile that genuinely supports them. TRT integration is separate.
"""

from .trt_vae_contract import output_bytes, output_shape, positive_int, temporal_plan, tile_axis


def _blend(a, b, extent, dim):
    import torch

    extent = min(a.shape[dim], b.shape[dim], extent)
    if extent <= 0:
        raise ValueError("Overlap must be nonempty")
    positions = torch.arange(extent, device=b.device, dtype=b.dtype) / extent
    shape = [1] * b.ndim
    shape[dim] = extent
    weight = positions.view(shape)
    previous, current, rest = [[slice(None)] * b.ndim for _ in range(3)]
    previous[dim] = slice(-extent, None)
    current[dim] = slice(0, extent)
    rest[dim] = slice(extent, None)
    mixed = a[tuple(previous)] * (1 - weight) + b[tuple(current)] * weight
    return torch.cat((mixed, b[tuple(rest)]), dim=dim)


def _spatial_decode(z, decode_tile, check):
    import torch

    height, width = z.shape[-2] * 16, z.shape[-1] * 16
    y, x = tile_axis(height), tile_axis(width)
    canvas = None
    row_tails = []
    out_y = 0
    for i, (top, tile_h) in enumerate(zip(y.starts, y.lengths)):
        new_tails, left_tail, out_x = [], None, 0
        for j, (left, tile_w) in enumerate(zip(x.starts, x.lengths)):
            check()
            source = z[..., top // 16:(top + tile_h) // 16, left // 16:(left + tile_w) // 16]
            tile = decode_tile(source.contiguous())
            expected = (1, 3, z.shape[2] * 4, tile_h, tile_w)
            if not isinstance(tile, torch.Tensor) or tuple(tile.shape) != expected:
                raise ValueError(f"Decoder returned incorrect tile shape; expected {expected}")
            if not tile.is_floating_point() or not bool(torch.isfinite(tile).all()):
                raise ValueError("Decoder tile must contain finite floating-point RGB")
            check()
            if i < len(y.starts) - 1:
                new_tails.append(tile[..., -y.overlaps[i]:, :].clone())
            next_left = tile[..., -x.overlaps[j]:].clone() if j < len(x.starts) - 1 else None
            if i:
                tile = _blend(row_tails[j], tile, y.overlaps[i - 1], -2)
            if j:
                tile = _blend(left_tail, tile, x.overlaps[j - 1], -1)
            left_tail = next_left
            if i < len(y.starts) - 1:
                tile = tile[..., :-y.overlaps[i], :]
            if j < len(x.starts) - 1:
                tile = tile[..., :-x.overlaps[j]]
            if canvas is None:
                canvas = torch.empty((1, 3, z.shape[2] * 4, height, width), dtype=tile.dtype, device=tile.device)
            elif tile.dtype != canvas.dtype or tile.device != canvas.device:
                raise ValueError("Decoder changed dtype or device between tiles")
            canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
            out_x += tile.shape[-1]
        if out_x != width:
            raise RuntimeError("Spatial assembly did not fill the output width")
        row_tails = new_tails
        out_y += tile.shape[-2]
    if out_y != height:
        raise RuntimeError("Spatial assembly did not fill the output height")
    return canvas


def pixel_statistics(pixel_mean, pixel_std):
    """Copy actual model buffers; preserve FP16-rounded values when promoting."""
    import torch
    values = []
    for value in (pixel_mean, pixel_std):
        value = torch.as_tensor(value, dtype=torch.float32, device="cpu").detach().clone()
        if tuple(value.shape) not in ((3,), (1,3,1,1,1)) or not bool(torch.isfinite(value).all()):
            raise ValueError("Pixel normalization requires three finite channel constants")
        values.append(value.reshape(1,3,1,1,1))
    if not bool((values[1] > 0).all()):
        raise ValueError("Pixel normalization scales must be positive")
    return values


def decode_raw_latent(z, decode_tile, *, max_output_bytes, check=lambda: None,
                      pixel_mean=(0.485,0.456,0.406), pixel_std=(0.229,0.224,0.225)):
    """Decode unnormalized H3 video latent, streaming finalized RGB to CPU.

The caller owns engine/device lifecycle and denormalization. Output budget
limits only the CPU RGB allocation, NOT total process/GPU peak memory.
Any backend failure/cancellation propagates; never return a partial video.
"""
    import torch

    shape = output_shape(z.shape)
    positive_int(max_output_bytes, "max_output_bytes")
    if output_bytes(z.shape) > max_output_bytes:
        raise ValueError("Decoded CPU RGB exceeds the explicit output byte budget")
    if not z.is_floating_point() or not bool(torch.isfinite(z).all()):
        raise ValueError("Latent must be finite floating-point data")
    mean, std = pixel_statistics(pixel_mean, pixel_std)
    check()
    plan = temporal_plan(z.shape[2])
    output = torch.empty(shape, dtype=torch.float32, device="cpu")
    for batch in range(z.shape[0]):
        source = z[batch:batch + 1]
        pos = 0

        def write(part):
            nonlocal pos
            count = min(part.shape[2], plan.frames - pos)
            if count <= 0:
                return
            part = part[:, :, :count].float()
            output[batch:batch + 1, :, pos:pos + count].copy_((part * std.to(part.device) + mean.to(part.device)).clamp(0, 1))
            pos += count

        if plan.tokens == 1:
            write(_spatial_decode(source, decode_tile, check)[:, :, -1:])
        else:
            overlap = None
            for window in range(plan.windows):
                check()
                start = window * 5
                clip = source[:, :, start:start + 7]
                missing = 7 - clip.shape[2]
                if missing:
                    clip = torch.cat((clip, source[:, :, -1:].expand(-1, -1, missing, -1, -1)), dim=2)
                decoded = _spatial_decode(clip, decode_tile, check)
                current = decoded[:, :, 3:20]
                if overlap is not None:
                    current = _blend(overlap, current, 5, -3)
                write(current)
                overlap = decoded[:, :, 23:28].clone()
                del decoded, current, clip
            write(overlap)
        if pos != plan.frames:
            raise RuntimeError("Temporal assembly did not fill the planned frame count")
    check()
    return output
