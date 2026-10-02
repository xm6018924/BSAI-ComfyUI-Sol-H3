"""Independent internal TST kernel; public MODEL composition is separate.

Equations 1-7: https://arxiv.org/html/2609.08505v1 .
H3 profile uses frame-mean post-RoPE Q/K, NOT exact spatial attention mass.
No external SelfLift source, backend replacement, or model execution.
"""

import math

import torch

PROFILE = 'signed_query_mean_frame_v1'


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def effective_strength(tau, *, layer_index, layer_count, step_index, total_steps):
    """Explicit endpoint convention: last layer/first step strongest; last step zero.

    Layer fraction=(index+1)/count, step fraction=index/(count-1).
    Caller supplies the index in the FULL schedule, including after resume.
    """
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau) or not 0 <= tau <= 2:
        raise ValueError('tau must be finite within [0,2]')
    _integer(layer_count, 'layer_count', 1)
    _integer(total_steps, 'total_steps', 2)
    _integer(layer_index, 'layer_index')
    _integer(step_index, 'step_index')
    if layer_index >= layer_count or step_index >= total_steps:
        raise ValueError('Layer/step index outside full schedule')
    depth = .5 * (1 - math.cos(math.pi * (layer_index + 1) / layer_count))
    early = .5 * (1 + math.cos(math.pi * step_index / (total_steps - 1)))
    return float(tau) * depth * early


def transport_statistics(matrix):
    """Signed spectral tension for batches of square row-stochastic matrices."""
    if (not isinstance(matrix, torch.Tensor) or matrix.layout != torch.strided or
            matrix.ndim < 2 or matrix.shape[-1] < 2 or matrix.shape[-2] != matrix.shape[-1] or
            matrix.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64) or
            matrix.device.type not in ('cpu', 'cuda') or matrix.numel() == 0):
        raise ValueError('Transport must be a nonempty materialized floating square matrix, F>=2')
    dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
    a = matrix.to(dtype=dtype)
    if not torch.isfinite(a).all() or (a < 0).any():
        raise ValueError('Transport must be finite and nonnegative')
    # Low precision storage rounds probabilities; callers must still supply a
    # stochastic matrix rather than having arbitrary scores silently normalized.
    tolerance = 1e-2 if matrix.dtype == torch.bfloat16 else 1e-3 if matrix.dtype == torch.float16 else 2e-6
    if not torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), rtol=0, atol=tolerance):
        raise ValueError('Transport rows must sum to one')
    log_frames = math.log(a.shape[-1])
    row = -torch.special.xlogy(a, a).sum(-1).mean(-1) / log_frames
    gram = a @ a.transpose(-2, -1)
    density = gram / gram.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
    eigenvalues = torch.linalg.eigvalsh(density).clamp_min(0)
    eigenvalues = eigenvalues / eigenvalues.sum(-1, keepdim=True)
    spectral = -torch.special.xlogy(eigenvalues, eigenvalues).sum(-1) / log_frames
    tension = row - spectral
    if not torch.isfinite(tension).all():
        raise ValueError('Nonfinite spectral tension')
    return {'row_entropy': row, 'spectral_entropy': spectral, 'tension': tension}


@torch.inference_mode()
def correct_video_queries(q, k, *, video_start, frames, spatial_tokens, mode='disabled',
                          tau=.2, layer_index=0, layer_count=1, step_index=0, total_steps=8,
                          max_workspace_mib=256):
    """Return fresh corrected Q or the original object; never modify inputs.

    Input layout B,H,packed_sequence,D. Caller authenticates actual video span
    and global clock. This function does not infer them from token counts.
    Workspace is a conservative explicit-tensor estimate, not CUDA peak proof.
    """
    if mode not in ('disabled', 'report_only', 'apply_exp'):
        raise ValueError('Unknown TST mode')
    if mode == 'disabled':
        return q, {'profile': PROFILE, 'mode': mode, 'applied': False, 'estimated_tensor_bytes': 0}
    strength = effective_strength(tau, layer_index=layer_index, layer_count=layer_count,
                                  step_index=step_index, total_steps=total_steps)
    _integer(video_start, 'video_start')
    _integer(frames, 'frames', 2)
    _integer(spatial_tokens, 'spatial_tokens', 1)
    _integer(max_workspace_mib, 'max_workspace_mib', 1)
    if (not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor) or q.ndim != 4 or
            q.shape != k.shape or q.device != k.device or q.dtype != k.dtype or q.numel() == 0 or
            q.layout != torch.strided or k.layout != torch.strided or
            q.device.type not in ('cpu', 'cuda') or
            q.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)):
        raise ValueError('Q/K require matching materialized B,H,S,D floating tensors')
    batch, heads, sequence, channels = q.shape
    end = video_start + frames * spatial_tokens
    if end > sequence:
        raise ValueError('Video span exceeds packed sequence')
    itemsize = 8 if q.dtype == torch.float64 else 4
    video_elements = batch * heads * frames * spatial_tokens * channels
    # Allow reshape copies, full corrected Q and finite checks, pooled Q/K,
    # logits/probabilities/Gram/density/eigensolver-visible tensors. Library
    # private workspace and allocator fragmentation are not predicted here.
    estimate = (2 * video_elements * q.element_size() + q.numel() * (q.element_size() + 1)
                + batch * heads * (4 * frames * channels + 10 * frames * frames) * itemsize)
    if estimate > max_workspace_mib * 1024**2:
        raise ValueError('TST explicit tensor workspace budget exceeded')
    if not torch.isfinite(q).all() or not torch.isfinite(k).all():
        raise ValueError('Q/K must be finite')
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    shape = (batch, heads, frames, spatial_tokens, channels)
    qv = q[:, :, video_start:end].reshape(shape).mean(-2, dtype=dtype)
    kv = k[:, :, video_start:end].reshape(shape).mean(-2, dtype=dtype)
    operator = torch.softmax((qv @ kv.transpose(-2, -1)) / math.sqrt(channels), dim=-1)
    statistics = transport_statistics(operator)
    gamma = torch.exp(strength * statistics['tension'])
    applied = mode == 'apply_exp' and bool((gamma != 1).any())
    result = q
    if applied:
        result = q.clone()
        result[:, :, video_start:end].mul_(gamma.to(q.dtype)[..., None, None])
        if not torch.isfinite(result).all():
            raise ValueError('TST query correction overflowed input dtype')
    return result, dict(profile=PROFILE, mode=mode, applied=applied,
        effective_strength=strength, layer_index=layer_index, step_index=step_index,
        total_steps=total_steps, video_start=video_start, video_end=end,
        # Do not retain an F-by-F operator per layer/step in diagnostics.
        estimated_tensor_bytes=estimate, gamma=gamma, **statistics)
