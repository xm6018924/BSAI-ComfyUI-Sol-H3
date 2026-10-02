"""Exact SDPA dispatch for VDN without the known SM120 efficient-kernel abort."""

from contextlib import nullcontext

import torch


def vdn_sdpa_context(device):
    device = torch.device(device)
    if device.type != "cuda":
        return nullcontext()
    major, _minor = torch.cuda.get_device_capability(device)
    if major != 12:
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # Issue #16: mem-efficient SDPA can abort the process before Python can
    # handle an exception. Do not enter it as a fallback, and do not fall back
    # to the quadratic score matrix. Flash is safe when the build supports it.
    return sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION])


def exact_sdpa_rows(q, k, v, scale):
    with vdn_sdpa_context(q.device):
        out = torch.nn.functional.scaled_dot_product_attention(
            q.permute(1, 0, 2).unsqueeze(0),
            k.permute(1, 0, 2).unsqueeze(0),
            v.permute(1, 0, 2).unsqueeze(0),
            dropout_p=0.0, is_causal=False, scale=scale,
        )
    return out.squeeze(0).permute(1, 0, 2)
