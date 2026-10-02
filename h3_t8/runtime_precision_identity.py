"""Read PyTorch precision state without mixing legacy and backend-specific APIs."""
from __future__ import annotations

import torch


def matmul_precision_identity(matmul_backend=None, legacy_global_getter=None) -> dict[str, object]:
    """Return the active CUDA matmul precision API and value.

    PyTorch 2.9 added ``cuda.matmul.fp32_precision`` and explicitly rejects
    mixing it with the legacy global/``allow_tf32`` getters.  Prefer the new
    backend-specific getter whenever it exists; touch the legacy getters only
    on older PyTorch builds where the new attribute is absent.
    """

    backend = torch.backends.cuda.matmul if matmul_backend is None else matmul_backend
    try:
        value = backend.fp32_precision
    except AttributeError:
        getter = torch.get_float32_matmul_precision if legacy_global_getter is None else legacy_global_getter
        return {
            "api": "legacy_allow_tf32",
            "global_matmul_precision": str(getter()),
            "cuda_matmul_allow_tf32": bool(backend.allow_tf32),
        }
    return {
        "api": "backend_specific_fp32_precision",
        "cuda_matmul_fp32_precision": str(value),
    }
