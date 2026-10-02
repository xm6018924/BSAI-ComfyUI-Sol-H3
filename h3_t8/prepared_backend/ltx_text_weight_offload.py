"""Use the verified weight lease for isolated native LTX text/connector modules.

Does not quantize/cast weights or alter the native forward implementation.
Kept separate from the live Tao request's frozen source files.
"""
from contextlib import contextmanager
from types import SimpleNamespace

from taomate_weight_offload import offload_transformer_blocks


@contextmanager
def offload_native_module(model, blocks, device, *, minimum_free_bytes):
    if getattr(model, "_t8_ltx_offload_active", False):
        raise RuntimeError("Native LTX module already has an active weight lease")
    blocks = tuple(blocks)
    owners = {id(module) for module in model.modules()}
    if not blocks or any(id(block) not in owners for block in blocks):
        raise ValueError("Every leased block must belong to this model")
    if len({id(block) for block in blocks}) != len(blocks):
        raise ValueError("Repeated leased blocks")
    # A non-Module view exposes the existing generic lease interface without
    # registering aliases that would change the native model state_dict paths.
    view = SimpleNamespace(training=model.training, blocks=blocks,
                           parameters=model.parameters, modules=model.modules)
    model._t8_ltx_offload_active = True
    try:
        with offload_transformer_blocks(view, device, minimum_free_bytes=minimum_free_bytes) as receipt:
            receipt["boundary"] = "native module weights only; INT8/FP32/BF16 retained"
            yield receipt
    finally:
        del model._t8_ltx_offload_active
