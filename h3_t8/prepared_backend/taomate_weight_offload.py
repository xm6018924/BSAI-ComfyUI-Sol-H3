"""Inference-only host weights with fixed small modules and serial DiT blocks.

No quantization, no full-model CUDA move, and no device-to-host weight copy on
release. Original CPU tensor storage is retained and restored even on failure.
Only use with an isolated model/worker; output and KV lifetimes are separate.
"""
from contextlib import contextmanager
import time

import torch


class WeightLease:
    def __init__(self, parameters, buffers, device, *, minimum_free_bytes=0):
        self.parameters = list(parameters)
        self.buffers = list(buffers)  # (owning module, local buffer name, tensor)
        self.device = torch.device(device)
        self.minimum_free_bytes = int(minimum_free_bytes)
        self.bytes = sum(p.numel() * p.element_size() for p in self.parameters)
        self.bytes += sum(t.numel() * t.element_size() for _, _, t in self.buffers)
        self._saved = []
        self.active = False
        if self.device.type not in ('cpu', 'cuda') or self.minimum_free_bytes < 0:
            raise ValueError('Unsupported offload device or reserve')
        tensors = self.parameters + [t for _, _, t in self.buffers]
        if any(t.device.type != 'cpu' or t.is_meta for t in tensors):
            raise ValueError('Original offload weights must be real CPU tensors')
        if len({id(t) for t in tensors}) != len(tensors):
            raise ValueError('Duplicate tensor ownership within a weight lease')

    def __enter__(self):
        if torch.is_grad_enabled() or self.active:
            raise RuntimeError('Weight offload requires inference/no-grad and an idle lease')
        if self.device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(self.device)
            if free < self.bytes + self.minimum_free_bytes:
                raise RuntimeError('Insufficient device headroom for the next weight stage')
        self.active = True
        try:
            for parameter in self.parameters:
                original = parameter.data
                moved = original.to(device=self.device, copy=True)
                self._saved.append(('parameter', parameter, original))
                parameter.data = moved
            for owner, name, original in self.buffers:
                moved = original.to(device=self.device, copy=True)
                self._saved.append(('buffer', (owner, name), original))
                owner._buffers[name] = moved
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for kind, owner, original in reversed(self._saved):
            if kind == 'parameter':
                owner.data = original
            else:
                module, name = owner
                module._buffers[name] = original
        self._saved.clear()
        self.active = False


def module_tensors(module):
    parameters = list(module.parameters())
    buffers = [(owner, name, tensor) for owner in module.modules()
               for name, tensor in owner._buffers.items() if tensor is not None]
    return parameters, buffers


@contextmanager
def offload_transformer_blocks(model, device, *, minimum_free_bytes):
    if torch.is_grad_enabled() or model.training:
        raise RuntimeError('Offload transformer requires eval and inference/no-grad')
    if getattr(model, '_t8_host_weight_offload_active', False):
        raise RuntimeError('Model is already under host-weight offload')
    if getattr(model, '_inference_static_timestep_cache_enabled', False):
        raise ValueError('Static timestep caches must be disabled for this initial offload route')
    blocks = list(model.blocks)
    if not blocks:
        raise ValueError('Transformer has no blocks')
    block_tensors = [module_tensors(block) for block in blocks]
    block_ids = [id(t) for ps, bs in block_tensors for t in ps + [x[2] for x in bs]]
    if len(set(block_ids)) != len(block_ids):
        raise ValueError('Shared weight storage between offloaded blocks is unsupported')
    all_parameters, all_buffers = module_tensors(model)
    ids = set(block_ids)
    static = WeightLease([p for p in all_parameters if id(p) not in ids],
        [b for b in all_buffers if id(b[2]) not in ids], device, minimum_free_bytes=minimum_free_bytes)
    leases = [WeightLease(ps, bs, device, minimum_free_bytes=minimum_free_bytes) for ps, bs in block_tensors]
    saved = []
    active = []
    # Lazy registered caches (e.g. TimeEmbedder._frequency_cache) start as None
    # and are not part of WeightLease. Restore their original ownership too,
    # otherwise the first CUDA forward can leave an unleased device tensor.
    empty_buffers = [(owner, name) for owner in model.modules()
                     for name, tensor in owner._buffers.items() if tensor is None]
    receipt = {'device': str(torch.device(device)), 'static_bytes': static.bytes,
               'largest_block_bytes': max(lease.bytes for lease in leases), 'block_calls': [],
               'policy': 'unchanged CPU weights; static modules resident; one block copied per call'}
    sentinel = object()
    model._t8_host_weight_offload_active = True
    try:
        with static:
            for index, (block, lease) in enumerate(zip(blocks, leases)):
                original = block.forward
                saved.append((block, block.__dict__.get('forward', sentinel)))
                def forwarded(*args, _index=index, _lease=lease, _forward=original, **kwargs):
                    if active:
                        raise RuntimeError('Offloaded transformer blocks must execute serially')
                    active.append(_index)
                    start = time.perf_counter()
                    success = False
                    try:
                        with _lease:
                            result = _forward(*args, **kwargs)
                            success = True
                            return result
                    finally:
                        receipt['block_calls'].append({'block': _index, 'completed': success,
                            'host_call_seconds_including_transfer': time.perf_counter() - start})
                        active.pop()
                block.forward = forwarded
            yield receipt
    finally:
        for owner, name in empty_buffers:
            owner._buffers[name] = None
        for block, original in reversed(saved):
            if original is sentinel:
                del block.forward
            else:
                block.forward = original
        del model._t8_host_weight_offload_active
