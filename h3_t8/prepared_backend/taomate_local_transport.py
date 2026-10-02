"""Explicit single-process TP=1 adapter. Not a fake torch process group.

Only use in an isolated, single-threaded Tao worker. All replaced aliases are
restored on exit. Multi-rank topology is rejected rather than dropping reduction.
"""
from contextlib import contextmanager


class LocalOnlyGroup:
    """Marker accepted by upstream context, never passed to torch.distributed."""


def validate_local(context):
    if (context.tp_world_size, context.tp_rank, context.ulysses_world_size, context.ulysses_rank) != (1, 0, 1, 0):
        raise ValueError('Local transport requires TP=1 and Ulysses=1, rank=0')
    if not isinstance(context.tp_group, LocalOnlyGroup) or not isinstance(context.ulysses_group, LocalOnlyGroup):
        raise ValueError('Local transport requires explicitly local group markers')


def local_reduce(value, *, context):
    validate_local(context)
    return value.clone()


def local_reduce_inplace(value, *, context):
    validate_local(context)
    return value


def local_gather(value, *, dim=-1, context):
    validate_local(context)
    # A singleton concatenation preserves values with independent storage.
    import torch
    return torch.cat((value,), dim=dim)


def local_gather_rows(value, *, context):
    return local_gather(value, dim=0, context=context)


def local_sequence_head_swap(value, *, context, sequence_to_heads):
    validate_local(context)
    if value.ndim < 3:
        raise ValueError('Ulysses input must have sequence/head/dimension axes')
    import torch
    return value.clone(memory_format=torch.contiguous_format)


@contextmanager
def local_transport():
    from taomate_h3.distributed import ParallelContext
    from taomate_h3.distributed import linear, ulysses
    from taomate_h3.model import layers, dit
    context = ParallelContext(1, 0, 1, 0, LocalOnlyGroup(), LocalOnlyGroup())
    replacements = [(linear, 'tp_all_reduce', local_reduce),
        (linear, 'tp_all_reduce_inplace_', local_reduce_inplace),
        (linear, 'tp_all_gather', local_gather), (layers, 'tp_all_gather', local_gather),
        (dit, 'tp_all_gather', local_gather), (dit, 'ulysses_all_gather_rows', local_gather_rows),
        (ulysses, '_swap', local_sequence_head_swap)]
    saved = [(module, name, getattr(module, name)) for module, name, _ in replacements]
    if any(getattr(module, name).__module__ == __name__ for module, name, _ in replacements):
        raise RuntimeError('Local transport is already installed')
    try:
        for module, name, replacement in replacements:
            setattr(module, name, replacement)
        yield context
    finally:
        for module, name, original in reversed(saved):
            setattr(module, name, original)
