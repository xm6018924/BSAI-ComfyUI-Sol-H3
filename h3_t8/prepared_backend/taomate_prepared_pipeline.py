"""Reuse the pinned native generate method with already prepared CPU weights/text.

No tokenizer/TE reload, CUDA work, downloads, algorithm substitution, or global
patches. The caller provides the owned offloading runtime to generate().
"""


def prepared_pipeline(model, context):
    import torch
    from taomate_h3.model.pipeline import MiniMaxH3NativePipeline
    from taomate_local_transport import validate_local
    validate_local(context)
    if model.training or model.parallel_context is not context:
        raise ValueError('Prepared pipeline requires the same local eval model/context')
    if any(p.device.type != 'cpu' or p.is_meta for p in model.parameters()):
        raise ValueError('Prepared pipeline must start from fully loaded CPU parameters')
    # Alternate construction intentionally avoids native multi-GPU weight/TE
    # allocation. All generation/noise/branch/unpacking methods stay unchanged.
    pipeline = MiniMaxH3NativePipeline.__new__(MiniMaxH3NativePipeline)
    pipeline.device = torch.device('cpu')
    pipeline.parallel_context = context
    pipeline.transformer = model
    return pipeline
