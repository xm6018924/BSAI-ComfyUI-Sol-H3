"""Verified, resumable source preparation using the actual loaded VAE state."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import os
from pathlib import Path
import tempfile
import uuid

import torch

from .long_video_delivery import _atomic_write_bytes, _manifest_lock
from .video_outpaint_media import validate_outpaint_source
from .video_outpaint_plan import canonical
from .video_outpaint_prepare import iter_encode_outpaint_source, SequentialOutpaintFrameReader
from .video_outpaint_source_store import OutpaintSourceStore
from .patch_stack_policy import _execution_selection, model_identity_matches, warn_patch_stack


@contextmanager
def outpaint_gpu_lease():
    """Serialize this route across same-user processes, including distinct Comfy roots.

    This is not a lock over third-party applications or unrelated Comfy workflows.
    The operating system releases ownership after an interpreter crash.
    """
    root = Path(tempfile.gettempdir()) / "t8-h3-outpaint-gpu-lease-v1"
    with _manifest_lock(root, timeout_seconds=0.1):
        yield


def loaded_video_vae_identity(vae, *, interrupt_check=None):
    """Hash loaded tensor bytes in <=4MiB copies; never trust a model filename.

    Includes nonpersistent normalization buffers, implementation files and dtype/
    tiling configuration. Opaque VAE patches remain executable but use a fresh
    execution cache: state_dict alone cannot certify portable cache reuse.
    This is a computed state identity, not the original safetensors file's SHA.
    """
    stage = getattr(vae, "first_stage_model", None)
    if stage is None or not hasattr(stage, "state_dict"):
        raise ValueError("loaded VAE does not expose a verifiable tensor state")
    patcher = getattr(vae, "patcher", None)
    live = {name: getattr(patcher, name, None) for name in (
        'patches', 'object_patches', 'wrappers', 'callbacks', 'injections',
        'weight_wrapper_patches', 'hook_patches', 'forced_hooks', 'current_hooks')
        if getattr(patcher, name, None)}
    for label, obj in (('wrapper', vae), ('stage', stage)):
        selected = {name: vars(obj)[name] for name in ('encode', 'decode', 'encode_tiled', 'decode_tiled', 'forward')
                    if name in vars(obj) and callable(vars(obj)[name])}
        selected.update({name: getattr(obj, name) for name in ('_forward_hooks', '_forward_pre_hooks')
                         if getattr(obj, name, None)})
        if selected:
            live[label] = selected
    if callable(getattr(stage, 'named_modules', None)):
        nested = {name: {key: value for key, value in vars(module).items()
                         if (key == 'forward' and callable(value))
                         or (key in {'_forward_hooks', '_forward_pre_hooks'} and value)}
                  for name, module in stage.named_modules()}
        if any(nested.values()):
            live['stage_modules'] = {key: value for key, value in nested.items() if value}
    implementations = {}
    for label, cls in (("wrapper", type(vae)), ("stage", type(stage))):
        source = inspect.getsourcefile(cls)
        if source is None or not Path(source).is_file():
            live[label + '_class'] = cls
        else:
            implementations[label] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    configuration = {
        "wrapper": {key: str(getattr(vae, key, None)) for key in (
            "vae_dtype", "output_dtype", "latent_dim", "latent_channels", "crop_input")},
        "stage": {key: str(getattr(stage, key, None)) for key in (
            "clip_length", "token_drop", "tokens_chunk_size", "vae_ratio", "tile_size", "tile_overlap_min", "tiling")},
        "implementation": implementations, "torch_version": torch.__version__,
        "outpaint_prepare_sha256": hashlib.sha256(Path(__file__).with_name("video_outpaint_prepare.py").read_bytes()).hexdigest(),
        "outpaint_decode_sha256": hashlib.sha256(Path(__file__).with_name("video_outpaint_decode.py").read_bytes()).hexdigest(),
    }
    digest = hashlib.sha256(canonical(configuration).encode())
    tensors = dict(stage.state_dict())
    if hasattr(stage, "named_buffers"):
        tensors.update(dict(stage.named_buffers()))
    if patcher is not None and (getattr(patcher, 'backup', None) or getattr(patcher, 'backup_buffers', None)):
        from .long_video_dual_identity import _original_state
        tensors = _original_state(patcher, tensors)
    if not tensors:
        raise ValueError("loaded VAE tensor state is empty")
    total_bytes = 0
    for name, tensor in sorted(tensors.items()):
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta" or tensor.layout != torch.strided
                or not tensor.is_contiguous() or tensor.is_quantized):
            raise ValueError(f"VAE tensor {name!r} cannot be fingerprinted without an unbounded conversion")
        digest.update(canonical({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}).encode())
        flat = tensor.detach().reshape(-1)
        elements = max(1, 4 * 1024 * 1024 // tensor.element_size())
        for offset in range(0, flat.numel(), elements):
            if interrupt_check:
                interrupt_check()
            chunk = flat[offset:offset+elements].cpu()
            if (chunk.is_floating_point() or chunk.is_complex()) and not bool(torch.isfinite(chunk).all()):
                raise ValueError(f'VAE tensor {name!r} contains NaN or Inf')
            raw = chunk.view(torch.uint8).numpy().tobytes()
            digest.update(raw)
            total_bytes += len(raw)
    result = {"schema": "t8.h3.video_outpaint.loaded_vae_identity/v1", "sha256": digest.hexdigest(),
            "tensor_count": len(tensors), "tensor_bytes": total_bytes, "max_copy_bytes": 4 * 1024 * 1024,
            "configuration": configuration, "model_filename_trusted": False}
    if live:
        warn_patch_stack('VAE execution patches retained; use a fresh execution cache, not a portable identity')
        result.update(execution_weight_sha256=result['sha256'], sha256=hashlib.sha256(uuid.uuid4().bytes).hexdigest(),
                      portable_cache_reuse=False, execution_selection=_execution_selection(live),
                      opaque_internal_state_verified=False)
    return result


def source_vae_identity_matches(store, current):
    """An in-memory source provider can bind opaque patches; disk SHA alone cannot."""
    bound = getattr(store, 'loaded_vae_identity', None)
    if bound is not None:
        return model_identity_matches(bound, current)
    return current['sha256'] == store.identity['video_vae_sha256']


def prepare_outpaint_source_cache(vae, inspection, plan, cache_root, *, interrupt_check=None, progress=None):
    """Prepare/reuse the source-video cache; not a generated-output completion state."""
    source, checked = validate_outpaint_source(inspection, plan)
    root = Path(cache_root).resolve()
    poison = root / "invalid_source_preparation.json"
    if poison.exists():
        raise ValueError("source preparation was invalidated; inspect it and choose a new cache directory")
    encoded = 0
    with outpaint_gpu_lease():
        with _manifest_lock(root / "source-worker", timeout_seconds=0.1):
            if poison.exists():
                raise ValueError("source preparation was invalidated while waiting for its worker")
            validate_outpaint_source(inspection, checked)
            identity = loaded_video_vae_identity(vae, interrupt_check=interrupt_check)
            if identity.get('portable_cache_reuse') is False:
                root = root / ('execution-' + identity['sha256'])
                poison = root / 'invalid_source_preparation.json'
            store = OutpaintSourceStore(root, checked, video_vae_sha256=identity["sha256"])
            store.loaded_vae_identity = identity
            # Verify all reused assets before invoking a learned encoder.
            for shot, start, stop in store.expected:
                if store.position() == (shot, start, stop):
                    break
                store.read_range(shot, start, stop)
            position = store.position()
            initial_position = position
            initial_stat = source.stat()
            initial_signature = (initial_stat.st_size, initial_stat.st_mtime_ns, initial_stat.st_ctime_ns)
            try:
                if position is not None:
                    first_shot, first_token, _ = position
                    with SequentialOutpaintFrameReader(source, checked, interrupt_check=interrupt_check) as reader:
                        for shot_index in range(first_shot, len(checked["shots"])):
                            for start, tensor, report in iter_encode_outpaint_source(
                                vae, reader, checked, shot_index=shot_index,
                                start_token=first_token if shot_index == first_shot else 0,
                                interrupt_check=interrupt_check,
                            ):
                                current_stat = source.stat()
                                if (current_stat.st_size, current_stat.st_mtime_ns, current_stat.st_ctime_ns) != initial_signature:
                                    raise ValueError("source file changed while preparing latents")
                                store.append(shot_index, start, tensor)
                                encoded += 1
                                if progress:
                                    progress({"stage": "source_video_encode", "chunks_encoded_this_call": encoded, **report})
            finally:
                # A cancelled preparation must not leave mixed-identity committed chunks reusable.
                # These final integrity checks deliberately do not take the cancellation callback.
                try:
                    validate_outpaint_source(inspection, checked)
                    after = loaded_video_vae_identity(vae)
                    final_stat = source.stat()
                    if (not model_identity_matches(identity, after) or
                            (final_stat.st_size, final_stat.st_mtime_ns, final_stat.st_ctime_ns) != initial_signature):
                        raise ValueError("source or loaded VAE changed during preparation")
                except BaseException as error:
                    _atomic_write_bytes(poison, canonical({"reason": str(error), "pid": os.getpid(),
                                                          "plan_sha256": checked["plan_sha256"]}).encode())
                    raise
            report = {"schema": "t8.h3.video_outpaint.source_prepared/v1", "plan_sha256": checked["plan_sha256"],
                      "source_sha256": inspection["sha256"], "loaded_vae_identity": identity,
                      "chunks_encoded_this_call": encoded, "resume_from": initial_position,
                      "all_source_video_chunks_prepared": store.position() is None,
                      "cache_manifest_sha256": hashlib.sha256(store.path.read_bytes()).hexdigest(),
                      "source_audio_prepared": False, "generated_video_complete": False}
            _atomic_write_bytes(root / "source_prepared_receipt.json", canonical(report).encode())
    return store, report
