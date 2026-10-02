"""Native progressive LOW-boundary checkpoints, not completed AV latents.

One OS-exclusive session owns each cache directory. A checkpoint contains clean
video prediction AND noisy audio_next in sampler coordinates. It is not an
accepted segment, and it must never be loaded through the legacy low_x0 cache.
Condition preparation still runs before lookup; its actual outputs are hashed.
Full producer/chain identity and automatic multi-segment delivery are separate.
"""

from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import uuid

import torch
from safetensors.torch import load_file, save_file

from .long_video_dual_identity import content_identity, stage_model_identity
from .patch_stack_policy import model_identity_matches
from .long_video_delivery import (
    _atomic_write_json, _sha256_file, _open_advisory_lock, _try_advisory_lock, _release_advisory_lock,
)
from .progressive_continuation_runtime import _input_identity


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def native_model_identity(model, sampler):
    from .tst_model import detach_tst_model
    original, tst = detach_tst_model(model)
    if tst is not None:
        return {'tst_model': json.loads(canonical(tst)), 'source': native_model_identity(original, sampler)}
    from comfy.model_sampling import CONST, ModelSamplingDiscreteFlow, ModelSamplingAV
    from comfy.latent_formats import LatentFormat
    from .progressive_sampling_runtime import validate_native_model
    sampling = validate_native_model(model, sampler)
    if inspect.getattr_static(type(sampling), 'audio_scale') is not ModelSamplingAV.audio_scale:
        raise ValueError('Progressive checkpoint requires the native audio coordinate property')
    extra = set(vars(sampling)) - set(vars(torch.nn.Module())) - {'shift', 'audio_shift', 'multiplier', 'noise_scale'}
    if extra:
        raise ValueError('Progressive checkpoint sampling has unbound custom state')
    latent_format = model.get_model_object('latent_format')
    for name in ('process_in', 'process_out'):
        if getattr(getattr(latent_format, name), '__func__', None) is not getattr(LatentFormat, name):
            raise ValueError('Progressive checkpoint latent coordinate method was replaced')
    # Native coordinate methods, not just a class name or nominal shift label.
    for name, owner in (('calculate_input', CONST), ('calculate_denoised', CONST),
                        ('noise_scaling', CONST), ('inverse_noise_scaling', CONST),
                        ('timestep', ModelSamplingDiscreteFlow), ('sigma', ModelSamplingDiscreteFlow),
                        ('percent_to_sigma', ModelSamplingDiscreteFlow)):
        if getattr(getattr(sampling, name), '__func__', None) is not getattr(owner, name):
            raise ValueError(f'Progressive checkpoint cannot identify modified sampling method {name}')
    inspection = model.clone()
    inspection.object_patches.pop('model_sampling', None)
    # This clone is inspection-only. Execution still uses the actual selected
    # model_sampling object, whose coordinates/buffers are separately bound.
    weights = stage_model_identity(inspection)
    coordinates = content_identity({name: getattr(sampling, name, None) for name in
        ('shift', 'audio_shift', 'audio_scale', 'multiplier', 'noise_scale', 'sigmas')})
    placement = content_identity(dict(load_device=model.load_device, offload_device=model.offload_device,
        force_cast_weights=getattr(model, 'force_cast_weights', None)))
    # Persisted JSON turns the audited KJ FFN settings tuple into a list.
    # Normalize before *both* binding and verification; equal executions must
    # not compare a live tuple against a deserialized list and fail spuriously.
    return json.loads(canonical(dict(weights=weights, coordinates=coordinates, placement=placement,
        latent_format=content_identity(dict(instance=vars(latent_format), scale=latent_format.scale_factor)))))


def implementation_identity():
    import comfy.model_base
    import comfy.model_sampling
    import comfy.samplers
    import comfy.sample
    import comfy.latent_formats
    import comfy.model_patcher
    import comfy.utils
    from . import progressive_sampling_runtime as runtime
    # Conservative invalidation: any top-level T8 implementation change starts
    # a new cache identity. Do not hash documentation/ROADMAP or upload sources.
    root = Path(__file__).parent
    files = {path.name: _sha256_file(path) for path in sorted(root.glob('*.py'))}
    for obj in (comfy.model_base.MiniMaxH3, comfy.model_sampling.CONST,
                comfy.samplers.KSAMPLER, comfy.sample.prepare_noise, comfy.utils.pack_latents,
                comfy.latent_formats.MiniMaxH3AV, comfy.model_patcher.ModelPatcher,
                runtime._native_stage, runtime._lift_video, runtime._lifter_identity):
        path = inspect.getsourcefile(obj)
        if path is None:
            raise ValueError('Progressive checkpoint implementation source is unavailable')
        files[obj.__module__ + '.' + obj.__qualname__] = _sha256_file(Path(path))
    return files


class ProgressiveCheckpointSession:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.active = False
        self.contract = None

    @contextmanager
    def exclusive(self):
        if self.active:
            raise RuntimeError('Progressive checkpoint session already active')
        self.root.mkdir(parents=True, exist_ok=True)
        handle = _open_advisory_lock(self.root / 'progressive_boundary.lock')
        acquired = False
        try:
            if not _try_advisory_lock(handle):
                raise RuntimeError('Another progressive checkpoint owner holds this directory')
            acquired = True
            self.active = True
            self.owner = os.getpid(), threading.get_ident()
            yield self
        finally:
            self.active = False
            self.contract = None
            try:
                if acquired:
                    _release_advisory_lock(handle)
            finally:
                handle.close()

    def bind(self, low_model, high_model, sampler, plan, *, inputs, settings, lifter, continuation=None, producers=None,
             relay_binding=None):
        if (not self.active or self.contract is not None
                or self.owner != (os.getpid(), threading.get_ident())):
            raise RuntimeError('Bind one active checkpoint session exactly once')
        self.models = low_model, high_model
        self.sampler, self.inputs, self.continuation = sampler, inputs, continuation
        self.producers = producers
        self.relay_binding = relay_binding
        if relay_binding is not None:
            from .progressive_relay_checkpoint import ProgressiveRelayCheckpointBinding
            if type(relay_binding) is not ProgressiveRelayCheckpointBinding:
                raise ValueError('Checkpoint Relay requires actual bound inputs')
        from .progressive_producers import verify_producers
        self.plan = plan
        self.contract = dict(schema='t8.progressive.low_boundary.v1', plan=plan.report(),
            models=[native_model_identity(m, sampler) for m in self.models],
            inputs=_input_identity(inputs), settings=settings, lifter=lifter,
            producers=verify_producers(producers) if producers is not None else None,
            relay=relay_binding.verify() if relay_binding is not None else None,
            implementation=implementation_identity())
        self.contract = json.loads(canonical(self.contract))
        self.identity = digest(self.contract)
        self.filename = 'low-boundary-' + self.identity
        return self.identity

    def verify(self):
        if (not self.active or self.contract is None
                or self.owner != (os.getpid(), threading.get_ident())):
            raise RuntimeError('Checkpoint session is not bound and locked')
        if self.continuation is not None:
            self.continuation.verify()
        if self.relay_binding is not None and self.relay_binding.verify() != self.contract['relay']:
            raise ValueError('Progressive checkpoint Relay binding changed')
        if self.producers is not None:
            from .progressive_producers import verify_producers
            if verify_producers(self.producers) != self.contract['producers']:
                raise ValueError('Progressive checkpoint producer binding changed')
        if (digest(self.contract) != self.identity or _input_identity(self.inputs) != self.contract['inputs']
                or not model_identity_matches(
                    self.contract['models'], [native_model_identity(m, self.sampler) for m in self.models])
                or implementation_identity() != self.contract['implementation']):
            raise ValueError('Progressive checkpoint execution inputs changed')

    def _validate_boundary(self, tensors):
        shapes = {'clean_video': (*self.plan.video_shape[:-2], self.plan.low_height // 16, self.plan.low_width // 16),
                  'audio_next': tuple(self.plan.audio_shape)}
        if set(tensors) != set(shapes):
            raise ValueError('Progressive boundary requires clean_video AND noisy audio_next')
        for key, tensor in tensors.items():
            if (type(tensor) is not torch.Tensor or tensor.dtype != torch.float32 or
                    tuple(tensor.shape) != shapes[key] or not bool(torch.isfinite(tensor).all())):
                raise ValueError(f'Invalid progressive native boundary {key}')

    def load_low(self):
        self.verify()
        path = self.root / (self.filename + '.json')
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('schema') != 1 or record.get('contract') != self.contract:
            raise ValueError('Progressive boundary receipt does not match execution')
        name = record['tensor_file']
        if (not isinstance(name, str) or Path(name).name != name or
                not name.startswith(self.filename + '-') or not name.endswith('.safetensors')):
            raise ValueError('Progressive boundary tensor path is invalid')
        tensor_path = (self.root / name).resolve(strict=True)
        if tensor_path.parent != self.root or _sha256_file(tensor_path) != record['tensor_sha256']:
            raise ValueError('Progressive boundary tensor integrity failed')
        tensors = load_file(str(tensor_path), device='cpu')
        self._validate_boundary(tensors)
        report = record['low_report']
        if (digest(report) != record['report_sha256'] or report['callbacks'] != self.plan.low_evaluations
                or report['actual_forwards'] != self.plan.low_evaluations):
            raise ValueError('Progressive boundary missing completed LOW execution evidence')
        self.verify()
        return tensors, report, {'contract_sha256': self.identity, 'tensor_sha256': record['tensor_sha256']}

    def save_low(self, tensors, report):
        self.verify()
        self._validate_boundary(tensors)
        if report['callbacks'] != self.plan.low_evaluations or report['actual_forwards'] != self.plan.low_evaluations:
            raise ValueError('Cannot checkpoint an unfinished LOW stage')
        path = self.root / (self.filename + '.json')
        if path.exists():
            raise FileExistsError('Progressive boundary is immutable; load it instead')
        tensor_path = self.root / (self.filename + '-' + uuid.uuid4().hex + '.safetensors')
        save_file({key: tensor.detach().cpu().contiguous().clone() for key, tensor in tensors.items()}, str(tensor_path))
        with tensor_path.open('r+b') as handle:
            handle.flush()
            os.fsync(handle.fileno())
        record = dict(schema=1, contract=self.contract, tensor_file=tensor_path.name,
            tensor_sha256=_sha256_file(tensor_path), low_report=report, report_sha256=digest(report))
        self.verify()
        # The atomic receipt is the only completion marker. Interrupted/orphan
        # tensor files are never treated as completed, nor silently deleted.
        _atomic_write_json(path, record)
        return {'contract_sha256': self.identity, 'tensor_sha256': record['tensor_sha256']}
