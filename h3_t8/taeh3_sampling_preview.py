"""Optional bounded x0 observer. No sampler, global preview override or media output."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import logging
from pathlib import Path
import time
import uuid

import torch
import torch.nn.functional as F

from .preview_execution_context import CONTEXT

EVENT = 't8-taeh3-sampling-preview'
OWNER = 't8_taeh3_read_only_preview_v1'


@dataclass(frozen=True)
class PreviewSettings:
    model_path: str
    phase: str = 'low'
    update_every_steps: int = 2
    min_interval_ms: int = 500
    max_resolution: int = 256
    latent_prefix: int = 7
    frames: int = 12
    fps: int = 12
    jpeg_quality: int = 75

    def validate(self):
        if self.phase not in {'low', 'high', 'all'} or self.latent_prefix not in {2, 7, 12}:
            raise ValueError('Invalid preview phase/prefix')
        bounds = ((self.update_every_steps, 1, 1000), (self.min_interval_ms, 0, 60000),
                  (self.max_resolution, 64, 512), (self.frames, 1, 24),
                  (self.fps, 1, 24), (self.jpeg_quality, 30, 95))
        if any(type(value) is not int or not lo <= value <= hi for value, lo, hi in bounds):
            raise ValueError('Preview settings exceed bounded observer limits')


def video_prediction(prediction, shapes=None):
    if getattr(prediction, 'is_nested', False):
        parts = prediction.unbind()
        if len(parts) != 2:
            raise ValueError('Expected joint AV x0')
        video = parts[0]
    elif shapes is not None:
        from comfy.utils import unpack_latents
        if len(shapes) != 2:
            raise ValueError('Expected native packed AV shapes')
        video = unpack_latents(prediction, shapes)[0]
    else:
        video = prediction
    if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
        raise ValueError('Preview requires actual H3 video x0 [1,24,T,H,W], not audio/noisy state')
    return video


def bounded_prefix(video, settings):
    t, h, w = video.shape[-3:]
    if min(t, h, w) <= 0:
        raise ValueError('Empty preview dimensions')
    prefix = min(settings.latent_prefix, t)
    prefix = 1 if prefix == 1 else 2 + (prefix-2)//5*5
    scale = min(1., settings.max_resolution / (max(h, w)*16))
    nh, nw = max(2, int(h*scale)//2*2), max(2, int(w*scale)//2*2)
    # Copy only a contiguous temporal prefix, not the full AV/video tensor.
    value = video[:, :, :prefix].detach()
    if (nh, nw) != (h, w):
        value = F.interpolate(value, size=(prefix, nh, nw), mode='trilinear', align_corners=False).float()
    else:
        value = value.float().clone()
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Preview prefix contains nonfinite values')
    return value, dict(latent_prefix=prefix, original_video_shape=list(video.shape),
        observer_shape=list(value.shape), spatially_resampled=(nh, nw) != (h, w),
        coverage='contiguous first temporal prefix, not the full clip', approximate=True)


class TinyDecoder:
    def __init__(self, path, device):
        from comfy.taesd.taehv import TAEHV
        import comfy.utils
        path = Path(path).resolve(strict=True)
        if not 0 < path.stat().st_size <= 100*1024**2:
            raise ValueError('Tiny preview checkpoint size is incompatible')
        state = comfy.utils.load_torch_file(str(path), safe_load=True)
        model = TAEHV(latent_channels=24, latent_format=None, show_progress_bar=False).eval()
        decoder = {key.removeprefix('decoder.'): value for key, value in state.items() if key.startswith('decoder.')}
        if set(decoder) != set(model.decoder.state_dict()):
            from .taeh3_2d_preview import build_decoder, decoder_state
            model = build_decoder(decoder_state(state))
            self.family = 'independent_2d_latent_frames'
        else:
            model.decoder.load_state_dict(decoder, strict=True)
            self.family = 'temporal_taehv'
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        if self.family == 'temporal_taehv':
            model.decoder.to(device=device, dtype=dtype)
        else:
            model.to(device=device, dtype=dtype)
        self.model, self.device, self.dtype = model, device, dtype

    @torch.inference_mode()
    def decode(self, value):
        if self.family == 'independent_2d_latent_frames':
            from .taeh3_2d_preview import decode_frames
            return decode_frames(self.model, value, self.device, self.dtype, _check)
        return self.model.decode(value.to(device=self.device, dtype=self.dtype))[0].movedim(0, -1).cpu()


def _environment():
    from comfy_execution.utils import get_executing_context
    from server import PromptServer
    context = get_executing_context()
    server = getattr(PromptServer, 'instance', None)
    client = getattr(server, 'client_id', None)
    if context is None or not client or server is None:
        return None
    return context, server, client


def _check():
    from comfy.model_management import throw_exception_if_processing_interrupted
    throw_exception_if_processing_interrupted()


def _cancel_exception(error):
    import asyncio
    from comfy.model_management import InterruptProcessingException
    return isinstance(error, (InterruptProcessingException, asyncio.CancelledError))


def _jpeg_frames(images, settings, source_size=None):
    from PIL import Image
    if images.ndim != 4 or images.shape[-1] != 3 or len(images) < 1:
        raise ValueError('Tiny decoder returned an invalid image sequence')
    height, width = images.shape[1:3]
    if min(height, width) < 1:
        raise ValueError('Tiny decoder returned empty spatial dimensions')
    source_width, source_height = source_size or (width, height)
    if min(source_width, source_height) < 1:
        raise ValueError('Invalid source aspect ratio')
    # Even latent-grid rounding can distort a small temporal preview. Restore
    # only the displayed JPEG aspect; never resize or mutate sampler x0.
    longest = min(settings.max_resolution, max(height, width))
    ratio = longest / max(source_width, source_height)
    display_size = (max(1, round(source_width * ratio)), max(1, round(source_height * ratio)))
    indices = torch.linspace(0, len(images)-1, min(settings.frames, len(images))).round().long().tolist()
    frames, total = [], 0
    for i in indices:
        _check()
        data = (images[i].float().clamp(0, 1)*255).round().to(torch.uint8).numpy()
        buffer = BytesIO()
        picture = Image.fromarray(data)
        if picture.size != display_size:
            picture = picture.resize(display_size, Image.Resampling.BILINEAR)
        picture.save(buffer, format='JPEG', quality=settings.jpeg_quality)
        raw = buffer.getvalue()
        total += len(raw)
        if total > 1024**2:
            raise ValueError('Preview payload exceeds1MiB; disable this observer run')
        frames.append('data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii'))
    return frames, indices


class TAEH3PreviewWrapper:
    def __init__(self, settings, node_id):
        settings.validate()
        self.settings, self.node_id = settings, str(node_id)

    def __call__(self, executor, *args, **kwargs):
        environment = _environment()
        if environment is None:
            # No browser/client means no private frame delivery and no allocation.
            return executor(*args, **kwargs)
        context, server, client = environment
        run_id, epoch = uuid.uuid4().hex, str(time.time_ns())
        settings = self.settings
        shapes = kwargs.get('latent_shapes')
        decoder, last, callbacks, sequence, stopped = None, -float('inf'), 0, 0, False
        labels = CONTEXT.get() or {}
        phase = labels.get('phase', 'unidentified')

        def emit(kind, **data):
            nonlocal sequence, stopped
            sequence += 1
            event = dict(kind=kind, preview_node_id=self.node_id, sampler_node_id=context.node_id,
                prompt_id=context.prompt_id, run_id=run_id, epoch_ns=epoch, sequence=sequence,
                phase=phase, segment=labels.get('segment'), window=labels.get('window'),
                global_offset=labels.get('global_offset'), global_total=labels.get('global_total'), **data)
            try:
                # Never broadcast to client=None or a different active client.
                server.send_sync(EVENT, event, client)
            except Exception as error:
                if _cancel_exception(error):
                    raise
                stopped = True
                logging.warning('T8 private preview transport unavailable: %s', error)

        if phase not in {'unidentified', settings.phase} and settings.phase != 'all':
            return executor(*args, **kwargs)
        emit('start', active=True, fps=settings.fps,
             note='Approximate contiguous-prefix x0 preview; not final VAE/audio/media. Unidentified stage is not assumedLOW.')
        original = args[5] if len(args) > 5 else kwargs.get('callback')

        def observe(step, prediction, state, total):
            nonlocal callbacks, decoder, last, stopped
            # Every original consumer runs with the same objects/count/order.
            if original is not None:
                original(step, prediction, state, total)
            callbacks += 1
            _check()
            now = time.monotonic()
            final = step + 1 == total
            if stopped or (not final and (step % settings.update_every_steps or (now-last)*1000 < settings.min_interval_ms)):
                return
            try:
                video = video_prediction(prediction, shapes)
                prefix, info = bounded_prefix(video, settings)
                _check()
                if decoder is None:
                    decoder = TinyDecoder(settings.model_path, video.device)
                images = decoder.decode(prefix)
                _check()
                frames, indices = _jpeg_frames(images, settings, (video.shape[-1], video.shape[-2]))
                _check()
                family = getattr(decoder, 'family', 'temporal_taehv')
                two_dimensional = family == 'independent_2d_latent_frames'
                if two_dimensional:
                    info['coverage'] = 'independent latent frames in the first prefix, not reconstructed pixel-frame timing'
                emit('frames', frames=frames, decoded_frame_indices=indices,
                     decoded_prefix_frames=len(images), source_prefix_fps=None if two_dimensional else 24,
                     decoder_family=family,
                     decoded_latent_frame_indices=indices if two_dimensional else None,
                     display_aspect_restored=True, source_aspect=[video.shape[-1], video.shape[-2]],
                     step=step+1, steps=total, global_step=(labels['global_offset']+step+1 if labels.get('global_offset') is not None else None),
                     **info)
                last = now
            except Exception as error:
                if _cancel_exception(error) or any(message in str(error).lower() for message in
                    ('illegal memory access', 'device-side assert', 'unspecified launch failure')):
                    raise
                stopped, decoder = True, None
                logging.warning('T8 observer disabled; original sampling continues: %s', error)
                emit('unavailable', reason=f'{type(error).__name__}: {error}', active=True)

        values = list(args)
        options = dict(kwargs)
        if len(values) > 5:
            values[5] = observe
        else:
            options['callback'] = observe
        status = 'finished'
        primary_error = None
        try:
            return executor(*values, **options)
        except BaseException as error:
            primary_error = error
            status = 'cancelled' if _cancel_exception(error) else 'sampling_failed'
            raise
        finally:
            decoder = None
            # No global unload, empty_cache, sampler replacement or saved x0.
            try:
                emit('end', status=status, active=False, callbacks=callbacks,
                     coverage='no callback observed' if callbacks == 0 else 'actual callback observed', unavailable=stopped)
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                if hasattr(primary_error, 'add_note'):
                    primary_error.add_note(f'Private preview final notification failed: {cleanup_error}')


def attach_preview(model, settings, node_id, enabled=True):
    if not enabled:
        return model
    settings.validate()
    from comfy.patcher_extension import WrappersMP
    clone = model.clone()
    # Re-executing our same node replaces only its own slot; other nodes and
    # all foreign LoRA/Sage/Sol/wrappers keep their existing order and state.
    key = OWNER + ':' + str(node_id)
    clone.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, key)
    clone.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, TAEH3PreviewWrapper(settings, node_id))
    return clone


_OWN_CALL = TAEH3PreviewWrapper.__call__


def is_read_only_preview(wrapper):
    return (type(wrapper) is TAEH3PreviewWrapper and set(vars(wrapper)) == {'settings', 'node_id'}
            and TAEH3PreviewWrapper.__call__ is _OWN_CALL and type(wrapper.settings) is PreviewSettings)


def cache_projection(model):
    """Ignore only our exact immutable observer slots; preserve unknown owners."""
    from comfy.patcher_extension import WrappersMP
    groups = getattr(model, 'wrappers', {}).get(WrappersMP.OUTER_SAMPLE, {})
    keys = [key for key, values in groups.items() if values and all(
        is_read_only_preview(value) and key == OWNER + ':' + value.node_id for value in values)]
    if not keys:
        return model
    clone = model.clone()
    for key in keys:
        clone.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, key)
    return clone
