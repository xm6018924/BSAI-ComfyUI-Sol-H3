"""Serial source-audio cache preparation and verified sampler-facing provider."""
from __future__ import annotations

from contextlib import closing
import hashlib
import os
from pathlib import Path

import torch

from .long_video_delivery import _atomic_write_bytes, _manifest_lock
from .video_outpaint_audio import loaded_audio_vae_identity, iter_encode_outpaint_audio_vae
from .video_outpaint_audio_file import prepare_outpaint_audio_file, OutpaintAudioIntegrityError
from .video_outpaint_audio_store import OutpaintAudioStore
from .video_outpaint_media import validate_outpaint_source
from .video_outpaint_plan import canonical
from .video_outpaint_source_runtime import outpaint_gpu_lease
from .patch_stack_policy import model_identity_matches


class OutpaintAudioProvider:
    """Construct from a verified complete cache, not a user-supplied audio SHA label."""
    def __init__(self, inspection, store):
        self.inspection = dict(inspection)
        self.store = store
        self.manifest_sha256 = hashlib.sha256(store.path.read_bytes()).hexdigest()
        self.verify()
        store.verify_assets()

    def verify(self):
        validate_outpaint_source(self.inspection, self.store.plan)
        if (hashlib.sha256(self.store.path.read_bytes()).hexdigest() != self.manifest_sha256
                or self.store.snapshot()["status"] != "prepared"):
            raise ValueError("bound audio provider cache changed or is incomplete")
        return self.manifest_sha256

    def __call__(self, shot, window):
        self.verify()
        return self.store.window(shot, window)


def prepare_outpaint_audio_cache(vae, inspection, plan, cache_root, *, stream_position=0,
                                  block_tokens=64, resume=False, interrupt_check=None, progress=None):
    """Resume preserves committed chunks and replays a partial shot's causal prefix.

    Prefix KV is scratch, not a restart artifact. On a partial-shot retry, replay
    encoded chunks are compared exactly to their old commits before new chunks
    are appended. Completed shots are skipped. This trades restart compute for
    bounded scratch/state; it never claims zero recomputation after cancellation.
    """
    source, checked = validate_outpaint_source(inspection, plan)
    root = Path(cache_root).resolve()
    poison = root / "invalid_audio_preparation.json"
    new_chunks = replayed = 0
    with outpaint_gpu_lease(), _manifest_lock(root / "audio-worker", timeout_seconds=0.1):
        if poison.exists():
            raise ValueError("audio preparation was invalidated; choose a new cache directory")
        initial_stat = source.stat()
        signature = (initial_stat.st_size, initial_stat.st_mtime_ns, initial_stat.st_ctime_ns)
        has_audio = bool(inspection["audio_pcm"])
        identity = loaded_audio_vae_identity(vae, interrupt_check=interrupt_check) if has_audio else None
        if identity and identity.get('portable_cache_reuse') is False:
            root = root / ('execution-' + identity['sha256'])
            poison = root / 'invalid_audio_preparation.json'
            root.mkdir(parents=True, exist_ok=True)
        store = None
        try:
            with prepare_outpaint_audio_file(inspection, checked, stream_position=stream_position,
                                             scratch_parent=root, interrupt_check=interrupt_check) as audio:
                store = OutpaintAudioStore(root, checked,
                    audio_vae_sha256=identity["sha256"] if identity else None,
                    pcm_sha256=audio.pcm_sha256 if audio else None,
                    stream_position=audio.stream_position if audio else None,
                    encoding_device=str(vae.device) if audio else None, block_tokens=block_tokens)
                store.verify_assets()
                resume_from = store.begin(resume=resume)
                if resume_from < len(store.expected):
                    first_shot = store.expected[resume_from][0]
                    for shot in range(first_shot, len(checked["shots"])):
                        count, reader = audio.shot(shot)
                        with closing(iter_encode_outpaint_audio_vae(
                            vae, reader, count, block_tokens=block_tokens, scratch_parent=root,
                            interrupt_check=interrupt_check,
                        )) as chunks:
                            for start, tensor in chunks:
                                stat = source.stat()
                                if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != signature:
                                    raise ValueError("source file changed while preparing audio")
                                committed = len(store.snapshot()["chunks"])
                                next_position = store.expected[committed] if committed < len(store.expected) else None
                                if next_position is not None and (shot, start) == next_position[:2]:
                                    store.append(shot, start, tensor)
                                    new_chunks += 1
                                else:
                                    previous = store.read_range(shot, start, start+tensor.shape[-1])
                                    if not torch.equal(previous, tensor):
                                        raise OutpaintAudioIntegrityError("replayed audio prefix differs from its committed posterior")
                                    replayed += 1
                                if progress:
                                    progress({"stage": "source_audio_encode", "shot": shot, "start_token": start,
                                              "new_chunks": new_chunks, "replayed_prefix_chunks": replayed})
        except BaseException as error:
            if store is not None:
                store.mark_interrupted()
            if isinstance(error, OutpaintAudioIntegrityError):
                _atomic_write_bytes(poison, canonical({"reason": str(error), "pid": os.getpid(),
                                                      "plan_sha256": checked["plan_sha256"]}).encode())
            raise
        finally:
            try:
                validate_outpaint_source(inspection, checked)
                stat = source.stat()
                after = loaded_audio_vae_identity(vae) if has_audio else None
                if not model_identity_matches(identity, after) or (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != signature:
                    raise ValueError("source or loaded audio VAE changed during preparation")
            except BaseException as error:
                _atomic_write_bytes(poison, canonical({"reason": str(error), "pid": os.getpid(),
                                                      "plan_sha256": checked["plan_sha256"]}).encode())
                raise
        provider = OutpaintAudioProvider(inspection, store)
        report = {"schema": "t8.h3.outpaint.audio_prepared/v1", "plan_sha256": checked["plan_sha256"],
                  "source_sha256": inspection["sha256"], "audio_vae_identity": identity,
                  "audio_manifest_sha256": provider.manifest_sha256, "resume_from": resume_from,
                  "chunks_encoded_this_call": new_chunks, "replayed_prefix_chunks": replayed,
                  "source_has_audio": has_audio, "source_audio_prepared": True,
                  "audio_output_policy": "original_file_tracks_only", "generated_video_complete": False}
        _atomic_write_bytes(root / "source_audio_prepared_receipt.json", canonical(report).encode())
    return provider, report
