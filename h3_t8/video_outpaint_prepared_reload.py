"""Reopen verified preparation caches without loading CLIP or either VAE.

Read-only store subclasses refuse the constructors' normal create-if-missing
path. Existing implementations and their audio/sampling cache hashes are kept
unchanged. This restores recorded conditioning; it does not encode new prompts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .video_outpaint_audio_runtime import OutpaintAudioProvider
from .video_outpaint_audio_store import OutpaintAudioStore
from .video_outpaint_conditioning import OutpaintConditioningProvider
from .video_outpaint_regional_conditioning import (
    REGIONAL_MANIFEST,
    OutpaintRegionalConditioningProvider,
)
from .video_outpaint_media import validate_outpaint_source
from .video_outpaint_source_runtime import outpaint_gpu_lease
from .video_outpaint_source_store import OutpaintSourceStore


class _ExistingSourceStore(OutpaintSourceStore):
    def _write(self, data):
        raise FileNotFoundError("source reload cannot create or rewrite a preparation manifest")


class _ExistingAudioStore(OutpaintAudioStore):
    def __init__(self, *args, interrupt_check=None, **kwargs):
        self._reload_interrupt = interrupt_check
        super().__init__(*args, **kwargs)

    def _write(self, data):
        raise FileNotFoundError("audio reload cannot create or rewrite a preparation manifest")

    def verify_assets(self):
        for record in self.snapshot()["chunks"]:
            if self._reload_interrupt:
                self._reload_interrupt()
            self._load(record)


class _ExistingConditioningProvider(OutpaintConditioningProvider):
    def __init__(self, *args, interrupt_check=None, **kwargs):
        self._reload_interrupt = interrupt_check
        super().__init__(*args, **kwargs)

    def __call__(self, shot, window):
        if self._reload_interrupt:
            self._reload_interrupt()
        return super().__call__(shot, window)


def _manifest(path):
    with path.open("rb") as handle:
        raw = handle.read(16*1024*1024 + 1)
    if len(raw) > 16*1024*1024:
        raise ValueError("preparation manifest exceeds the 16MiB metadata limit")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("preparation manifest must be an object")
    return raw, data


def load_prepared_outpaint(plan_handle, cache_root, *, interrupt_check=None, progress=None):
    def checkpoint():
        if interrupt_check:
            interrupt_check()

    root = Path(cache_root).resolve()
    plain_text = root / "text/outpaint_conditioning.json"
    regional_text = root / "text" / REGIONAL_MANIFEST
    if not plain_text.exists() and not regional_text.exists():
        raise FileNotFoundError("prepared cache has no plain or regional conditioning manifest")
    if plain_text.exists() and regional_text.exists():
        raise ValueError("prepared cache must contain exactly one plain or regional conditioning manifest")
    is_regional = regional_text.exists()
    paths = {"source": root / "source/outpaint_source_latents.json",
             "audio": root / "audio/outpaint_source_audio.json",
             "text": regional_text if is_regional else plain_text}
    with outpaint_gpu_lease():
        checkpoint()
        _, plan = validate_outpaint_source(plan_handle["inspection"], plan_handle["plan"])
        # Check all inputs first. No constructor should create a missing cache.
        records = {key: _manifest(path) for key, path in paths.items()}
        source_identity = records["source"][1].get("identity", {})
        audio_identity = records["audio"][1].get("identity", {})
        source = _ExistingSourceStore(root / "source", plan,
                                     video_vae_sha256=source_identity.get("video_vae_sha256"))
        if source.position() is not None:
            raise ValueError("source preparation is incomplete; finish or resume Prepare first")
        # Verify all video chunks individually, not a full long-video allocation.
        for index, (shot, start, stop) in enumerate(source.expected):
            checkpoint()
            source.read_range(shot, start, stop)
            if progress:
                progress({"stage": "verify_prepared_source", "chunks_verified": index+1,
                          "total_chunks": len(source.expected)})
        checkpoint()
        keys = {"audio_vae_sha256", "pcm_sha256", "stream_position", "encoding_device", "block_tokens"}
        if set(audio_identity) != keys | {"plan_sha256", "implementation_sha256"}:
            raise ValueError("saved audio preparation identity has unexpected fields")
        audio_store = _ExistingAudioStore(root / "audio", plan, interrupt_check=checkpoint,
                                         **{key: audio_identity[key] for key in keys})
        tracks = plan_handle["inspection"]["audio_pcm"]
        if audio_store.has_audio != bool(tracks) or (
            audio_store.has_audio and audio_identity["stream_position"] >= len(tracks)
        ):
            raise ValueError("saved audio preparation track differs from the original input")
        audio = OutpaintAudioProvider(plan_handle["inspection"], audio_store)
        checkpoint()
        if is_regional:
            text = OutpaintRegionalConditioningProvider(root / "text", plan, interrupt_check=checkpoint)
        else:
            text = _ExistingConditioningProvider(root / "text", plan, interrupt_check=checkpoint)
        checkpoint()
        for key, path in paths.items():
            if _manifest(path)[0] != records[key][0]:
                raise ValueError("preparation manifest changed during reload")
        validate_outpaint_source(plan_handle["inspection"], plan)
        handle = {**plan_handle, "root": root, "source": source, "audio": audio, "conditioning": text}
        if is_regional:
            handle["guidance"] = text.guidance
        report = {"schema": "t8.h3.video_outpaint.prepared_reload/v1", "plan_sha256": plan["plan_sha256"],
            "manifest_sha256": {key: hashlib.sha256(raw).hexdigest() for key, (raw, _) in records.items()},
            "source_chunks_verified": len(source.expected), "saved_conditioning_reused": True,
            "conditioning_mode": "regional" if is_regional else "plain",
            "models_loaded_by_reload": False, "vae_or_clip_called": False,
            "sampler_called": False, "cache_manifests_rewritten": False, "perceptual_acceptance": False}
        return handle, report
