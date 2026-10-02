"""Bind real native MODEL, text and source providers before serial sampling."""
from __future__ import annotations

import hashlib

from .patch_stack_policy import model_identity_matches

from .video_outpaint_identity import native_stock_model_identity
from .video_outpaint_noise import COORDINATE_NOISE, NOISE_ALGORITHMS
from .video_outpaint_sampling_runtime import sample_prepared_outpaint_windows
from .video_outpaint_source_runtime import outpaint_gpu_lease
from .video_outpaint_window_store import OutpaintWindowStore


def sample_verified_outpaint(*, model, conditioning, source_store, audio, cache_root,
                            seed=20260808, steps=20, sampler_name="res_multistep", scheduler="simple",
                            resume=False, interrupt_check=None, progress=None, sample_function=None,
                            noise_algorithm=COORDINATE_NOISE):
    if noise_algorithm not in NOISE_ALGORITHMS:
        raise ValueError("unrecognized outpaint noise algorithm")
    if conditioning.plan != source_store.plan or audio.store.plan != source_store.plan:
        raise ValueError("outpaint MODEL inputs must share the same source/shot plan")
    # Sampling runtime owns the later full-run lease and revalidates this snapshot
    # inside it. This preliminary lease prevents concurrent preparation while hashing.
    with outpaint_gpu_lease():
        identity = native_stock_model_identity(model, interrupt_check=interrupt_check)
        source_sha = hashlib.sha256(source_store.path.read_bytes()).hexdigest()
        text_sha, audio_sha = conditioning.verify(), audio.verify()
        store = OutpaintWindowStore(cache_root, source_store.plan, execution_identity={
            "model_sha256": identity["sha256"], "conditioning_sha256": text_sha,
            "source_cache_sha256": source_sha, "audio_source_sha256": audio_sha,
            "seed": seed, "steps": steps, "sampler_name": sampler_name, "scheduler": scheduler,
            "noise_algorithm": noise_algorithm})

    def verify():
        if (not model_identity_matches(identity, native_stock_model_identity(model, interrupt_check=interrupt_check))
                or conditioning.verify() != text_sha or audio.verify() != audio_sha):
            raise ValueError("actual MODEL or conditioning/audio inputs changed during sampling")
        return store.identity

    report = sample_prepared_outpaint_windows(model=model, conditioning_for_window=conditioning,
        audio_for_window=audio, source_store=source_store, window_store=store, verify_execution=verify,
        resume=resume, interrupt_check=interrupt_check, progress=progress, sample_function=sample_function)
    return store, {**report, "loaded_model_identity": identity, "caller_supplied_identity_trusted": False}
