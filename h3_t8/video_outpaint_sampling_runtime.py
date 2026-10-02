"""Serial prepared-source sampling coordinator, independent of VIDEO file saving.

The workflow layer must provide a verified model/conditioning/audio identity and
phase-consistent audio windows. This coordinator never invents silent audio for a
source which has sound and never marks sampling as final video acceptance.
"""
from __future__ import annotations

import hashlib

from .video_outpaint_noise import NATIVE_NOISE, outpaint_window_noise, native_outpaint_window_noise
from .video_outpaint_sampling import sample_outpaint_window
from .video_outpaint_source_runtime import outpaint_gpu_lease


def sample_prepared_outpaint_windows(
    *, model, conditioning_for_window, audio_for_window, source_store, window_store,
    verify_execution, resume=False, interrupt_check=None, progress=None, sample_function=None,
):
    """verify_execution must revalidate actual live state and return the bound identity.

    Provider callbacks are internal adapter boundaries, not trusted serialized user
    reports. Native model/audio/provider construction is still a separate concern.
    """
    def checkpoint():
        if interrupt_check:
            interrupt_check()

    def verify():
        if verify_execution() != window_store.identity:
            raise ValueError("live sampling execution identity differs from its checkpoint")
        if source_store.plan != window_store.plan or source_store.position() is not None:
            raise ValueError("sampling requires the complete matching source latent store")
        actual = hashlib.sha256(source_store.path.read_bytes()).hexdigest()
        if actual != window_store.identity["source_cache_sha256"]:
            raise ValueError("source latent manifest changed after sampling was bound")

    settings = window_store.identity
    sampled_now = 0
    with outpaint_gpu_lease():
        checkpoint()
        verify()
        snapshot = window_store.snapshot()
        for previous in range(len(snapshot["committed"])):
            checkpoint()
            window_store.load(previous)
        start = window_store.begin(resume=resume)
        try:
            for index in range(start, len(window_store.windows)):
                checkpoint()
                verify()
                shot, local = window_store.windows[index]
                window = window_store.plan["shots"][shot]["windows"][local]
                vt = (window["render_frames"]-5)//17*5+2
                source = source_store.read_range(shot, window["video_start"], window["video_start"]+vt)
                audio = audio_for_window(shot, local)
                audio_mask = None
                if isinstance(audio, dict):
                    if set(audio) != {"samples", "noise_mask"}:
                        raise ValueError("audio window provider returned an invalid samples/mask contract")
                    audio, audio_mask = audio["samples"], audio["noise_mask"]
                if settings["noise_algorithm"] == NATIVE_NOISE:
                    video_noise, audio_noise = native_outpaint_window_noise(
                        window_store.plan, shot, local, settings["seed"], interrupt_check=checkpoint)
                else:
                    video_noise, audio_noise = outpaint_window_noise(window_store.plan, shot, local, settings["seed"])
                sampled, _, report = sample_outpaint_window(
                    model=model, conditioning=conditioning_for_window(shot, local), plan=window_store.plan,
                    shot_index=shot, window_index=local, video_latent=source, audio_latent=audio,
                    video_noise=video_noise, audio_noise=audio_noise, context=window_store.context_for(index),
                    seed=settings["seed"], steps=settings["steps"], sampler_name=settings["sampler_name"],
                    scheduler=settings["scheduler"], sample_function=sample_function, audio_noise_mask=audio_mask,
                )
                checkpoint()
                verify()
                video, conditioning_audio = sampled["samples"].unbind()
                window_store.commit(index, video, conditioning_audio)
                sampled_now += 1
                if progress:
                    progress({"stage": "sample_window", "window_global_index": index,
                              "windows_sampled_this_call": sampled_now, **report})
        except BaseException:
            window_store.mark_interrupted()
            raise
    return {"schema": "t8.h3.video_outpaint.sampled_windows/v1", "resume_from": start,
            "windows_sampled_this_call": sampled_now, "all_windows_sampled": window_store.snapshot()["status"] == "sampled",
            "generated_video_complete": False, "perceptual_acceptance": False,
            "window_manifest_sha256": hashlib.sha256(window_store.path.read_bytes()).hexdigest()}
