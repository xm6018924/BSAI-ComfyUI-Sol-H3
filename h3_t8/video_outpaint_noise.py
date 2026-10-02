"""Stateless shot/token-addressed noise, reproducible across resume/window sizes.

This is explicitly T8 coordinate noise v1, not the upstream whole-tensor RNG order.
No global RNG state is read or changed and no full-video noise is allocated.
"""
from __future__ import annotations

import hashlib

import torch

from .video_outpaint_plan import canonical, _integer, validate_outpaint_plan


COORDINATE_NOISE = "t8.outpaint.coordinate_noise/v1"
NATIVE_NOISE = "t8.outpaint.native_cpu_noise/v1"
NOISE_ALGORITHMS = (COORDINATE_NOISE, NATIVE_NOISE)
_BLOCK_FLOATS = 262144


def native_outpaint_window_noise(plan, shot_index, window_index, seed, *, interrupt_check=None):
    """Stream the native FP32 CPU video-then-audio RNG sequence into one window.

    Only the requested window and one 1MiB temporary block are allocated. Each
    call replays the shot's global stream: bounded memory, not reduced CPU work.
    H3's 32px canvas grid makes both whole-stream counts multiples of 16, avoiding
    the different tail behavior of CPU randn calls of arbitrary lengths. Exact
    parity is tested against Comfy's prepare_noise_inner on the installed runtime.
    Shots restart the same seed, as independent native runs would. No global RNG
    state is touched; this does not promise parity for arbitrary stochastic hooks.
    """
    checked = validate_outpaint_plan(plan)
    _integer(seed, "seed", 0, 2**64-1)
    _integer(shot_index, "shot_index", 0, len(checked["shots"])-1)
    shot = checked["shots"][shot_index]
    _integer(window_index, "window_index", 0, len(shot["windows"])-1)
    window = shot["windows"][window_index]
    global_video_t = (shot["aligned_frames"]-5)//17*5+2
    global_audio_t = round(shot["aligned_frames"]/24*40)
    video_t = (window["render_frames"]-5)//17*5+2
    audio_t = round(window["render_frames"]/24*40)
    height, width = checked["sampling"]["height"]//16, checked["sampling"]["width"]//16
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def stream(channels, global_t, local_t, start, plane):
        whole_size = channels*global_t*plane
        if whole_size % 16:
            raise ValueError("native outpaint noise requires 16-aligned whole streams")
        result = torch.empty(channels*local_t*plane, dtype=torch.float32, device="cpu")
        ranges = [(c*global_t*plane+start*plane, c*global_t*plane+(start+local_t)*plane,
                   c*local_t*plane) for c in range(channels)]
        range_index = 0
        for offset in range(0, whole_size, _BLOCK_FLOATS):
            if interrupt_check:
                interrupt_check()
            count = min(_BLOCK_FLOATS, whole_size-offset)
            block = torch.randn(count, generator=generator, dtype=torch.float32, device="cpu")
            while range_index < channels and ranges[range_index][1] <= offset:
                range_index += 1
            current = range_index
            while current < channels and ranges[current][0] < offset+count:
                first, last, destination = ranges[current]
                lo, hi = max(first, offset), min(last, offset+count)
                if hi > lo:
                    result[destination+lo-first:destination+hi-first].copy_(block[lo-offset:hi-offset])
                current += 1
            del block
        return result

    video = stream(24, global_video_t, video_t, window["video_start"], height*width)
    audio = stream(64, global_audio_t, audio_t, window["audio_start"], 1)
    return video.reshape(1, 24, video_t, height, width), audio.reshape(1, 32, 2, audio_t)


def outpaint_window_noise(plan, shot_index, window_index, seed):
    checked = validate_outpaint_plan(plan)
    _integer(seed, "seed", 0, 2**64-1)
    _integer(shot_index, "shot_index", 0, len(checked["shots"])-1)
    shot = checked["shots"][shot_index]
    _integer(window_index, "window_index", 0, len(shot["windows"])-1)
    window = shot["windows"][window_index]
    video_t = (window["render_frames"]-5)//17*5+2
    audio_t = round(window["render_frames"]/24*40)
    height, width = checked["sampling"]["height"]//16, checked["sampling"]["width"]//16
    identity = {"algorithm": "t8.outpaint.coordinate_noise/v1", "source": checked["source"]["sha256"],
                "shot_start": shot["start"], "shot_stop": shot["stop"], "seed": seed,
                "height": height, "width": width}
    video = torch.empty((1, 24, video_t, height, width))
    audio = torch.empty((1, 32, 2, audio_t))
    generator = torch.Generator(device="cpu")
    for stream, start, count, shape in (("video", window["video_start"], video_t, (24, height, width)),
                                       ("audio", window["audio_start"], audio_t, (32, 2))):
        for local in range(count):
            digest = hashlib.sha256(canonical({**identity, "stream": stream, "token": start+local}).encode()).digest()
            generator.manual_seed(int.from_bytes(digest[:8], "little"))
            value = torch.randn(shape, generator=generator, dtype=torch.float32)
            if stream == "video":
                video[0, :, local] = value
            else:
                audio[0, :, :, local] = value
    return video, audio
