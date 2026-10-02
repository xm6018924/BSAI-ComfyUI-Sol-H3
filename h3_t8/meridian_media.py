"""Selected-window VIDEO decoding, normalized24fps and explicit original-PCM policy."""

from fractions import Fraction
import math
from pathlib import Path
import tempfile

import numpy as np


def display_pixels(pixels, sar, rotation):
    """Honor residual display metadata; unspecified SAR0 means square pixels."""
    ratio = Fraction(sar) if sar is not None else Fraction(1)
    if ratio < 0:
        raise ValueError("Negative VIDEO sample aspect ratio")
    if ratio not in (0, 1):
        from PIL import Image

        height, width = pixels.shape[:2]
        pixels = np.array(
            Image.fromarray(pixels).resize(
                (max(1, round(width * ratio)), height), Image.Resampling.BICUBIC
            )
        )
    angle = float(rotation or 0)
    if not math.isfinite(angle) or abs(angle / 90 - round(angle / 90)) > 1e-6:
        raise ValueError("VIDEO rotation must be a finite multiple of90degrees")
    quadrants = int(round(angle / 90)) % 4
    return np.rot90(pixels, k=quadrants).copy() if quadrants else pixels


def image_u8(image):
    import torch

    if (
        not isinstance(image, torch.Tensor)
        or image.ndim != 4
        or image.shape[0] != 1
        or image.shape[-1] < 3
    ):
        raise ValueError(
            "Meridian IMAGE must contain exactly one RGB frame; use VIDEO for a clip"
        )
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("IMAGE requires finite RGB values in0..1")
    return (image[..., :3].detach().cpu() * 255).round().byte().contiguous()


def selected_frames(path, start, count, check=lambda: None):
    """Decode to target24fps via hold at actual PTS. Retain selected window only.

    Caller first exports the Core VIDEO view to preserve trim/crop/SAR/rotation.
    No metadata frame-count trust, no silent shortening of requested window.
    """
    import av
    import torch

    if type(start) is not int or type(count) is not int or start < 0 or count < 1:
        raise ValueError("Nonnegative integer start and positive window count required")
    targets = [Fraction(start + i, 24) for i in range(count)]
    out, previous, previous_time, first_time, last_time = [], None, None, None, None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # This window's identity is the decoded RGB, not merely file bytes.
        # The installed FFmpeg/PyAV frame-threaded decoder produced different
        # pixels across fresh opens of the same valid H264 clip. A single
        # decoder thread makes the selected source deterministic; this is local
        # to our container, not a global codec/Core setting or GPU control.
        stream.codec_context.thread_count = 1
        for frame in container.decode(stream):
            check()
            if frame.pts is None:
                raise ValueError(
                    "VIDEO frame has no timestamp; cannot normalize source time"
                )
            t = Fraction(frame.pts) * Fraction(frame.time_base)
            if first_time is None:
                first_time = t
            t -= first_time
            if previous_time is not None and t <= previous_time:
                raise ValueError("VIDEO timestamps must strictly increase")
            # Copy only when target has been reached; preceding frame is the held source.
            while len(out) < count and targets[len(out)] < t and previous is not None:
                out.append(previous.copy())
            if len(out) == count:
                break
            # Core may stream-copy an untrimmed same-codec file. Honor remaining display metadata.
            previous = display_pixels(
                frame.to_ndarray(format="rgb24"),
                stream.sample_aspect_ratio,
                frame.rotation,
            )
            previous_time, last_time = t, t
        duration = (
            Fraction(stream.duration) * Fraction(stream.time_base)
            if stream.duration is not None
            else None
        )
        if duration is None and last_time is not None:
            rate = stream.average_rate
            if rate is None:
                raise ValueError(
                    "VIDEO end duration unavailable; cannot validate full window"
                )
            duration = last_time + Fraction(1, 1) / Fraction(rate)
        while (
            len(out) < count
            and previous is not None
            and duration is not None
            and targets[len(out)] < duration
        ):
            out.append(previous.copy())
    if len(out) != count:
        raise ValueError(
            f"Requested window is longer than VIDEO: decoded {len(out)}/{count} normalized frames"
        )
    dimensions = out[0].shape
    if any(x.shape != dimensions for x in out):
        raise ValueError(
            "Variable source dimensions unsupported in one geometry window"
        )
    return torch.from_numpy(np.stack(out)), dict(
        first_PTS=str(first_time),
        normalized_fps="24",
        selection="hold_at_actual_PTS",
        requested_start=start,
        requested_count=count,
    )


def video_window(video, start, count, check=lambda: None):
    from comfy_api.latest import Types

    with tempfile.TemporaryDirectory(prefix="t8-meridian-source-") as directory:
        path = Path(directory) / "active-view.mp4"
        check()
        # Even a file view must honor active trim/crop, not get_stream_source shortcuts.
        video.save_to(
            str(path), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264
        )
        check()
        return selected_frames(path, start, count, check)


def original_audio_window(video, start_frame, frames, check=lambda: None):
    """Original audio is media-level delivery only; never claimed generated lipsync.

    Stream resample retains channels and uses audio PTS, with exact N/24 output length.
    """
    import av
    import torch
    from comfy_api.latest import Types

    check()
    with tempfile.TemporaryDirectory(prefix="t8-meridian-audio-") as directory:
        path = Path(directory) / "active-view.mp4"
        video.save_to(
            str(path), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264
        )
        check()
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError("source_1to1 selected but source VIDEO has no audio")
            stream = container.streams.audio[0]
            rate = stream.codec_context.sample_rate
            channels = stream.codec_context.channels
            length = math.ceil(Fraction(frames * rate, 24))
            offset = Fraction(start_frame * rate, 24)
            waveform = np.zeros((channels, length), dtype=np.float32)
            resampler = av.AudioResampler(
                format="fltp", layout=stream.codec_context.layout, rate=rate
            )
            origin = None
            # Audio and video must share the same exported view origin, not separate first-audio PTS.
            with av.open(str(path)) as clock:
                first = next(clock.decode(video=0))
                origin = Fraction(first.pts) * Fraction(first.time_base)

            def copy(frame):
                if frame.pts is None:
                    raise ValueError("Source PCM timestamp missing")
                left = round(
                    (Fraction(frame.pts) * Fraction(frame.time_base) - origin) * rate
                    - offset
                )
                arr = frame.to_ndarray()
                a, b = max(0, left), min(length, left + arr.shape[1])
                if a < b:
                    waveform[:, a:b] = arr[:, a - left : b - left]
                return left >= length

            finished = False
            for frame in container.decode(stream):
                check()
                for pcm in resampler.resample(frame):
                    if copy(pcm):
                        finished = True
                        break
                if finished:
                    break
            if not finished:
                for pcm in resampler.resample(None):
                    copy(pcm)
    check()
    return dict(waveform=torch.from_numpy(waveform)[None], sample_rate=rate)
