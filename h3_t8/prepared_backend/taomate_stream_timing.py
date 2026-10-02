"""Exact rational native/published timing for the pinned direct5s Tao recipe."""

from fractions import Fraction


def stream_timing(request_count):
    if type(request_count) is not int or request_count < 1:
        raise ValueError("Native stream request count must be a positive integer")
    frames = 124 + 119 * (request_count - 1)
    audio = round(Fraction(frames * 40, 24))
    return dict(
        requests=request_count,
        native_frames=frames,
        published_frames=120 * request_count,
        fps=24,
        video_latents=37 + 35 * (request_count - 1),
        audio_latents=audio,
        native_samples=audio * 800,
        published_samples=160000 * request_count,
        sample_rate=32000,
    )


def validate_stream_timing(values, request_count):
    expected = stream_timing(request_count)
    if (
        not isinstance(values, dict)
        or set(values) != set(expected)
        or any(type(value) is not int for value in values.values())
        or values != expected
    ):
        raise ValueError("Stream timing differs from the pinned native global timeline")
    return expected
