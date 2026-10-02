"""Isolated CPU PyAV conditioning decoder. No Comfy/Torch/package imports.

Run as a subprocess because native FFmpeg/PyAV faults cannot be caught by Python
exceptions in the parent Comfy process. Output is a newly created scratch file.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path


def decode(source, target, *, origin, samples, track):
    import av
    import numpy as np

    if samples <= 0 or track < 0:
        raise ValueError("invalid conditioning audio bounds")
    with target.open("x+b") as output, av.open(str(source), mode="r") as container:
        output.truncate(samples*8)
        stream = container.streams.audio[track]
        stream.codec_context.thread_count = 1
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=32000)

        def copy(frame):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("audio conditioning requires timestamped resampler frames")
            cursor = round((Fraction(frame.pts)*Fraction(frame.time_base)-origin)*32000)
            array = frame.to_ndarray()
            if array.shape != (2, frame.samples) or not np.isfinite(array).all():
                raise ValueError("invalid decoded stereo conditioning frame")
            a, b = max(0, -cursor), min(frame.samples, samples-cursor)
            if b > a:
                output.seek((cursor+a)*8)
                output.write(array[:, a:b].T.astype("<f4", copy=False).tobytes())

        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                copy(converted)
            if (frame.pts is not None and frame.time_base is not None
                    and Fraction(frame.pts)*Fraction(frame.time_base) > origin+Fraction(samples, 32000)):
                break
        for converted in resampler.resample(None):
            copy(converted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin", type=Fraction, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--track", type=int, required=True)
    args = parser.parse_args()
    decode(args.source, args.output, origin=args.origin, samples=args.samples, track=args.track)
