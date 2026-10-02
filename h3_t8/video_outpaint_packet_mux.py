"""Isolated PyAV packet mux worker. Intentionally imports no Torch or ComfyUI."""
from contextlib import ExitStack
from fractions import Fraction
import heapq
import json
from pathlib import Path
import sys

import av


def mux(video, source, target):
    with ExitStack() as stack:
        vc = stack.enter_context(av.open(str(video), mode="r"))
        ac = stack.enter_context(av.open(str(source), mode="r"))
        output = stack.enter_context(av.open(str(target), mode="w", format="mp4",
            options={"movflags": "use_metadata_tags+faststart", "avoid_negative_ts": "disabled",
                     "max_interleave_delta": "1000000"}))
        if len(vc.streams.video) != 1 or vc.streams.audio:
            raise ValueError("packet mux candidate must contain exactly one video and no audio")
        streams = [vc.streams.video[0], *ac.streams.audio]
        targets = [output.add_stream_from_template(stream, opaque=True) for stream in streams]
        for original, destination in zip(streams, targets):
            destination.time_base = original.time_base
        output.metadata.update({str(k): str(v) for k, v in ac.metadata.items()})
        iterators = [iter(vc.demux(vc.streams.video[0]))]
        # Independent readers keep one pending packet per stream. A single
        # interleaved demux iterator can have non-monotonic cross-track DTS.
        for index in range(len(ac.streams.audio)):
            reader = stack.enter_context(av.open(str(source), mode="r"))
            iterators.append(iter(reader.demux(reader.streams.audio[index])))
        pending = []
        def advance(index):
            for packet in iterators[index]:
                if packet.dts is not None:
                    heapq.heappush(pending, (Fraction(packet.dts)*packet.time_base, index, packet))
                    return
        for index in range(len(iterators)):
            advance(index)
        counts = [0]*len(iterators)
        maximum = 0
        while pending:
            _, index, packet = heapq.heappop(pending)
            maximum = max(maximum, packet.size)
            packet.stream = targets[index]
            output.mux(packet)
            counts[index] += 1
            advance(index)
    return {"backend": "isolated_pyav_packet_copy", "av_version": av.__version__,
            "library_versions": av.library_versions, "packet_counts": counts,
            "max_packet_bytes": maximum, "pending_packets_bound": len(iterators),
            "interleave_delta_us": 1000000, "audio_reencoded": False}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: packet_mux.py VIDEO_ONLY SOURCE OUTPUT")
    paths = [Path(p).resolve() for p in sys.argv[1:]]
    if paths[2] in paths[:2]:
        raise ValueError("packet mux must not overwrite an input")
    if paths[2].exists() and paths[2].stat().st_size:
        raise FileExistsError("packet mux needs a new or empty private output")
    print(json.dumps(mux(*paths)), flush=True)
