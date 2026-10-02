"""Trusted CPU codec worker: bounded PCM envelopes and unchanged video packets.

No Core/model import, external commands, source overwrite or frame-batch decode.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

import av
import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def boundary(path, n):
    """First actual displayed PTS and start of frame N (or last frame end)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        first = previous = None
        count = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('Opening fade requires actual video frame timestamps')
            pts = frame.pts * Fraction(frame.time_base)
            if first is None:
                first = pts
            if previous is not None and pts <= previous:
                raise ValueError('Video display timestamps must increase')
            if count == n:
                return first, pts, count, 'actual_frame_pts'
            previous = pts
            last_duration = frame.duration * Fraction(frame.time_base) if frame.duration else None
            count += 1
        if first is None:
            raise ValueError('Video contains no displayed frames')
        if last_duration is None or last_duration <= 0:
            # Do not manufacture the final VFR duration from average FPS.
            if stream.duration is None:
                raise ValueError('Last displayed frame duration is unknown')
            end = ((stream.start_time or 0) + stream.duration) * Fraction(stream.time_base)
            if end <= previous:
                raise ValueError('Video end timestamp is inconsistent')
        else:
            end = previous + last_duration
        return first, end, count, 'actual_last_frame_end'


def video_signature(path):
    """Encoded bytes plus exact timestamp values, not container-byte equality."""
    result = {}
    with av.open(str(path)) as container:
        for packet in container.demux(*container.streams.video):
            if packet.dts is None:
                continue
            stream = packet.stream
            rows = result.setdefault(stream.index, [])
            tb = Fraction(packet.time_base)
            rows.append((hashlib.sha256(bytes(packet)).hexdigest(),
                         str(packet.pts * tb) if packet.pts is not None else None,
                         str(packet.dts * tb), str(packet.duration * tb)))
    return list(result.values())


def process_file(source, destination, n, fade_ms):
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if destination == source or destination.exists():
        raise ValueError('Choose a new task-owned media path, never overwrite source')
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError('mute_first_frames must be a nonnegative integer')
    fade_ms = Fraction(str(fade_ms))
    if fade_ms < 0:
        raise ValueError('fade_in_ms must be nonnegative')
    before = digest(source)
    origin, mute_end, observed_frames, basis = boundary(source, n)
    counts = []
    with av.open(str(source)) as probe:
        timescale = math.lcm(*(Fraction(s.time_base).denominator for s in probe.streams))
        if timescale > 2**31 - 1:
            raise ValueError('Exact source timing exceeds MP4 movie-timescale capacity')
    with av.open(str(source)) as inp, av.open(str(destination), mode='w', format='mp4',
             options={'movflags': '+faststart', 'movie_timescale': str(timescale)}) as out:
        out.metadata.update(inp.metadata)
        tracks = {}
        for stream in inp.streams:
            if stream.type == 'video':
                copied = out.add_stream_from_template(stream, opaque=True)
                tracks[stream.index] = ('copy', copied)
            elif stream.type == 'audio':
                ctx = stream.codec_context
                rate, layout = ctx.sample_rate, ctx.layout.name
                if not rate or not layout:
                    raise ValueError('Audio stream must declare its real sample rate/layout')
                encoded = out.add_stream('aac', rate=rate)
                encoded.layout = layout
                encoded.bit_rate = min(512000, 128000 * ctx.channels)
                encoded.time_base = Fraction(1, rate)
                encoded.metadata.update(stream.metadata)
                stats = dict(source_stream=stream.index, sample_rate=rate, layout=layout,
                             samples=0, decoded_source_samples=0, zero_samples=0, fade_samples=0,
                             first_pcm_pts_seconds=None, last_pcm_end_seconds=None)
                if stream.duration is None:
                    raise ValueError('Audio duration is unknown; codec padding cannot be mapped accurately')
                stats['_start'] = (stream.start_time or 0) * Fraction(stream.time_base)
                stats['_end'] = stats['_start'] + stream.duration * Fraction(stream.time_base)
                stats['effective_start_seconds'] = str(stats['_start'])
                stats['effective_end_seconds'] = str(stats['_end'])
                counts.append(stats)
                tracks[stream.index] = ('audio', encoded,
                    av.AudioResampler(format='fltp', layout=layout, rate=rate), stats)
            else:
                # Explicit failure avoids silently deleting subtitles/data tracks.
                raise ValueError(f'Opening fade MP4 adapter does not silently discard {stream.type} streams')

        def mux_audio(packet, track):
            # Drop the encoder's extra priming packet, not source samples. AAC's
            # first effective packet starts at the original PCM PTS. This avoids
            # MP4 moving a delayed track earlier to accommodate new priming.
            stats = track[3]
            start, end = packet.pts * Fraction(packet.time_base), stats['_end']
            if start < stats['_start'] or start >= end:
                return
            duration = min(packet.duration * Fraction(packet.time_base), end - start)
            exact_ticks = duration / Fraction(packet.time_base)
            if exact_ticks.denominator != 1:
                raise ValueError('AAC packet timebase cannot represent exact audio end')
            packet.duration = exact_ticks.numerator
            out.mux(packet)

        def encode_pcm(frame, track):
            _, encoded, resampler, stats = track
            for pcm in resampler.resample(frame):
                if pcm.pts is None:
                    raise ValueError('Audio has no PTS; cannot infer synchronization from average FPS')
                rate = pcm.sample_rate
                start = pcm.pts * Fraction(pcm.time_base)
                array = pcm.to_ndarray().copy()
                if not np.isfinite(array).all():
                    raise ValueError('Audio PCM contains nonfinite values')
                # Absolute sample times respect leading audio, offsets and gaps.
                # ceil excludes samples before the video opening from the envelope.
                def ceil(value):
                    return -(-value.numerator // value.denominator)
                stats['decoded_source_samples'] += pcm.samples
                keep_start = max(0, min(pcm.samples, ceil((stats['_start'] - start) * rate)))
                keep_end = max(0, min(pcm.samples, ceil((stats['_end'] - start) * rate)))
                if keep_end <= keep_start:
                    continue
                array = array[:, keep_start:keep_end]
                start += Fraction(keep_start, rate)
                samples = array.shape[-1]
                ticks = start * rate
                if ticks.denominator != 1:
                    raise ValueError('Audio sample PTS is not aligned to its real sample rate')
                lo = max(0, min(samples, ceil((origin - start) * rate)))
                hi = max(0, min(samples, ceil((mute_end - start) * rate)))
                array[:, lo:hi] = 0
                fade_count = ceil(min(fade_ms / 1000, max(Fraction(0), stats['_end'] - mute_end)) * rate)
                fade_start = ceil((mute_end - start) * rate)
                left, right = max(0, fade_start), min(samples, fade_start + fade_count)
                if right > left:
                    positions = np.arange(left, right, dtype=np.float64) - fade_start
                    gain = np.zeros_like(positions) if fade_count <= 1 else (1 - np.cos(np.pi * positions / (fade_count - 1))) / 2
                    array[:, left:right] *= gain.astype(array.dtype)
                    stats['fade_samples'] += right - left
                result = av.AudioFrame.from_ndarray(np.ascontiguousarray(array), format='fltp', layout=pcm.layout.name)
                result.sample_rate, result.pts, result.time_base = rate, ticks.numerator, Fraction(1, rate)
                for packet in encoded.encode(result):
                    mux_audio(packet, track)
                stats['samples'] += samples
                stats['zero_samples'] += max(0, hi - lo)
                if stats['first_pcm_pts_seconds'] is None:
                    stats['first_pcm_pts_seconds'] = str(start)
                stats['last_pcm_end_seconds'] = str(start + Fraction(samples, rate))

        for packet in inp.demux():
            track = tracks[packet.stream.index]
            if track[0] == 'copy':
                if packet.dts is not None:
                    packet.stream = track[1]
                    out.mux(packet)
            else:
                for frame in packet.decode():
                    encode_pcm(frame, track)
        for track in tracks.values():
            if track[0] == 'audio':
                encode_pcm(None, track)
                for packet in track[1].encode(None):
                    mux_audio(packet, track)
    for stats in counts:
        del stats['_start'], stats['_end']
    if before != digest(source):
        raise RuntimeError('Source media changed while processing')
    unchanged = video_signature(source) == video_signature(destination)
    if not unchanged:
        raise RuntimeError('Video encoded bytes/timestamps changed; candidate is not deliverable')
    # Full bounded decoding verifies both streams, not just a successful mux.
    decoded = {'video': 0, 'audio_samples': {}}
    with av.open(str(destination)) as container:
        for packet in container.demux():
            for frame in packet.decode():
                if packet.stream.type == 'video':
                    decoded['video'] += 1
                else:
                    key = str(packet.stream.index)
                    decoded['audio_samples'][key] = decoded['audio_samples'].get(key, 0) + frame.samples
    return dict(schema='t8.audio.opening_mute_fade.file.v1', status='processed', source_sha256=before,
                video_packet_bytes_and_timestamps_unchanged=unchanged,
                video_origin_seconds=str(origin), mute_end_seconds=str(mute_end),
                boundary_basis=basis, boundary_frames_observed=observed_frames,
                fade_in_ms=str(fade_ms), audio_tracks=counts, full_decode=decoded,
                sampling_started=False, input_overwritten=False,
                limitation='AAC is lossy. Decoder block padding can differ; effective PCM interval/duration/PTS, not encoded sample identity, is preserved. Encoder priming packet is discarded to preserve delayed-track start. No loudness normalization.')


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
    report = process_file(**request)
    Path(sys.argv[2]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
