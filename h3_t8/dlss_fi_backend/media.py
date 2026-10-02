"""CPU file adapter for fixed-2x FI research; no worker startup or node import.

First route deliberately accepts only square-pixel, unrotated, 8-bit ordinary
SDR CFR files. Audio is packet copied with the existing NR identity checks.
RGB frame accounting is pre-encode; H.264 encoding is not pixel lossless.
"""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path

from ..dlss_fi_contract import TwoXTimeline


COLOR_FIELDS = ('color_primaries', 'color_trc', 'colorspace', 'color_range')


def _video_contract(stream):
    context = stream.codec_context
    if stream.sample_aspect_ratio not in (None, Fraction(1)):
        raise ValueError('FI first file route requires square pixels')
    if str(stream.metadata.get('rotate', '0')) not in ('0', '0.0'):
        raise ValueError('Rotated input requires an explicit prior conversion')
    width, height = context.width, context.height
    if min(width, height) < 64 or width % 2 or height % 2 or width*height > 4096*2160:
        raise ValueError('First FI encoder requires even bounded dimensions, at least 64x64')
    fmt = context.format
    if fmt is None or not fmt.components or any(c.bits != 8 for c in fmt.components):
        raise ValueError('First FI route is 8-bit only, not HDR/10-bit processing')
    colors = {key: int(getattr(context, key)) for key in COLOR_FIELDS}
    if any(colors[key] not in (1, 2) for key in COLOR_FIELDS[:3]) or colors['color_range'] not in (0, 1):
        raise ValueError('First FI file route accepts only limited/unspecified Rec.709 SDR')
    rate = Fraction(stream.average_rate) if stream.average_rate else None
    if rate is None or not 0 < rate <= 120:
        raise ValueError('Missing/unsupported exact source FPS')
    if stream.base_rate and Fraction(stream.base_rate) != rate:
        raise ValueError('Source average and real rates disagree')
    return width, height, rate, colors


def _checked_frame(frame, width, height):
    if (frame.width, frame.height) != (width, height) or frame.pts is None or frame.time_base is None:
        raise ValueError('Decoded frame geometry/timestamp changed or is missing')
    for data in frame.side_data:
        if any(token in str(data.type).upper() for token in ('DISPLAYMATRIX', 'MASTERING', 'CONTENT_LIGHT', 'DYNAMIC_HDR')):
            raise ValueError('Display transform/HDR side data is unsupported in the first FI route')
    if any(int(getattr(frame, key, 0) or 0) != 0 for key in ('crop_top', 'crop_bottom', 'crop_left', 'crop_right')):
        raise ValueError('Cropped source requires explicit geometry conversion')
    return Fraction(frame.pts)*Fraction(frame.time_base)


def inspect_source(path, *, cuts=(), check=lambda: None):
    import av
    from ..dlss_nr_advanced import _audio_packet_digests, _audio_pcm_digests
    from .resources import file_identity

    path = Path(path).resolve(strict=True)
    cuts = tuple(cuts)
    if len(set(cuts)) != len(cuts):
        raise ValueError('Duplicate cut markers are not a valid explicit cut list')
    identity = file_identity(path)
    with av.open(str(path), options={'err_detect': 'explode'}) as container:
        if len(container.streams.video) != 1 or any(s.type not in ('video', 'audio') for s in container.streams):
            raise ValueError('FI first route requires one video, optional audio, no additional tracks')
        if any(s.codec.name not in ('aac', 'mp3', 'alac', 'ac3', 'eac3') for s in container.streams.audio):
            raise ValueError('Source audio cannot be copied into this MP4 route; no implicit transcoding')
        stream = container.streams.video[0]
        width, height, rate, colors = _video_contract(stream)
        count, origin = 0, None
        for frame in container.decode(stream):
            check()
            pts = _checked_frame(frame, width, height)
            if origin is None:
                origin = pts
            if pts != origin+Fraction(count)/rate:
                raise ValueError('Source is not exact CFR; no silent timestamp remapping')
            count += 1
            if count > 1_000_000:
                raise ValueError('Frame count exceeds bounded first-route contract')
        if stream.frames and stream.frames != count:
            raise ValueError('Decoded count differs from declared count')
        if stream.duration is not None and Fraction(stream.duration)*stream.time_base != Fraction(count)/rate:
            raise ValueError('Source duration is not exactly its CFR frame duration')
    plan = TwoXTimeline(count, rate, origin, frozenset(cuts))
    audio_packets, audio_pcm = _audio_packet_digests(path), _audio_pcm_digests(path)
    check()
    if file_identity(path) != identity:
        raise ValueError('Source changed during preflight')
    return {'file': identity, 'width': width, 'height': height, 'plan': plan,
            'colors': colors, 'audio_packets': audio_packets, 'audio_pcm': audio_pcm}


def decode_frames(source, *, check=lambda: None):
    import av
    plan = source['plan']
    with av.open(source['file']['path'], options={'err_detect': 'explode'}) as container:
        stream = container.streams.video[0]
        count = 0
        for frame in container.decode(stream):
            check()
            stamp = _checked_frame(frame, source['width'], source['height'])
            if count >= plan.source_count or stamp != plan.origin+Fraction(count)/Fraction(plan.source_rate):
                raise ValueError('Source frame sequence changed since preflight')
            count += 1
            yield frame.to_ndarray(format='rgba')
        if count != plan.source_count:
            raise ValueError('Source decode stopped before declared end')


def encoder_time_base(plan):
    denominator = math.lcm(Fraction(plan.origin).denominator, Fraction(plan.target_rate).numerator)
    if denominator > 2_000_000_000:
        raise ValueError('Exact target clock exceeds the encoder time-base contract')
    return Fraction(1, denominator)


def encode_video(path, source, outputs, *, check=lambda: None):
    import av
    import numpy as np
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    plan, width, height = source['plan'], source['width'], source['height']
    clock = encoder_time_base(plan)
    # Exclusive creation prevents accidental replacement even if called directly.
    with path.open('xb') as file, av.open(file, 'w', format='mp4', options={'movflags': 'faststart', 'avoid_negative_ts': 'disabled'}) as container:
        stream = container.add_stream('libx264', rate=plan.target_rate)
        stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
        stream.time_base = clock
        stream.codec_context.time_base = clock
        stream.codec_context.max_b_frames = 0
        stream.codec_context.thread_count = 1
        stream.options = {'crf': '18', 'preset': 'medium', 'threads': '1'}
        for key, value in source['colors'].items():
            setattr(stream.codec_context, key, value)
        count = 0
        for slot, payload in outputs:
            check()
            if slot.index != count or slot.pts != plan.origin+Fraction(count)/Fraction(plan.target_rate):
                raise ValueError('Encoder received an out-of-order frame/timestamp')
            if count >= plan.output_count or len(payload) != width*height*4:
                raise ValueError('Encoder payload dimensions/count mismatch')
            ticks = slot.pts/clock
            if ticks.denominator != 1:
                raise ValueError('Frame timestamp cannot be exactly encoded')
            rgba = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 4)
            frame = av.VideoFrame.from_ndarray(rgba, format='rgba')
            frame.pts, frame.time_base = int(ticks), clock
            for packet in stream.encode(frame):
                container.mux(packet)
            count += 1
        if count != plan.output_count:
            raise ValueError('Incomplete output cannot qualify')
        for packet in stream.encode():
            container.mux(packet)
    check()
    return {'encoded_frames': count, 'time_base': str(clock), 'codec': 'libx264', 'crf': 18,
            'source_pixels_preserved_after_lossy_encode': False}


def mux_and_validate(video_only, source, target, *, check=lambda: None):
    from ..dlss_nr_advanced import _packet_copy_video_and_audio, _validate_audio_identity
    from .resources import file_identity
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    check()
    _packet_copy_video_and_audio(Path(video_only), Path(source['file']['path']), target)
    result = inspect_source(target, check=check)
    wanted, got = source['plan'], result['plan']
    if (got.source_count, got.source_rate, got.origin, got.duration) != (wanted.output_count, wanted.target_rate, wanted.origin, wanted.duration):
        raise ValueError('Final mux changed frame count/FPS/origin/duration')
    if (result['width'], result['height'], result['colors']) != (source['width'], source['height'], source['colors']):
        raise ValueError('Final mux changed geometry/SDR color metadata')
    audio = _validate_audio_identity(source['audio_packets'], result['audio_packets'], source['audio_pcm'], result['audio_pcm'])
    if file_identity(source['file']['path']) != source['file']:
        raise ValueError('Source changed while processing')
    check()
    return {'file': result['file'], 'frames': got.source_count, 'fps': str(got.source_rate),
            'origin': str(got.origin), 'duration': str(got.duration), 'audio': audio,
            'audio_streams': len(result['audio_packets']), 'quality_qualified': False}
