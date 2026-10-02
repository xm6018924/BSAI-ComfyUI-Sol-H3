"""One whole-source audio encode for Dance, without altering video packets."""
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import av
import numpy as np
import torch

from .core import validate_audio
from .long_video_delivery import _run_isolated_ffmpeg, _sha256_file, _strict_validate_mp4

POLICY = 'dance_whole_source_audio_aac_v1'


def prepare_source_audio(audio, frame_count, fps=24):
    waveform, rate = validate_audio(audio, 'Dance original audio')
    if isinstance(audio['sample_rate'], bool) or float(audio['sample_rate']) != rate:
        raise ValueError('Dance original audio sample rate must be an exact integer')
    if rate not in (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000):
        raise ValueError('Dance original audio sample rate is unsupported by this AAC route')
    if waveform.shape[1] not in (1, 2):
        raise ValueError('Dance original audio must be mono or stereo; no implicit downmix')
    if int(frame_count) != frame_count or frame_count <= 0 or fps <= 0:
        raise ValueError('Dance audio delivery needs a positive frame count and fps')
    samples = round(Fraction(int(frame_count), 1) * rate / Fraction(fps))
    if waveform.shape[-1] < samples:
        raise ValueError('Original music is shorter than the delivered Dance video; provide a complete track')
    pcm = waveform[0, :, :samples].detach().float().cpu()
    if not torch.isfinite(pcm).all():
        raise ValueError('Original music contains NaN or Inf')
    interleaved = np.ascontiguousarray(pcm.numpy().T, dtype='<f4')
    contract = dict(policy=POLICY, sample_rate=rate, channels=waveform.shape[1], samples=samples,
                    frame_count=int(frame_count), fps=str(Fraction(fps)),
                    pcm_sha256=hashlib.sha256(interleaved.tobytes()).hexdigest())
    contract['identity'] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return interleaved, contract


def video_packet_identity(path):
    digest = hashlib.sha256()
    count = 0
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        info = {'width': video.width, 'height': video.height, 'codec': video.codec_context.name}
        for packet in container.demux(video):
            if not packet.size:
                continue
            timing = [str(value * packet.time_base) if value is not None else None for value in (packet.pts, packet.dts, packet.duration)]
            digest.update(json.dumps(timing).encode())
            digest.update(bytes(packet))
            count += 1
    return {**info, 'packets': count, 'sha256': digest.hexdigest()}


def finalize_source_audio(video_path, audio, frame_count, fps=24, cached_report=None):
    """Return a new MP4 with copied video and one AAC encode of the original PCM."""
    pcm, contract = prepare_source_audio(audio, frame_count, fps)
    source = Path(video_path).resolve(strict=True)
    if cached_report and cached_report.get('identity') == contract['identity']:
        if cached_report.get('output_path') == str(source) and cached_report.get('output_sha256') == _sha256_file(source):
            return str(source), cached_report
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('FFmpeg is required for Dance original music delivery')
    before = video_packet_identity(source)
    if before['packets'] != frame_count:
        raise ValueError('Dance video packet count does not match its declared frames')
    output = source.parent / f"dance_music_{before['sha256'][:16]}_{contract['identity'][:16]}.mp4"
    receipt_path = output.with_suffix('.audio.json')
    if output.exists():
        if receipt_path.is_file():
            saved = json.loads(receipt_path.read_text(encoding='utf8'))
            if saved.get('identity') == contract['identity'] and saved.get('output_sha256') == _sha256_file(output) and saved.get('video_packets') == before:
                return str(output), saved
        raise FileExistsError('Dance music output already exists without a matching cache receipt')
    with tempfile.TemporaryDirectory(prefix='.dance-audio-', dir=source.parent) as temp:
        root = Path(temp)
        raw, candidate, log = root / 'audio.f32', root / 'result.mp4', root / 'ffmpeg.log'
        raw.write_bytes(pcm.tobytes())
        _run_isolated_ffmpeg([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
            '-i', str(source), '-f', 'f32le', '-ar', str(contract['sample_rate']), '-ac', str(contract['channels']), '-i', str(raw),
            '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(candidate)], log,
            operation='Dance whole original audio mux')
        _strict_validate_mp4(candidate)
        after = video_packet_identity(candidate)
        if before != after:
            raise RuntimeError('Dance whole-audio mux changed video packets or timing')
        with av.open(str(candidate)) as media:
            audio_stream = media.streams.audio[0]
            duration = float(audio_stream.duration * audio_stream.time_base)
        if abs(duration - frame_count / float(fps)) > 1 / contract['sample_rate']:
            raise RuntimeError('Dance audio stream duration differs from the intended timeline')
        report = {**contract, 'output_path': str(output), 'output_sha256': _sha256_file(candidate),
                  'video_packets': after, 'audio_duration_seconds': duration,
                  'source_pcm_unchanged_before_encoding': True, 'aac_bitexact': False,
                  'boundary_processing': 'whole track; no segment bridge, repeat or resampling'}
        with candidate.open('r+b') as stream:
            os.fsync(stream.fileno())
        receipt = root / 'receipt.json'
        receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        os.replace(receipt, receipt_path)
        os.replace(candidate, output)
    return str(output), report
