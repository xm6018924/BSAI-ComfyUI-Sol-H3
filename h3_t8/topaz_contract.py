"""Official Topaz configuration and command contracts; no automatic execution.

Regular enhancement is pixel-domain postprocessing, not H3 latent sampling.
Starlight definitions are never routed to the regular tvai_up filter.
"""
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re


MODEL_ID = re.compile(r'[a-z][a-z0-9-]*(?:\.[0-9]+)*\Z')
REGULAR_PARAMETERS = frozenset({'preblur', 'noise', 'details', 'halo', 'blur',
    'compression', 'prenoise', 'grain', 'gsize', 'blend', 'estimate', 'kcolor'})
DELIVERY_OUTPUT_PROFILES = frozenset({'delivery_h264', 'delivery_hevc_main10'})
REGULAR_OUTPUT_PROFILES = DELIVERY_OUTPUT_PROFILES | {'lossless_master'}


@dataclass(frozen=True)
class OfficialTopaz:
    install: Path
    definitions: Path
    data: Path

    def __post_init__(self):
        reference_only = Path('G:/star2.6').resolve()
        for name in ('install', 'definitions', 'data'):
            path = Path(getattr(self, name)).resolve(strict=True)
            if not path.is_dir() or path == reference_only or path.is_relative_to(reference_only):
                raise ValueError('Use official Topaz directories, never the reference-only star2.6 package')
            object.__setattr__(self, name, path)
        for name in ('Topaz Video.exe', 'ffmpeg.exe', 'ffprobe.exe'):
            self.executable(name)

    def executable(self, name):
        if name not in ('Topaz Video.exe', 'ffmpeg.exe', 'ffprobe.exe'):
            raise ValueError('Unsupported official executable request')
        # Topaz renamed the GUI binary to ``Topaz Video AI.exe`` in newer
        # Windows packages while the tvai/FFmpeg contract still refers to the
        # historical ``Topaz Video.exe`` slot.  Keep the public slot stable and
        # resolve the signed alias locally; fixture installations and older
        # packages continue to use the original name.
        names = ('Topaz Video.exe', 'Topaz Video AI.exe') if name == 'Topaz Video.exe' else (name,)
        candidate = next((self.install / value for value in names
                          if (self.install / value).is_file()), self.install / names[0])
        path = candidate.resolve(strict=True)
        if path.parent != self.install or not path.is_file():
            raise ValueError('Executable leaves the selected official installation')
        return path

    def child_environment(self, inherited):
        # Keep the formal installation's normal licensing state. Never inherit
        # another distribution's engine/model/license routing, read a license,
        # or change the parent's environment. Windows names are case-insensitive.
        controlled = {'TVAI_MODEL_DIR', 'TVAI_MODEL_DATA_DIR', 'TOPAZ_MODEL_STORE',
                      'TOPAZ_ENGINE_MODE', 'TOPAZLABS_LICENSE', 'PATH'}
        env = {key: value for key, value in inherited.items() if key.upper() not in controlled}
        paths = [value for key, value in inherited.items() if key.upper() == 'PATH']
        if len(paths) > 1:
            raise ValueError('Ambiguous inherited PATH aliases')
        env['PATH'] = str(self.install) + (os.pathsep + paths[0] if paths and paths[0] else '')
        env.update(TVAI_MODEL_DIR=str(self.definitions), TVAI_MODEL_DATA_DIR=str(self.data))
        return env

    def model(self, model_id):
        if not isinstance(model_id, str) or len(model_id) > 64 or not MODEL_ID.fullmatch(model_id):
            raise ValueError('Invalid Topaz model ID; choose an installed model definition')
        path = (self.definitions / (model_id + '.json')).resolve(strict=True)
        if path.parent != self.definitions or path.stat().st_size > 4 * 1024**2:
            raise ValueError('Model definition leaves its directory or is unbounded')
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or not isinstance(data.get('backends'), dict):
            raise ValueError('Not a Topaz model definition')
        return path, data


def resolve_output_geometry(source_width, source_height, width, height, scale, size_mode='scale'):
    """Resolve exact pixel geometry without stretching, rounding or model loading."""
    if any(type(x) is not int or x <= 0 for x in (source_width, source_height)):
        raise ValueError('Source dimensions must be positive integers')
    if type(scale) is not int or scale not in (1, 2, 4):
        raise ValueError('Fixed scale selector must be 1, 2 or 4')
    if size_mode not in ('scale', 'target_dimensions'):
        raise ValueError('Unknown Topaz size mode')
    if size_mode == 'scale':
        expected = (source_width * scale, source_height * scale)
        if width is None and height is None:
            width, height = expected
        elif (width, height) != expected:
            raise ValueError('Output size is not the requested fixed model upscale factor')
    if any(type(x) is not int or x < 32 or x > 8192 or x % 2 for x in (width, height)):
        raise ValueError('Regular Topaz target dimensions must be even integers in32..8192')
    ratio = Fraction(width, source_width)
    if ratio != Fraction(height, source_height) or not 1 <= ratio <= 4:
        raise ValueError('Target dimensions must preserve aspect ratio and be between1x and4x')
    return width, height, {'size_mode': size_mode, 'ratio': str(ratio),
        'source_width': source_width, 'source_height': source_height,
        'target_width': width, 'target_height': height,
        'fixed_scale_selector_ignored': size_mode == 'target_dimensions'}


def regular_filter(runtime, model_id, width, height, *, device=0, vram=.8, instances=0, parameters=None):
    _, definition = runtime.model(model_id)
    if definition.get('isNeuroserverModel'):
        raise ValueError('Starlight requires the official Neuroserver route, not tvai_up')
    if definition.get('changesFPS') or definition.get('modelType', 1) != 1:
        raise ValueError('Selected model is not a regular enhancement model; frame interpolation and auxiliary models are outside this route')
    if any(type(x) is not int or x < 32 or x > 8192 or x % 2 for x in (width, height)):
        raise ValueError('Regular Topaz target dimensions must be even integers in32..8192')
    if type(device) is not int or device < 0 or device > 15:
        raise ValueError('Select an explicit nonnegative Topaz GPU index')
    if type(vram) not in (float, int) or not math.isfinite(vram) or not .1 <= vram <= 1:
        raise ValueError('Topaz vram must be finite in0.1..1')
    if type(instances) is not int or not 0 <= instances <= 3:
        raise ValueError('Topaz extra instances must be an integer in0..3')
    parameters = {} if parameters is None else parameters
    if not isinstance(parameters, dict) or set(parameters) - REGULAR_PARAMETERS:
        raise ValueError('Unsupported regular Topaz parameter; arbitrary filter text is not accepted')
    declared = {str(item.get('name', '')).lower(): item for item in definition.get('parameters', [])}
    parts = [f'model={model_id}', 'scale=0', f'w={width}', f'h={height}',
        f'device={device}', f'vram={vram:.6g}', f'instances={instances}', 'download=0']
    for name, value in sorted(parameters.items()):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Topaz parameters must be finite numbers, not booleans/expressions')
        if name in ('estimate', 'kcolor'):
            limit = 100 if name == 'estimate' else 1
            if type(value) is not int or not 0 <= value <= limit:
                raise ValueError('Invalid Topaz ' + name)
        elif name in ('blend', 'grain', 'prenoise', 'gsize'):
            limit = {'blend': 1, 'grain': 1, 'prenoise': .1, 'gsize': 5}[name]
            if not 0 <= value <= limit:
                raise ValueError(f'Topaz {name} must be in0..{limit}')
        else:
            param = declared.get(name)
            if param is None or not param.get('min', -1) <= value <= param.get('max', 1):
                raise ValueError('Parameter is unsupported by the selected model or out of range: ' + name)
        parts.append(f'{name}={value:.9g}')
    return 'tvai_up=' + ':'.join(parts)


def native_dimension_evidence(stderr, expected_frames):
    rows = re.findall(r'\[showinfo@t8_topaz_native[^\]]*\].*?\bn:\s*(\d+)\b.*?\bs:(\d+)x(\d+)\b', stderr)
    if len(rows) != expected_frames or [int(row[0]) for row in rows] != list(range(expected_frames)):
        raise ValueError('Missing or inconsistent native Topaz frame-size trace')
    sizes = {(int(row[1]), int(row[2])) for row in rows}
    if len(sizes) != 1:
        raise ValueError('Topaz native output dimensions changed within the clip')
    width, height = sizes.pop()
    return {'width': width, 'height': height, 'frames': len(rows),
        'exact_weight_variant_verified': False, 'source': 'post_tvai_up_pre_resample_showinfo'}


def regular_command(runtime, source, destination, model_id, width, height, *, size_mode='scale',
                    output_profile='delivery_h264', audio_mode='copy', **settings):
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    suffix = destination.suffix.lower()
    if output_profile not in REGULAR_OUTPUT_PROFILES:
        raise ValueError('Unknown Topaz output profile')
    expected_suffixes = ('.mp4',) if output_profile in DELIVERY_OUTPUT_PROFILES else ('.mkv', '.mov')
    if not source.is_file() or destination.exists() or source == destination or suffix not in expected_suffixes:
        raise ValueError('Expected a source file and a new destination matching the output profile')
    filter_text = regular_filter(runtime, model_id, width, height, **settings)
    if size_mode not in ('scale', 'target_dimensions'):
        raise ValueError('Unknown Topaz size mode')
    if size_mode == 'target_dimensions':
        # tvai_up w/h estimate a native model scale, not an exact output size.
        # Explicit post-AI sizing is part of this mode, never an error fallback.
        filter_text += f',showinfo@t8_topaz_native,scale=w={width}:h={height}:flags=lanczos:threads=2'
    if output_profile == 'delivery_h264':
        # Topaz internally works in RGB48, but storing every 16-bit frame as PNG
        # made a 15-second phone clip exceed 12 GiB and starved the following
        # SaveVideo node. Convert once and encode directly on NVIDIA hardware.
        filter_text += ',format=yuv420p'
        encoder = ['-c:v', 'h264_nvenc', '-preset', 'p5', '-tune', 'hq',
            '-rc', 'vbr', '-cq', '16', '-b:v', '0', '-profile:v', 'high',
            '-pix_fmt', 'yuv420p']
        if suffix == '.mp4':
            encoder += ['-movflags', '+faststart']
    elif output_profile == 'delivery_hevc_main10':
        filter_text += ',format=p010le'
        encoder = ['-c:v', 'hevc_nvenc', '-preset', 'p5', '-tune', 'hq',
            '-rc', 'vbr', '-cq', '16', '-b:v', '0', '-profile:v', 'main10',
            '-pix_fmt', 'p010le']
        if suffix == '.mp4':
            encoder += ['-tag:v', 'hvc1', '-movflags', '+faststart']
    else:
        # Optional audit/archive master. This is intentionally large and is no
        # longer the user-facing default.
        filter_text += ',format=rgb48be'
        encoder = (['-c:v', 'png', '-pix_fmt', 'rgb48be'] if suffix == '.mov' else
            ['-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'gbrp16le'])
    if audio_mode not in ('copy', 'aac') or (audio_mode == 'aac' and suffix != '.mp4'):
        raise ValueError('Invalid automatic Topaz audio delivery mode')
    audio_encoder = (['-c:a', 'copy'] if audio_mode == 'copy'
        else ['-c:a', 'aac', '-b:a', '192k'])
    # No -r, frame interpolation or silent FPS rounding. MP4-incompatible
    # audio is automatically encoded to AAC; compatible audio remains copied.
    return [str(runtime.executable('ffmpeg.exe')), '-hide_banner', '-nostdin', '-n',
        '-protocol_whitelist', 'file,pipe', '-copyts', '-start_at_zero', '-i', str(source),
        '-map', '0:v:0', '-map', '0:a?', '-vf', filter_text, '-fps_mode', 'passthrough',
        '-enc_time_base:v', 'demux',
        *encoder, '-threads', '2', *audio_encoder,
        '-map_metadata', '0', '-progress', 'pipe:1', '-nostats', str(destination)]


def interpolation_filter(runtime, model_id, output_fps, *, device=0, vram=.8, instances=0,
                         duplicate_threshold=.01):
    _, definition = runtime.model(model_id)
    if not definition.get('changesFPS') or definition.get('modelType') != 2:
        raise ValueError('Selected model is not a Topaz frame-interpolation model')
    rate = Fraction(output_fps)
    if rate <= 0 or rate > 240:
        raise ValueError('Topaz interpolation output FPS must be in0..240')
    if type(device) is not int or not 0 <= device <= 15:
        raise ValueError('Select an explicit nonnegative Topaz GPU index')
    if type(vram) not in (float, int) or not math.isfinite(vram) or not .1 <= vram <= 1:
        raise ValueError('Topaz vram must be finite in0.1..1')
    if type(instances) is not int or not 0 <= instances <= 3:
        raise ValueError('Topaz extra instances must be an integer in0..3')
    if (type(duplicate_threshold) not in (float, int)
            or not math.isfinite(duplicate_threshold) or not -.01 <= duplicate_threshold <= .2):
        raise ValueError('Duplicate-frame threshold must be finite in-0.01..0.2')
    return ('tvai_fi=' + ':'.join([f'model={model_id}', f'device={device}',
        f'instances={instances}', 'download=0', f'vram={vram:.6g}', 'slowmo=1',
        f'rdt={duplicate_threshold:.6g}', f'fps={rate.numerator}/{rate.denominator}']))


def interpolation_command(runtime, source, destination, model_id, output_fps, *,
                          output_profile='delivery_h264', audio_mode='copy', **settings):
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if output_profile not in DELIVERY_OUTPUT_PROFILES:
        raise ValueError('Unknown Topaz interpolation output profile')
    if (not source.is_file() or destination.exists() or source == destination
            or destination.suffix.lower() != '.mp4'):
        raise ValueError('Expected a source file and a new MP4 interpolation destination')
    filter_text = interpolation_filter(runtime, model_id, output_fps, **settings)
    if output_profile == 'delivery_h264':
        filter_text += ',format=yuv420p'
        encoder = ['-c:v', 'h264_nvenc', '-preset', 'p5', '-tune', 'hq',
            '-rc', 'vbr', '-cq', '16', '-b:v', '0', '-profile:v', 'high', '-pix_fmt', 'yuv420p']
    else:
        filter_text += ',format=p010le'
        encoder = ['-c:v', 'hevc_nvenc', '-preset', 'p5', '-tune', 'hq',
            '-rc', 'vbr', '-cq', '16', '-b:v', '0', '-profile:v', 'main10', '-pix_fmt', 'p010le']
        if destination.suffix.lower() == '.mp4':
            encoder += ['-tag:v', 'hvc1']
    if audio_mode not in ('copy', 'aac') or (audio_mode == 'aac' and destination.suffix.lower() != '.mp4'):
        raise ValueError('Invalid automatic Topaz audio delivery mode')
    audio_encoder = (['-c:a', 'copy'] if audio_mode == 'copy'
        else ['-c:a', 'aac', '-b:a', '192k'])
    return [str(runtime.executable('ffmpeg.exe')), '-hide_banner', '-nostdin', '-n',
        '-protocol_whitelist', 'file,pipe', '-copyts', '-start_at_zero', '-i', str(source),
        '-map', '0:v:0', '-map', '0:a?', '-vf', filter_text, '-fps_mode', 'passthrough',
        '-enc_time_base:v', 'filter', *encoder,
        *(['-movflags', '+faststart'] if destination.suffix.lower() == '.mp4' else []),
        *audio_encoder, '-map_metadata', '0', '-progress', 'pipe:1', '-nostats', str(destination)]


def validate_cfr_timeline(points, rate):
    """Require exact rational display timestamps, not rounded average FPS."""
    rate = Fraction(rate)
    if rate <= 0 or rate > 240 or len(points) < 2:
        raise ValueError('Regular video needs at least two frames and a valid rational frame rate')
    origin = Fraction(points[0])
    for index, value in enumerate(points):
        if Fraction(value) - origin != Fraction(index, 1) / rate:
            raise ValueError('VFR, missing/duplicate frame, or nonuniform presentation timestamps')
    return {'frames': len(points), 'fps': str(rate), 'origin': str(origin), 'duration': str(len(points) / rate)}
