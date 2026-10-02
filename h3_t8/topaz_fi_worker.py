"""Private official Topaz frame-interpolation worker."""
import importlib.util
import json
import os
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import sys


def local_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    contract, media = local_module('topaz_contract'), local_module('topaz_media')
    request = Path(sys.argv[1]).resolve(strict=True)
    spec = json.loads(request.read_text(encoding='utf8'))
    job = Path(spec['job']).resolve(strict=True)
    if request.parent != job:
        raise ValueError('Task request must belong to its own output directory')
    runtime = contract.OfficialTopaz(spec['install'], spec['definitions'], spec['data'])
    identities = [spec['source'], *spec['installation']['executables'],
        spec['model']['definition'], *spec['model']['candidate_weights']]

    def revalidate():
        for expected in identities:
            if media.file_identity(expected['path']) != expected:
                raise RuntimeError('Source, runtime, model definition or weight changed after task preparation')

    env = runtime.child_environment(os.environ)

    def run(command, name):
        with (job / (name + '.stdout')).open('xb') as out, (job / (name + '.stderr')).open('xb') as err:
            result = subprocess.run(command, cwd=runtime.install, env=env, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, shell=False,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            raise RuntimeError(f'Official Topaz {name} failed with exit code{result.returncode}; see task stderr. No fallback/download was attempted.')
        return (job / (name + '.stdout')).read_text(encoding='utf8') if 'probe' in name else None

    revalidate()
    settings = spec['settings']
    output_profile = settings.get('output_profile', 'delivery_h264')
    source_bit_depths = media.qualified_input_bit_depths(output_profile)
    output_bit_depths = (10,) if output_profile == 'delivery_hevc_main10' else (8,)
    source = Path(spec['source']['path'])
    source_video = media.analyze_video(json.loads(run(
        media.probe_command(runtime, source, frames=True), 'source_video_probe')),
        allowed_bit_depths=source_bit_depths)
    source_audio = json.loads(run(media.probe_command(runtime, source, packets=True), 'source_audio_probe'))
    source_pcm = media.decoded_pcm_digests(runtime, source, source_audio, env, job / 'source_pcm.stderr')
    multiplier = settings['multiplier']
    output_fps = Fraction(source_video['fps']) * multiplier
    if output_fps > 240:
        raise ValueError('Requested interpolation would exceed 240fps')
    encoder = 'hevc_nvenc' if output_profile == 'delivery_hevc_main10' else 'h264_nvenc'
    run([str(runtime.executable('ffmpeg.exe')), '-hide_banner', '-h', 'encoder=' + encoder],
        'encoder_probe')
    if ('Encoder ' + encoder + ' ') not in (job / 'encoder_probe.stdout').read_text(encoding='utf8'):
        raise RuntimeError('Official FFmpeg lacks the required NVIDIA output encoder')
    estimate = max(256 * 1024**2,
        source_video['width'] * source_video['height'] * source_video['frames'] * multiplier // 8)
    available = shutil.disk_usage(job).free
    (job / 'disk_preflight.json').write_text(json.dumps({
        'output_directory': str(job), 'required_bytes': estimate + 512 * 1024**2,
        'conservative_delivery_working_estimate_bytes': estimate,
        'available_bytes': available, 'safety_margin_bytes': 512 * 1024**2,
        'status': 'advisory', 'blocking': False,
        'runtime_stop_floor_bytes': 256 * 1024**2}, indent=2), encoding='utf8')
    suffix = media.delivery_suffix(source_audio)
    audio_mode = media.delivery_audio_mode(source_audio)
    pending = job / ('enhanced.pending' + suffix)
    command = contract.interpolation_command(runtime, source, pending, settings['model_id'],
        output_fps, device=settings['device'], vram=settings['vram'],
        instances=settings.get('instances', 0),
        duplicate_threshold=settings['duplicate_threshold'], output_profile=output_profile,
        audio_mode=audio_mode)
    (job / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf8')
    revalidate()
    run(command, 'interpolation')
    output_video = media.analyze_video(json.loads(run(
        media.probe_command(runtime, pending, frames=True), 'output_video_probe')),
        allowed_bit_depths=output_bit_depths)
    output_audio = json.loads(run(media.probe_command(runtime, pending, packets=True), 'output_audio_probe'))
    video_audit = media.compare_interpolated_video(source_video, output_video, multiplier)
    output_pcm = media.decoded_pcm_digests(runtime, pending, output_audio, env, job / 'output_pcm.stderr')
    if audio_mode == 'copy':
        audio_audit = media.compare_audio_packets(source_audio, output_audio, video_audit['common_shift'])
        audio_audit['pcm'] = media.compare_pcm_digests(source_pcm, output_pcm)
    else:
        audio_audit = media.compare_transcoded_audio(source_audio, output_audio,
            video_audit['common_shift'], source_pcm, output_pcm)
    run([str(runtime.executable('ffmpeg.exe')), '-v', 'error', '-xerror', '-err_detect', 'explode',
        '-nostdin', '-protocol_whitelist', 'file,pipe', '-i', str(pending), '-map', '0:v:0',
        '-map', '0:a?', '-f', 'null', '-'], 'strict_decode')
    revalidate()
    target = job / ('enhanced' + suffix)
    if target.exists():
        raise RuntimeError('Refusing to replace an existing completed output')
    pending.rename(target)
    report = {'status': 'media_audit_pass_human_pending', 'operation': 'frame_interpolation',
        'output': media.file_identity(target), 'source': spec['source'], 'video': video_audit,
        'audio': audio_audit, 'model': spec['model'], 'installation': spec['installation'],
        'settings': settings,
        'format_conversion': {
            'mode': ('automatic_h264_sdr' if output_profile == 'delivery_h264'
                     else 'explicit_output_profile'),
            'source_bit_depth': source_video['bit_depth'],
            'output_bit_depth': output_video['bit_depth'],
            'output_codec': ('h264' if output_profile == 'delivery_h264' else 'hevc'),
            'output_container': suffix.lstrip('.'),
            'audio_mode': audio_mode,
        },
        'downloads': False, 'quality_verdict': 'pending',
        'video_encoding': ('high_quality_hevc_main10_nvenc'
                           if output_profile == 'delivery_hevc_main10' else 'high_quality_h264_nvenc')}
    (job / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({'status': report['status'], 'output': str(target)}))


if __name__ == '__main__':
    main()
