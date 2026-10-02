"""Private official Topaz worker. Launched only behind the owned Windows job gate."""
import importlib.util
import json
import os
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
    revalidate()
    env = runtime.child_environment(os.environ)
    def run(command, name):
        with (job / (name + '.stdout')).open('xb') as out, (job / (name + '.stderr')).open('xb') as err:
            result = subprocess.run(command, cwd=runtime.install, env=env, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, shell=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            raise RuntimeError(f'Official Topaz {name} failed with exit code{result.returncode}; see task stderr. No fallback/download was attempted.')
        return (job / (name + '.stdout')).read_text(encoding='utf8') if 'probe' in name else None
    output_profile = spec['settings'].get('output_profile', 'delivery_h264')
    source_bit_depths = media.qualified_input_bit_depths(output_profile)
    output_bit_depths = ({'delivery_h264': (8,), 'delivery_hevc_main10': (10,),
                          'lossless_master': (16,)}[output_profile])
    source = Path(spec['source']['path'])
    source_video = media.analyze_video(json.loads(run(
        media.probe_command(runtime, source, frames=True), 'source_video_probe')),
        allowed_bit_depths=source_bit_depths)
    source_audio = json.loads(run(media.probe_command(runtime, source, packets=True), 'source_audio_probe'))
    if output_profile in contract.DELIVERY_OUTPUT_PROFILES:
        suffix = media.delivery_suffix(source_audio)
        audio_mode = media.delivery_audio_mode(source_audio)
    else:
        suffix, audio_mode = media.lossless_master_suffix(source_audio), 'copy'
    encoder = ({'delivery_h264': 'h264_nvenc', 'delivery_hevc_main10': 'hevc_nvenc'}[output_profile]
        if output_profile in contract.DELIVERY_OUTPUT_PROFILES else ('png' if suffix == '.mov' else 'ffv1'))
    # Capability check runs before any enhancement or model load.
    run([str(runtime.executable('ffmpeg.exe')), '-hide_banner', '-h', 'encoder=' + encoder], 'encoder_probe')
    encoder_help = (job / 'encoder_probe.stdout').read_text(encoding='utf8')
    if ('Encoder ' + encoder + ' ') not in encoder_help:
        raise RuntimeError('Official FFmpeg lacks the required Topaz output encoder')
    source_pcm = media.decoded_pcm_digests(runtime, source, source_audio, env, job / 'source_pcm.stderr')
    settings = dict(spec['settings'])
    scale = settings.pop('scale')
    size_mode = settings.pop('size_mode', 'scale')
    output_profile = settings.pop('output_profile', 'delivery_h264')
    width, height, geometry = contract.resolve_output_geometry(
        source_video['width'], source_video['height'], settings['width'], settings['height'], scale, size_mode)
    settings.update(width=width, height=height)
    if output_profile in contract.DELIVERY_OUTPUT_PROFILES:
        estimated_payload = max(256 * 1024**2, width * height * source_video['frames'] // 8)
        safety_margin = 512 * 1024**2
        estimate_name = 'conservative_delivery_working_estimate_bytes'
        runtime_stop_floor = 256 * 1024**2
    else:
        estimated_payload = width * height * source_video['frames'] * 6
        safety_margin = 2 * 1024**3
        estimate_name = 'raw_rgb48_upper_bound_bytes'
        runtime_stop_floor = 1024**3
    required = estimated_payload + safety_margin
    available = shutil.disk_usage(job).free
    disk_preflight = {'output_directory': str(job), 'required_bytes': required,
        'available_bytes': available, estimate_name: estimated_payload,
        'safety_margin_bytes': safety_margin, 'width': width, 'height': height,
        'frames': source_video['frames'],
        'status': ('profile_estimate_available' if available >= required
                   else 'advisory_below_profile_estimate'),
        'blocking': False, 'runtime_stop_floor_bytes': runtime_stop_floor,
        'output_profile': output_profile}
    (job / 'disk_preflight.json').write_text(json.dumps(disk_preflight, indent=2), encoding='utf8')
    pending = job / ('enhanced.pending' + suffix)
    command = contract.regular_command(runtime, source, pending, size_mode=size_mode,
        output_profile=output_profile, audio_mode=audio_mode, **settings)
    (job / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf8')
    revalidate()
    run(command, 'enhancement')
    if size_mode == 'target_dimensions':
        geometry['native_ai_output'] = contract.native_dimension_evidence(
            (job / 'enhancement.stderr').read_text(encoding='utf8'), source_video['frames'])
        geometry['post_ai_resampling'] = 'lanczos_exact_target_dimensions'
    else:
        geometry['post_ai_resampling'] = 'none'
    output_video = media.analyze_video(json.loads(run(
        media.probe_command(runtime, pending, frames=True), 'output_video_probe')),
        allowed_bit_depths=output_bit_depths)
    output_audio = json.loads(run(media.probe_command(runtime, pending, packets=True), 'output_audio_probe'))
    video_audit = media.compare_video(source_video, output_video, width, height)
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
    report = {'status': 'media_audit_pass_human_pending', 'output': media.file_identity(target),
        'source': spec['source'], 'video': video_audit, 'audio': audio_audit,
        'model': spec['model'], 'installation': spec['installation'],
        'parameter_audit': spec.get('parameter_audit', {}),
        'settings': {**settings, 'scale': scale, 'size_mode': size_mode,
                     'output_profile': output_profile}, 'geometry': geometry,
        'format_conversion': {
            'mode': ('automatic_h264_sdr' if output_profile == 'delivery_h264'
                     else 'explicit_output_profile'),
            'source_bit_depth': source_video['bit_depth'],
            'output_bit_depth': output_video['bit_depth'],
            'output_codec': ('h264' if output_profile == 'delivery_h264'
                             else ('hevc' if output_profile == 'delivery_hevc_main10'
                                   else 'lossless')),
            'output_container': suffix.lstrip('.'),
            'audio_mode': audio_mode,
        },
        'downloads': False, 'interpolation': False, 'quality_verdict': 'pending',
        'video_encoding': ({'delivery_h264': 'high_quality_h264_nvenc',
                            'delivery_hevc_main10': 'high_quality_hevc_main10_nvenc'}
                           .get(output_profile, 'lossless_audit_master')),
        'model_loading_evidence': 'inspect enhancement stderr; candidate hashes alone are not engine-load proof'}
    (job / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({'status': report['status'], 'output': str(target)}))


if __name__ == '__main__':
    main()
