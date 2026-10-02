"""Explicit migration of evidenced local pilot artifacts, never filename-only reuse.

This binds a previously qualified computation to the new controller contract.
It is not fresh GPU validation of the packaged wrappers or a signed attestation.
"""
from copy import deepcopy
import json
from pathlib import Path
import struct

from .prepared_generation_contract import BACKEND, engine_sources, fingerprint, validate_bundle
from .prepared_backend.backend_files import sha


def read_json(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or not 0 < path.stat().st_size <= 32 * 1024**2:
        raise ValueError('Evidence JSON is missing or unbounded')
    value = json.loads(path.read_text(encoding='utf8'))
    if not isinstance(value, dict):
        raise ValueError('Evidence JSON must be an object')
    return value


def header(path):
    with Path(path).open('rb') as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError('Truncated latent header')
        size = struct.unpack('<Q', size_bytes)[0]
        if not 0 < size <= 16 * 1024**2:
            raise ValueError('Unbounded latent header')
        return json.loads(stream.read(size))


def same_path(a, b):
    return Path(a).resolve(strict=True) == Path(b).resolve(strict=True)


def migration(bundle, generation_directory, decode_directory, *, seed, media_directory=None):
    validate_bundle(bundle)
    if 'checkpoint' in bundle:
        raise ValueError('Migrate from a plain bundle, never an already adopted checkpoint')
    gen_dir = Path(generation_directory).resolve(strict=True)
    dec_dir = Path(decode_directory).resolve(strict=True)
    media_dir = Path(media_directory).resolve(strict=True) if media_directory else dec_dir
    old_gen = read_json(gen_dir / 'request.json')
    old_dec = read_json(dec_dir / 'request.json')
    generated, decoded = read_json(gen_dir / 'report.json'), read_json(dec_dir / 'report.json')
    terminal = read_json(gen_dir / 'terminal.json')
    expected_terminal = ('Tao_first_request_latents_complete_decode_pending' if bundle['kind'] == 'tao5s'
        else 'LTX_joint_refinement_latents_complete_decode_pending')
    if terminal.get('status') != expected_terminal or terminal.get('cleanup', {}).get('exit_code') != 0 or terminal['cleanup'].get('owned_children_remaining') != []:
        raise ValueError('Original generation process did not complete and clean up')
    media = read_json(media_dir / 'report.json')
    gen, dec = bundle['generation'], bundle['decode']
    assets = {str(Path(a['path']).resolve(strict=True)): a for a in bundle['assets']}
    evidence = [gen_dir / 'request.json', gen_dir / 'report.json', dec_dir / 'request.json',
        dec_dir / 'report.json', media_dir / 'report.json', gen_dir / 'terminal.json']

    def bind_file(field, old_request, request):
        if not same_path(old_request[field], request[field]):
            raise ValueError(f'Original {field} is not the prepared input')
        key = str(Path(request[field]).resolve(strict=True))
        old_identity = {str(Path(p).resolve(strict=True)): digest for p, digest in old_request['identities'].items()}
        if old_identity.get(key) != assets[key]['sha256']:
            raise ValueError(f'Original {field} hash differs from the verified bundle')

    # The same math helpers used by the original GPU job must be in this backend.
    helpers = (['taomate_prepared_pipeline.py', 'taomate_local_runtime.py', 'taomate_local_transport.py',
        'taomate_local_session.py', 'taomate_weight_offload.py', 'taomate_host_cache.py',
        'taomate_leased_hook.py', 'taomate_bounded_retention.py'] if bundle['kind'] == 'tao5s' else
        ['ltx_dequantized_lora.py', 'ltx_lora_contract.py', 'ltx_gpu_fusion_gate.py',
         'ltx_text_weight_offload.py', 'taomate_weight_offload.py'])
    helper_receipts = []
    for name in helpers:
        matches = [(p, digest) for p, digest in old_gen['identities'].items() if Path(p).name == name]
        if len(matches) != 1 or sha(BACKEND / name) != matches[0][1]:
            raise ValueError(f'Packaged math helper differs from original GPU evidence: {name}')
        helper_receipts.append({'name': name, 'sha256': matches[0][1]})

    if bundle['kind'] == 'tao5s':
        if old_gen['schema'] != 't8-taomate-video-first-request-v1' or generated['status'] != 'first_Tao_request_latents_pass':
            raise ValueError('Tao original generation is not qualified')
        if decoded['status'] != 'Tao_first_request_decode_media_pass_pending_review' or media_dir != dec_dir:
            raise ValueError('Tao original decoded media is not qualified')
        for field in ('source', 'base', 'adapter', 'teacher'):
            if not same_path(old_gen[field], gen[field]):
                raise ValueError(f'Tao original {field} differs')
        if old_gen['source_revision'] != gen['source_revision']:
            raise ValueError('Tao source revision differs')
        for field in ('text_features', 'milestones', 'download_receipt', 'cpu_receipt'):
            bind_file(field, old_gen, gen)
        for field in ('audio', 'vae', 'milestones'):
            bind_file(field, old_dec, dec)
        # The original base receipt supplies all shard hashes, not a model name.
        receipt = read_json(gen['download_receipt'])
        if receipt['status'] != 'download_and_transformer_checksums_pass':
            raise ValueError('Tao original base download receipt failed')
        for item in receipt['files']:
            file = (Path(gen['base']) / item['file']).resolve(strict=True)
            if not file.is_relative_to(Path(gen['base']).resolve()) or assets[str(file)]['sha256'] != item['actual_sha256']:
                raise ValueError('Tao base shard identity differs')
        for directory in (gen['adapter'], gen['teacher']):
            for path, digest in old_gen['identities'].items():
                if Path(path).resolve().is_relative_to(Path(directory).resolve()) and assets[str(Path(path).resolve())]['sha256'] != digest:
                    raise ValueError('Tao adapter/teacher identity differs')
        latent = gen_dir / 'tao-normalized-latents.safetensors'
        movie = media_dir / 'tao-first-request-5s.mp4'
        metadata = header(latent).get('__metadata__', {})
        if metadata.get('video_noise_seed') != str(seed) or metadata.get('audio_noise_seed') != str(gen['audio_seed']):
            raise ValueError('Tao checkpoint seed differs')
        if generated['video_shape'] != [1, 24, 37, 30, 54] or not generated['clean_audio_bitexact'] or not generated['retained_state_released']:
            raise ValueError('Tao original latent/audio contract failed')
        if not media['media']['full_ffmpeg_decode_pass'] or media['media']['video_frames'] != 120:
            raise ValueError('Tao final media report failed')
    else:
        if old_gen['schema'] != 't8-ltx-real-joint-refinement-v1' or generated['status'] != 'real_three_update_LTX_joint_refinement_latents_pass':
            raise ValueError('LTX original generation is not qualified')
        if old_gen['seed'] != seed or generated['seed'] != seed or generated['sigmas'] != [.909375, .725, .421875, 0] or generated['lora_strength'] != .8:
            raise ValueError('LTX checkpoint seed/sampling settings differ')
        if generated['joint_forwards'] != 3 or generated['block_calls'] != 144:
            raise ValueError('LTX original forward audit failed')
        if gen['geometry'] != {'frames': 73, 'width': 2048, 'height': 1024, 'fps': 24} or gen['prompt'] != '4K, refined, high quality, cinematic detail, clean textures, natural motion.':
            raise ValueError('This original fixed LTX pilot has different geometry/prompt')
        for field in ('inputs', 'text_cache'):
            bind_file(field, old_gen, gen)
        for field in ('vae', 'original_video'):
            bind_file(field, old_dec, dec)
        for field in ('base', 'lora'):
            if not same_path(old_gen[field], gen[field]):
                raise ValueError('LTX model path differs')
            identities = [a for a in old_gen['model_identities'] if same_path(a['path'], gen[field])]
            if len(identities) != 1 or identities[0]['sha256'] != assets[str(Path(gen[field]).resolve())]['sha256']:
                raise ValueError('LTX model identity differs')
        if old_gen['isolated_paths'] != gen['isolated_paths']:
            raise ValueError('LTX original import environment differs')
        pins = {str(Path(source['path']).resolve()): source['revision'] for source in bundle['source_revisions']}
        for suffix, revision in [('packages/ltx-core/src', 'd151147788a9284cca791edc6ce898007e727fe6'),
                ('packages/ltx-pipelines/src', 'd151147788a9284cca791edc6ce898007e727fe6'),
                ('models/minimax_h3/Sol-H3-Spark', '144085566a866f9784f3798d4c8d1603f3adbccf')]:
            matches = [Path(path).resolve() for path in gen['isolated_paths'] if Path(path).as_posix().endswith('/' + suffix)]
            if len(matches) != 1 or pins.get(str(matches[0].parents[2])) != revision:
                raise ValueError('LTX actual imported source revision differs from original controller')
        if media['status'] != 'real_LTX_refined_media_pass_CPU_reencode_pending_review' or media_dir == dec_dir:
            raise ValueError('Use the qualified LTX CPU-recovered media, not the failed original MP4')
        if not same_path(media['source_decode_report'], dec_dir / 'report.json') or media['GPU_rerun'] is not False:
            raise ValueError('LTX media recovery provenance differs')
        if not media['media']['full_decode_pass'] or not media['media']['audio_packets_and_timestamps_exact']:
            raise ValueError('LTX recovered media audit failed')
        rgb = dec_dir / 'refined-rgb.safetensors'
        if sha(rgb) != media['rgb_sha256']:
            raise ValueError('LTX media recovery RGB source changed')
        latent = gen_dir / 'refined-latents.safetensors'
        movie = media_dir / 'h3-learned2x-ltx-refined-original-audio.mp4'
        evidence.append(rgb)
    if not same_path(old_dec['latent'], latent) or not same_path(old_dec['core'], dec['core']):
        raise ValueError('Original decode was not for this latent/Core')
    if sha(latent) != generated['output_sha256'] or sha(movie) != media['video_sha256']:
        raise ValueError('Original latent/movie content changed')
    if header(latent)['video']['shape'] != generated['video_shape']:
        raise ValueError('Actual saved latent shape differs from generation report')
    receipt = {'status': 'explicit_original_pilot_checkpoint_migration', 'kind': bundle['kind'],
        'seed': seed, 'GPU_rerun': False, 'new_wrapper_GPU_qualified': False,
        'math_helpers_identical': helper_receipts, 'human_review': 'pending',
        'evidence': [{'path': str(p), 'sha256': sha(p)} for p in dict.fromkeys(evidence)],
        'limits': 'Original evidenced pilot reuse, not arbitrary historical job migration or new wrapper GPU proof.'}
    result = deepcopy(bundle)
    result['checkpoint'] = {'fingerprint': fingerprint(bundle, seed, engine_sources()),
        'stages': {'generation': {'path': str(latent), 'sha256': generated['output_sha256'], 'report': receipt},
            'decode': {'path': str(movie), 'sha256': media['video_sha256'], 'report': receipt}}}
    return result, receipt
