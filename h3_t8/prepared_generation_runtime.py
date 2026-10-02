"""Owned serial subprocess generation/decode with explicit identity-bound cache."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from .prepared_generation_contract import (BACKEND, engine_sources, fingerprint, generation_request,
    decode_request, job_directory, validate_bundle)
from .prepared_backend.backend_files import sha
from .prepared_backend.resource_guard import SerialProbeLease, NvmlResourceReader, ResourceGuard, GuardPolicy
from .prepared_process import execute_owned, note_original
from .prepared_identity import inventory, linked

ROUTES = {
    'tao_stream': [('taomate_stream_worker.py', 'native_Tao_stream_latents_pass', 'tao-stream-normalized-latents.safetensors', 7200),
                   ('taomate_stream_decode_worker.py', 'native_Tao_stream_decode_pass_pending_review', 'tao-stream.mp4', 900)],
    'tao5s': [('taomate_video_worker.py', 'first_Tao_request_latents_pass', 'tao-normalized-latents.safetensors', 7200),
              ('taomate_video_decode_worker.py', 'Tao_first_request_decode_media_pass_pending_review', 'tao-first-request-5s.mp4', 900)],
    'ltx_refine': [('ltx_prepared_refinement_worker.py', 'prepared_LTX_three_update_latents_pass', 'refined-latents.safetensors', 3600),
                   ('ltx_refined_decode_worker.py', 'real_H3_x2_LTX_refined_media_pass_pending_review', 'h3-learned2x-ltx-refined-original-audio.mp4', 900)],
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_assets(bundle, interrupt):
    for tree in bundle['trees']:
        interrupt()
        if inventory(tree['path']) != sorted(tree['files']):
            raise ValueError('Prepared directory inventory changed; rebuild the bundle')
    for asset in bundle['assets']:
        interrupt()
        path = Path(asset['path']).resolve(strict=True)
        stat = path.stat()
        if linked(Path(asset['path'])) or (stat.st_size, stat.st_mtime_ns) != (asset['bytes'], asset['mtime_ns']) or sha(path) != asset['sha256']:
            raise ValueError(f'Prepared asset changed; rebuild the bundle: {path.name}')
    for source in bundle['source_revisions']:
        interrupt()
        command = ['git', '-C', source['path']]
        revision = subprocess.check_output([*command, 'rev-parse', 'HEAD'], text=True, timeout=15).strip()
        # Source pin covers tracked implementation. Model/output directories and
        # unrelated untracked Comfy custom projects must not become inputs.
        dirty = subprocess.check_output([*command, 'status', '--porcelain', '--untracked-files=no'], text=True, timeout=15).strip()
        if revision != source['revision'] or dirty:
            raise ValueError('Prepared upstream source revision or working tree changed')


def artifact_record(path, report=None):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path), 'report': report or {}}


def verify_record(record, root=None):
    if (not isinstance(record, dict) or set(record) != {'path', 'sha256', 'report'}
            or not isinstance(record['report'], dict) or not isinstance(record['path'], str)
            or not Path(record['path']).is_absolute()):
        raise ValueError('Invalid cached artifact record')
    path = Path(record['path']).resolve(strict=True)
    if root is not None and not path.is_relative_to(root.resolve()):
        raise ValueError('Cached stage leaves its owned job directory')
    if sha(path) != record['sha256']:
        raise ValueError('Cached stage content changed; no silent reuse')
    return path


def validate_stages(stages):
    if not isinstance(stages, dict) or set(stages) - {'generation', 'decode'}:
        raise ValueError('Unknown cached stage')
    if 'decode' in stages and 'generation' not in stages:
        raise ValueError('Cached decode requires its generation artifact')


def worker_budget(spec, request):
    """Reserve bounded time for the decoder's two mandatory full hash scans.

    Do not weaken identity verification or extend generation/unknown workers.
    The byte estimate is diagnostic, not a guarantee of filesystem throughput.
    """
    base = spec[3]
    if spec[0] not in {route[1][0] for route in ROUTES.values()}:
        return {'base_seconds': base, 'verification_bytes': 0, 'seconds': base}
    identities = request.get('identities')
    if not isinstance(identities, dict) or not identities:
        raise ValueError('Decoder timeout accounting requires its bound identities')
    size = 0
    for name in identities:
        path = Path(name).resolve(strict=True)
        if not path.is_file():
            raise ValueError('Decoder identity must refer to a file')
        size += path.stat().st_size
    return {'base_seconds': base, 'verification_bytes': size,
            'hash_scans': 2, 'assumed_hash_bytes_per_second': 32 * 1024**2,
            'seconds': min(7200, base + math.ceil(2 * size / (32 * 1024**2)))}


def run_worker(stage, spec, request, reader, guard, interrupt):
    """Only a fixed packaged worker in an owned Job Object may run."""
    name, success, output_name, _ = spec
    budget = worker_budget(spec, request)
    stage.mkdir(parents=True, exist_ok=False)
    write_json(stage / 'request.json', request)
    result = {'status': 'incomplete', 'timeout_budget': budget}
    try:
        with (stage / 'resources.jsonl').open('x', encoding='utf8') as telemetry:
            def observe():
                sample = reader.sample()
                telemetry.write(json.dumps(sample) + '\n')
                telemetry.flush()
                if reason := guard.observe(sample):
                    raise RuntimeError('Prepared generation resource guard: ' + reason)
            result['process'] = execute_owned(BACKEND / name, stage / 'request.json', timeout=budget['seconds'],
                observe=observe, interrupt=interrupt, write_json=write_json)
        report = json.loads((stage / 'report.json').read_text(encoding='utf8'))
        if report['status'] != success:
            raise RuntimeError('Worker did not return its required completion report')
        output = stage / output_name
        record = artifact_record(output, report)
        expected_digest = report.get('output_sha256') if output.suffix == '.safetensors' else report.get('video_sha256')
        if record['sha256'] != expected_digest:
            raise RuntimeError('Worker output and completion report disagree')
        result.update(status='complete', artifact=record)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        original = sys.exc_info()[1]
        try:
            write_json(stage / 'terminal.json', result)
        except BaseException as error:
            if original is None:
                raise
            note_original(original, f'Could not write prepared terminal receipt: {error}')
    return result['artifact']


def run_prepared(bundle, *, output_directory, chain_id, noise_seed, resume_existing, lease_path, interrupt):
    validate_bundle(bundle)
    if type(resume_existing) is not bool:
        raise ValueError('resume_existing must be a boolean')
    sources = engine_sources()
    identity = fingerprint(bundle, noise_seed, sources)
    root = job_directory(output_directory, chain_id)
    root.mkdir(parents=True, exist_ok=True)
    # Per-job lock prevents two callers racing cached adoption or stage promotion.
    with SerialProbeLease(root / 'job.lock'):
        state_file = root / 'state.json'
        if state_file.exists():
            state = json.loads(state_file.read_text(encoding='utf8'))
            if (not isinstance(state, dict) or state.get('schema') != 1 or state.get('kind') != bundle['kind']
                    or set(state) != {'schema', 'fingerprint', 'kind', 'stages', 'human_review'}):
                raise ValueError('Invalid prepared chain state')
            if not resume_existing or state.get('fingerprint') != identity:
                raise ValueError('Existing chain has different settings or resume is disabled; use a new chain_id')
        else:
            state = {'schema': 1, 'fingerprint': identity, 'kind': bundle['kind'], 'stages': {}, 'human_review': 'pending'}
        verify_assets(bundle, interrupt)
        validate_stages(state['stages'])
        # Bind failed/empty attempts too, before guards or model processes start.
        write_json(state_file, state)
        cache_hits = []
        for name, record in state['stages'].items():
            verify_record(record, root)
            cache_hits.append(name)
        checkpoint = bundle.get('checkpoint')
        if not state['stages'] and resume_existing and checkpoint and checkpoint.get('fingerprint') == identity:
            # Explicit one-time migration, never a filename-only cache guess.
            validate_stages(checkpoint['stages'])
            # Validate every source before copying any checkpoint artifact.
            for record in checkpoint['stages'].values():
                verify_record(record)
            for name, record in checkpoint['stages'].items():
                if name not in ('generation', 'decode'):
                    raise ValueError('Unknown checkpoint stage')
                source = verify_record(record)
                destination = root / f'adopted-{name}{source.suffix}'
                if destination.exists():
                    if sha(destination) != record['sha256']:
                        raise ValueError('Conflicting adoption artifact')
                else:
                    shutil.copyfile(source, destination)
                if sha(destination) != record['sha256']:
                    raise ValueError('Checkpoint changed during adoption')
                state['stages'][name] = artifact_record(destination, {**record.get('report', {}),
                    'cache_origin': 'explicit_verified_checkpoint', 'source_path': str(source)})
                cache_hits.append(name)
            write_json(state_file, state)
        if 'decode' not in state['stages']:
            Path(lease_path).parent.mkdir(parents=True, exist_ok=True)
            with SerialProbeLease(lease_path), NvmlResourceReader() as reader:
                for index, stage_name in enumerate(('generation', 'decode')):
                    if stage_name in state['stages']:
                        continue
                    interrupt()
                    ram = (92 if bundle['kind'] in ('tao5s', 'tao_stream') else 64) if stage_name == 'generation' else 16
                    guard = ResourceGuard(GuardPolicy(startup_free_ram_bytes=ram * 1024**3,
                        minimum_free_gpu_bytes=2 * 1024**3, minimum_free_ram_bytes=8 * 1024**3))
                    sample = reader.sample()
                    if reason := guard.observe(sample, startup=True):
                        raise RuntimeError('Prepared stage startup guard: ' + reason)
                    request = generation_request(bundle, noise_seed, sample['gpu_uuid'], sources) if index == 0 else decode_request(
                        bundle, verify_record(state['stages']['generation'], root), sample['gpu_uuid'], sources)
                    stage = root / f'{stage_name}-{uuid.uuid4().hex}'
                    record = run_worker(stage, ROUTES[bundle['kind']][index], request, reader, guard, interrupt)
                    # Worker is gone before promotion or the next VAE/model starts.
                    if engine_sources() != sources:
                        raise RuntimeError('Prepared backend changed while running')
                    if fingerprint(bundle, noise_seed, sources) != identity:
                        raise RuntimeError('Prepared environment or settings changed while running')
                    state['stages'][stage_name] = record
                    write_json(state_file, state)
        movie = verify_record(state['stages']['decode'], root)
        report = {'status': 'prepared_video_ready_pending_review', 'fingerprint': identity,
            'kind': bundle['kind'], 'noise_seed': noise_seed, 'cache_hits': cache_hits,
            'movie': str(movie), 'sha256': sha(movie), 'generation_ran': 'generation' not in cache_hits,
            'decode_ran': 'decode' not in cache_hits, 'human_review': 'pending', 'state_path': str(state_file)}
        write_json(root / 'last_execution.json', report)
        return movie, report
