"""Serial native progressive job -> durable candidates -> streaming assembly.

Mechanical acceptance is an execution checkpoint, never human quality approval.
Uses existing VAE decode, trim, context writer, isolated encoder and composer.
Caller must construct a real content-bound NativeProgressiveJob first.
"""

import json
import math
from pathlib import Path
import uuid

from . import long_video_delivery as delivery
from .audio_ops import decode_av_latent, trim_av_output
from .progressive_checkpoint import digest
from .progressive_job import NativeProgressiveJob
from .long_video_in_node_loop_advanced import LOOP_LOCK_NAME


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _write_receipt(path, body):
    if path.exists():
        raise FileExistsError('Progressive delivery receipt is immutable')
    delivery._atomic_write_json(path, dict(body=body, sha256=digest(body)))


def _read_receipt(path):
    receipt = _read(path)
    if set(receipt) != {'body', 'sha256'} or digest(receipt['body']) != receipt['sha256']:
        raise ValueError('Progressive delivery receipt integrity failed')
    return receipt['body']


class ProgressiveChainDelivery:
    def __init__(self, job, *, filename_prefix='H3_Progressive', audio_seam_policy='cosine_bridge',
                 bridge_ms=5., crf=18):
        if type(job) is not NativeProgressiveJob:
            raise ValueError('Use a content-bound NativeProgressiveJob')
        if (type(filename_prefix) is not str or not filename_prefix or
                delivery._safe_token(filename_prefix, fallback_prefix='H3') != filename_prefix):
            raise ValueError('Use a nonempty sanitized filename_prefix')
        if audio_seam_policy not in {'none', 'cosine_bridge'}:
            raise ValueError('Unknown audio seam policy')
        if (isinstance(bridge_ms, bool) or not isinstance(bridge_ms, (int, float)) or
                not math.isfinite(bridge_ms) or not 0 <= bridge_ms <= 50):
            raise ValueError('bridge_ms must be finite and within0..50')
        if type(crf) is not int or not 0 <= crf <= 51:
            raise ValueError('crf must be an integer within0..51')
        self.job = job
        self.chain_id = job.plan_inputs['chain_id']
        self.root = delivery.long_video_chain_root(self.chain_id)
        self.settings = dict(filename_prefix=filename_prefix, audio_seam_policy=audio_seam_policy,
                             bridge_ms=float(bridge_ms), crf=crf)
        self.contract = dict(schema='t8.progressive.delivery.v1', job_sha256=job.sha256,
                             settings=dict(self.settings), human_qualified=False)
        self.contract_sha = digest(self.contract)

    def _inside(self, *parts):
        return delivery._resolve_inside(self.root, self.root.joinpath(*parts))

    def _check_paths(self):
        if (delivery.long_video_chain_root(self.chain_id) != self.root
                or self.job.sha256 != self.contract['job_sha256']
                or self.chain_id != self.job.plan_inputs['chain_id']
                or digest(self.contract) != self.contract_sha or self.settings != self.contract['settings']):
            raise ValueError('Progressive delivery root/settings changed')
        for name in (LOOP_LOCK_NAME, 'progressive_job.json', delivery.MANIFEST_NAME,
                     delivery.MANIFEST_BACKUP_NAME, delivery.LOCK_NAME, delivery.ADVISORY_LOCK_NAME,
                     'progressive_delivery', 'candidates', 'accepted', 'assembled', 'progressive_segments'):
            self._inside(name)
        for segment in self.job.segments:
            self._inside('progressive_segments', str(segment.index), 'completed')
            self._inside('progressive_segments', str(segment.index), 'progressive_boundary.lock')
            self._inside('candidates', f'segment_{segment.index:05d}')

    def _check(self):
        self.job._locked()
        self.job.verify()
        self._check_paths()
        if self.job.root != self.root:
            raise ValueError('Progressive job holds a different delivery root')
        delivery._interruption_check()

    def _manifest(self):
        manifest, source = delivery.load_delivery_manifest(self.chain_id, allow_new=True)
        # A stale backup must not silently discard the last committed segment.
        if source == 'backup':
            raise ValueError('Progressive delivery requires recovery of the primary manifest first')
        if (manifest['revision'] != len(manifest['segments']) or manifest.get('invalidated') or
                len(manifest['segments']) > len(self.job.segments)):
            raise ValueError('Progressive manifest is not this append-only planned chain')
        return manifest

    def _expected(self, index, parent_id):
        segment = self.job.segments[index]
        plan = segment.plan
        return dict(chain_id=self.chain_id, index=index, parent_candidate_id=parent_id,
            parent_manifest_revision=index, sampling_summary=self.job.sha256, model_id=self.job.sha256,
            seed=segment.seed, width=self.job.width, height=self.job.height, fps=24,
            timeline_start_frame=round(plan.timeline_start_seconds * 24),
            timeline_end_frame=round(plan.timeline_end_seconds * 24), frame_count=plan.final_frame_count,
            is_final_segment=plan.is_final_segment, crf=self.settings['crf'], bit_depth=8)

    def _candidate(self, index, parent_id):
        receipt = self._inside('progressive_delivery', f'segment_{index:05d}.json')
        if not receipt.exists():
            return None
        body = _read_receipt(receipt)
        expected = self._expected(index, parent_id)
        if body['delivery_sha256'] != self.contract_sha or body['expected'] != expected:
            raise ValueError('Progressive candidate belongs to another delivery contract')
        descriptor = delivery._resolve_inside(self.root, body['candidate_path'])
        if delivery._sha256_file(descriptor) != body['descriptor_sha256']:
            raise ValueError('Progressive candidate descriptor changed')
        candidate, root, _ = delivery._load_candidate(str(descriptor))
        if root != self.root or any(candidate.get(k) != v for k, v in expected.items()):
            raise ValueError('Progressive candidate differs from the planned segment')
        sampling = body['sampling']
        historical = sampling.get('historical_sampling', sampling)
        if candidate['prompt'] != historical['prepared_delivery']['conditioned_prompt']:
            raise ValueError('Progressive candidate prompt differs from actual conditioning')
        if (historical['job']['sha256'] != self.job.sha256 or historical['job']['segment_index'] != index
                or historical['counts']['actual_forwards']['high'] != self.job.plan.high_evaluations
                or historical['counts']['actual_forwards']['low'] not in (0, self.job.plan.low_evaluations)):
            raise ValueError('Progressive candidate lacks matching completed sampling evidence')
        return descriptor, candidate, body

    def _verify_accepted(self, manifest):
        parent_id = ''
        for index, entry in enumerate(manifest['segments']):
            checked = self._candidate(index, parent_id)
            if checked is None:
                raise ValueError('Accepted progressive segment lacks its candidate audit')
            _, candidate, _ = checked
            for key in self._expected(index, parent_id):
                if key in {'parent_manifest_revision', 'chain_id'}:  # original candidate only
                    continue
                if entry.get(key) != candidate[key]:
                    raise ValueError('Accepted progressive segment contract changed')
            if entry['candidate_id'] != candidate['candidate_id']:
                raise ValueError('Accepted progressive candidate selection changed')
            if (not entry.get('video_path') or
                    bool(entry.get('context_path')) != bool(candidate.get('context_path'))):
                raise ValueError('Accepted progressive asset path is missing or unexpected')
            for kind in ('video', 'context'):
                if entry.get(kind + '_sha256', '') != candidate.get(kind + '_sha256', ''):
                    raise ValueError('Accepted progressive asset differs from its candidate')
                value = entry.get(kind + '_path', '')
                if value:
                    path = delivery._resolve_inside(self.root, value)
                    if delivery._sha256_file(path) != entry[kind + '_sha256']:
                        raise ValueError('Accepted progressive asset checksum failed')
            parent_id = entry['candidate_id']

    def _generate_candidate(self, index, parent_id, callback):
        segment = self.job.segments[index]
        notify = None if callback is None else lambda *args: callback(index, *args)
        sampled, text = (self.job.sample_first(callback=notify) if index == 0 else
                         self.job.sample_continuation(index, callback=notify))
        report = json.loads(text)
        historical = report.get('historical_sampling', report)
        self.job._validate_completed(sampled, historical)
        self._check()
        frames, generated, _, _ = decode_av_latent(sampled, self.job.producers.components['video_vae'],
                                                   self.job.producers.components['audio_vae'])
        selected, audio_report = self.job.delivery_audio(index, generated)
        prepared = historical['prepared_delivery']
        # A generated result replaces no mux override. An explicit source must
        # match the exact object contents selected by the actual HIGH builder.
        if ((audio_report['origin'] == 'generated_audio' and prepared['mux_audio_identity'] is not None)
                or (audio_report['origin'] != 'generated_audio'
                    and audio_report['identity'] != prepared['mux_audio_identity'])):
            raise ValueError('Progressive delivery audio differs from actual conditioning output')
        frames, audio, trim = trim_av_output(frames, segment.plan.trim_start_seconds,
                                            segment.plan.final_duration_seconds, selected, 24.)
        if audio is None or len(frames) != segment.plan.final_frame_count:
            raise ValueError('Progressive delivery lacks the exact planned AV segment')
        self._check()
        # Every encode attempt has a new namespace. An orphan after power loss
        # stays available for diagnosis, never overwritten or auto-adopted.
        candidate_id = f'progressive_{index:05d}_{uuid.uuid4().hex}'
        prompt = prepared['conditioned_prompt']
        descriptor, _, _ = delivery.save_long_video_candidate(frames, audio, sampled,
            self.chain_id, index, segment.plan.timeline_start_seconds, segment.plan.save_context,
            parent_id, index, candidate_id, self.job.sha256, self.job.sha256, prompt, segment.seed,
            24, 8, self.settings['crf'])
        self._check()
        path = Path(descriptor)
        body = dict(delivery_sha256=self.contract_sha, expected=self._expected(index, parent_id),
            candidate_path=path.relative_to(self.root).as_posix(),
            descriptor_sha256=delivery._sha256_file(path), sampling=report,
            delivery_audio=audio_report, trim=json.loads(trim), human_qualified=False)
        _write_receipt(self._inside('progressive_delivery', f'segment_{index:05d}.json'), body)
        return self._candidate(index, parent_id)

    def run(self, *, callback=None, resume_existing=True):
        # Validate before exclusive() creates its lock or job receipt.
        self._check_paths()
        with self.job.exclusive(self.root, resume_existing=resume_existing):
            self._check()
            contract_path = self._inside('progressive_delivery', 'contract.json')
            if contract_path.exists():
                if _read_receipt(contract_path) != self.contract:
                    raise ValueError('Existing delivery settings differ; use a new chain')
            else:
                if self._manifest()['segments']:
                    raise ValueError('Cannot adopt accepted media without its delivery contract')
                _write_receipt(contract_path, self.contract)
            manifest = self._manifest()
            self._verify_accepted(manifest)
            current = []
            for index in range(len(manifest['segments']), len(self.job.segments)):
                self._check()
                parent_id = manifest['segments'][-1]['candidate_id'] if manifest['segments'] else ''
                checked = self._candidate(index, parent_id)
                reused = checked is not None
                if checked is None:
                    checked = self._generate_candidate(index, parent_id, callback)
                descriptor, candidate, body = checked
                self._check()
                stem = f"segment_{index:05d}_{candidate['candidate_id']}"
                self._inside('accepted', stem + '.mp4')
                self._inside('accepted', stem + '.context.safetensors')
                _, accepted, _, _ = delivery.accept_long_video_candidate(str(descriptor), True,
                                                                         'reject_existing', True)
                if not accepted:
                    raise RuntimeError('Progressive candidate was not mechanically accepted')
                current.append(dict(segment_index=index, reused_candidate=reused,
                    actual_forwards=dict(low=0, high=0) if reused else body['sampling']['counts']['actual_forwards']))
                manifest = self._manifest()
                self._verify_accepted(manifest)
            self._check()
            final_contract = dict(delivery_sha256=self.contract_sha, manifest_sha256=digest(manifest))
            final_receipt = self._inside('progressive_delivery', 'assembled.json')
            reused_final = final_receipt.exists()
            if reused_final:
                final = _read_receipt(final_receipt)
                if final['contract'] != final_contract:
                    raise ValueError('Progressive assembly contract changed')
                output = delivery._resolve_inside(self.root, final['output_path'])
                if delivery._sha256_file(output) != final['output_sha256']:
                    raise ValueError('Progressive assembled media checksum failed')
            else:
                # Unique path avoids overwriting an unreceipted encode attempt.
                options = {**self.settings, 'filename_prefix': self.settings['filename_prefix'] + '_' + uuid.uuid4().hex[:12]}
                output, assembly = delivery.compose_accepted_long_video(self.chain_id, **options)
                output = Path(output)
                self._check()
                if digest(self._manifest()) != final_contract['manifest_sha256']:
                    raise ValueError('Progressive manifest changed during assembly')
                final = dict(contract=final_contract, output_path=output.relative_to(self.root).as_posix(),
                    output_sha256=delivery._sha256_file(output), assembly=json.loads(assembly), human_qualified=False)
                _write_receipt(final_receipt, final)
            self._check()
            return str(output), json.dumps(dict(status='assembled_mechanical_only_quality_unverified',
                job_sha256=self.job.sha256, delivery_sha256=self.contract_sha, current_segments=current,
                reused_final=reused_final, assembly=final['assembly'], human_qualified=False))
