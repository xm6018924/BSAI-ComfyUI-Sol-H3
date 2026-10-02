"""Content-bound progressive chain job and locked continuation entry.

Actual producers/models own the job SHA. This is an internal execution API,
not yet an automatic encode/accept/assembly runner or UI node.
"""

from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import threading

import torch

from . import long_video_delivery as delivery
from .long_video_in_node_loop_advanced import LOOP_LOCK_NAME
from .long_video_orchestration import build_long_video_chain_plan
from .long_video import sanitize_chain_id
from .core import temporal_shape
from .progressive_checkpoint import canonical, digest, implementation_identity, native_model_identity, ProgressiveCheckpointSession
from .progressive_producers import NativeProgressiveProducers, verify_producers
from .progressive_continuation import capture_continuation_source
from .progressive_continuation_runtime import _input_identity, sample_progressive_continuation
from .progressive_sampling_contract import plan_progressive_first_sample
from .progressive_stage_models import validate_stage_pair
from .long_video_dual_stage_cache import AVStageCache
from . import progressive_sampling_runtime as runtime
from .progressive_media import validate_condition_options, resolve_segment_options, select_delivery_audio
from .patch_stack_policy import model_identity_matches


class NativeProgressiveJob:
    def __init__(self, model, model_hires, sampler, sigmas, *, clip, video_vae, audio_vae,
                 chain_id, total_duration_seconds, width, height, upscaler_model,
                 global_prompt='', segment_prompts_json='', base_seed=0, seed_policy='increment',
                 render_window_frames=124, context_frames=22, low_evaluations=4, low_scale=.5,
                 precision='fp16', prompt_relay_plan=None, query_chunk_rows=256,
                 segment_condition_options=None, sampling_options=None, guide_resize='legacy_bilinear',
                 shared_condition_options=None):
        self.models = model, model_hires if model_hires is not None else model
        self.sampler, self.sigmas = sampler, sigmas
        self.producers = NativeProgressiveProducers(clip=clip, video_vae=video_vae, audio_vae=audio_vae)
        self.plan_inputs = dict(chain_id=chain_id, total_duration_seconds=total_duration_seconds,
            render_window_frames=render_window_frames, context_frames=context_frames,
            global_prompt=global_prompt, segment_prompts_json=segment_prompts_json,
            base_seed=base_seed, seed_policy=seed_policy)
        if sanitize_chain_id(chain_id) != chain_id:
            raise ValueError('Use an already sanitized progressive chain_id')
        if context_frames not in (22, 39):
            raise ValueError('Progressive accepted-picture chain requires context22 or39')
        if any(type(v) is not int or v < 32 or v % 32 for v in (width, height)):
            raise ValueError('Progressive job canvases require positive multiples of32')
        self.width, self.height = width, height
        if guide_resize not in {'legacy_bilinear', 'preserve_mean'}:
            raise ValueError('Unknown progressive first-frame guide resize')
        self.guide_resize = guide_resize
        self.segments = build_long_video_chain_plan(**self.plan_inputs)
        self.relay_plan, self.query_chunk_rows = prompt_relay_plan, query_chunk_rows
        if type(query_chunk_rows) is not int or not 32 <= query_chunk_rows <= 2048:
            raise ValueError('query_chunk_rows must be32..2048')
        # Local media stays preselected; shared audio is windowed lazily using
        # the established global timeline/context selector, never all at once.
        self.shared_condition_options = {} if shared_condition_options is None else shared_condition_options
        validate_condition_options(self.shared_condition_options)
        self.condition_options = {} if segment_condition_options is None else segment_condition_options
        if (type(self.condition_options) is not dict or any(type(i) is not int or
                not 0 <= i < len(self.segments) for i in self.condition_options)):
            raise ValueError('Condition options must use actual segment indices')
        for value in self.condition_options.values():
            validate_condition_options(value)
        signature = inspect.signature(sample_progressive_continuation)
        allowed = ('reserve_vram_mib', 'eav_mode', 'eav_tau', 'eav_start_video_progress',
                   'eav_end_video_progress', 'eav_max_workspace_mib', 'eav_g_hard_limit',
                   'tst_mode', 'tst_tau', 'tst_max_workspace_mib')
        self.options = {key: signature.parameters[key].default for key in allowed}
        if sampling_options is not None:
            if type(sampling_options) is not dict or set(sampling_options) - set(allowed):
                raise ValueError('Unknown progressive job sampling option')
            self.options.update(sampling_options)
        self.options.update(upscaler_model=upscaler_model, precision=precision,
                            low_evaluations=low_evaluations, low_scale=low_scale)
        _, vt, at = temporal_shape(render_window_frames)
        # Explicit CPU planning only. No intermediate_device/GPU allocation.
        self.plan = plan_progressive_first_sample(torch.zeros(1, 24, vt, height // 16, width // 16),
            torch.zeros(1, 32, 2, at), sigmas, low_evaluations=low_evaluations, low_scale=low_scale)
        low_sampling, high_sampling = (runtime.validate_native_model(m, sampler) for m in self.models)
        validate_stage_pair(*self.models, low_sampling, high_sampling)
        self._contract_json = canonical(self._capture())
        self._sha = digest(json.loads(self._contract_json))
        self.active = False

    @property
    def identity(self):
        return json.loads(self._contract_json)

    @property
    def sha256(self):
        return self._sha

    def _capture(self):
        if self.relay_plan is not None:
            from .prompt_relay_advanced import _validate_plan
            plan = _validate_plan(self.relay_plan)
            if plan['frame_count'] < round(self.segments[-1].plan.timeline_end_seconds * 24):
                raise ValueError('Global Relay plan is shorter than the progressive chain')
        segments = build_long_video_chain_plan(**self.plan_inputs)
        if segments != self.segments:
            raise ValueError('Progressive job segment plan changed')
        return dict(schema='t8.progressive.job.v1',
            models=[native_model_identity(model, self.sampler) for model in self.models],
            producers=verify_producers(self.producers),
            sigmas=_input_identity(self.sigmas), geometry=dict(width=self.width, height=self.height),
            progressive_plan=self.plan.report(),
            segments=[segment.as_report() for segment in segments], plan_inputs=self.plan_inputs,
            inputs=_input_identity({str(i): value for i, value in self.condition_options.items()}),
            shared_inputs=_input_identity(self.shared_condition_options),
            relay=_input_identity(self.relay_plan), query_chunk_rows=self.query_chunk_rows,
            options=self.options, guide_resize=self.guide_resize,
            lifter=runtime._lifter_identity(self.options['upscaler_model'], self.options['precision']),
            implementation=implementation_identity())

    def verify(self):
        # Opaque owners have a new nonportable nonce on inspection, not a new
        # user selection. Compare actual objects/weights within this job while
        # keeping the initial nonce in the persisted identity for no cross-run
        # reuse. All remaining inputs/config/source fingerprints stay exact.
        current = self._capture()
        bound = self.identity
        models_match = model_identity_matches(bound['models'], current['models'])
        current['models'] = bound['models']
        if not models_match or canonical(current) != self._contract_json or digest(bound) != self._sha:
            raise ValueError('Progressive job execution contract changed')
        return self.sha256

    def _locked(self):
        if not self.active or self.owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError('Progressive job requires its actual exclusive chain lock')

    def resolved_condition_options(self, segment_index):
        self._locked()
        self.verify()
        if type(segment_index) is not int or not 0 <= segment_index < len(self.segments):
            raise ValueError('Select an actual planned segment')
        return resolve_segment_options(self.shared_condition_options,
            self.condition_options.get(segment_index, {}), self.segments[segment_index])

    def delivery_audio(self, segment_index, generated_audio):
        """Resolve delivery again from bound sources, including completed-cache hits."""
        selected, origin = select_delivery_audio(self.resolved_condition_options(segment_index), generated_audio)
        self.verify()
        return selected, dict(origin=origin, identity=_input_identity(selected))

    @contextmanager
    def exclusive(self, root, *, resume_existing=True):
        if type(resume_existing) is not bool:
            raise ValueError('resume_existing must be a boolean')
        if self.active:
            raise RuntimeError('Progressive job is already active')
        self.verify()
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        handle = delivery._open_advisory_lock(root / LOOP_LOCK_NAME)
        acquired = False
        try:
            if not delivery._try_advisory_lock(handle):
                raise RuntimeError('Another runner owns this long-video chain')
            acquired = True
            # Decide while holding the real OS lock, before any job/manifest
            # write. Refuse reuse, never delete data or race another runner.
            if not resume_existing and any(path.name != LOOP_LOCK_NAME for path in root.iterdir()):
                raise ValueError('resume_existing is false and chain contains data; use a new chain_id')
            self.root, self.owner, self.active = root, (os.getpid(), threading.get_ident()), True
            manifest = root / delivery.MANIFEST_NAME
            if manifest.exists():
                saved = delivery._validate_manifest(json.loads(manifest.read_text(encoding='utf-8')),
                                                     self.plan_inputs['chain_id'])
                if any(entry['sampling_summary'] != self.sha256 for entry in saved['segments']):
                    raise ValueError('Accepted chain belongs to another progressive job')
            path = root / 'progressive_job.json'
            if path.exists():
                if json.loads(path.read_text(encoding='utf-8')) != self.identity:
                    raise ValueError('Existing progressive job contract differs; use a new chain')
            else:
                self.verify()
                delivery._atomic_write_json(path, self.identity)
            yield self
        finally:
            self.active = False
            try:
                if acquired:
                    delivery._release_advisory_lock(handle)
            finally:
                handle.close()

    def source(self, segment_index):
        self._locked()
        self.verify()
        if type(segment_index) is not int or not 1 <= segment_index < len(self.segments):
            raise ValueError('Select a planned continuation segment')
        manifest = delivery._validate_manifest(json.loads((self.root / delivery.MANIFEST_NAME).read_text(
            encoding='utf-8')), self.plan_inputs['chain_id'])
        if len(manifest['segments']) != segment_index:
            raise ValueError('Continuation must follow the latest accepted segment')
        parent = manifest['segments'][-1]
        previous = self.segments[segment_index - 1].plan
        if (parent['timeline_start_frame'], parent['timeline_end_frame'], parent['frame_count']) != (
                round(previous.timeline_start_seconds * 24), round(previous.timeline_end_seconds * 24),
                previous.final_frame_count):
            raise ValueError('Accepted predecessor differs from the actual job timeline')
        return capture_continuation_source(self.root, chain_id=self.plan_inputs['chain_id'],
            segment_index=segment_index, parent_candidate_id=parent['candidate_id'],
            parent_revision=manifest['revision'], job_sha256=self.sha256,
            context_frames=self.plan_inputs['context_frames'], width=self.width, height=self.height,
            low_width=self.plan.low_width, low_height=self.plan.low_height)

    def sample_continuation(self, segment_index, *, callback=None):
        source = self.source(segment_index)
        segment = self.segments[segment_index]
        cache = AVStageCache(self.root / 'progressive_segments' / str(segment_index) / 'completed')
        contract = dict(schema='t8.progressive.completed_av.v1', job_sha256=self.sha256,
                        source=source.binding, segment=segment.as_report())
        hit = cache.load('high_output', contract)
        if hit is not None:
            output, receipt = hit
            stored = receipt['report']
            historical = stored['sampling']
            if digest(historical) != stored['report_sha256'] or historical['job']['sha256'] != self.sha256:
                raise ValueError('Progressive completed-stage report integrity failed')
            self._validate_completed(output, historical)
            self._locked()
            self.verify()
            source.revalidate()
            return output, json.dumps(dict(job=historical['job'],
                counts=dict(actual_forwards=dict(low=0, high=0)),
                checkpoint=dict(reused_completed=True, tensor_sha256=receipt['tensor_sha256'],
                    scope='completed_av_stage_not_accepted_media'),
                historical_sampling=historical,
                status='restored_completed_av_no_current_sampling_quality_unverified'))
        with ProgressiveCheckpointSession(self.root / 'progressive_segments' / str(segment_index)).exclusive() as checkpoint:
            output, text = sample_progressive_continuation(source, self.models[0], self.sampler, self.sigmas,
                **self.producers.components, prompt=None if self.relay_plan is not None else segment.prompt,
                length=segment.plan.render_frames, seed=segment.seed, model_hires=self.models[1],
                condition_options=self.resolved_condition_options(segment_index),
                prompt_relay_plan=self.relay_plan, query_chunk_rows=self.query_chunk_rows,
                accepted_end_frame=round(segment.plan.timeline_end_seconds * 24) if self.relay_plan is not None else None,
                producers=self.producers, checkpoint=checkpoint, callback=callback, **self.options)
        self._locked()
        self.verify()
        source.revalidate()
        report = json.loads(text)
        report['job'] = dict(sha256=self.sha256, segment_index=segment_index,
                            scope='actual_bound_job_and_single_continuation_not_automatic_delivery')
        report['prepared_delivery'] = report['continuation']['prepared_delivery']
        self._validate_completed(output, report)
        cache.save('high_output', contract, output, dict(sampling=report, report_sha256=digest(report)))
        return output, json.dumps(report)

    def sample_first(self, *, callback=None):
        """Execute segment0 under the job lock; cache AV, not accepted media."""
        from .progressive_first_segment import prepare_first_segment, bind_first_relay
        self._locked()
        self.verify()
        manifest_path = self.root / delivery.MANIFEST_NAME
        if manifest_path.exists():
            manifest = delivery._validate_manifest(json.loads(manifest_path.read_text(encoding='utf-8')),
                                                  self.plan_inputs['chain_id'])
            if manifest['segments']:
                raise ValueError('Segment0 is already accepted; resume the next planned segment')
        segment = self.segments[0]
        cache = AVStageCache(self.root / 'progressive_segments' / '0' / 'completed')
        contract = dict(schema='t8.progressive.completed_first_av.v1', job_sha256=self.sha256,
                        segment=segment.as_report())
        hit = cache.load('high_output', contract)
        if hit is not None:
            output, receipt = hit
            stored = receipt['report']
            historical = stored['sampling']
            if digest(historical) != stored['report_sha256'] or historical['job']['sha256'] != self.sha256:
                raise ValueError('Progressive completed-stage report integrity failed')
            self._validate_completed(output, historical)
            self._locked()
            self.verify()
            return output, json.dumps(dict(job=historical['job'],
                counts=dict(actual_forwards=dict(low=0, high=0)),
                checkpoint=dict(reused_completed=True, tensor_sha256=receipt['tensor_sha256'],
                                scope='completed_av_stage_not_accepted_media'),
                historical_sampling=historical,
                status='restored_completed_av_no_current_sampling_quality_unverified'))
        verify_producers(self.producers)
        result, projected = prepare_first_segment(**self.producers.components,
            chain_id=self.plan_inputs['chain_id'], segment=segment, width=self.width, height=self.height,
            condition_options=self.resolved_condition_options(0), prompt_relay_plan=self.relay_plan)
        verify_producers(self.producers)
        model, positive, negative, projection = bind_first_relay(
            self.models[0], result, projected, self.producers.components['clip'], self.query_chunk_rows)
        with ProgressiveCheckpointSession(self.root / 'progressive_segments' / '0').exclusive() as checkpoint:
            output, text = runtime.sample_progressive_h3(model, positive, negative, result[1],
                self.sampler, self.sigmas, model_hires=self.models[1], seed=segment.seed, cfg=1.,
                task=result[6]['resolved_task'], input_mode='initialized_av_exp',
                guide_resize=self.guide_resize,
                producers=self.producers, checkpoint=checkpoint, callback=callback, **self.options)
        self._locked()
        self.verify()
        report = json.loads(text)
        report['job'] = dict(sha256=self.sha256, segment_index=0,
                            scope='actual_bound_job_first_window_not_automatic_delivery')
        report['first_preparation'] = dict(conditioning=json.loads(result[5]), media_map=result[4],
            conditioned_prompt=result[3], mux_audio_identity=_input_identity(result[2]), relay=projection)
        report['prepared_delivery'] = {key: report['first_preparation'][key] for key in
                                      ('conditioning', 'media_map', 'conditioned_prompt', 'mux_audio_identity')}
        self._validate_completed(output, report)
        cache.save('high_output', contract, output, dict(sampling=report, report_sha256=digest(report)))
        return output, json.dumps(report)

    def _validate_completed(self, output, report):
        parts = output['samples'].unbind()
        for tensor, shape in zip(parts, (self.plan.video_shape, self.plan.audio_shape), strict=True):
            if tuple(tensor.shape) != tuple(shape) or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError('Progressive completed AV differs from native job geometry/dtype')
        counts = report['counts']['actual_forwards']
        if counts['high'] != self.plan.high_evaluations or counts['low'] not in (0, self.plan.low_evaluations):
            raise ValueError('Progressive completed stage lacks actual HIGH execution evidence')
