"""Serial dual-model segment execution for the existing durable delivery loop."""
from __future__ import annotations

import json
import math
import time

import torch
from comfy.nested_tensor import NestedTensor

from .enhance_a_video_advanced import (
    build_eav_long_video_model, build_eav_prompt_relay_long_video_model, finalize_eav_runtime,
)
from .learned_latent_upscale_advanced import (
    build_learned_two_pass_parity_plan, learned_upscale_h3_av_latent, reconcile_two_pass_h3_latent,
    PIXELS_PER_H3_LATENT,
)
from .long_video import build_long_video_conditioning, patch_long_video_model
from .long_video_delivery import (
    _load_accepted_context_file, _resolve_inside, _sha256_file, _write_context_candidate,
)
from .long_video_dual_model_stages import bind_stage_conditioning, sample_model_stage
from .long_video_dual_stage_cache import AVStageCache
from .long_video_dual_residency import release_stage_residency
from .long_video_in_node_loop_effects_advanced import _load_effects_audit
from .prompt_relay_long_video_advanced import build_prompt_relay_long_video_conditioning
from .sampling import setup_dual_clock_sampling
from .execution_timing import WallTimings
from .preview_execution_context import preview_scope
from . import long_video_dual_picture_context as picture_context
from .long_video_dual_color import COLOR_MATCH_MODES


def lock_high_video_prefix(latent, context, *, chain_id, segment_index, context_frames,
                           mode='high_native_mask_exp'):
    """Research-only native high-pass mask; retain completed joint-audio policy."""
    from .native_masked_context_advanced import (
        require_native_h3_av_mask_support, _validated_context, _native_video_mask, _native_audio_mask,
    )
    from .core import nested_av_parts, split_noise_masks
    from .long_video import CONTEXT_FRAME_STEPS
    require_native_h3_av_mask_support()
    if segment_index <= 0 or context_frames not in CONTEXT_FRAME_STEPS:
        raise ValueError('High video prefix lock requires a continuation with5/22/39 context frames')
    video, audio = nested_av_parts(latent)
    steps = CONTEXT_FRAME_STEPS[context_frames]
    if steps >= video.shape[2]:
        raise ValueError('High video prefix must leave a generated region')
    tail = _validated_context(context, chain_id=chain_id, segment_index=segment_index,
        context_frames=context_frames, target_video=video)
    vm, am = split_noise_masks(latent, video, audio)
    vm, am = _native_video_mask(vm, video), _native_audio_mask(am, audio)
    if vm is not None and not bool((vm[:, :, :steps] == 1).all()):
        raise ValueError('High video prefix is already owned by another visual mask')
    target = video.clone()
    target[:, :, :steps] = tail.to(target)
    video_mask = torch.ones_like(video) if vm is None else vm.expand_as(video).clone()
    video_mask[:, :, :steps] = 0
    ramp_values = ()
    if mode == 'high_native_mask_ramp_exp':
        ramp_values = (.25, .5, .75)
        ramp_end = steps + len(ramp_values)
        if ramp_end >= video.shape[2]:
            raise ValueError('High video mask ramp must leave a fully generated region')
        if vm is not None and not bool((vm[:, :, steps:ramp_end] == 1).all()):
            raise ValueError('High video mask ramp is already owned by another visual mask')
        for offset, value in enumerate(ramp_values):
            video_mask[:, :, steps + offset] = value
    elif mode != 'high_native_mask_exp':
        raise ValueError('Unknown high video prefix mode')
    # Native full-shape masks satisfy AVStageCache and the two-pass reconciler.
    # Preserve an existing full audio mask object; expanding singleton channels
    # only changes its view, never its values or the audio sample object.
    audio_mask = torch.ones_like(audio) if am is None else (am if am.shape == audio.shape else am.expand_as(audio))
    result = {**latent, 'samples': NestedTensor((target, audio)),
        'noise_mask': NestedTensor((video_mask, audio_mask))}
    return result, {'mode': mode, 'context_steps': steps,
        'release_ramp_values': list(ramp_values),
        'source': 'accepted_completed_high_video_tail', 'audio_touched': False,
        'scope': 'known prefix constrained in sampler; not a human seam-quality pass'}


class DualModelSegmentRunner:
    def __init__(self, first_model, second_model, *, contract, low_width, low_height,
                 upscaler_model, coarse_steps=4, refine_steps=4,
                 first_shift_video=12., first_shift_audio=3., second_shift_video=12., second_shift_audio=3.,
                 prompt_relay_mode="disabled", query_chunk_rows=256,
                 second_audio_source="auto", second_audio_strength=0., eav_config=None, color_match=True,
                 video_context_mode='reference_only', low_context_source='independent_low_x0',
                 color_match_mode='bounded_spatial_v2', fast_h3_v2_profile=None,
                 semantic_bridge_pass1=None, semantic_bridge_pass2=None):
        from .semantic_bridge import preflight_bridge
        self.bridge_configs = (semantic_bridge_pass1, semantic_bridge_pass2)
        self.bridge_identities = tuple(preflight_bridge(config) for config in self.bridge_configs)
        self.fast_h3_v2_profile = fast_h3_v2_profile
        if fast_h3_v2_profile is not None:
            from .fast_h3_v2_advanced import _gates, capture_fast_h3_v2_owner
            if fast_h3_v2_profile not in ('trained_vsa_exp', 'dense_compat_exp'):
                raise ValueError('FastH3 V2 split loop requires trained DMD or explicit Dense EXP, not the template ladder')
            if (coarse_steps, refine_steps) != (4, 4):
                raise ValueError('FastH3 V2 split loop requires four plus four forwards')
            if (first_shift_video, first_shift_audio, second_shift_video, second_shift_audio) != (10., 3., 10., 3.):
                raise ValueError('FastH3 V2 split loop requires both independent clocks10/3')
            if prompt_relay_mode == 'apply_exp' and fast_h3_v2_profile != 'dense_compat_exp':
                raise ValueError('FastH3 V2 Relay requires explicit Dense EXP; no discarded timeline bias')
            for model in (first_model, second_model):
                _gates(model)
                if capture_fast_h3_v2_owner(model) is not None:
                    raise ValueError('Connect two bare V2 student MODEL branches to the loop, not a latent-bound single setup')
        if low_context_source not in (picture_context.LEGACY, picture_context.NAME):
            raise ValueError('Unknown low video context source')
        self.low_context_source = low_context_source
        if video_context_mode not in ('reference_only', 'high_native_mask_exp',
                                      'high_native_mask_ramp_exp'):
            raise ValueError('Unknown dual video context mode')
        self.video_context_mode = video_context_mode
        self.color_match = bool(color_match)
        if color_match_mode not in COLOR_MATCH_MODES:
            raise ValueError('Unknown dual Color Match mode')
        self.color_match_mode = color_match_mode
        self.motion_color_runtime_identity = None
        if self.color_match and color_match_mode == 'bounded_motion_color_exp':
            from .long_video_motion_color import runtime_identity
            self.motion_color_runtime_identity = runtime_identity()
        if coarse_steps not in (4, 20) or refine_steps not in (3, 4, 5):
            raise ValueError("Dual-model loop currently supports Turbo4 or Stock20 then3/4/5 refine steps")
        if low_width < 32 or low_height < 32 or low_width % 32 or low_height % 32:
            raise ValueError("Low-resolution size must be positive multiples of32")
        self.eav_config = dict(eav_config or {"mode": "disabled"})
        if self.eav_config["mode"] != "disabled" and coarse_steps != 20:
            raise ValueError("EAV requires full Stock20; disable EAV for the4+4 candidate")
        self.models = (first_model, second_model)
        self.contract = dict(contract)
        self.low_size = (low_width, low_height)
        self.upscaler_model = upscaler_model
        self.steps = (coarse_steps, refine_steps)
        self.shifts = ((first_shift_video, first_shift_audio), (second_shift_video, second_shift_audio))
        self.relay_mode = prompt_relay_mode
        self.query_chunk_rows = query_chunk_rows
        if second_audio_source not in ('auto', 'legacy_policy', 'first_pass', 'highres_template'):
            raise ValueError('Unknown second-pass audio source')
        strength = float(second_audio_strength)
        if not math.isfinite(strength) or not 0. <= strength <= 1.:
            raise ValueError('Second-pass audio strength must be finite and within0..1')
        source = second_audio_source
        migration = None
        if source == 'auto':
            source = 'legacy_policy' if coarse_steps == 4 else 'first_pass'
            strength = 0.
        elif source == 'first_pass' and strength == 0. and coarse_steps == 4:
            # Older EXP graphs froze a partial4-of8 x0 estimate as finished audio.
            # Keep those graphs loadable, but restore the native joint AV route.
            source = 'legacy_policy'
            migration = 'partial_first_pass_zero_lock_to_joint_continuation'
        self.audio = (source, strength)
        self.audio_policy = {'version': 2, 'requested_source': second_audio_source,
            'requested_strength': float(second_audio_strength), 'effective_source': source,
            'effective_strength': strength, 'migration': migration,
            'first_pass_complete_trajectory': coarse_steps == 20}
        if source == 'legacy_policy':
            self.audio_policy['context_handoff'] = 'coarse_unlocked_template_locked_v1'

    def color_match_frames(self, frames, root, chain_id, segment_index, parent_candidate_id):
        from .long_video_dual_color import correct_dual_segment_color
        return correct_dual_segment_color(frames, root, chain_id, segment_index, parent_candidate_id,
                                          self.color_match, self.color_match_mode)

    def _reconcile(self, enlarged, template, positive):
        joint = self.audio[0] == 'legacy_policy'
        prepared, positive, report_json = reconcile_two_pass_h3_latent(
            enlarged, template, positive, 'first_pass' if joint else 'auto',
            second_pass_audio_source=self.audio[0], second_pass_audio_strength=self.audio[1])
        if joint and 'noise_mask' in template:
            video, coarse_audio = prepared['samples'].unbind()
            template_audio = template['samples'].unbind()[1].to(coarse_audio)
            mask = template['noise_mask'].unbind()[1].to(coarse_audio.device)
            if (not torch.isfinite(mask).all() or torch.any(mask < 0) or torch.any(mask > 1)
                    or not torch.isfinite(template_audio).all()):
                raise ValueError('Continuation audio template/mask is invalid')
            # Keep generated-region coarse x0 even when a context prefix is
            # locked. The native mask applies partial denoise strength once.
            audio = torch.where(mask == 0, template_audio, coarse_audio)
            prepared['samples'] = NestedTensor((video, audio))
            report = json.loads(report_json)
            report['dual_audio_handoff'] = 'coarse_unlocked_template_locked_v1'
            report_json = json.dumps(report)
        return prepared, positive, report_json

    def _conditions(self, model, context, inputs, projected_plan):
        arguments = {**inputs, "context": context}
        if projected_plan is None:
            positive, latent, mux, prompt, _media, conditioning_report = build_long_video_conditioning(**arguments)
            return patch_long_video_model(model), positive, latent, mux, prompt, conditioning_report, {"status": "disabled"}
        arguments.pop("prompt")
        result = bind_stage_conditioning(model, build_prompt_relay_long_video_conditioning,
            prompt_relay_plan=projected_plan, execution_mode=self.relay_mode,
            query_chunk_rows=self.query_chunk_rows, **arguments)
        patched, positive, latent, mux, prompt, _media, relay_json = result
        report = json.loads(relay_json)
        return patch_long_video_model(patched), positive, latent, mux, prompt, json.dumps(report["long_video_report"]), report

    def _stage_sampling(self, model, latent, first):
        if self.fast_h3_v2_profile is not None:
            from .fast_h3_v2_advanced import build_fast_h3_v2_setup
            model, sampler, full, report = build_fast_h3_v2_setup(model, latent, self.fast_h3_v2_profile)
            start, end = (0, 4) if first else (4, 8)
            sampler.extra_options.update(stage_start=start, stage_end=end)
            sigmas = full[start:end+1]
            payload = json.loads(report)
            payload.update(mode='fasth3_v2_split_4plus4_exp', stage_start=start, stage_end=end,
                           recipe_nfe=8, nfe=end-start,
                           sigmas=sigmas.tolist(), first_output='denoised_x0',
                           first_pass_audio_complete=False, second_audio='joint_continuation',
                           warning='Split/learned upscale/reference conditioning are not distilled training support')
            return model, sampler, sigmas, json.dumps(payload)
        model, sampler, sigmas = setup_dual_clock_sampling(model, latent,
            20 if first and self.steps[0] == 20 else 8, *self.shifts[0 if first else 1],
            'dual_clock_euler', 'native_flow')
        if first and self.steps[0] == 20:
            schedule = json.dumps({'mode': 'full_stock20', 'sigmas': sigmas.tolist()})
        else:
            low, high, schedule = build_learned_two_pass_parity_plan(model, 8, 4, self.steps[1])
            sigmas = low if first else high
        return model, sampler, sigmas, schedule

    def _low_context(self, root, chain_id, segment, high_context, parent_candidate_id, job_sha256):
        if segment.index == 0:
            return high_context, "none"
        candidate_json = _resolve_inside(root, root / "candidates" / f"segment_{segment.index - 1:05d}"
                                         / parent_candidate_id / "candidate.json")
        audit = _load_effects_audit(candidate_json, contract_sha256=job_sha256,
                                   segment_index=segment.index - 1, candidate_id=parent_candidate_id)
        record = audit["sampling_plan"]["dual_model"]["low_context"]
        path = _resolve_inside(root, record["path"])
        if _sha256_file(path) != record["sha256"]:
            raise ValueError("Accepted first-pass context checksum changed")
        context, _ = _load_accepted_context_file(path, chain_id, segment.index - 1, segment.index)
        if context["metadata"]["sampling_summary"] != job_sha256:
            raise ValueError("First-pass continuation belongs to another dual-model job")
        if tuple(context["video_tail"].shape[-2:]) != (self.low_size[1] // PIXELS_PER_H3_LATENT, self.low_size[0] // PIXELS_PER_H3_LATENT):
            raise ValueError("First-pass continuation is not at the configured low resolution")
        return context, record["sha256"]

    def run(self, *, root, chain_id, job_sha256, segment, candidate_id, base_candidate_id,
            high_context, parent_candidate_id, parent_revision, projected_plan, inputs):
        from .semantic_bridge import preflight_bridge
        current_bridges = tuple(preflight_bridge(config) for config in self.bridge_configs)
        if current_bridges != self.bridge_identities:
            raise ValueError("Bridge configuration changed within the dual-model chain")
        started = time.perf_counter()
        timings = WallTimings()
        low_context, low_parent_sha = self._low_context(root, chain_id, segment, high_context,
                                                      parent_candidate_id, job_sha256)
        cache_root = _resolve_inside(root, root / "dual_stages" / f"segment_{segment.index:05d}" / base_candidate_id)
        cache = AVStageCache(cache_root)
        contract = {"job": job_sha256, "segment": segment.index, "parent_high": parent_candidate_id,
                    "audio_policy_version": 2, "audio_policy": self.audio_policy,
                    "parent_revision": parent_revision, "parent_low_sha256": low_parent_sha,
                    "seed": segment.seed, "projected_plan": projected_plan["plan_hash"] if projected_plan else None}
        if any(identity is not None for identity in self.bridge_identities):
            contract["semantic_bridges"] = list(self.bridge_identities)
        if self.fast_h3_v2_profile is not None:
            from .fast_h3_v2_advanced import RUNG_STEPS, SCHEMA
            contract['fasth3_v2_recipe'] = {'schema': SCHEMA, 'profile': self.fast_h3_v2_profile,
                'rungs': list(RUNG_STEPS), 'shifts': self.shifts, 'stage_cut': 4,
                'first_init': 'raw_target_gaussian_fp32', 'partial_init': 'independent_av_rebase',
                'trained_reference_support': False, 'first_pass_audio_complete': False}
        if self.video_context_mode != 'reference_only':
            contract['video_context_mode'] = self.video_context_mode
        picture_source = None
        if segment.index > 0 and self.low_context_source == picture_context.NAME:
            # Cache hits skip decode/VAE, never accepted-media identity checks.
            # Job identity binds VAE, geometry, settings and implementation too.
            picture_media, picture_source = picture_context.accepted_source(
                root, parent_candidate_id, segment.index, chain_id)
            contract['low_picture_context'] = picture_source
        low_hit = cache.load("low_x0", contract)
        if low_hit is None:
            picture_report = None
            if picture_source is not None:
                low_context, picture_report = timings.call('accepted_picture_context',
                    picture_context.prepare_context, low_context, picture_media, picture_source,
                    inputs['video_vae'], *self.low_size)
            low_inputs = {**inputs, "width": self.low_size[0], "height": self.low_size[1]}
            if self.bridge_configs[0] is not None:
                low_inputs["semantic_bridge"] = self.bridge_configs[0]
            low_model, positive, low_latent, _mux, _prompt, low_condition_report, low_relay = timings.call('first_conditioning', self._conditions,
                self.models[0], low_context, low_inputs, projected_plan)
            low_condition_release = release_stage_residency(inputs['clip'], inputs['video_vae'], inputs['audio_vae'])
            low_model, sampler, sigmas, schedule = self._stage_sampling(low_model, low_latent, True)
            eav_runtime = None
            eav_setup = {"status": "disabled"}
            if self.eav_config["mode"] != "disabled":
                composer = build_eav_prompt_relay_long_video_model if low_relay.get("status") == "applied_exp" else build_eav_long_video_model
                low_model, eav_runtime, eav_json = composer(low_model, sigmas, segment_index=segment.index,
                    context_frames=segment.plan.context_frames, **self.eav_config)
                eav_setup = json.loads(eav_json)
            with preview_scope(phase='low', segment=segment.index, global_offset=0, global_total=sum(self.steps)):
                low_x0, low_report = timings.call('first_sampling', sample_model_stage, low_model, positive, low_latent,
                    sampler=sampler, sigmas=sigmas, seed=segment.seed, segment_index=segment.index)
            eav_audit = {"status": "disabled"}
            if eav_runtime is not None:
                low_x0, eav_json = finalize_eav_runtime(low_x0, eav_runtime)
                eav_audit = json.loads(eav_json)
                composed_backend = eav_audit["config"].get("composed_attention_backend")
                if composed_backend is not None:
                    low_report["backend"] = composed_backend
                    low_report["backend_observation"] = "execution-local EAV owner's retained delegate; after successful EAV finalization"
            low_report.update(schedule=json.loads(schedule), conditioning=json.loads(low_condition_report),
                              relay=low_relay, eav_setup=eav_setup, eav_audit=eav_audit,
                              conditioning_residency_release=low_condition_release,
                              sampling_residency_release=release_stage_residency(low_model))
            if picture_report is not None:
                low_report['accepted_picture_context'] = picture_report
            low_receipt = cache.save("low_x0", contract, low_x0, low_report)
            del positive, low_latent, low_model, sampler
        else:
            low_x0, low_receipt = low_hit
            low_report = low_receipt["report"]
        high_inputs = dict(inputs)
        if self.bridge_configs[1] is not None:
            high_inputs["semantic_bridge"] = self.bridge_configs[1]
        high_model, positive, template, mux, prompt, condition_report, relay_report = timings.call('second_conditioning', self._conditions,
            self.models[1], high_context, high_inputs, projected_plan)
        high_condition_release = release_stage_residency(inputs['clip'], inputs['video_vae'], inputs['audio_vae'])
        high_contract = {**contract, "low_tensor_sha256": low_receipt["tensor_sha256"]}
        high_hit = cache.load("high_output", high_contract)
        if high_hit is None:
            prepared_hit = cache.load("high_input", high_contract)
            if prepared_hit is None:
                upscale_started = time.perf_counter()
                enlarged, width, height, upscale_json = timings.call('learned_upscale', learned_upscale_h3_av_latent, low_x0,
                    self.upscaler_model, "target_dimensions", 2., 1., inputs["width"], inputs["height"],
                    "honor_dimensions_exp", 1.05, "fp16", "offload_after")
                if (width, height) != (inputs["width"], inputs["height"]):
                    raise RuntimeError("Learned upscaler geometry differs from the high-resolution template")
                prepared, positive, reconcile_json = timings.call('reconcile', self._reconcile, enlarged, template, positive)
                prepare_report = {"upscale": json.loads(upscale_json), "reconcile": json.loads(reconcile_json),
                                  "upscale_seconds": time.perf_counter() - upscale_started}
                cache.save("high_input", high_contract, prepared, prepare_report)
                del enlarged
            else:
                prepared, prepare_receipt = prepared_hit
                prepare_report = prepare_receipt["report"]
                # Revalidate/reconcile against newly built high-res conditions.
                prepared, positive, _ = timings.call('cached_input_reconcile', self._reconcile, prepared, template, positive)
            video_context_report = {'mode': self.video_context_mode, 'applied': False}
            if segment.index > 0 and self.video_context_mode != 'reference_only':
                # Run AFTER fresh/cached reconciliation: that operation selects
                # upscaled video and can discard a cached mask. Never mark a
                # zero mask while leaving the wrong low-derived prefix values.
                prepared, video_context_report = lock_high_video_prefix(prepared, high_context,
                    chain_id=chain_id, segment_index=segment.index, context_frames=segment.plan.context_frames,
                    mode=self.video_context_mode)
                video_context_report['applied'] = True
            high_model, sampler, sigmas, schedule = self._stage_sampling(high_model, prepared, False)
            with preview_scope(phase='high', segment=segment.index, global_offset=self.steps[0], global_total=sum(self.steps)):
                output, high_report = timings.call('second_sampling', sample_model_stage, high_model, positive, prepared, sampler=sampler,
                    sigmas=sigmas, seed=segment.seed, segment_index=segment.index, output_kind="zero_sigma_output")
            if self.audio == ("first_pass", 0.):
                # Zero-mask sampling still makes a native latent normalization
                # roundtrip. Restore the exact first-pass audio for delivery,
                # separately from the audio used during high-pass conditioning.
                video, sampled_audio = output['samples'].unbind()
                first_audio = low_x0['samples'].unbind()[1]
                if sampled_audio.shape != first_audio.shape or sampled_audio.dtype != first_audio.dtype:
                    raise RuntimeError('Locked first-pass audio geometry/dtype changed')
                delta = float((sampled_audio - first_audio).abs().max())
                if not torch.isfinite(torch.tensor(delta)) or delta > 1e-6:
                    raise RuntimeError('Locked audio changed beyond native normalization roundtrip tolerance')
                output = {**output, 'samples': NestedTensor((video, first_audio.clone()))}
                high_report['audio_delivery'] = {'source': 'first_pass_exact_latent',
                    'sampled_roundtrip_max_absolute_delta': delta,
                    'scope': 'audio latent bit identity; not a claim of compressed PCM or lip-sync equality'}
            else:
                high_report['audio_delivery'] = {'source': 'completed_second_pass_output',
                    'scope': 'native second-pass AV output; perceptual speech quality requires review'}
            # No latent endpoint offsets or generated-video blending here.
            # The accepted-picture fix changes LOW guidance, not this output.
            high_report.update(schedule=json.loads(schedule), preparation=prepare_report,
                               video_context=video_context_report,
                               conditioning_residency_release=high_condition_release,
                               sampling_residency_release=release_stage_residency(high_model))
            cache.save("high_output", high_contract, output, high_report)
        else:
            output, receipt = high_hit
            high_report = receipt["report"]
        context_record = None
        if segment.plan.save_context:
            # Low-resolution video keeps its own geometry/trajectory, but audio
            # must come from the completed output, never the partial coarse x0.
            low_video, low_audio = low_x0['samples'].unbind()
            final_audio = output['samples'].unbind()[1]
            if final_audio.shape != low_audio.shape or final_audio.dtype != low_audio.dtype:
                raise RuntimeError('Completed audio cannot populate low-resolution continuation')
            continuation = {**low_x0, 'samples': NestedTensor((low_video, final_audio))}
            low_path = _resolve_inside(root, root / "candidates" / f"segment_{segment.index:05d}" / candidate_id / "low.context.safetensors")
            context_record = _write_context_candidate(continuation, low_path, chain_id, segment.index,
                self.contract["first_model"]["sha256"], job_sha256)
            context_record.update(path=low_path.relative_to(root).as_posix(),
                                  audio_source='completed_second_pass_output', audio_policy_version=2)
        return {"sampled": output, "mux_audio": mux, "conditioned_prompt": prompt,
                "conditioning_report_json": condition_report, "relay_report": relay_report,
                "eav_setup": low_report.get("eav_setup", {"status": "disabled"}),
                "eav_audit": low_report.get("eav_audit", {"status": "disabled"}),
                "sampling_report": {"mode": "independent_model_learned_upscale", "dual_model": {
                    "low_context": context_record, "first_pass": low_report, "second_pass": high_report,
                    "audio_policy": self.audio_policy,
                    "low_context_source": self.low_context_source,
                    "low_reused": low_hit is not None, "high_reused": high_hit is not None,
                    "segment_compute_seconds": time.perf_counter() - started,
                    "execution_timings": timings.report(),
                    "source_contexts": (
                        "accepted movie RGB tail resized/re-encoded for low guidance; high-output tail and completed audio unchanged"
                        if self.low_context_source == picture_context.NAME else
                        "independent low-x0/high-output video tails with completed output audio; no spatial context resize")}}}
