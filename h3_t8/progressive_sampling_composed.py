"""Configured native progressive stages; legacy empty-input route stays separate.

Connect existing checkpoint/producer, Relay/EAV and stage-TST contracts to actual
execution. No external SelfLift code or paused research modules are restored.
User-selected patches remain delegated; real execution errors are not caught.
"""

from __future__ import annotations

import json
import math
import time

import torch

from .patch_stack_policy import warn_patch_stack
from .progressive_sampling_contract import EvaluationLedger, euler_sampler_space_step, plan_progressive_first_sample


@torch.inference_mode()
def sample_progressive_configured(model, positive, negative, av_latent, sampler, sigmas, *,
                          upscaler_model, seed, cfg=1., low_evaluations=6,
                          low_scale=0.5, task="t2va", precision="fp16",
                          reserve_vram_mib=1024, callback=None, model_hires=None,
                          guide_resize='legacy_bilinear', eav_mode='disabled', eav_tau=4.,
                          eav_start_video_progress=0., eav_end_video_progress=1.,
                          eav_max_workspace_mib=32, eav_g_hard_limit=1.5,
                          input_mode='empty', continuation=None, checkpoint=None, producers=None,
                          tst_mode='disabled', tst_tau=.2, tst_max_workspace_mib=256):
    """Execute learned-only native AV stages, returning LATENT and a report.

    The report covers this sampler node, not text/VAE/output end-to-end time.
    There is no automatic algorithm fallback, pixel-anchor path, tiling or
    global attention mutation; requested EAV/Relay/TST stay stage-scoped.
    """
    import comfy.model_management as mm
    import comfy.nested_tensor
    import comfy.patcher_extension
    import comfy.sample
    from .core import nested_av_parts
    from .learned_latent_upscale_advanced import learned_upscale_geometry
    from .progressive_stage_models import validate_stage_pair
    from .progressive_attention import backend_phase_report, backend_snapshot, prepare_progressive_attention
    from .progressive_relay import detach_relay_input, prepare_relay_stage, strip_paired_conditioning
    from .progressive_eav import audit_progressive_eav_stage, prepare_progressive_eav, summarize_progressive_eav
    from .progressive_sampling_contract import plan_progressive_initialized_sample
    from .progressive_masking import normalize_av_masks, resize_video_source, sampler_with_clean_anchor
    # Resolve helpers at execution time so the public runtime remains the
    # authoritative sampler/lifter boundary (and its test doubles are honest).
    from . import progressive_sampling_runtime as runtime
    _lifter_identity, _lift_video = runtime._lifter_identity, runtime._lift_video
    _native_stage, _resource_snapshot = runtime._native_stage, runtime._resource_snapshot
    validate_native_model = runtime.validate_native_model
    prepare_stage_conditioning = runtime.prepare_stage_conditioning
    started = time.perf_counter()
    producer_identity = None
    if producers is not None:
        from .progressive_producers import verify_producers
        producer_identity = verify_producers(producers)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be unsigned 64-bit")
    if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not math.isfinite(cfg) or not 0 <= cfg <= 100:
        raise ValueError("cfg must be finite and between 0 and 100")
    if type(reserve_vram_mib) is not int or reserve_vram_mib < 512:
        raise ValueError("GPU reserve must be an integer of at least 512 MiB")
    if eav_mode not in {'disabled', 'report_only', 'apply_exp'}:
        raise ValueError('Unknown progressive EAV mode')
    eav_enabled = eav_mode != 'disabled'
    if tst_mode not in {'disabled', 'report_only', 'apply_exp'}:
        raise ValueError('Unknown progressive TST mode')
    tst_enabled = tst_mode != 'disabled'
    if tst_enabled and cfg != 1.:
        raise ValueError('Progressive TST requires native CFG1 for the exact forward/layer clock')
    if input_mode not in {'empty', 'initialized_av_exp'}:
        raise ValueError('Unknown progressive input_mode')
    initialized = input_mode == 'initialized_av_exp'
    if continuation is not None:
        from .progressive_continuation_runtime import PreparedProgressiveContinuation
        if type(continuation) is not PreparedProgressiveContinuation:
            raise ValueError('Unknown progressive continuation contract')
        if not initialized or cfg != 1. or guide_resize != 'legacy_bilinear':
            raise ValueError('Continuation requires native initialized CFG1 without guide override')
        continuation.verify(positive=positive, av_latent=av_latent)
    if eav_enabled and cfg != 1.:
        raise ValueError('Progressive EAV requires CFG1 for native batch-1 routing')
    if not isinstance(av_latent, dict) or "samples" not in av_latent:
        raise ValueError("Expected an initial AV LATENT dictionary with samples")
    if set(av_latent) - {"samples", "noise_mask"}:
        warn_patch_stack("Progressive sampling retains additional user LATENT metadata; composition unverified")
    video, audio = nested_av_parts(av_latent)
    if initialized:
        plan = plan_progressive_initialized_sample(video, audio, sigmas, low_evaluations=low_evaluations,
                                                    low_scale=low_scale, task=task)
        high_mask = normalize_av_masks(av_latent.get('noise_mask'), video, audio)
        # The crop is a shape/finite-value view only, never a LOW source image.
        # Bind this exact normalized mask for both the EAV guard and sampling.
        if continuation is None:
            low_mask = normalize_av_masks(av_latent.get('noise_mask'),
                video[..., :plan.low_height // 16, :plan.low_width // 16], audio)
        else:
            continuation.verify(plan=plan)
            # LOW has its own reference-guided template. HIGH known-prefix
            # masks must never be resized into LOW's generation region.
            low_mask = normalize_av_masks(continuation.low[1].get('noise_mask'),
                                          *continuation.low[1]['samples'].unbind())
    else:
        plan = plan_progressive_first_sample(video, audio, sigmas, low_evaluations=low_evaluations,
                                             low_scale=low_scale, task=task, noise_mask=av_latent.get("noise_mask"))
        high_mask = None
        low_mask = None
    lift_geometry = learned_upscale_geometry(plan.low_width // 16, plan.low_height // 16,
        "target_dimensions", 2., .5, plan.target_width, plan.target_height, "honor_dimensions_exp", 1.1)
    schedule = sigmas.detach().clone()
    source_model = model
    from .tst_model import detach_tst_model
    tst_inputs = [model] + ([model_hires] if model_hires is not None and model_hires is not model else [])
    model, low_tst_spec = detach_tst_model(model)
    if model_hires is None or model_hires is source_model:
        model_hires_clean, high_tst_spec = model, low_tst_spec
    else:
        model_hires_clean, high_tst_spec = detach_tst_model(model_hires)
    if low_tst_spec is not None or high_tst_spec is not None:
        if tst_enabled:
            raise ValueError('Choose TST MODEL nodes or sampler TST options, not both')
        if cfg != 1.:
            raise ValueError('Progressive TST MODEL nodes require CFG1')
        for spec in (low_tst_spec, high_tst_spec):
            if spec is not None and tuple(spec['full_sigmas']) != plan.sigmas:
                raise ValueError('TST MODEL full schedule differs from progressive SIGMAS')
        tst_enabled = True
        tst_specs = {'low': low_tst_spec, 'high': high_tst_spec}
    else:
        spec = dict(mode=tst_mode, tau=tst_tau, workspace=tst_max_workspace_mib) if tst_enabled else None
        tst_specs = {'low': spec, 'high': spec}
    # Relay checkpoint identity uses its actual unwrapped inputs; TST config is
    # separately bound below and the original MODEL owners are verified again.
    source_model = model
    source_positive, source_negative = positive, negative
    model, relay_contract = detach_relay_input(model)
    if model_hires_clean is source_model:
        high_model, high_relay_contract = model, relay_contract
    else:
        high_model, high_relay_contract = detach_relay_input(model_hires_clean)
    if relay_contract is None and high_relay_contract is not None:
        raise ValueError('Progressive Relay requires paired low MODEL/CONDITIONING inputs')
    if relay_contract is not None:
        if continuation is not None:
            raise ValueError('Continuation requires a dedicated projected Relay composer')
        if cfg != 1.:
            raise ValueError('Progressive Relay currently requires CFG1 for native batch-1 routing')
        if high_relay_contract is not None and high_relay_contract['binding'] != relay_contract['binding']:
            raise ValueError('Progressive low/high Relay bindings differ')
        positive = strip_paired_conditioning(positive, relay_contract, required=True)
        negative = strip_paired_conditioning(negative, relay_contract, required=False)
    sampling = validate_native_model(model, sampler)
    high_sampling = sampling if high_model is model else validate_native_model(high_model, sampler)
    stage_models = validate_stage_pair(model, high_model, sampling, high_sampling)
    stage_models['separate_input'] = model_hires is not None
    stage_models['shared_base_object'] = model.model is high_model.model
    tst_runtimes, tst_reports = {}, {}
    if tst_enabled:
        from .tst_runtime import TSTQueryRuntime, TST_RUNTIME_KEY
        for phase, stage_model, start, end, width, height in (
                ('low', model, 0, plan.low_evaluations, plan.low_width, plan.low_height),
                ('high', high_model, plan.low_evaluations, plan.total_evaluations, plan.target_width, plan.target_height)):
            spec = tst_specs[phase]
            if spec is None:
                tst_reports[phase] = {'mode': 'disabled', 'identity': True}
                continue
            tst_runtimes[phase] = TSTQueryRuntime(plan.sigmas, stage_start=start, stage_end=end,
                layer_count=len(stage_model.model.diffusion_model.blocks), frames=plan.video_shape[2],
                spatial_tokens=(height // 32) * (width // 32), mode=spec['mode'], tau=spec['tau'],
                max_workspace_mib=spec['workspace'])
    checkpoint_lifter = None
    if checkpoint is not None:
        from .progressive_checkpoint import ProgressiveCheckpointSession
        if type(checkpoint) is not ProgressiveCheckpointSession or cfg != 1.:
            raise ValueError('Checkpoint requires its exclusive native CFG1 session')
        relay_binding = None
        if relay_contract is not None:
            from .progressive_relay_checkpoint import ProgressiveRelayCheckpointBinding
            relay_binding = ProgressiveRelayCheckpointBinding(source_model,
                model_hires_clean,
                source_positive, source_negative, sampler)
        checkpoint_lifter = _lifter_identity(upscaler_model, precision)
        checkpoint.bind(model, high_model, sampler, plan,
            inputs={'positive': positive, 'negative': negative, 'av_latent': av_latent,
                    'prepared': continuation.identity if continuation is not None else None},
            settings=dict(seed=seed, cfg=cfg, guide_resize=guide_resize, input_mode=input_mode,
                eav_mode=eav_mode, eav_tau=eav_tau, eav_start=eav_start_video_progress,
                eav_end=eav_end_video_progress, eav_workspace=eav_max_workspace_mib,
                eav_hard_limit=eav_g_hard_limit, reserve_vram_mib=reserve_vram_mib,
                **({'tst': {phase: owner.config for phase, owner in tst_runtimes.items()}} if tst_enabled else {})),
            lifter=checkpoint_lifter, continuation=continuation, producers=producers, relay_binding=relay_binding)
    if continuation is None:
        low_positive, high_positive = prepare_stage_conditioning(positive, plan, positive=True, guide_resize=guide_resize)
        low_negative, high_negative = prepare_stage_conditioning(negative, plan, positive=False, guide_resize=guide_resize)
    else:
        low_positive, high_positive = continuation.low[0], continuation.high[0]
        low_negative, high_negative = low_positive, high_positive  # explicit CFG1
        model, high_model = continuation.phase_models(model, high_model)
    ledger = EvaluationLedger(plan)
    relay_reports = None
    if continuation is not None and continuation.relay is not None:
        from .progressive_continuation_relay import prepare_continuation_relay_stage
        branch, low_positive, low_negative, low_backend, low_relay_report = prepare_continuation_relay_stage(
            model, continuation, low=True)
        high_branch, high_positive, high_negative, high_backend, high_relay_report = prepare_continuation_relay_stage(
            high_model, continuation, low=False)
        relay_reports = {'low': low_relay_report, 'high': high_relay_report}
    elif relay_contract is not None:
        # Check the original target before spending GPU time on the low stage.
        high_branch, high_positive, high_negative, high_backend, high_relay_report = prepare_relay_stage(
            high_model, relay_contract, high_positive, high_negative, plan, low=False,
            backend_contract=high_relay_contract)
        branch, low_positive, low_negative, low_backend, low_relay_report = prepare_relay_stage(
            model, relay_contract, low_positive, low_negative, plan, low=True,
            backend_contract=relay_contract)
        relay_reports = {'low': low_relay_report, 'high': high_relay_report}
    elif eav_enabled:
        # EAV builds its own authenticated backend owner; do not stack the
        # standalone progressive attention wrapper underneath it.
        branch, high_branch = model.clone(), high_model.clone()
        low_backend = high_backend = None
    else:
        branch, low_backend = prepare_progressive_attention(model, tst_enabled='low' in tst_runtimes)
        if high_model is model and bool(tst_specs['low']) == bool(tst_specs['high']):
            high_branch, high_backend = branch, low_backend
        else:
            high_branch, high_backend = prepare_progressive_attention(high_model, tst_enabled='high' in tst_runtimes)
    eav_runtimes = {}
    eav_reports = {}
    if eav_enabled:
        low_mask_contract = high_mask_contract = None
        if initialized:
            from .progressive_eav_masks import NativeProgressiveMaskContract
            low_mask_contract = NativeProgressiveMaskContract(branch, low_mask,
                (*video.shape[:-2], plan.low_height // 16, plan.low_width // 16), audio.shape, continuation=continuation)
            high_mask_contract = NativeProgressiveMaskContract(high_branch, high_mask, video.shape, audio.shape,
                                                                continuation=continuation)
        eav_options = dict(mode=eav_mode, tau=eav_tau, start=eav_start_video_progress,
                           end=eav_end_video_progress, workspace=eav_max_workspace_mib,
                           hard_limit=eav_g_hard_limit)
        branch, eav_runtimes['low'] = prepare_progressive_eav(
            branch, schedule, plan, relay_report=relay_reports['low'] if relay_reports else None,
            mask_contract=low_mask_contract, **eav_options)
        high_branch, eav_runtimes['high'] = prepare_progressive_eav(
            high_branch, schedule, plan, relay_report=relay_reports['high'] if relay_reports else None,
            mask_contract=high_mask_contract, **eav_options)
    identity = checkpoint_lifter if checkpoint_lifter is not None else _lifter_identity(upscaler_model, precision)
    owned_branches = [branch] if high_branch is branch else [branch, high_branch]
    device = torch.device(branch.load_device)
    intermediate = mm.intermediate_device()
    snapshots = []
    active_stage = "low"
    branch_evaluations = {"low": 0, "high": 0}
    apply_calls = {"low": 0, "high": 0}
    network_shapes = {}
    boundary = {}
    timings = {}
    attention_reports = {}
    checkpoint_report = {'enabled': checkpoint is not None, 'reused_low': False, 'saved_low': False}

    def tst_snapshot(phase):
        observed = tst_runtimes[phase].snapshot()
        if not observed['completed']:
            warn_patch_stack(f'Progressive {phase} TST observer was bypassed; execution coverage unverified')
            observed['composition_verified'] = False
        return observed

    def checked_resources():
        for tst_input in tst_inputs:
            detach_tst_model(tst_input)
        active_device = device if active_stage == 'low' else torch.device(high_branch.load_device)
        snapshots.append(_resource_snapshot(active_device, reserve_vram_mib * 1024**2))

    prior_wrappers = {phase: owned.model_options.get("model_function_wrapper")
                      for phase, owned in (("low", branch), ("high", high_branch))}
    if any(owner is not None and not callable(owner) for owner in prior_wrappers.values()):
        raise TypeError("Progressive sampling: model_function_wrapper must be callable")

    def measured_forward(apply_model, arguments):
        branch_evaluations[active_stage] += len(arguments.get("cond_or_uncond", ()))
        def counted(*args, **kwargs):
            apply_calls[active_stage] += 1
            return apply_model(*args, **kwargs)
        prior = prior_wrappers[active_stage]
        if prior is not None:
            return prior(counted, arguments)
        return counted(arguments["input"], arguments["timestep"], **arguments["c"])

    def measured_network(executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
        # The native BaseModel boundary unpacks these shapes and invokes H3
        # once. Keeping observation outside the DiT leaves its sole owner slot
        # available for Relay/EAV; do not weaken their inner-wrapper checks.
        shapes = kwargs.get('latent_shapes')
        if not isinstance(shapes, (list, tuple)) or len(shapes) != 2:
            raise RuntimeError('Progressive observer requires native packed AV shapes')
        if active_stage in tst_runtimes:
            if not isinstance(t, torch.Tensor) or t.numel() != 1 or not bool(torch.isfinite(t).all()):
                raise RuntimeError('TST requires a single finite actual native video sigma')
            options = transformer_options or {}
            if TST_RUNTIME_KEY in options:
                raise RuntimeError('TST refuses an existing or foreign query owner')
            with tst_runtimes[active_stage].forward(float(t.detach().cpu().item())):
                result = executor(x, t, c_concat, c_crossattn, control,
                    {**options, TST_RUNTIME_KEY: tst_runtimes[active_stage]}, **kwargs)
        else:
            result = executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
        ledger.record(active_stage, forward=True)
        network_shapes.setdefault(active_stage, [list(shape) for shape in shapes])
        return result

    def progress(step, prediction, state, total):
        expected = ledger.expected[active_stage]
        if step != ledger.callbacks[active_stage] or total != expected:
            raise RuntimeError("Sampler callbacks no longer match the qualified stage contract")
        ledger.record(active_stage)
        checked_resources()
        if active_stage == "low" and step + 1 == plan.low_evaluations:
            predicted_video, predicted_audio = prediction.unbind()
            _, audio_state = state.unbind()
            boundary["clean_video"] = predicted_video.detach().to(device=intermediate, dtype=torch.float32, copy=True)
            boundary["audio_next"] = euler_sampler_space_step(audio_state, predicted_audio,
                                                               plan.prediction_sigma, plan.resume_sigma).to(intermediate)
        if callback is not None:
            offset = 0 if active_stage == "low" else plan.low_evaluations
            callback(step + offset, prediction, state, plan.total_evaluations)

    wrapper_kind = comfy.patcher_extension.WrappersMP.APPLY_MODEL
    wrapper_key = "t8_progressive_forward_counter_v1"
    for owned in owned_branches:
        owned.set_model_unet_function_wrapper(measured_forward)
        owned.add_wrapper_with_key(wrapper_kind, wrapper_key, measured_network)
    try:
        checked_resources()
        if continuation is not None:
            low_video = continuation.low[1]['samples'].unbind()[0].to(intermediate)
        else:
            low_video = (resize_video_source(video, plan.low_height // 16, plan.low_width // 16).to(intermediate)
                         if initialized else torch.zeros(
                             (*video.shape[:-2], plan.low_height // 16, plan.low_width // 16),
                             dtype=video.dtype, device=intermediate))
        low_template = comfy.nested_tensor.NestedTensor((low_video, audio))
        # Native CPU noise layout/seed; distinct low/high shapes are recorded.
        low_noise = comfy.sample.prepare_noise(low_template, seed)
        anchor_audio_noise = low_noise.unbind()[1] if high_mask is not None else None
        attention_before = backend_snapshot(low_backend)
        start = time.perf_counter()
        restored = checkpoint.load_low() if checkpoint is not None else None
        if restored is None:
            _native_stage(branch, sampler, schedule[:plan.low_evaluations + 1], low_template,
                          low_noise, low_positive, low_negative, cfg, seed, progress, denoise_mask=low_mask,
                          preview_phase='low', preview_offset=0, preview_total=plan.total_evaluations)
        else:
            tensors, historical, receipt = restored
            boundary.update({key: value.to(intermediate) for key, value in tensors.items()})
            # Counts describe THIS execution. Do not forge callbacks/model
            # forwards for the already completed LOW stage loaded from disk.
            ledger.expected['low'] = 0
            checkpoint_report.update(reused_low=True, receipt=receipt, historical_low_report=historical)
        timings["low_sampling_including_model_prepare"] = time.perf_counter() - start
        attention_reports['low'] = backend_phase_report(low_backend, attention_before)
        if 'low' in tst_runtimes:
            if restored is None:
                tst_reports['low'] = tst_snapshot('low')
            else:
                previous = historical.get('tst')
                if not isinstance(previous, dict) or not previous.get('completed'):
                    raise RuntimeError('Restored LOW lacks completed TST execution evidence')
                tst_reports['low'] = {**previous, 'execution_scope': 'restored_low_not_current_forwards'}
        if eav_enabled:
            if restored is None:
                eav_reports['low'] = audit_progressive_eav_stage(eav_runtimes['low'], plan, 'low', branch)
            else:
                eav_reports['low'] = {**historical['eav'], 'execution_scope': 'restored_low_not_current_forwards'}
            if restored is None and low_backend is None and 'composed_attention_backend' in eav_runtimes['low'].config:
                attention_reports['low'] = {**eav_runtimes['low'].config['composed_attention_backend'],
                                            'counter_scope': 'this_phase_completed_calls_only'}
        if checkpoint is not None and restored is None:
            if (ledger.callbacks['low'] == plan.low_evaluations
                    and ledger.forwards['low'] == plan.low_evaluations
                    and ('low' not in tst_runtimes or tst_reports['low']['completed'])):
                checkpoint_report['receipt'] = checkpoint.save_low(boundary,
                    dict(callbacks=ledger.callbacks['low'], actual_forwards=ledger.forwards['low'],
                         eav=eav_reports.get('low'), attention=attention_reports['low'],
                         relay=relay_reports['low'] if relay_reports else None,
                         **({'tst': tst_reports['low']} if tst_enabled else {})))
                checkpoint_report.update(saved_low=True, save_status='completed_low_written')
            else:
                warn_patch_stack('Progressive LOW forward evidence is incomplete; not publishing a checkpoint')
                checkpoint_report['save_status'] = 'incomplete_forward_evidence_not_cached'
        del low_template, low_noise, low_video, low_positive, low_negative, low_mask
        if ledger.callbacks["low"] != ledger.expected['low'] or set(boundary) != {"clean_video", "audio_next"}:
            raise RuntimeError("Low stage ended without a complete boundary prediction")
        checked_resources()
        start = time.perf_counter()
        latent_format = model.get_model_object("latent_format")
        clean_vae = latent_format.process_out(boundary.pop("clean_video"))
        lifted_vae, lift_report = _lift_video(clean_vae, torch.zeros_like(audio), plan, identity)
        del clean_vae
        if tuple(lifted_vae.shape) != tuple(video.shape) or not bool(torch.isfinite(lifted_vae).all()):
            raise RuntimeError("Learned lift produced an invalid video tensor")
        clean_high = latent_format.process_in(lifted_vae.float())
        del lifted_vae
        high_seed = (seed + 1) % 2**64
        high_noise = comfy.sample.prepare_noise(clean_high, high_seed).to(clean_high)
        sigma = schedule[plan.low_evaluations].to(clean_high)
        high_state = sampling.noise_scaling(sigma, high_noise, clean_high)
        high_sampler = sampler
        if high_mask is not None:
            # Clean target-space source is NOT the lifted LOW prediction or the
            # noisy HIGH restart. Native H3 handles all mask/time/audio math.
            high_sampler = sampler_with_clean_anchor(sampler,
                comfy.nested_tensor.NestedTensor((video, audio)),
                comfy.nested_tensor.NestedTensor((high_noise.to(intermediate), anchor_audio_noise)))
        del anchor_audio_noise
        del clean_high, high_noise
        # Encode states so Core's zero-noise restart reconstructs both marginals.
        restart_video = latent_format.process_out(sampling.inverse_noise_scaling(sigma, high_state))
        del high_state
        audio_next = boundary.pop("audio_next")
        restart_audio = sampling.inverse_noise_scaling(sigma.to(audio_next), audio_next) / float(sampling.audio_scale)
        del audio_next
        restart = comfy.nested_tensor.NestedTensor((restart_video.to(intermediate), restart_audio.to(intermediate)))
        del restart_video, restart_audio
        restart_noise = comfy.nested_tensor.NestedTensor([torch.zeros_like(part) for part in restart.unbind()])
        timings["transition_including_lift_and_reload_policy"] = time.perf_counter() - start
        active_stage = "high"
        checked_resources()
        attention_before = backend_snapshot(high_backend)
        start = time.perf_counter()
        result = _native_stage(high_branch, high_sampler, schedule[plan.low_evaluations:], restart,
                               restart_noise, high_positive, high_negative, cfg, seed, progress,
                               denoise_mask=high_mask, preview_phase='high',
                               preview_offset=plan.low_evaluations, preview_total=plan.total_evaluations)
        timings["high_sampling_including_model_prepare"] = time.perf_counter() - start
        attention_reports['high'] = backend_phase_report(high_backend, attention_before)
        if 'high' in tst_runtimes:
            tst_reports['high'] = tst_snapshot('high')
        if eav_enabled:
            eav_reports['high'] = audit_progressive_eav_stage(eav_runtimes['high'], plan, 'high', high_branch)
            if high_backend is None and 'composed_attention_backend' in eav_runtimes['high'].config:
                attention_reports['high'] = {**eav_runtimes['high'].config['composed_attention_backend'],
                                             'counter_scope': 'this_phase_completed_calls_only'}
        output_video, output_audio = nested_av_parts({"samples": result})
        if tuple(output_video.shape) != tuple(video.shape) or tuple(output_audio.shape) != tuple(audio.shape):
            raise RuntimeError("Final AV shapes differ from the plan")
        if not all(bool(torch.isfinite(value).all()) for value in (output_video, output_audio)):
            raise RuntimeError("Final samples contain NaN or Inf")
        counts = ledger.finish(allow_incomplete_evidence=True)
        if counts["actual_forwards"] != apply_calls:
            warn_patch_stack("Progressive user stack bypassed a network/apply_model observer; counts unverified")
            counts["forward_evidence_complete"] = False
        if relay_reports is not None:
            for phase, phase_model in (('low', branch), ('high', high_branch)):
                expected = {'forward': ledger.expected[phase],
                            'routed_attention': ledger.expected[phase] * len(phase_model.model.diffusion_model.blocks)}
                if relay_reports[phase]['completed_calls'] != expected:
                    warn_patch_stack(f"Progressive {phase} Relay block coverage is incomplete")
                    relay_reports[phase]["composition_verified"] = False
                relay_reports[phase]['status'] = ('restored_low_no_current_stage_routing'
                    if phase == 'low' and checkpoint_report['reused_low']
                    else 'executed_user_stack_unverified' if relay_reports[phase].get('composition_verified') is False
                    else 'verified_native_stage_routing_quality_unverified')
        counts.update(apply_model_calls=apply_calls, forward_boundary="completed_native_h3_apply_model_executor")
        continuation_report = continuation.verify() if continuation is not None else None
        if checkpoint is not None:
            checkpoint.verify()
        report = {**plan.report(), "status": "sampled_quality_unverified", "runtime_identity": "native_h3_euler_av",
                  "sampler_seconds": time.perf_counter() - started, "timings": timings,
                  "timing_scope": "sampler_node_not_end_to_end_no_forced_sync", "counts": counts,
                  "cfg_branch_evaluations": branch_evaluations, "upscaler": identity,
                  "stage_models": stage_models,
                  "initialization": {'mode': input_mode, 'mask_present': high_mask is not None,
                      'start_sigma': plan.sigmas[0], 'unmasked_source_erased_at_sigma1': plan.sigmas[0] == 1,
                      'known_region_policy': 'native_clean_source_anchor_not_noisy_restart' if high_mask is not None
                                             else 'no_known_region_lock',
                      'continuation_and_resume_qualified': False},
                  "prompt_relay": relay_reports,
                  "eav": {**eav_reports, 'summary': summarize_progressive_eav(eav_reports, plan)} if eav_enabled
                         else {'mode': 'disabled', 'identity': True},
                  "tst": tst_reports if tst_enabled else {'mode': 'disabled', 'identity': True},
                  "attention": {**attention_reports, 'shared_counter': False,
                                'shared_backend_instance': high_backend is low_backend and low_backend is not None},
                  "lift_geometry": lift_geometry,
                  "network_input_shapes": network_shapes,
                  "lift_report": lift_report, "resource_observations": snapshots,
                  "noise": {"low_seed": seed, "high_seed": high_seed,
                            "low_video_shape": [*video.shape[:-2], plan.low_height // 16, plan.low_width // 16],
                            "audio_shape": list(audio.shape), "high_video_shape": list(video.shape),
                            "policy": "core_prepare_noise_cpu_low_joint_high_video_seed_plus_one"},
                  "reference_policy": ('first_frame_latent_mean_preserving_low_original_high' if guide_resize == 'preserve_mean'
                                       else 'first_frame_latent_bilinear_low_original_high'),
                  "pixel_anchor": False, "highres_tiling": False,
                  "existing_vdn_two_pass_modified": False}
        if checkpoint is not None:
            checkpoint_report['reused_evaluations'] = plan.low_evaluations if restored is not None else 0
            checkpoint_report['scope'] = 'native_low_boundary_only_not_complete_chain_resume'
            report['checkpoint'] = checkpoint_report
        if producers is not None:
            if verify_producers(producers) != producer_identity:
                raise ValueError('Progressive producer binding changed during sampling')
            report['producers'] = producer_identity
        if continuation_report is not None:
            report['continuation'] = continuation_report
            report['reference_policy'] = 'accepted_rgb_low_completed_av_high_native_prefix'
        output = {key: value for key, value in av_latent.items() if key not in {"samples", "noise_mask"}}
        output["samples"] = result.to(intermediate)
        return output, json.dumps(report, ensure_ascii=False, allow_nan=False)
    finally:
        for owned in owned_branches:
            owned.model_options.pop("model_function_wrapper", None)
            owned.remove_wrappers_with_key(wrapper_kind, wrapper_key)
        boundary.clear()
