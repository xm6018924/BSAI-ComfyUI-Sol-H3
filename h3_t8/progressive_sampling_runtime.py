"""T8 learned progressive first sampling on native H3 FLOW_AV/Euler.

Uses Comfy's sampler and the existing T8 learned resizer. Not SelfLift-zero/rich,
not a completed-first-pass restart, and no external SelfLift code is imported.
All execution wrappers belong to a disposable MODEL clone, never global state.
"""

from __future__ import annotations

from .patch_stack_policy import warn_patch_stack

import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .progressive_sampling_contract import EvaluationLedger, euler_sampler_space_step, plan_progressive_first_sample


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lifter_identity(model_name, precision):
    import folder_paths
    from safetensors import safe_open
    from .learned_latent_upscale_advanced import PRECISIONS, EXPECTED_STATE_CONTRACT
    if not isinstance(model_name, str) or not model_name.strip() or model_name == "none":
        raise ValueError("Select an installed H3 learned upscaler; no nearest fallback")
    if precision not in PRECISIONS:
        raise ValueError("Unsupported learned-upscaler precision")
    path = Path(folder_paths.get_full_path_or_raise("latent_upscale_models", model_name)).resolve()
    # Check architecture before any low-stage sampling, without allocating weights.
    # The folder also contains non-H3 models, which must fail before spending GPU time.
    try:
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            if set(checkpoint.keys()) != set(EXPECTED_STATE_CONTRACT):
                raise ValueError("expected the 322-tensor H3 3D learned-resizer architecture")
            for key, shape in EXPECTED_STATE_CONTRACT.items():
                tensor = checkpoint.get_slice(key)
                if tuple(tensor.get_shape()) != shape or tensor.get_dtype() not in {"F16", "BF16", "F32"}:
                    raise ValueError(f"incompatible learned-resizer tensor: {key}")
    except Exception as error:
        raise ValueError(f"Not a compatible H3 learned upscaler: {path.name}: {error}") from error
    return {"name": model_name, "path": str(path), "sha256": _sha256_file(path), "precision": precision,
            "header_tensor_count": len(EXPECTED_STATE_CONTRACT)}


def _empty_option(value):
    """Do not coerce unknown tensors/callables to bool while checking patches."""
    return value is None or value is False or (isinstance(value, (dict, list, tuple)) and not value)


def validate_native_model(model, sampler):
    import comfy.k_diffusion.sampling
    import comfy.latent_formats
    import comfy.model_base
    import comfy.model_sampling
    import comfy.samplers
    from .h3_core_compat import plain_attention_backend
    av_type = getattr(comfy.model_sampling, "ModelSamplingAV", None)
    if av_type is None or type(getattr(model, "model", None)) is not comfy.model_base.MiniMaxH3:
        raise ValueError("Progressive first sampling requires current Core native H3 FLOW_AV; old routes remain unchanged")
    sampling = model.get_model_object("model_sampling")
    if not isinstance(sampling, (av_type,)) or not isinstance(sampling, comfy.model_sampling.CONST):
        raise ValueError("Configure native H3 Euler/AV sampling, not dual_clock_euler")
    if getattr(sampling, "multiplier", None) != 1000:
        raise ValueError("Native H3 timestep multiplier must be 1000")
    if getattr(sampling.noise_scaling, "__func__", None) is not comfy.model_sampling.CONST.noise_scaling:
        raise ValueError("Custom noise scaling is not qualified")
    if getattr(sampling.inverse_noise_scaling, "__func__", None) is not comfy.model_sampling.CONST.inverse_noise_scaling:
        raise ValueError("Custom restart scaling is not qualified")
    if type(model.get_model_object("latent_format")) is not comfy.latent_formats.MiniMaxH3AV:
        raise ValueError("Native H3 AV latent format is required")
    if not math.isfinite(float(sampling.audio_scale)) or float(sampling.audio_scale) <= 0:
        raise ValueError("Invalid native audio scale")
    noise_scale = getattr(sampling, "noise_scale", 1.0)
    if isinstance(noise_scale, bool) or not isinstance(noise_scale, (int, float)) or not math.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("Native noise scale must be finite and positive")
    if set(getattr(model, "object_patches", {})) - {"model_sampling"}:
        warn_patch_stack('Unqualified MODEL object patches; only native model_sampling is supported')
    if not isinstance(sampler, comfy.samplers.KSAMPLER) or sampler.sampler_function is not comfy.k_diffusion.sampling.sample_euler:
        raise ValueError("Only native Euler is qualified for this first-sampling route")
    if sampler.inpaint_options or any(k != "s_churn" or v != 0 for k, v in sampler.extra_options.items()):
        raise ValueError("Custom Euler options are not qualified")
    for name in ("wrappers", "callbacks", "injections", "hook_patches", "additional_models"):
        candidate = model
        if name == 'wrappers':
            from .taeh3_sampling_preview import cache_projection
            candidate = cache_projection(model)
        if any(bool(value) for value in getattr(candidate, name, {}).values()):
            warn_patch_stack(f'Progressive first sampling does not yet compose MODEL {name}')
    for name in ("t8_minimax_h3_openvdn_contract_v2", "t8_fast_h3_vsa_gate_contract_v1"):
        if getattr(model, "get_attachment", lambda key: None)(name) is not None:
            warn_patch_stack('Keep VDN/FastH3 on their existing independent workflows')
    options = model.model_options
    if any(not _empty_option(value) for key, value in options.items() if key != "transformer_options"):
        warn_patch_stack('A sampler/model wrapper already owns this MODEL branch')
    transformer = options.get("transformer_options", {})
    shifts = {"minimax_h3_sigma_shift_video": ("sigma_shift_video", sampling.shift),
              "minimax_h3_sigma_shift_audio": ("sigma_shift_audio", sampling.audio_shift)}
    for key, (attribute, expected) in shifts.items():
        value = transformer.get(key, getattr(model.model.diffusion_model, attribute, None))
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value <= 0 or value != expected):
            raise ValueError(f"Native AV shift mismatch: {key}")
    for key, value in transformer.items():
        if key in shifts or _empty_option(value):
            continue
        if key == "optimized_attention_override" and plain_attention_backend(value) is not None:
            continue
        warn_patch_stack(f'Unqualified transformer option: {key}; use a separate native MODEL branch')
    return sampling


def prepare_stage_conditioning(conditioning, plan, *, positive, guide_resize='legacy_bilinear'):
    if guide_resize not in {'legacy_bilinear', 'preserve_mean'}:
        raise ValueError('Unknown guide_resize policy')
    if not isinstance(conditioning, list) or len(conditioning) != 1:
        raise ValueError("Initial progressive scope requires one conditioning entry")
    item = conditioning[0]
    if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[1], dict):
        raise ValueError("Invalid CONDITIONING")
    embedding, metadata = item
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 3 or embedding.shape[0] != 1:
        raise ValueError("Expected batch-1 text conditioning")
    allowed = {"pooled_output", "guidance", "minimax_keyframes", "minimax_frame_count", "minimax_token_tags"}
    if set(metadata) - allowed:
        warn_patch_stack(f"Reference/area/hook conditioning retained without qualification: {sorted(set(metadata) - allowed)}")
    # Native Qwen H3 uses one tag per embedding token: text=1, vision=0.
    # These select AdaLN modality rows, not a spatial mask. Both stages use the
    # same text/vision embeddings, so preserve the tags exactly (including int64).
    tags = metadata.get("minimax_token_tags")
    if tags is not None:
        if (not isinstance(tags, torch.Tensor) or tags.dtype != torch.long or
                tuple(tags.shape) not in {(embedding.shape[1],), (1, embedding.shape[1])} or
                not bool(((tags == 0) | (tags == 1)).all())):
            raise ValueError("minimax_token_tags must be batch-1 int64 text/vision tags matching the embedding length")
    high_meta = metadata.copy()
    low_meta = metadata.copy()
    keyframes = metadata.get("minimax_keyframes", [])
    if not isinstance(keyframes, list):
        raise ValueError("minimax_keyframes must be a list")
    if plan.task == "t2va" and keyframes:
        raise ValueError("T2VA cannot contain keyframe conditioning")
    if plan.task == "i2va" and positive and not keyframes:
        raise ValueError("I2VA requires a first-frame conditioning reference")
    if keyframes:
        if len(keyframes) != 1 or not isinstance(keyframes[0], dict) or set(keyframes[0]) != {"resolved_frame_index", "latent"}:
            raise ValueError("Initial I2VA supports exactly one native first-frame reference")
        frame = keyframes[0]
        if type(frame["resolved_frame_index"]) is not int or frame["resolved_frame_index"] != 0:
            raise ValueError("Only first-frame I2VA is qualified")
        reference = frame["latent"]
        expected = (1, 24, 1, plan.target_height // 16, plan.target_width // 16)
        if not isinstance(reference, torch.Tensor) or tuple(reference.shape) != expected:
            raise ValueError(f"Reference latent must match target canvas: {expected}")
        if not reference.is_floating_point() or not bool(torch.isfinite(reference).all()):
            raise ValueError("Reference latent must be finite floating point")
        original_frame = reference[:, :, 0].float()
        resized = F.interpolate(original_frame, size=(plan.low_height // 16, plan.low_width // 16),
                                mode="bilinear", align_corners=False)
        if guide_resize == 'preserve_mean':
            resized = resized - resized.mean((-2, -1), keepdim=True) + original_frame.mean((-2, -1), keepdim=True)
        resized = resized.to(reference.dtype).unsqueeze(2)
        high_meta["minimax_keyframes"] = [frame.copy()]
        low_meta["minimax_keyframes"] = [{**frame, "latent": resized}]
        expected_frames = 5 + (plan.video_shape[2] - 2) // 5 * 17
        if metadata.get("minimax_frame_count") != expected_frames:
            raise ValueError("Keyframe frame-count metadata does not match the latent")
    return [[embedding, low_meta]], [[embedding, high_meta]]


def _lift_video(video_vae_space, audio_placeholder, plan, identity):
    import comfy.nested_tensor
    from .learned_latent_upscale_advanced import learned_upscale_h3_av_latent
    latent = {"samples": comfy.nested_tensor.NestedTensor((video_vae_space, audio_placeholder))}
    output, width, height, receipt = learned_upscale_h3_av_latent(
        latent, identity["name"], "target_dimensions", 2., 0.5,
        plan.target_width, plan.target_height, "honor_dimensions_exp", 1.1,
        identity["precision"], "offload_after")
    report = json.loads(receipt)
    if (width, height) != (plan.target_width, plan.target_height) or report.get("status") != "ok":
        raise RuntimeError("Learned lift did not execute the planned target canvas")
    if report.get("model", {}).get("sha256") != identity["sha256"]:
        raise RuntimeError("Learned weights changed or cached identity is stale")
    if _sha256_file(identity["path"]) != identity["sha256"]:
        raise RuntimeError("Learned checkpoint changed during sampling")
    return output["samples"].unbind()[0], report


def _resource_snapshot(device, reserve_bytes):
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()
    result = {"kind": "boundary_snapshot_not_peak", "device": str(device)}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        result.update(free_bytes=int(free), total_bytes=int(total),
                      torch_allocated_bytes=torch.cuda.memory_allocated(device))
        if free < reserve_bytes:
            raise RuntimeError("Insufficient free GPU memory at progressive stage boundary; no automatic retry")
    return result


def _native_stage(model, sampler, sigmas, latent, noise, positive, negative, cfg, seed, callback,
                  denoise_mask=None, *, preview_phase=None, preview_offset=None, preview_total=None):
    import comfy.samplers
    from .preview_execution_context import preview_scope
    error = None
    try:
        labels = ({} if preview_phase is None else dict(phase=preview_phase,
                  global_offset=preview_offset, global_total=preview_total))
        with preview_scope(**labels):
            return comfy.samplers.sample(model, noise, positive, negative, cfg, model.load_device,
                                      sampler, sigmas, model.model_options, latent_image=latent,
                                      callback=callback, disable_pbar=True, seed=seed, denoise_mask=denoise_mask)
    except BaseException as caught:
        error = caught
        raise
    finally:
        # The disposable patcher shares modules with its caller. Restore only
        # objects this stage actually installed, including on cancellation.
        try:
            _restore_stage_objects(model)
        except BaseException as cleanup_error:
            if error is None:
                raise
            message = f'Progressive stage object cleanup failed: {cleanup_error}'
            if hasattr(error, 'add_note'):
                error.add_note(message)
            else:
                import logging
                logging.warning(message)


def _restore_stage_objects(model):
    import comfy.utils
    backups = model.object_patches_backup
    if not backups:
        return
    if any(path not in model.object_patches
           or comfy.utils.get_attr(model.model, path) is not model.object_patches[path]
           for path in backups):
        raise RuntimeError('Progressive stage cannot restore objects owned by another MODEL')
    model.unpatch_model(unpatch_weights=False)


@torch.inference_mode()
def sample_progressive_h3(model, positive, negative, av_latent, sampler, sigmas, *,
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
    The default route has no automatic fallback, pixel-anchor path or tiling;
    explicitly requested compositions are delegated to the scoped executor.
    """
    from .tst_model import TST_MODEL_KEY
    from .prompt_relay_advanced import PROMPT_RELAY_WRAPPER_KEY
    get_attachment = getattr(model, 'get_attachment', lambda key: None)
    high_attachment = getattr(model_hires, 'get_attachment', lambda key: None)
    configured = (model_hires is not None or guide_resize != 'legacy_bilinear'
        or eav_mode != 'disabled' or input_mode != 'empty' or continuation is not None
        or checkpoint is not None or producers is not None or tst_mode != 'disabled'
        or any(get_attachment(key) is not None or high_attachment(key) is not None
               for key in (TST_MODEL_KEY, PROMPT_RELAY_WRAPPER_KEY)))
    if configured:
        from .progressive_sampling_composed import sample_progressive_configured
        return sample_progressive_configured(model, positive, negative, av_latent, sampler, sigmas,
            upscaler_model=upscaler_model, seed=seed, cfg=cfg, low_evaluations=low_evaluations,
            low_scale=low_scale, task=task, precision=precision, reserve_vram_mib=reserve_vram_mib,
            callback=callback, model_hires=model_hires, guide_resize=guide_resize,
            eav_mode=eav_mode, eav_tau=eav_tau, eav_start_video_progress=eav_start_video_progress,
            eav_end_video_progress=eav_end_video_progress, eav_max_workspace_mib=eav_max_workspace_mib,
            eav_g_hard_limit=eav_g_hard_limit, input_mode=input_mode, continuation=continuation,
            checkpoint=checkpoint, producers=producers, tst_mode=tst_mode, tst_tau=tst_tau,
            tst_max_workspace_mib=tst_max_workspace_mib)
    import comfy.model_management as mm
    import comfy.nested_tensor
    import comfy.patcher_extension
    import comfy.sample
    from .core import nested_av_parts
    from .learned_latent_upscale_advanced import learned_upscale_geometry
    started = time.perf_counter()
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be unsigned 64-bit")
    if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not math.isfinite(cfg) or not 0 <= cfg <= 100:
        raise ValueError("cfg must be finite and between 0 and 100")
    if type(reserve_vram_mib) is not int or reserve_vram_mib < 512:
        raise ValueError("GPU reserve must be an integer of at least 512 MiB")
    if not isinstance(av_latent, dict) or "samples" not in av_latent:
        raise ValueError("Expected an initial AV LATENT dictionary with samples")
    if set(av_latent) - {"samples", "noise_mask"}:
        warn_patch_stack("Progressive sampling retains additional user LATENT metadata; composition unverified")
    video, audio = nested_av_parts(av_latent)
    plan = plan_progressive_first_sample(video, audio, sigmas, low_evaluations=low_evaluations,
                                         low_scale=low_scale, task=task, noise_mask=av_latent.get("noise_mask"))
    lift_geometry = learned_upscale_geometry(plan.low_width // 16, plan.low_height // 16,
        "target_dimensions", 2., .5, plan.target_width, plan.target_height, "honor_dimensions_exp", 1.1)
    schedule = sigmas.detach().clone()
    sampling = validate_native_model(model, sampler)
    low_positive, high_positive = prepare_stage_conditioning(positive, plan, positive=True)
    low_negative, high_negative = prepare_stage_conditioning(negative, plan, positive=False)
    identity = _lifter_identity(upscaler_model, precision)
    ledger = EvaluationLedger(plan)
    branch = model.clone()
    device = torch.device(branch.load_device)
    intermediate = mm.intermediate_device()
    snapshots = []
    active_stage = "low"
    branch_evaluations = {"low": 0, "high": 0}
    apply_calls = {"low": 0, "high": 0}
    network_shapes = {}
    boundary = {}
    timings = {}
    prior_wrapper = branch.model_options.get("model_function_wrapper")
    if prior_wrapper is not None and not callable(prior_wrapper):
        raise TypeError("Progressive sampling: model_function_wrapper must be callable")

    def checked_resources():
        snapshots.append(_resource_snapshot(device, reserve_vram_mib * 1024**2))

    def measured_forward(apply_model, arguments):
        branch_evaluations[active_stage] += len(arguments.get("cond_or_uncond", ()))
        def counted(*args, **kwargs):
            apply_calls[active_stage] += 1
            return apply_model(*args, **kwargs)
        if prior_wrapper is not None:
            return prior_wrapper(counted, arguments)
        return counted(arguments["input"], arguments["timestep"], **arguments["c"])

    def measured_network(executor, x, *args, **kwargs):
        ledger.record(active_stage, forward=True)
        network_shapes.setdefault(active_stage, [list(part.shape) for part in x])
        return executor(x, *args, **kwargs)

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

    branch.set_model_unet_function_wrapper(measured_forward)
    wrapper_kind = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    wrapper_key = "t8_progressive_forward_counter_v1"
    branch.add_wrapper_with_key(wrapper_kind, wrapper_key, measured_network)
    try:
        checked_resources()
        low_video = torch.zeros((*video.shape[:-2], plan.low_height // 16, plan.low_width // 16),
                                 dtype=video.dtype, device=intermediate)
        low_template = comfy.nested_tensor.NestedTensor((low_video, audio))
        # Native CPU noise layout/seed; distinct low/high shapes are recorded.
        low_noise = comfy.sample.prepare_noise(low_template, seed)
        start = time.perf_counter()
        _native_stage(branch, sampler, schedule[:plan.low_evaluations + 1], low_template,
                      low_noise, low_positive, low_negative, cfg, seed, progress,
                      preview_phase='low', preview_offset=0, preview_total=plan.total_evaluations)
        timings["low_sampling_including_model_prepare"] = time.perf_counter() - start
        del low_template, low_noise, low_video, low_positive, low_negative
        if ledger.callbacks["low"] != plan.low_evaluations or set(boundary) != {"clean_video", "audio_next"}:
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
        start = time.perf_counter()
        result = _native_stage(branch, sampler, schedule[plan.low_evaluations:], restart,
                               restart_noise, high_positive, high_negative, cfg, seed, progress,
                               preview_phase='high', preview_offset=plan.low_evaluations,
                               preview_total=plan.total_evaluations)
        timings["high_sampling_including_model_prepare"] = time.perf_counter() - start
        output_video, output_audio = nested_av_parts({"samples": result})
        if tuple(output_video.shape) != tuple(video.shape) or tuple(output_audio.shape) != tuple(audio.shape):
            raise RuntimeError("Final AV shapes differ from the plan")
        if not all(bool(torch.isfinite(value).all()) for value in (output_video, output_audio)):
            raise RuntimeError("Final samples contain NaN or Inf")
        counts = ledger.finish(allow_incomplete_evidence=True)
        if counts["actual_forwards"] != apply_calls:
            warn_patch_stack('Progressive user stack bypassed a network/apply_model observer; counts unverified')
            counts['forward_evidence_complete'] = False
        counts.update(apply_model_calls=apply_calls, forward_boundary="native_h3_diffusion_model_executor")
        report = {**plan.report(), "status": "sampled_quality_unverified", "runtime_identity": "native_h3_euler_av",
                  "sampler_seconds": time.perf_counter() - started, "timings": timings,
                  "timing_scope": "sampler_node_not_end_to_end_no_forced_sync", "counts": counts,
                  "cfg_branch_evaluations": branch_evaluations, "upscaler": identity,
                  "lift_geometry": lift_geometry,
                  "network_input_shapes": network_shapes,
                  "lift_report": lift_report, "resource_observations": snapshots,
                  "noise": {"low_seed": seed, "high_seed": high_seed,
                            "low_video_shape": [*video.shape[:-2], plan.low_height // 16, plan.low_width // 16],
                            "audio_shape": list(audio.shape), "high_video_shape": list(video.shape),
                            "policy": "core_prepare_noise_cpu_low_joint_high_video_seed_plus_one"},
                  "reference_policy": "first_frame_latent_bilinear_low_original_high",
                  "pixel_anchor": False, "highres_tiling": False,
                  "existing_vdn_two_pass_modified": False}
        output = {key: value for key, value in av_latent.items() if key not in {'samples', 'noise_mask'}}
        output['samples'] = result.to(intermediate)
        return output, json.dumps(report, ensure_ascii=False, allow_nan=False)
    finally:
        branch.model_options.pop("model_function_wrapper", None)
        branch.remove_wrappers_with_key(wrapper_kind, wrapper_key)
        boundary.clear()
