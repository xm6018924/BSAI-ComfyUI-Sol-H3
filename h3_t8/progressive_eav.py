"""EAV on native progressive stages with an explicit full-schedule audit.

Each stage executes only its assigned interval. EAV progress remains 1-sigma
on the original video clock, not a phase-local step percentage. Legacy EAV
audit/profile and long-video Stock20 gates are not replaced by this audit.
"""

import math

import torch

from . import enhance_a_video_advanced as eav
from .h3_core_compat import plain_attention_backend
from .progressive_attention import _PlainDelegate
from .patch_stack_policy import warn_patch_stack


def prepare_progressive_eav(model, sigmas, plan, *, mode, tau, start, end,
                            workspace, hard_limit, relay_report=None, mask_contract=None):
    if mask_contract is None and plan.total_evaluations not in (8, 20):
        raise ValueError('Progressive EAV requires a complete8 or20 evaluation schedule')
    profile = (eav.PROGRESSIVE_INITIALIZED_EAV_PROFILE if mask_contract is not None else
               'stock20' if plan.total_evaluations == 20 else eav.PROGRESSIVE_EAV_PROFILE)
    kwargs = dict(mode=mode, tau=tau, start_video_progress=start, end_video_progress=end,
                  max_workspace_mib=workspace, g_hard_limit=hard_limit, sampling_profile=profile,
                  progressive_mask_contract=mask_contract)
    if relay_report is None:
        override = model.model_options.get('transformer_options', {}).get('optimized_attention_override')
        plain = plain_attention_backend(override) if override is not None else None
        delegate = _PlainDelegate(override, plain) if plain is not None else None
        long_video = mask_contract.long_video_contract if mask_contract is not None else None
        if long_video is None:
            scope = dict(allowed_tasks=(plan.task.upper(),), composer_profile='progressive_native_av_v1')
        else:
            from .long_video import LONG_VIDEO_PATCH_VERSION
            scope = dict(allowed_tasks=('LongVideoMotion',), allow_reference_blocks=True,
                long_video_contract=long_video, allowed_live_extra_conds_patch_versions=(LONG_VIDEO_PATCH_VERSION,),
                composer_profile='progressive_continuation_native_av_v1')
        model, runtime, _ = eav.build_eav_model(model, sigmas,
            composed_backend_override=delegate, **scope, **kwargs)
    else:
        def observe(kind):
            relay_report['completed_calls'][kind] += 1

        model, runtime, _ = eav.build_eav_prompt_relay_model(
            model, sigmas, execution_observer=observe, **kwargs)
    runtime.config['progressive_schedule'] = {
        'full_sigmas': list(plan.sigmas), 'low_evaluations': plan.low_evaluations,
        'high_evaluations': plan.high_evaluations,
        'progress_policy': 'one_minus_native_video_sigma_not_phase_step_fraction',
    }
    return model, runtime


def audit_progressive_eav_stage(runtime, plan, phase, model):
    if phase not in {'low', 'high'}:
        raise ValueError('Unknown progressive EAV phase')
    report = runtime.snapshot(consume=True)
    if report['aborted']:
        raise RuntimeError('Progressive EAV aborted: ' + report['aborted'])
    if (report['config']['sigma_contract']['nfe'] != plan.total_evaluations
            or report['config'].get('progressive_schedule', {}).get('full_sigmas') != list(plan.sigmas)):
        raise RuntimeError('Progressive EAV runtime no longer matches the complete schedule')
    expected = plan.sigmas[:plan.low_evaluations] if phase == 'low' else plan.sigmas[plan.low_evaluations:-1]
    observed = report['forwards']
    coverage_complete = len(observed) == len(expected)
    if len(observed) != len(expected):
        warn_patch_stack(f'Progressive EAV {phase} forward coverage is incomplete')
    block_count = len(model.model.diffusion_model.blocks)
    expected_spatial = ((plan.low_height if phase == 'low' else plan.target_height) // 32
                        * ((plan.low_width if phase == 'low' else plan.target_width) // 32))
    active_total = 0
    mask_contract = report['config'].get('progressive_mask_contract')
    if report['config']['sampling_profile'] == eav.PROGRESSIVE_INITIALIZED_EAV_PROFILE:
        expected_video_shape = [*plan.video_shape[:-2],
                                (plan.low_height if phase == 'low' else plan.target_height) // 16,
                                (plan.low_width if phase == 'low' else plan.target_width) // 16]
        if (not isinstance(mask_contract, dict) or mask_contract.get('video_shape') != expected_video_shape
                or mask_contract.get('audio_shape') != list(plan.audio_shape)):
            raise RuntimeError('Progressive EAV is missing the correct initialized stage mask contract')
    previous_index = -1
    for index, forward in enumerate(observed):
        if mask_contract is not None and forward.get('progressive_mask_contract') != mask_contract:
            raise RuntimeError('Progressive EAV did not verify the bound native masks on every forward')
        matches = [i for i, value in enumerate(expected)
                   if i > previous_index and math.isclose(forward['sigma_video'], value,
                                                         abs_tol=2e-6, rel_tol=2e-6)]
        if not matches:
            raise RuntimeError(f'Progressive EAV {phase} sigma mismatch at call{index}')
        previous_index = matches[0]
        sigma = expected[previous_index]
        progress = 1. - sigma
        if not math.isclose(forward['progress_video'], progress, abs_tol=2e-6, rel_tol=2e-6):
            raise RuntimeError('Progressive EAV changed the global video progress coordinate')
        # Reproduce the actual fp32 timestep division for comparisons at endpoints.
        runtime_progress = 1. - float((torch.tensor(sigma, dtype=torch.float32) * 1000.) / 1000.)
        active = report['config']['start_video_progress'] <= runtime_progress <= report['config']['end_video_progress']
        if forward['active'] != active:
            raise RuntimeError('Progressive EAV activation window differs from the native video clock')
        expected_count = block_count if active else 0
        if forward['attention_count'] != expected_count:
            warn_patch_stack('Progressive EAV active native H3 block coverage is incomplete')
            coverage_complete = False
        if forward['frames'] != plan.video_shape[2] or forward['spatial_tokens'] != expected_spatial:
            raise RuntimeError('Progressive EAV observed the wrong phase spatial/temporal layout')
        if forward['audio_rows'] != 2 * plan.audio_shape[-1]:
            raise RuntimeError('Progressive EAV changed the target audio layout')
        if forward['strict_sage_failure_count']:
            raise RuntimeError('Progressive EAV attention execution failed')
        active_total += int(active)
    report.update(status=('verified_stage_execution_quality_unverified' if coverage_complete
                          else 'executed_user_stack_unverified'), composition_verified=coverage_complete, phase=phase,
                  assigned_nfe=len(expected), observed_main_blocks=block_count,
                  active_stage_forwards=active_total,
                  sigma_scope='phase_subset_of_verified_full_schedule',
                  block_scope='actual_native_model_architecture_not_a_trained_model_identity')
    if mask_contract is not None:
        report['verified_native_mask_forwards'] = len(observed)
    return report


def summarize_progressive_eav(reports, plan):
    if set(reports) != {'low', 'high'}:
        raise RuntimeError('Progressive EAV requires both completed stage audits')
    total = sum(report['model_forward_count'] for report in reports.values())
    if total != plan.total_evaluations:
        warn_patch_stack('Progressive EAV total observed forward coverage is incomplete')
    verified = total == plan.total_evaluations and all(
        report.get('composition_verified', True) for report in reports.values())
    active = sum(report['active_forward_count'] for report in reports.values())
    gain = max((report['g_max'] or 1.) for report in reports.values())
    return {'full_schedule_nfe': total, 'active_forwards': active,
            'mode': reports['low']['config']['mode'],
            'output_gain_above_one_applied': reports['low']['config']['mode'] == 'apply_exp' and gain > 1.,
            'status': ('executed_user_stack_unverified' if not verified else
                       'verified_execution_quality_unverified' if active else 'inactive_window_no_enhancement'),
            'composition_verified': verified,
            'direct_audio_scaling': False, 'adds_model_forwards': False}
