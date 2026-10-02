"""Append-only public interfaces for the content-bound progressive chain.

No alternate sampling, continuation, decode, or assembly math lives here.
"""

import json

import folder_paths
from comfy_api.latest import io
from .core import MIN_TRAINED_FRAMES


class MiniMaxH3ProgressiveSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3ProgressiveSetupEXPT8',
            display_name='MiniMax H3 Progressive Dual Setup / 渐进双模型配置 (EXP/T8)',
            category='T8/MiniMax H3/Performance/Experimental', is_experimental=True,
            description=('给LOW/HIGH配置同一完整原生Euler时间表，不执行采样。'
                         '两路各接兼容LoRA和单一后端；TST应接在此节点输出之后。'
                         '不接model_hires则复用LOW源配置，不改动输入MODEL。'),
            inputs=[io.Model.Input('model'), io.Int.Input('steps', default=8, min=2, max=1000),
                io.Float.Input('shift_video', default=12., min=.01, max=100., step=.1),
                io.Float.Input('shift_audio', default=3., min=.01, max=100., step=.1),
                io.Model.Input('model_hires', optional=True)],
            outputs=[io.Model.Output('model'), io.Model.Output('model_hires'),
                     io.Sampler.Output('sampler'), io.Sigmas.Output('sigmas')])

    @classmethod
    def execute(cls, model, steps, shift_video, shift_audio, model_hires=None):
        import torch
        import comfy.nested_tensor
        from .sampling import setup_dual_clock_sampling
        from .progressive_sampling_runtime import validate_native_model
        from .progressive_stage_models import validate_stage_pair
        # Native Euler setup only checks AV channels; no conditioning or device
        # allocation is needed. This is NOT a latent used for generation.
        template = {'samples': comfy.nested_tensor.NestedTensor((
            torch.zeros(1, 24, 2, 2, 2, device='cpu'),
            torch.zeros(1, 32, 2, 8, device='cpu')))}
        high_source = model if model_hires is None else model_hires
        low, sampler, sigmas = setup_dual_clock_sampling(
            model, template, steps, shift_video, shift_audio, 'euler', 'native_flow')
        high, _, high_sigmas = setup_dual_clock_sampling(
            high_source, template, steps, shift_video, shift_audio, 'euler', 'native_flow')
        if not torch.equal(sigmas, high_sigmas):
            raise ValueError('Progressive phase schedules differ')
        validate_stage_pair(low, high, validate_native_model(low, sampler),
                            validate_native_model(high, sampler))
        return io.NodeOutput(low, high, sampler, sigmas)


class MiniMaxH3ProgressiveLongVideoEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3ProgressiveLongVideoEXPT8',
            display_name='MiniMax H3 Progressive Long Video / 渐进双采长视频 (EXP/T8)',
            category='T8/MiniMax H3/Long Video/Experimental', is_experimental=True,
            is_output_node=True,
            description=('独立渐进长视频实验：每段完整Euler表拆成LOW+HIGH，复用现有3D放大器。'
                '使用内容绑定的接受画面续接、原生AV遮罩与磁盘恢复；不替换旧双MODEL节点。'
                '完整权重成片验收尚未完成。只支持T2VA或单首帧I2VA；不宣称通用多参考支持。'
                '关闭resume_existing会拒绝已有链，不删除旧文件；改配置使用新chain_id。'),
            inputs=[io.Model.Input('model'), io.Clip.Input('clip'), io.Vae.Input('video_vae'),
                io.Vae.Input('audio_vae'), io.Sampler.Input('sampler'), io.Sigmas.Input('sigmas'),
                io.String.Input('chain_id', default='h3_progressive_long_video'),
                io.Float.Input('total_duration_seconds', default=8., min=.25, max=None, step=.01),
                io.Int.Input('width', default=896, min=64, max=8192, step=32),
                io.Int.Input('height', default=448, min=64, max=8192, step=32),
                io.Int.Input('render_window_frames', default=124, min=MIN_TRAINED_FRAMES,
                    max=None, step=17,
                    tooltip='内部窗口至少124帧、17n+5网格，无固定帧数上限；短片按总时长裁出。窗口越大，显存和耗时通常越高。'),
                io.Combo.Input('context_frames', options=[22, 39], default=22),
                io.String.Input('global_prompt', default='', multiline=True),
                io.String.Input('segment_prompts_json', default='', multiline=True, advanced=True),
                io.Combo.Input('upscaler_model', options=folder_paths.get_filename_list('latent_upscale_models')),
                io.Int.Input('base_seed', default=2609032101, min=0, max=2**64 - 1, control_after_generate=True),
                io.Combo.Input('seed_policy', options=['increment', 'fixed'], default='increment'),
                io.Int.Input('low_evaluations', default=4, min=1, max=999,
                    tooltip='完整8步表中填4为4+4，填6为6+2；至少留1步HIGH。'),
                io.Float.Input('low_scale', default=.5, min=.25, max=.99, step=.01),
                io.Combo.Input('precision', options=['fp16', 'bf16', 'fp32'], default='fp16'),
                io.Combo.Input('guide_resize', options=['legacy_bilinear', 'preserve_mean'], default='legacy_bilinear'),
                io.Combo.Input('context_audio', options=['video_and_audio', 'video_only'], default='video_and_audio'),
                io.Combo.Input('audio_mode', options=['native', 'lock_source', 'remix_source'], default='native'),
                io.Float.Input('audio_denoise_strength', default=.35, min=0., max=1., step=.01),
                io.Boolean.Input('resume_existing', default=True,
                    tooltip='相同输入重新执行会核验后复用阶段/成片。关闭则拒绝已有数据链；不清理旧结果。'),
                io.String.Input('filename_prefix', default='H3_Progressive_UNREVIEWED'),
                io.Combo.Input('audio_seam_policy', options=['none', 'cosine_bridge'], default='cosine_bridge'),
                io.Float.Input('audio_bridge_ms', default=5., min=0., max=50., step=1.),
                io.Int.Input('crf', default=18, min=0, max=51, advanced=True),
                io.Int.Input('query_chunk_rows', default=256, min=32, max=2048, advanced=True),
                io.Int.Input('reserve_vram_mib', default=2048, min=512, max=32768, advanced=True),
                io.Combo.Input('eav_mode', options=['disabled', 'report_only', 'apply_exp'], default='disabled'),
                io.Float.Input('eav_tau', default=4., min=-32., max=32., step=.1, advanced=True),
                io.Float.Input('eav_start_video_progress', default=.15, min=0., max=1., step=.01, advanced=True),
                io.Float.Input('eav_end_video_progress', default=.90, min=0., max=1., step=.01, advanced=True),
                io.Int.Input('eav_max_workspace_mib', default=32, min=4, max=512, advanced=True),
                io.Float.Input('eav_g_hard_limit', default=1.5, min=1., max=3., step=.01, advanced=True),
                io.Model.Input('model_hires', optional=True),
                io.Custom('H3_T8_PROMPT_RELAY_PLAN').Input('prompt_relay_plan', optional=True),
                io.Image.Input('first_frame', optional=True),
                io.Audio.Input('drive_audio', optional=True), io.Audio.Input('final_audio', optional=True)],
            outputs=[io.Video.Output('video'), io.String.Output('video_path'), io.String.Output('report_json')])

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Cached UI output must not bypass on-disk receipt/hash verification.
        return float('nan')

    @classmethod
    def execute(cls, **kwargs):
        import comfy.model_management as mm
        import comfy.utils
        from .progressive_job import NativeProgressiveJob
        from .progressive_delivery import ProgressiveChainDelivery
        from .nodes_long_video_in_node_loop_advanced import _preview_video
        values = dict(kwargs)
        resume = values.pop('resume_existing')
        delivery_options = {key: values.pop(key) for key in ('filename_prefix', 'audio_seam_policy', 'crf')}
        delivery_options['bridge_ms'] = values.pop('audio_bridge_ms')
        options = {key: values.pop(key) for key in ('context_audio', 'audio_mode', 'audio_denoise_strength')}
        options.update(add_source_as_reference=False, first_frame_reuse='segment0_only')
        for key in ('first_frame', 'drive_audio', 'final_audio'):
            if values.get(key) is not None:
                options[key] = values.pop(key)
            else:
                values.pop(key, None)
        sampling_options = {key: values.pop(key) for key in tuple(values)
                            if key.startswith('eav_') or key == 'reserve_vram_mib'}
        values.setdefault('model_hires', None)
        job = NativeProgressiveJob(**values, shared_condition_options=options, sampling_options=sampling_options)
        steps = len(job.sigmas) - 1
        progress = comfy.utils.ProgressBar(steps * len(job.segments))

        def notify(segment, step, prediction, state, total):
            mm.throw_exception_if_processing_interrupted()
            progress.update_absolute(segment * steps + step + 1, steps * len(job.segments))

        path, report = ProgressiveChainDelivery(job, **delivery_options).run(callback=notify, resume_existing=resume)
        if json.loads(report).get('human_qualified') is not False:
            raise RuntimeError('Generated media must remain explicitly unreviewed')
        video, preview = _preview_video(path)
        return io.NodeOutput(video, path, report, ui=preview)


PROGRESSIVE_LONG_VIDEO_NODE_CLASSES = [MiniMaxH3ProgressiveSetupEXPT8, MiniMaxH3ProgressiveLongVideoEXPT8]
