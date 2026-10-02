"""Independent experimental H3 first sampler; legacy node schemas are unchanged."""

import folder_paths
from comfy_api.latest import io


class MiniMaxH3ProgressiveSamplerEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ProgressiveSamplerEXPT8",
            display_name="MiniMax H3 Progressive Sampler / 渐进分辨率首采 (EXP/T8)",
            category="T8/MiniMax H3/Performance/Experimental",
            is_experimental=True,
            description=("先以较小画幅采样，再用学习型模型放大潜空间，完成剩余步数。"
                         "仅当前 Core 原生 Euler 的 T2VA/首帧 I2VA；输入必须是空 AV 潜空间。"
                         "不是完成一采后的 VDN 二采，不锁音轨，不启用像素锚或分块。"
                         "已有本机短片速度对照；画质仍需按素材审看，不保证省显存。"
                         "可选HIGH模型及分阶段EAV；Relay接配对MODEL/positive，TST可接MODEL配置节点。"
                         "缺失放大模型时不会改用普通插值。"),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
                io.Latent.Input("av_latent"),
                io.Sampler.Input("sampler", tooltip="连接原生 Setup 的 euler，不使用 dual_clock_euler。"),
                io.Sigmas.Input("sigmas", tooltip="完整首采时间表：从 1 到 0，至少 2 步。"),
                io.Combo.Input("upscaler_model", options=folder_paths.get_filename_list("latent_upscale_models"),
                               tooltip="models/latent_upscale_models 内的 H3 学习型潜空间放大模型。必须显式选择。"),
                io.Int.Input("seed", default=20260909, min=0, max=2**64-1, control_after_generate=True),
                io.Float.Input("cfg", default=1., min=0., max=100., step=.1),
                io.Int.Input("low_evaluations", default=6, min=1, max=999,
                             tooltip="小画幅的步数。例：完整 8 步中选 6，则放大后还采 2 步。至少留 1 步。"),
                io.Float.Input("low_scale", default=.5, min=.25, max=.99, step=.01,
                               tooltip="小画幅宽高比例，按 32 像素对齐；0.5 不是保证 4 倍加速。"),
                io.Combo.Input("task", options=["t2va", "i2va"], default="t2va"),
                io.Combo.Input("precision", options=["fp16", "bf16", "fp32"], default="fp16", advanced=True),
                io.Int.Input("reserve_vram_mib", default=1024, min=512, max=32768, advanced=True,
                             tooltip="阶段边界和采样回调检查的显存余量，不代表峰值预测或不会 OOM。"),
                io.Model.Input('model_hires', optional=True,
                               tooltip='可选HIGH阶段MODEL；可独立串联LoRA/注意力补丁，原有补丁保留。结构和AV时钟须匹配；不接沿用model。'),
                io.Combo.Input('guide_resize', options=['legacy_bilinear', 'preserve_mean'],
                               default='legacy_bilinear', advanced=True, optional=True,
                               tooltip='首帧LOW参考缩放；preserve_mean显式保持通道均值，HIGH原参考不变。'),
                io.Combo.Input('eav_mode', options=['disabled', 'report_only', 'apply_exp'],
                               default='disabled', advanced=True, optional=True,
                               tooltip='分阶段EAV使用完整8/20步原视频sigma时钟，CFG1。report_only只测量，apply_exp实际增强；不是画质认证。'),
                io.Float.Input('eav_tau', default=4., min=-32., max=32., step=.1, advanced=True, optional=True),
                io.Float.Input('eav_start_video_progress', default=.15, min=0., max=1., step=.01,
                               advanced=True, optional=True, tooltip='1-原视频sigma；切换分辨率不重置。旧API省略该参数仍保留原0/1合同。'),
                io.Float.Input('eav_end_video_progress', default=.90, min=0., max=1., step=.01,
                               advanced=True, optional=True),
                io.Int.Input('eav_max_workspace_mib', default=32, min=4, max=512, advanced=True, optional=True),
                io.Float.Input('eav_g_hard_limit', default=1.5, min=1., max=3., step=.01,
                               advanced=True, optional=True, tooltip='真实超限正常报错，不截断或重试。关闭选disabled，tau=0不是关闭。'),
            ],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        # Current-Core-specific runtime checks are delayed until explicitly used.
        import comfy.utils
        from .progressive_sampling_runtime import sample_progressive_h3
        steps = max(1, len(kwargs["sigmas"]) - 1)
        progress = comfy.utils.ProgressBar(steps)

        def notify(step, prediction, state, total):
            progress.update_absolute(step + 1, total)

        return io.NodeOutput(*sample_progressive_h3(**kwargs, callback=notify))
