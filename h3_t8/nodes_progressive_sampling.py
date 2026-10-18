"""Independent experimental H3 first sampler; legacy node schemas are unchanged."""

import folder_paths
from comfy_api.latest import io


class MiniMaxH3ProgressiveSamplerEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ProgressiveSamplerEXPT8",
            display_name="MiniMax H3 Progressive Sampler (EXP/T8)",
            category="T8/MiniMax H3/Performance/Experimental",
            is_experimental=True,
            description=("First samples at a smaller resolution, then uses a learned model to upscale the latent and completes the remaining steps."
                         "Only current Core native Euler T2VA / first-frame I2VA; input must be an empty AV latent."
                         "This is not a VDN second pass after a first pass; it does not lock the audio track, enable pixel anchors, or chunking."
                         "Local short-film speed benchmarks exist; quality must still be reviewed per footage; VRAM savings are not guaranteed."
                         "Optional HIGH model and staged EAV; Relay connects the paired MODEL/positive; TST can connect a MODEL config node."
                         "If the upscale model is missing, it does not fall back to plain interpolation."),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
                io.Latent.Input("av_latent"),
                io.Sampler.Input("sampler", tooltip="Connect the native Setup euler; do not use dual_clock_euler."),
                io.Sigmas.Input("sigmas", tooltip="Full first-pass schedule: from 1 to 0, at least 2 steps."),
                io.Combo.Input("upscaler_model", options=folder_paths.get_filename_list("latent_upscale_models"),
                               tooltip="The H3 learned latent upscale model from models/latent_upscale_models. Must be explicitly selected."),
                io.Int.Input("seed", default=20260909, min=0, max=2**64-1, control_after_generate=True),
                io.Float.Input("cfg", default=1., min=0., max=100., step=.1),
                io.Int.Input("low_evaluations", default=6, min=1, max=999,
                             tooltip="Number of steps at the small resolution. E.g. pick 6 of 8 total steps, then 2 more after upscale. Leave at least 1 step."),
                io.Float.Input("low_scale", default=.5, min=.25, max=.99, step=.01,
                               tooltip="Small-resolution width/height ratio, 32-pixel aligned; 0.5 does not guarantee 4x speedup."),
                io.Combo.Input("task", options=["t2va", "i2va"], default="t2va"),
                io.Combo.Input("precision", options=["fp16", "bf16", "fp32"], default="fp16", advanced=True),
                io.Int.Input("reserve_vram_mib", default=1024, min=512, max=32768, advanced=True,
                             tooltip="VRAM headroom checked at stage boundaries and sampling callbacks; this is not a peak prediction or an OOM guarantee."),
                io.Model.Input('model_hires', optional=True,
                               tooltip='Optional HIGH-stage MODEL; LoRA/attention patches can be chained independently while existing patches are retained. Structure and AV clock must match; if unconnected, model is reused.'),
                io.Combo.Input('guide_resize', options=['legacy_bilinear', 'preserve_mean'],
                               default='legacy_bilinear', advanced=True, optional=True,
                               tooltip='First-frame LOW reference scaling; preserve_mean explicitly keeps the channel mean, while the HIGH reference is unchanged.'),
                io.Combo.Input('eav_mode', options=['disabled', 'report_only', 'apply_exp'],
                               default='disabled', advanced=True, optional=True,
                               tooltip='Staged EAV uses the full 8/20-step source video sigma clock, CFG1. report_only measures; apply_exp actually enhances. This is not a quality certification.'),
                io.Float.Input('eav_tau', default=4., min=-32., max=32., step=.1, advanced=True, optional=True),
                io.Float.Input('eav_start_video_progress', default=.15, min=0., max=1., step=.01,
                               advanced=True, optional=True, tooltip='1 - source video sigma; changing resolution does not reset it. Omitting this legacy parameter preserves the original 0/1 contract.'),
                io.Float.Input('eav_end_video_progress', default=.90, min=0., max=1., step=.01,
                               advanced=True, optional=True),
                io.Int.Input('eav_max_workspace_mib', default=32, min=4, max=512, advanced=True, optional=True),
                io.Float.Input('eav_g_hard_limit', default=1.5, min=1., max=3., step=.01,
                               advanced=True, optional=True, tooltip='A real overflow raises an error normally; no truncation or retry. Choose disabled to turn off; tau=0 is not off.'),
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
