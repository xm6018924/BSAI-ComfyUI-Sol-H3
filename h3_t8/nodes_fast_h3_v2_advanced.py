from comfy_api.latest import io

from .fast_h3_v2_advanced import (
    PROFILES, build_fast_h3_v2_setup, capture_fast_h3_v2_owner,
)


class MiniMaxH3FastH3V2SetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3FastH3V2SetupEXPT8",
            is_experimental=True,
            display_name="FastH3 V2 · 8-Step Recipe (T8 EXP)", category="MiniMax H3 T8/Speed/Experimental",
            description="Full V2 student, not a four-step LoRA. Exact DMD8 / shift10+3 / learned VSA20%. "
                "Dense compatibility and official Comfy template are separate EXP recipes. "
                "Use the existing encoders/VAEs. Model: https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy",
            inputs=[io.Model.Input("model"), io.Latent.Input("av_latent"),
                    io.Combo.Input("profile", options=list(PROFILES), default="trained_vsa_exp"),
                    io.Int.Input("min_tokens", default=12288, min=0, max=1048576,
                                 tooltip="Below this packed-token count native VSA runs Dense and reports it; 0 forces eligibility testing.")],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"),
                     io.Sigmas.Output("sigmas"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, av_latent, profile="trained_vsa_exp", min_tokens=12288):
        return io.NodeOutput(*build_fast_h3_v2_setup(model, av_latent, profile, min_tokens))


class MiniMaxH3FastH3V2RuntimeAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3FastH3V2RuntimeAuditEXPT8",
            is_experimental=True,
            display_name="FastH3 V2 · Actual Dispatch Audit (T8 EXP)", category="MiniMax H3 T8/Speed/Experimental",
            description="Wire the sampler output to run after sampling. VSA profiles count sparse dispatch and Dense eligibility failures. "
                        "Dense compatibility preserves its backend; separate instrumentation is needed to measure its dispatch. Not quality acceptance.",
            inputs=[io.Model.Input("model"), io.Latent.Input("sampled_av_latent")],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, sampled_av_latent):
        import json
        receipt = capture_fast_h3_v2_owner(model)
        if receipt is None:
            raise ValueError("FastH3 V2 runtime owner missing")
        return io.NodeOutput(sampled_av_latent, json.dumps(receipt.runtime.snapshot(), indent=2))


class MiniMaxH3FastH3V2DualModelLongVideoEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from .nodes_long_video_dual_model import MiniMaxH3DualModelLongVideoEXPT8
        original = MiniMaxH3DualModelLongVideoEXPT8.define_schema()
        removed = {'coarse_steps', 'refine_steps', 'first_shift_video', 'first_shift_audio',
                   'second_shift_video', 'second_shift_audio'}
        inherited = [item for item in original.inputs if item.id not in removed]
        for item in inherited:
            if item.id == 'total_duration_seconds':
                item.default = 8.
            elif item.id == 'chain_id':
                item.default = 'fasth3_v2_dual_4plus4_exp'
            elif item.id == 'eav_mode':
                item.default = 'disabled'
        return io.Schema(node_id='MiniMaxH3FastH3V2DualModelLongVideoEXPT8',
            display_name='FastH3 V2 · Dual MODEL4+Upscale+4 Loop (T8 EXP)',
            category=original.category, is_experimental=True, is_output_node=True,
            description='Two full V2 student MODEL branches, optional independent ordinary LoRA and T8 memory nodes. '
                'Exact trained ladder cut4+4, AV clocks10/3, learned3D latent upscale. Reference/loop is EXP, not distilled training support. '
                'Relay apply_exp requires explicitly selected dense_compat_exp; no silent loss of timeline bias. '
                'Do not connect the latent-bound single V2 setup MODEL here. No old workflow migration.',
            inputs=[io.Combo.Input('profile', options=['trained_vsa_exp', 'dense_compat_exp'],
                                   default='trained_vsa_exp'), *inherited], outputs=original.outputs)

    @classmethod
    def execute(cls, profile='trained_vsa_exp', **kwargs):
        from .nodes_long_video_dual_model import MiniMaxH3DualModelLongVideoEXPT8
        return MiniMaxH3DualModelLongVideoEXPT8.execute(
            coarse_steps=4, refine_steps=4, first_shift_video=10., first_shift_audio=3.,
            second_shift_video=10., second_shift_audio=3., _fast_h3_v2_profile=profile, **kwargs)


FAST_H3_V2_NODE_CLASSES = [MiniMaxH3FastH3V2SetupEXPT8, MiniMaxH3FastH3V2RuntimeAuditEXPT8,
                         MiniMaxH3FastH3V2DualModelLongVideoEXPT8]
