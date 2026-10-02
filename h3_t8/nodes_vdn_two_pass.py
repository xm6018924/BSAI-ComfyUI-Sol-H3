from comfy_api.latest import io

from .vdn_two_pass import setup_vdn_refine


class MiniMaxH3VDNRefinePlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VDNRefinePlanT8Advanced",
            display_name="MiniMax H3 VDN Two-Pass Refine / VDN二采 (Advanced EXP/T8)",
            category="T8/MiniMax H3/Performance/Advanced",
            description="Restart VDN on its own stage sigma grid after a completed first pass and learned video latent upscale. Rebuild high-resolution Conditioning and reconcile audio first. Connect fresh RandomNoise; do not add another Turbo LoRA. Quality is experimental.",
            is_experimental=True,
            inputs=[io.Model.Input("model"), io.Latent.Input("av_latent"),
                    io.Latent.Input("first_pass_latent"),
                    io.Int.Input("refine_steps", default=4, min=1, max=49)],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"),
                     io.Sigmas.Output("sigmas"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, av_latent, first_pass_latent, refine_steps=4):
        return io.NodeOutput(*setup_vdn_refine(model, av_latent, first_pass_latent, refine_steps))
