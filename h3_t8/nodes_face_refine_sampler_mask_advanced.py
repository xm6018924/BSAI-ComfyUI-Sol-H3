from __future__ import annotations

from comfy_api.latest import io

from .face_refine_sampler_mask_advanced import apply_face_refine_sampler_mask_patch


CATEGORY = "T8/MiniMax H3/Quality/Experimental/Face Refine Parity"


class MiniMaxH3FaceRefineSamplerMaskPatchV11T8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FaceRefineSamplerMaskPatchV11T8Advanced",
            display_name=(
                "MiniMax H3 Face Refine Sampler-Mask Fix V1.1 / "
                "采样遮罩修正 (Advanced)"
            ),
            description=(
                "Optional, disabled by default: the reviewed fixture favored the original route. "
                "Disabled returns the exact input MODEL and LATENT unchanged. When enabled, "
                "Applies the audited H3 FaceRefine v1.1.1 correction on a cloned MODEL: "
                "the video noise mask remains on the sampler path only, held video is "
                "re-noised to the current sigma, and the locked audio mask still reaches "
                "MiniMax H3 unchanged. The connected latent and denoise report are hash-bound; "
                "unsupported models, stale reports and conflicting object patches fail closed."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("av_latent"),
                io.String.Input("denoise_report_json", force_input=True),
                io.Boolean.Input("enabled", default=False),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Latent.Output("av_latent"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*apply_face_refine_sampler_mask_patch(**kwargs))


FACE_REFINE_SAMPLER_MASK_ADVANCED_NODE_CLASSES = [
    MiniMaxH3FaceRefineSamplerMaskPatchV11T8Advanced,
]
