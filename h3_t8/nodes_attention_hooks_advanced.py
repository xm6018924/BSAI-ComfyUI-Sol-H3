from __future__ import annotations

from comfy_api.latest import io

from .attention_hooks_advanced import build_attention_hook_compatibility


CATEGORY = "T8/MiniMax H3/Compatibility/Advanced"


class MiniMaxH3AttentionHooksT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AttentionHooksT8Advanced",
            display_name="MiniMax H3 Attention Hooks (T8 Advanced)",
            category=CATEGORY,
            description=(
                "Expose standard attn1 and attn1-output patch hooks on older H3 cores. "
                "Native hook support is preserved. Official H3 Sol/SLA/VSA producers are adapted locally: "
                "VSA tiles and coarse gates remain active. Calls with hooks materialize full QKV and may "
                "use more VRAM; calls without hooks retain the original chunked producer."
            ),
            inputs=[io.Model.Input("model")],
            outputs=[
                io.Model.Output("model"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        return io.NodeOutput(*build_attention_hook_compatibility(model))


ATTENTION_HOOKS_ADVANCED_NODE_CLASSES = [MiniMaxH3AttentionHooksT8Advanced]
