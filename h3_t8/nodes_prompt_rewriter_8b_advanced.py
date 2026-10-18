from __future__ import annotations

from comfy_api.latest import io

from .prompt_rewriter_8b import (
    BASE_REPOSITORY,
    DEFAULT_ADAPTER_FOLDER,
    DEFAULT_BASE_FOLDER,
    TASK_LABELS,
    UPSTREAM_REPOSITORY,
    rewrite_prompt_8b,
    unload_prompt_rewriter_cache,
)


CATEGORY = "T8/MiniMax H3/Conditioning/Experimental"


class MiniMaxH3PromptRewriter8BT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptRewriter8BT8Advanced",
            display_name="MiniMax H3 Prompt Rewriter 8B (Advanced)",
            description=(
                "Runs the pinned LightX2V Qwen3-VL-8B LoRA for T2VA/I2VA/L2VA/FL2VA. "
                "It is separate from the H3 32B CLIP because their parameter shapes are "
                "not LoRA-compatible. Unload-after-generate is enabled by default."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="Original short prompt; dialogue, lyrics, and on-screen text are requested to be kept verbatim.",
                ),
                io.Combo.Input(
                    "task",
                    options=list(TASK_LABELS),
                    default="T2VA — text-to-AV",
                ),
                io.Combo.Input(
                    "resolution",
                    options=["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                    default="16:9",
                ),
                io.Int.Input("duration", default=10, min=4, max=None, step=1),
                io.String.Input(
                    "base_model_path",
                    default=DEFAULT_BASE_FOLDER,
                    tooltip=(
                        "Relative paths resolve from ComfyUI/models/text_encoders; an absolute directory may also be entered."
                        f"精确基座为 {BASE_REPOSITORY}。"
                    ),
                ),
                io.String.Input(
                    "adapter_path",
                    default=DEFAULT_ADAPTER_FOLDER,
                    tooltip=(
                        "Relative paths resolve from ComfyUI/models/loras; an absolute directory may also be entered."
                        f"精确LoRA为 {UPSTREAM_REPOSITORY}。"
                    ),
                ),
                io.Combo.Input(
                    "load_policy",
                    options=["auto_cpu_offload", "gpu_only", "cpu_only"],
                    default="auto_cpu_offload",
                    tooltip="Auto CPU offload is recommended for 16GB GPUs; gpu_only requires sufficient VRAM.",
                ),
                io.Combo.Input("dtype", options=["bfloat16", "float16"], default="bfloat16"),
                io.Combo.Input("decoding", options=["greedy", "sample"], default="greedy"),
                io.Int.Input("seed", default=42, min=0, max=0x7FFFFFFFFFFFFFFF),
                io.Int.Input(
                    "max_new_tokens",
                    default=1024,
                    min=128,
                    max=8192,
                    step=128,
                    tooltip=(
                        "Default 1024 balances completeness with 16GB offload speed; too low may truncate the three-part structure, "
                        "too high only raises the ceiling without forcing the model to generate up to it."
                    ),
                ),
                io.Float.Input("temperature", default=0.7, min=0.01, max=2.0, step=0.01, advanced=True),
                io.Float.Input("top_p", default=0.8, min=0.01, max=1.0, step=0.01, advanced=True),
                io.Int.Input(
                    "min_image_pixels",
                    default=256 * 256,
                    min=32 * 32,
                    max=2048 * 2048,
                    step=1024,
                    advanced=True,
                ),
                io.Int.Input(
                    "max_image_pixels",
                    default=1024 * 1024,
                    min=32 * 32,
                    max=4096 * 4096,
                    step=1024,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "unload_after_generate",
                    default=True,
                    tooltip="After generation, errors, or OOM, attempts to release only this node's model and cache.",
                ),
                io.Boolean.Input(
                    "free_comfy_models_before_load",
                    default=True,
                    tooltip="Releases ComfyUI's current model before loading the 8B to free space for 16GB GPUs; does not globally unload after generation.",
                ),
                io.Boolean.Input(
                    "allow_hub_download",
                    default=False,
                    advanced=True,
                    tooltip="Defaults to local models only, avoiding accidental runtime downloads of large files.",
                ),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[
                io.String.Output("enhanced_prompt"),
                io.String.Output("integrated_multimodal_description"),
                io.String.Output("overall_soundscape"),
                io.String.Output("non_diegetic_music"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*rewrite_prompt_8b(**kwargs))


class MiniMaxH3PromptRewriterUnloadT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptRewriterUnloadT8Advanced",
            display_name="MiniMax H3 Prompt Rewriter Unload (Advanced)",
            description="Releases only the optional Prompt Rewriter 8B cache retained by this node pack.",
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[io.Boolean.Input("unload_now", default=True)],
            outputs=[io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, unload_now):
        if not unload_now:
            return io.NodeOutput('{"status":"bypass"}')
        return io.NodeOutput(unload_prompt_rewriter_cache())


PROMPT_REWRITER_8B_ADVANCED_NODE_CLASSES = [
    MiniMaxH3PromptRewriter8BT8Advanced,
    MiniMaxH3PromptRewriterUnloadT8Advanced,
]
