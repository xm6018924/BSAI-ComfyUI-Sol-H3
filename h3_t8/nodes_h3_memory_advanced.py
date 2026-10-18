from __future__ import annotations

from comfy_api.latest import io

from .h3_memory_advanced import (
    canonical_json,
    configure_chunk_feed_forward,
    configure_low_vram_attention,
)


CATEGORY = "T8/MiniMax H3/Models/Experimental"


class MiniMaxH3LowVRAMAttentionT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LowVRAMAttentionT8Advanced",
            display_name="MiniMax H3 Low VRAM Attention (Advanced EXP/T8)",
            description=(
                "Standalone T8 low-VRAM attention node: releases H3 normalization inputs early and calls the current Core "
                "attention in head groups. Default 4 groups; the math is unchanged, but different kernel shapes may produce floating-point rounding differences."
                "Existing KJ/Sage/Sol or other callable patches are not hard-blocked; external block/attention "
                "forward layers may not get this node's head grouping. The installed/retained layer count is reported; combined effects are the user's responsibility."
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model"),
                io.Int.Input(
                    "head_chunks",
                    default=4,
                    min=1,
                    max=56,
                    step=1,
                    tooltip=(
                        "Number of attention head groups. Larger groups usually reduce per-attention temporary VRAM but increase call overhead; "
                        "1 still enables early release, just without grouping."
                    ),
                ),
            ],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, head_chunks):
        patched, report = configure_low_vram_attention(model, head_chunks)
        return io.NodeOutput(patched, canonical_json(report))


class MiniMaxH3ChunkFeedForwardT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ChunkFeedForwardT8Advanced",
            display_name="MiniMax H3 Chunk FeedForward (Advanced EXP/T8)",
            description=(
                "Standalone T8 H3 SwiGLU chunking node: chunks along the token axis only when packed token count exceeds the threshold, "
                "lowering FFN activation peak. chunks=1 is a passthrough that does not modify the MODEL."
                "Existing KJ/Sage/Sol, block, or MLP patches are not hard-blocked; a warning is emitted and the call is retained/delegated; "
                "combined-use risk is the user's responsibility; compatibility or actual VRAM savings are not guaranteed."
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model"),
                io.Int.Input(
                    "chunks",
                    default=2,
                    min=1,
                    max=64,
                    step=1,
                    tooltip=(
                        "FFN chunk count. Larger may reduce activation peak but increases GEMM call overhead; 1 returns the original MODEL."
                    ),
                ),
                io.Int.Input(
                    "seq_threshold",
                    default=4096,
                    min=256,
                    max=262144,
                    step=256,
                    tooltip=(
                        "Chunking only applies when packed token count is strictly greater than this; at or below the threshold, native single-pass FFN is used."
                    ),
                ),
            ],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, chunks, seq_threshold):
        patched, report = configure_chunk_feed_forward(model, chunks, seq_threshold)
        return io.NodeOutput(patched, canonical_json(report))


H3_MEMORY_ADVANCED_NODE_CLASSES = [
    MiniMaxH3LowVRAMAttentionT8Advanced,
    MiniMaxH3ChunkFeedForwardT8Advanced,
]
