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
            display_name="MiniMax H3 Low VRAM Attention / 低显存注意力 (Advanced EXP/T8)",
            description=(
                "独立 T8 低显存注意力节点：提前释放 H3 归一化输入，并按头分组调用当前 Core "
                "attention。默认 4 组；数值公式不变，但不同内核形状可能产生浮点舍入差异。"
                "已有 KJ/Sage/Sol 或其他 callable 补丁不硬拦截；保留外部 block/attention "
                "forward 的层可能不安装本节点头分组。报告实际安装/保留层数，组合效果由用户承担。"
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
                        "注意力头分组数。更大通常降低单次 attention 临时显存，但增加调用开销；"
                        "1 仍启用提前释放，只是不分组。"
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
            display_name="MiniMax H3 Chunk FeedForward / 分块前馈 (Advanced EXP/T8)",
            description=(
                "独立 T8 H3 SwiGLU 分块节点：仅在 packed token 数超过阈值时按 token 轴分块，"
                "降低 FFN 激活峰值。chunks=1 是完全不改 MODEL 的旁路。"
                "已有 KJ/Sage/Sol、block 或 MLP 补丁不作兼容性硬拦截，仅警告并保留/委托调用；"
                "组合风险由使用者承担，不保证兼容或实际省显存。"
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
                        "FFN 分块数。更大可能降低激活峰值，但会增加 GEMM 调用开销；1 返回原 MODEL。"
                    ),
                ),
                io.Int.Input(
                    "seq_threshold",
                    default=4096,
                    min=256,
                    max=262144,
                    step=256,
                    tooltip=(
                        "只有 packed token 数严格大于该值才分块；等于或低于阈值时保持单次原生 FFN。"
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
