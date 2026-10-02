from __future__ import annotations

import folder_paths
from comfy_api.latest import io

from .chunked_two_pass_upscale_advanced import (
    build_chunked_two_pass_plan,
    execute_chunked_two_pass_upscale,
)
from .learned_latent_upscale_advanced import ASPECT_POLICIES, PRECISIONS, RELEASE_POLICIES, SIZE_MODES


CATEGORY = "T8/MiniMax H3/Upscale/Advanced"
PLAN_TYPE = io.Custom("T8_H3_CHUNKED_TWO_PASS_PLAN")


def _upscaler_options() -> list[str]:
    names = list(folder_paths.get_filename_list("latent_upscale_models"))
    return names or ["minimax_h3_latent_upscaler_3d_fp16.safetensors"]


class MiniMaxH3ChunkedTwoPassPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ChunkedTwoPassPlanT8Advanced",
            display_name="MiniMax H3 Chunked Two-Pass Plan (Advanced EXP/T8)",
            description=(
                "Plans learned 3D latent upscale followed by temporal-chunk H3 "
                "re-sampling. The safe default keeps each temporal chunk full-frame; "
                "independent spatial canvases remain explicit EXP. There is no project "
                "pixel-area ceiling; memory and runtime remain user-owned."
                " Optional scale_by/target_megapixels sizing reads the actual first-pass LATENT "
                "and uses the ordinary learned-upscaler geometry, aspect and 32-pixel alignment rules. "
                "Connect width/height outputs to HIGH conditioning to keep its canvas synchronized."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Combo.Input("model_name", options=_upscaler_options()),
                io.Int.Input("target_width", default=1280, min=32, max=16384, step=32,
                    tooltip="仅target_dimensions模式使用。倍率/面积模式忽略此值；请用width输出连接HIGH条件。连接source_latent时会按aspect_policy计算实际目标尺寸。"),
                io.Int.Input("target_height", default=704, min=32, max=16384, step=32,
                    tooltip="仅target_dimensions模式使用。倍率/面积模式忽略此值；请用height输出连接HIGH条件。实际输出为32像素对齐后的尺寸。"),
                io.Int.Input(
                    "temporal_chunk_frames", default=136, min=17, max=3600, step=17
                ),
                io.Int.Input(
                    "temporal_overlap_frames", default=17, min=0, max=1700, step=17
                ),
                io.Float.Input(
                    "anchor_strength", default=0.999, min=0.0, max=1.0, step=0.001
                ),
                io.Int.Input("tile_width", default=512, min=32, max=16384, step=32),
                io.Int.Input("tile_height", default=512, min=32, max=16384, step=32),
                io.Int.Input(
                    "spatial_overlap", default=128, min=0, max=4096, step=32
                ),
                io.Int.Input("spatial_fade", default=32, min=0, max=4096, step=32),
                io.Int.Input(
                    "minimum_tile_size", default=256, min=32, max=4096, step=32
                ),
                io.Combo.Input(
                    "overlap_blend",
                    options=["smoothstep", "linear"],
                    default="smoothstep",
                ),
                io.Combo.Input("precision", options=list(PRECISIONS), default="fp16"),
                io.Combo.Input(
                    "release_policy",
                    options=list(RELEASE_POLICIES),
                    default="offload_after",
                ),
                io.Combo.Input(
                    "spatial_strategy",
                    options=["full_frame_safe", "independent_tiles_exp"],
                    default="full_frame_safe",
                ),
                io.Combo.Input('sampling_contract', optional=True,
                    options=['video_only_legacy', 'standard_joint_4plus4_exp'],
                    default='video_only_legacy', display_name='采样合同',
                    tooltip='旧默认只重采视频并保留输入音频。标准4+4选standard_joint_4plus4_exp，并连接Parity Plan的report_json；每窗后4步联合AV，交付二采音频。'),
                io.String.Input('parity_report_json', optional=True, force_input=True,
                    tooltip='标准4+4模式必接Learned Two-Pass Parity Plan的report_json；旧模式不需要。'),
                io.Combo.Input("size_mode", optional=True, options=list(SIZE_MODES), default="target_dimensions",
                    display_name="尺寸模式", tooltip="旧默认手填尺寸；scale_by按源LATENT倍率、target_megapixels按目标面积计算，均使用普通放大节点的数学与32像素对齐规则。"),
                io.Float.Input("scale_by", optional=True, default=2.0, min=1.0, max=4.0, step=0.01,
                    display_name="放大倍率", tooltip="scale_by模式生效，例如1.5或2.0。必接source_latent，target_width/height在倍率模式不参与计算。"),
                io.Float.Input("target_megapixels", optional=True, default=0.70, min=0.01, max=8.0, step=0.01),
                io.Combo.Input("aspect_policy", optional=True, options=list(ASPECT_POLICIES), default="preserve_source",
                    tooltip="连接源LATENT时preserve_source保持原比例；honor_dimensions_exp允许按手填尺寸改变比例，仍检查max_anisotropy。旧图不接源时仍保留原手填尺寸。"),
                io.Float.Input("max_anisotropy", optional=True, default=1.05, min=1.0, max=2.0, step=0.01, advanced=True),
                io.Latent.Input("source_latent", optional=True,
                    tooltip="接一采LATENT（标准4+4用denoised_output），用于读取实际原始宽高；不是HIGH条件的LATENT，不改变输入视频或音频。"),
            ],
            outputs=[PLAN_TYPE.Output("plan"), io.String.Output("report_json"),
                     io.Int.Output("width"), io.Int.Output("height")],
        )

    @classmethod
    def execute(cls, **kwargs):
        plan, report = build_chunked_two_pass_plan(**kwargs)
        return io.NodeOutput(plan, report, plan["target_width"], plan["target_height"])


class MiniMaxH3ChunkedTwoPassUpscaleT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ChunkedTwoPassUpscaleT8Advanced",
            display_name="MiniMax H3 Chunked Two-Pass Upscale (Advanced EXP/T8)",
            description=(
                "Runs learned latent upscale per temporal chunk and restores the exact "
                "input audio tensor. Full-frame mode preserves global H3 spatial context; "
                "independent spatial tiles are research-only because their content can diverge. "
                "Opt-in standard_joint_4plus4_exp instead upscales once, runs four joint AV "
                "refine intervals per window, and publishes refined audio. Multiple windows "
                "increase total model calls; this is not eight calls for the entire clip."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("latent"),
                io.Noise.Input("noise"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                PLAN_TYPE.Input("plan"),
                io.Conditioning.Input("negative", optional=True),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1),
            ],
            outputs=[io.Latent.Output("latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*execute_chunked_two_pass_upscale(**kwargs))


CHUNKED_TWO_PASS_UPSCALE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3ChunkedTwoPassPlanT8Advanced,
    MiniMaxH3ChunkedTwoPassUpscaleT8Advanced,
]
