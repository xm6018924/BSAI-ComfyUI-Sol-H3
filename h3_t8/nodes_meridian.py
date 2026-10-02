"""Four append-only Meridian V3 entries. Lazy external dependencies and no server GPU thread."""

import json

from comfy_api.latest import io

from .meridian_plan import (
    PRESETS,
    canonical_plan,
    plan_json,
    preset_plan,
    source_map,
    identity,
)

RuntimeIO = io.Custom("T8_MERIDIAN_RUNTIME")
MaterialIO = io.Custom("T8_MERIDIAN_MATERIAL")
PlanIO = io.Custom("T8_MERIDIAN_PREPARED")
CATEGORY = "T8/MiniMax H3/Meridian (EXP)"


def schema(cls, name, inputs, outputs, **kwargs):
    return io.Schema(
        node_id=cls.__name__,
        display_name=name,
        category=CATEGORY,
        is_experimental=True,
        inputs=inputs,
        outputs=outputs,
        **kwargs,
    )


class MiniMaxH3MeridianConfigEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Meridian 1 · 模型与资源（INT8/T8）",
            [
                io.String.Input(
                    "model_path",
                    default="",
                    tooltip="留空自动寻找models/meridian/meridian_dmd_int8_convrot_comfy.safetensors；支持MERIDIAN_MODEL环境变量。下载：https://huggingface.co/t8star/Meridian-Comfy；DMD已合并，不重复加载",
                ),
                io.String.Input(
                    "vae_path",
                    default="",
                    tooltip="默认models/vae/minimax_h3_video_vae_fp16.safetensors；可改用标准video_vae插口",
                ),
                io.String.Input(
                    "source_directory",
                    default="",
                    tooltip="默认models/meridian/source；固定2083d059源码与assets，支持MERIDIAN_SOURCE_DIR",
                ),
                io.String.Input(
                    "omega_source_directory",
                    default="",
                    tooltip="授权vggt-omega源码，默认models/meridian/vggt-omega；VGGT_OMEGA_DIR",
                ),
                io.String.Input(
                    "omega_checkpoint",
                    default="",
                    tooltip="正确Omega1B512原始PT，不是普通VGGT或INT8。默认models/meridian/vggt-omega/checkpoints/vggt_omega_1b_512.pt；VGGT_OMEGA_CKPT。下载：https://huggingface.co/t8star/Meridian-Comfy；单独遵守FAIR非商用研究许可",
                ),
                io.Float.Input(
                    "cache_gib",
                    default=20.0,
                    min=0.0,
                    step=1.0,
                    tooltip="自有磁盘缓存预算，不是显存准入；0关闭磁盘缓存",
                ),
                io.String.Input("cache_directory", default=""),
                io.Model.Input("model", optional=True),
                io.Vae.Input("video_vae", optional=True),
            ],
            [RuntimeIO.Output("runtime"), io.String.Output("report_json")],
            description="独立Meridian merged-DMD原生ConvRot INT8。懒加载，不更改旧双采或用户全局优化。外部MODEL/VAE补丁保留并提示未验证，关闭其不可靠的磁盘缓存。模型及Omega下载：https://huggingface.co/t8star/Meridian-Comfy；两者许可分别适用，源码/assets及H3 VAE另备。",
        )

    @classmethod
    def execute(
        cls,
        model_path,
        vae_path,
        source_directory,
        omega_source_directory,
        omega_checkpoint,
        cache_gib,
        cache_directory,
        model=None,
        video_vae=None,
    ):
        from .meridian_runtime import config

        value = config(
            model_path,
            vae_path,
            source_directory,
            omega_source_directory,
            omega_checkpoint,
            cache_directory,
            cache_gib,
            model,
            video_vae,
        )
        report = {k: v for k, v in value.items() if k not in ("model", "video_vae")}
        return io.NodeOutput(value, json.dumps(report, ensure_ascii=False))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float(
            "nan"
        )  # Validate actual asset bytes; file edits cannot inherit an old Core cache.


class MiniMaxH3MeridianMaterialEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Meridian 2 · 素材窗口与几何（T8）",
            [
                RuntimeIO.Input("runtime"),
                io.Int.Input("start_frame", default=0, min=0, max=None),
                io.Int.Input(
                    "window_frames",
                    default=73,
                    min=1,
                    max=None,
                    tooltip="源窗口，按24fps归一；IMAGE只重建一次，不复制成73张再重建",
                ),
                io.Image.Input("image", optional=True),
                io.Video.Input("video", optional=True),
            ],
            [
                MaterialIO.Output("material"),
                io.Image.Output("source_preview"),
                io.String.Output("report_json"),
            ],
            description="只接IMAGE或VIDEO之一。保留VIDEO活动trim/crop/画面比例，按真实PTS取所选窗口。Omega联合几何一次；1280letterbox后使用训练ROI，不拉伸。大窗口可能OOM，不自动缩小或重复运行。",
        )

    @classmethod
    def execute(cls, runtime, start_frame, window_frames, image=None, video=None):
        from .meridian_runtime import prepare, editor_payload

        material = prepare(runtime, image, video, start_frame, window_frames)
        payload = editor_payload(material)
        report = dict(
            identity=material["identity"],
            kind=material["kind"],
            start=material["start"],
            end=material["end"],
            canvas=material["canvas"],
            box=material["box"],
            geometry_stage=material["receipt"],
            clock=material["clock"],
        )
        return io.NodeOutput(
            material,
            material["full"][:1].float().div(255),
            json.dumps(report, ensure_ascii=False),
            ui={"meridian_material": [payload]},
        )


class MiniMaxH3MeridianCameraEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Meridian 3 · 运镜／时间编辑器（T8）",
            [
                MaterialIO.Input("material"),
                io.Int.Input("frames", default=73, min=22, max=None),
                io.Combo.Input("preset", options=list(PRESETS), default="slide"),
                io.Float.Input(
                    "strength",
                    default=0.08,
                    step=0.01,
                    tooltip="相对固定枢轴深度的位移；orbit为strength×100度。大位移补洞更多，非性能/画质保证",
                ),
                io.String.Input(
                    "camera_plan",
                    default="",
                    multiline=True,
                    tooltip="高级编辑器自动写入规范JSON；留空使用预设。空间轨与源时间轨独立。",
                ),
            ],
            [
                PlanIO.Output("prepared"),
                io.String.Output("canonical_plan"),
                io.Image.Output("warp_preview"),
                io.String.Output("report_json"),
            ],
            description="先运行预设获得真实几何预览，再打开大编辑器。固定首源相机坐标、深度单位、roll=0；无反向源时间或自动角色跟踪。改素材/窗口后显式重置旧路径。预览是灰洞点云warp，不是生成视频。",
        )

    @classmethod
    def execute(cls, material, frames, preset, strength, camera_plan):
        from .meridian_runtime import editor_payload, warp_material
        from .meridian_generate import fixed_assets

        plan = (
            canonical_plan(camera_plan, material)
            if camera_plan.strip()
            else preset_plan(material, frames, preset, strength)
        )
        if plan["frames"] != frames:
            raise ValueError(
                "Output frames changed while a saved camera plan exists: reset or explicitly update both tracks in the editor"
            )
        # Validate paired length assets before expensive warping.
        fixed_assets(material["runtime"], plan["frames"])
        value = warp_material(material, plan)
        payload = editor_payload(material)
        payload.update(
            plan=plan,
            source_map=source_map(plan),
            coverage=value["coverage"],
            revision=identity(plan),
        )
        report = dict(
            warp_stage=value["receipt"],
            coverage_min=min(value["coverage"]),
            coverage_mean=sum(value["coverage"]) / len(value["coverage"]),
            canvas=value["canvas"],
            final_native_generation_ran=False,
        )
        return io.NodeOutput(
            value,
            plan_json(plan),
            value["render"].float().div(255),
            json.dumps(report),
            ui={"meridian_editor": [payload]},
        )


class MiniMaxH3MeridianGenerateEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Meridian 4 · 原生3前向生成（T8）",
            [
                PlanIO.Input("prepared"),
                io.Int.Input("seed", default=1234, min=0, max=2**64 - 1),
                io.Combo.Input(
                    "audio_mode",
                    options=["silent", "source_1to1"],
                    default="silent",
                    tooltip="冻结/变速选silent；source_1to1只交付原声，不认证新生成口型",
                ),
                io.String.Input("output_directory", default=""),
            ],
            [
                io.Video.Output("video"),
                io.String.Output("saved_path"),
                io.String.Output("report_json"),
            ],
            is_output_node=True,
            description="原生INT8 merged-DMD＋两路480级视频token参考＋768级训练桶。CFG1、euler、shift3/3、三次前向，原生joint AV。自动完整解码核验H264 MP4后交付。取消遵循Core普通任务中断；不改旧节点或全局队列。",
        )

    @classmethod
    def execute(cls, prepared, seed, audio_mode, output_directory):
        from .meridian_generate import generate

        return io.NodeOutput(*generate(prepared, seed, audio_mode, output_directory))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float(
            "nan"
        )  # A returned filename cannot mask a deleted/changed final delivery.


MERIDIAN_NODE_CLASSES = [
    MiniMaxH3MeridianConfigEXPT8,
    MiniMaxH3MeridianMaterialEXPT8,
    MiniMaxH3MeridianCameraEXPT8,
    MiniMaxH3MeridianGenerateEXPT8,
]
