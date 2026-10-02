from __future__ import annotations

from pathlib import Path

import folder_paths
import torch
from comfy_api.latest import io, ui

from .dlss_nr_advanced import (
    MINIMUM_FREE_VRAM_MIB,
    MOTION_ENGINES,
    PROCESSING_MODES,
    PROBE_MODES,
    QUALITY_PROFILE_NAMES,
    RUNTIME_HANDLE_SCHEMA,
    SR_PRESETS,
    SUPPORTED_SCALES,
    audit_dlss_nr_runtime,
    available_runtime_versions,
    canonical_json,
    process_image_batch,
    process_video_file,
    process_video_frame_batch,
    revalidate_runtime_handle,
    runtime_root,
    runtime_handle_from_report,
    target_dimensions,
)


CATEGORY = "T8/MiniMax H3/Post FX/DLSS-NR"
DLSSNRRuntimeIO = io.Custom("T8_DLSS_NR_RUNTIME")


def _runtime_versions() -> list[str]:
    return available_runtime_versions(folder_paths.models_dir)


class MiniMaxH3DLSSNRRuntimeAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        versions = _runtime_versions()
        return io.Schema(
            node_id="MiniMaxH3DLSSNRRuntimeAuditT8Advanced",
            display_name="MiniMax H3 DLSS-NR Runtime Audit (T8 Advanced)",
            description=(
                "Read-only Windows/RTX audit for a user-supplied, allowlisted DLSS-NR runtime. "
                "It verifies the official release archive, installed EXE/DLL byte identity, "
                "driver, free VRAM and CUDA/DXGI adapter mapping. It never downloads, installs, "
                "unloads models, upgrades drivers or processes media in static-only mode."
            ),
            category=CATEGORY,
            is_experimental=False,
            is_output_node=True,
            inputs=[
                io.Combo.Input(
                    "runtime_version",
                    options=versions,
                    default=versions[-1],
                ),
                io.Combo.Input(
                    "probe_mode",
                    options=list(PROBE_MODES),
                    default="feature_probe_1_frame",
                    tooltip="feature_probe_1_frame：实际运行一帧自检，通过后可接超分。static_only：仅检查文件和环境，不输出可运行句柄。",
                ),
                io.Int.Input(
                    "dxgi_adapter_index", default=0, min=0, max=31, advanced=True,
                    tooltip="外部程序使用的 Windows 显卡编号，不是显存大小。单显卡一般用 0；必须与 CUDA 选择对应同一张卡。"
                ),
                io.Int.Input(
                    "cuda_device_index", default=0, min=0, max=31, advanced=True,
                    tooltip="ComfyUI/PyTorch 使用的 CUDA 设备编号。单显卡一般用 0；不是 Windows DXGI 编号的同义词。"
                ),
            ],
            outputs=[
                DLSSNRRuntimeIO.Output("dlss_nr_runtime"),
                io.Boolean.Output("ready"),
                io.String.Output("status"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        runtime_version,
        accept_external_runtime_license=None,
        probe_mode="feature_probe_1_frame",
        dxgi_adapter_index=0,
        cuda_device_index=0,
    ):
        root = runtime_root(folder_paths.models_dir, runtime_version)
        ready, report = audit_dlss_nr_runtime(
            root,
            runtime_version,
            accept_external_runtime_license=accept_external_runtime_license,
            probe_mode=probe_mode,
            dxgi_adapter_index=dxgi_adapter_index,
            cuda_device_index=cuda_device_index,
        )
        return io.NodeOutput(
            runtime_handle_from_report(report),
            ready,
            report["status"],
            canonical_json(report, indent=2),
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


def _scale(value) -> float:
    scale = float(value)
    if scale not in SUPPORTED_SCALES:
        raise ValueError(f"unsupported DLSS-NR scale: {value!r}")
    return scale


def _prepare_runtime(runtime) -> dict:
    if not isinstance(runtime, dict) or runtime.get("schema") != RUNTIME_HANDLE_SCHEMA:
        raise ValueError("connect a READY output from MiniMax H3 DLSS-NR Runtime Audit")
    from comfy import model_management

    device = torch.device(f"cuda:{int(runtime['cuda_device_index'])}")
    model_management.free_memory(
        int(MINIMUM_FREE_VRAM_MIB * 1024 * 1024),
        device,
    )
    return revalidate_runtime_handle(runtime)


def _quality_inputs():
    inputs = [
        io.Combo.Input(
            "quality_profile",
            options=list(QUALITY_PROFILE_NAMES),
            default="standard",
            tooltip=(
                "standard matches the reference ComfyUI wrapper. Named profiles override "
                "the advanced manual controls; choose custom to use those controls."
            ),
        ),
        io.Combo.Input(
            "sr_preset",
            options=list(SR_PRESETS),
            default="default",
            tooltip=(
                "DLSS Super Resolution network. default lets the v1.3 driver choose; "
                "K is DLSS 4 Transformer and L/M are newer DLSS 4.5 options."
            ),
        ),
        io.Combo.Input(
            "nr_style",
            options=["0 Default", "1 Natural", "2 Cinematic"],
            default="0 Default",
            advanced=True,
        ),
        io.Combo.Input(
            "nr_preset",
            options=["0 Default", "1", "2", "3"],
            default="0 Default",
            advanced=True,
        ),
        io.Float.Input(
            "nr_intensity", default=1.5, min=0.0, max=2.0, step=0.05, advanced=True
        ),
        io.Float.Input(
            "nr_detail", default=1.0, min=0.0, max=1.0, step=0.05, advanced=True
        ),
        io.Float.Input(
            "nr_color", default=1.0, min=0.0, max=1.0, step=0.05, advanced=True
        ),
        io.Float.Input(
            "nr_skin", default=-1.0, min=-1.0, max=2.0, step=0.05, advanced=True
        ),
        io.Float.Input(
            "nr_local_structure",
            default=1.0,
            min=0.0,
            max=2.0,
            step=0.05,
            advanced=True,
        ),
        io.Float.Input(
            "nr_local_tone", default=1.0, min=0.0, max=2.0, step=0.05, advanced=True
        ),
        io.Float.Input(
            "nr_global_tone",
            default=-1.0,
            min=-1.0,
            max=2.0,
            step=0.05,
            advanced=True,
        ),
        io.Boolean.Input("nr_ui_correction", default=False, advanced=True),
        io.Boolean.Input("nr_auto_mask", default=False, advanced=True),
    ]
    tooltips = {
        "quality_profile": "画质方案。命名预设覆盖下方所有 nr_* 手动值；仅 custom 使用手动值。 Named profiles override NR sliders; choose custom to edit.",
        "sr_preset": "SR 超分网络预设，default 由运行时选择；字母不是质量等级。 SR model preset, not a quality ranking.",
        "nr_style": "NR 风格：0 默认、1 自然、2 电影感。仅 custom 生效。",
        "nr_preset": "NR 内部预设编号，不是放大倍数或步数；通常保留 0。仅 custom 生效。",
        "nr_intensity": "神经渲染增强强度，越大不一定越自然。仅 custom 生效。",
        "nr_detail": "NR 合成占比：0 不混入 NR，1 完整采用 NR。sr_only 固定为 0。仅 custom 生效。",
        "nr_color": "采用 NR 色彩变化的程度；偏色时可降低。仅 custom 生效。",
        "nr_skin": "皮肤结构控制，-1 使用运行时默认值。仅 custom 生效。",
        "nr_local_structure": "局部结构和细纹理的调整强度。仅 custom 生效。",
        "nr_local_tone": "局部光影和对比的调整强度。仅 custom 生效。",
        "nr_global_tone": "整体色调控制，-1 使用运行时默认值。仅 custom 生效。",
        "nr_ui_correction": "UI/界面元素修正选项，不是 ComfyUI 界面开关。仅 custom 生效。",
        "nr_auto_mask": "自动遮罩，针对文字/UI 等区域；不保证所有字幕都不变。仅 custom 生效。",
    }
    for item in inputs:
        item.tooltip = tooltips[item.id]
    return inputs


def _manual_parameters(
    nr_style,
    nr_preset,
    nr_intensity,
    nr_detail,
    nr_color,
    nr_skin,
    nr_local_structure,
    nr_local_tone,
    nr_global_tone,
    nr_ui_correction,
    nr_auto_mask,
) -> dict:
    return {
        "nr_style": int(str(nr_style).split()[0]),
        "nr_preset": int(str(nr_preset).split()[0]),
        "nr_intensity": float(nr_intensity),
        "nr_detail": float(nr_detail),
        "nr_color": float(nr_color),
        "nr_skin": float(nr_skin),
        "nr_local_structure": float(nr_local_structure),
        "nr_local_tone": float(nr_local_tone),
        "nr_global_tone": float(nr_global_tone),
        "nr_ui_correction": bool(nr_ui_correction),
        "nr_auto_mask": bool(nr_auto_mask),
    }


class MiniMaxH3DLSSNRImageSuperResolutionT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DLSSNRImageSuperResolutionT8Advanced",
            display_name="MiniMax H3 DLSS-NR Image / 图片超分 (T8 Advanced)",
            description=(
                "Runs each SDR IMAGE independently through the audited external runtime. "
                "Supports 1x NR-only or 1.5x/2x/3x SR-only and SR+NR. RGB uses an explicit "
                "8-bit RGBA bridge; alpha and extra channels are resized from the source and "
                "reattached. Standard matches the reference wrapper; custom exposes every "
                "NR control and v1.3 SR model selector."
            ),
            category=CATEGORY,
            is_experimental=False,
            inputs=[
                DLSSNRRuntimeIO.Input("dlss_nr_runtime"),
                io.Image.Input("images"),
                io.Combo.Input("mode", options=list(PROCESSING_MODES), default="sr_nr"),
                io.Combo.Input(
                    "scale",
                    options=[_format_scale(value) for value in SUPPORTED_SCALES],
                    default="2.0",
                ),
                *_quality_inputs(),
            ],
            outputs=[
                io.Image.Output("candidate"),
                io.Image.Output("source"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        dlss_nr_runtime,
        images,
        mode,
        scale,
        quality_profile,
        sr_preset,
        nr_style,
        nr_preset,
        nr_intensity,
        nr_detail,
        nr_color,
        nr_skin,
        nr_local_structure,
        nr_local_tone,
        nr_global_tone,
        nr_ui_correction,
        nr_auto_mask,
    ):
        revalidation = _prepare_runtime(dlss_nr_runtime)
        candidate, source, report = process_image_batch(
            dlss_nr_runtime,
            images,
            mode=mode,
            scale=_scale(scale),
            quality_profile=quality_profile,
            sr_preset=sr_preset,
            manual_parameters=_manual_parameters(
                nr_style,
                nr_preset,
                nr_intensity,
                nr_detail,
                nr_color,
                nr_skin,
                nr_local_structure,
                nr_local_tone,
                nr_global_tone,
                nr_ui_correction,
                nr_auto_mask,
            ),
        )
        report["runtime_revalidation"] = revalidation
        return io.NodeOutput(candidate, source, canonical_json(report, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


def _format_scale(value: float) -> str:
    return f"{float(value):.1f}"


class MiniMaxH3DLSSNRVideoFramesT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DLSSNRVideoFramesT8Advanced",
            display_name="MiniMax H3 DLSS-NR Video Frames / 帧序列超分 (T8 Advanced)",
            description=(
                "Streams an ordered IMAGE batch through one persistent external process so "
                "temporal NR state is retained. It does not interpolate or reorder frames and "
                "returns the exact same AUDIO object. Partial pipes, crashes and cancellation fail."
            ),
            category=CATEGORY,
            is_experimental=False,
            inputs=[
                DLSSNRRuntimeIO.Input("dlss_nr_runtime"),
                io.Image.Input("frames"),
                io.Float.Input("fps", default=24.0, min=0.001, max=240.0, step=0.001),
                io.Audio.Input("audio", optional=True),
                io.Combo.Input("mode", options=list(PROCESSING_MODES), default="sr_nr"),
                io.Combo.Input(
                    "scale",
                    options=[_format_scale(value) for value in SUPPORTED_SCALES],
                    default="2.0",
                ),
                *_quality_inputs(),
                io.Combo.Input(
                    "motion_engine", options=list(MOTION_ENGINES), default="auto"
                ),
            ],
            outputs=[
                io.Image.Output("candidate_frames"),
                io.Image.Output("source_frames"),
                io.Audio.Output("audio"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        dlss_nr_runtime,
        frames,
        fps,
        mode,
        scale,
        quality_profile,
        sr_preset,
        nr_style,
        nr_preset,
        nr_intensity,
        nr_detail,
        nr_color,
        nr_skin,
        nr_local_structure,
        nr_local_tone,
        nr_global_tone,
        nr_ui_correction,
        nr_auto_mask,
        motion_engine,
        audio=None,
    ):
        revalidation = _prepare_runtime(dlss_nr_runtime)
        candidate, source, audio_output, report = process_video_frame_batch(
            dlss_nr_runtime,
            frames,
            fps=float(fps),
            audio=audio,
            mode=mode,
            scale=_scale(scale),
            motion_engine=motion_engine,
            quality_profile=quality_profile,
            sr_preset=sr_preset,
            manual_parameters=_manual_parameters(
                nr_style,
                nr_preset,
                nr_intensity,
                nr_detail,
                nr_color,
                nr_skin,
                nr_local_structure,
                nr_local_tone,
                nr_global_tone,
                nr_ui_correction,
                nr_auto_mask,
            ),
        )
        report["runtime_revalidation"] = revalidation
        return io.NodeOutput(
            candidate,
            source,
            audio_output,
            canonical_json(report, indent=2),
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


def _output_path(
    filename_prefix: str, width: int, height: int
) -> tuple[Path, str, str]:
    full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        str(filename_prefix), folder_paths.get_output_directory(), width, height
    )
    path = Path(full_folder) / f"{filename}_{counter:05}_.mp4"
    return path, path.name, subfolder


class MiniMaxH3DLSSNRVideoFileT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DLSSNRVideoFileT8Advanced",
            display_name="MiniMax H3 DLSS-NR Video File / 文件视频超分 (T8 Advanced)",
            description=(
                "Streams an untrimmed SDR 8-bit Rec.709 CFR file-backed VIDEO without building "
                "a full IMAGE batch. Candidate video is encoded separately, source audio packets "
                "are copied and packet/PCM checked, and publication is atomic. HDR, VFR, rotation, "
                "crop metadata, odd target sizes and unsupported audio fail closed."
            ),
            category=CATEGORY,
            is_experimental=False,
            is_output_node=True,
            inputs=[
                DLSSNRRuntimeIO.Input("dlss_nr_runtime"),
                io.Video.Input("source_video"),
                io.Combo.Input("mode", options=list(PROCESSING_MODES), default="sr_nr"),
                io.Combo.Input(
                    "scale",
                    options=[_format_scale(value) for value in SUPPORTED_SCALES],
                    default="2.0",
                ),
                *_quality_inputs(),
                io.Combo.Input(
                    "motion_engine", options=list(MOTION_ENGINES), default="auto"
                ),
                io.String.Input(
                    "filename_prefix", default="MiniMaxH3/DLSS-NR/dlss_nr_candidate"
                ),
                io.Float.Input("crf", default=18.0, min=0.0, max=51.0, step=1.0),
            ],
            outputs=[
                io.Video.Output("candidate_video"),
                io.Video.Output("source_video"),
                io.String.Output("saved_path"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        dlss_nr_runtime,
        source_video,
        mode,
        scale,
        quality_profile,
        sr_preset,
        nr_style,
        nr_preset,
        nr_intensity,
        nr_detail,
        nr_color,
        nr_skin,
        nr_local_structure,
        nr_local_tone,
        nr_global_tone,
        nr_ui_correction,
        nr_auto_mask,
        motion_engine,
        filename_prefix,
        crf,
    ):
        from comfy_api.latest import InputImpl

        revalidation = _prepare_runtime(dlss_nr_runtime)
        width, height = map(int, source_video.get_dimensions())
        target_width, target_height = target_dimensions(width, height, _scale(scale))
        output, saved_name, subfolder = _output_path(
            filename_prefix, target_width, target_height
        )
        published, source, report = process_video_file(
            dlss_nr_runtime,
            source_video,
            output_path=output,
            mode=mode,
            scale=_scale(scale),
            motion_engine=motion_engine,
            quality_profile=quality_profile,
            sr_preset=sr_preset,
            manual_parameters=_manual_parameters(
                nr_style,
                nr_preset,
                nr_intensity,
                nr_detail,
                nr_color,
                nr_skin,
                nr_local_structure,
                nr_local_tone,
                nr_global_tone,
                nr_ui_correction,
                nr_auto_mask,
            ),
            crf=float(crf),
        )
        report["runtime_revalidation"] = revalidation
        candidate = InputImpl.VideoFromFile(str(published))
        preview = ui.PreviewVideo(
            [ui.SavedResult(saved_name, subfolder, io.FolderType.output)]
        )
        return io.NodeOutput(
            candidate,
            source,
            str(published),
            canonical_json(report, indent=2),
            ui=preview,
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


DLSS_NR_ADVANCED_NODE_CLASSES = [
    MiniMaxH3DLSSNRRuntimeAuditT8Advanced,
    MiniMaxH3DLSSNRImageSuperResolutionT8Advanced,
    MiniMaxH3DLSSNRVideoFramesT8Advanced,
    MiniMaxH3DLSSNRVideoFileT8Advanced,
]
