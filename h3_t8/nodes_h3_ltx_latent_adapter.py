"""Append-only EXP standard-LATENT entry; no optional assets loaded at registration."""
import json

from comfy_api.latest import io

from .h3_ltx_adapter_runtime import owned_adapter, inspect_assets, MODEL_SHA, MODEL_REVISION, SOURCE_REVISION
from .h3_ltx_latent_contract import FRAME_POLICIES, convert_video_latent, timeline, extract_video


class MiniMaxH3LTXLatentAdapterEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3LTXLatentAdapterEXPT8",
            display_name="H3 → LTX 视频潜空间 / Learned Adapter (T8 EXP)",
            category="T8/MiniMax H3/Latent", is_experimental=True,
            description="仅转换H3视频潜空间；原AV原样另行输出。接已有learned放大之后，不再额外放大。LTX音频/条件不自动转换。",
            inputs=[io.Latent.Input("h3_latent"),
                io.Int.Input("source_frames", default=73, min=5, tooltip="实际H3输出帧数17n+5，不从潜空间猜时长。"),
                io.Float.Input("source_fps", default=24, min=1, tooltip="当前配方仅24fps；不自动重采样。"),
                io.Combo.Input("frame_policy", options=list(FRAME_POLICIES), default="exact",
                               tooltip="exact保持帧数；124帧需显式选pad→129或crop→121，原音需另行对应处理。"),
                io.String.Input("source_directory", default="", tooltip="固定Sana源码中h3_ltx_adapter目录，非整个仓库。"),
                io.String.Input("model_directory", default="", tooltip="包含官方config.json和model.safetensors的目录。"),
                io.Combo.Input("device", options=["cpu", "cuda"], default="cpu"),
                io.Combo.Input("precision", options=["float32", "bfloat16"], default="float32"),
                io.Int.Input("reference_prefix_latents", default=0, min=0, advanced=True),
                io.Combo.Input("normalization", options=["comfy_normalized", "raw_h3"], default="comfy_normalized", advanced=True)],
            outputs=[io.Latent.Output("ltx_video_latent"), io.Latent.Output("original_h3_av"),
                io.Int.Output("output_frames"), io.Float.Output("output_fps"), io.String.Output("report_json")])

    @classmethod
    def fingerprint_inputs(cls, source_directory, model_directory, **kwargs):
        inspect_assets(source_directory, model_directory)
        return MODEL_SHA + ":" + SOURCE_REVISION

    @classmethod
    def execute(cls, h3_latent, source_frames=73, source_fps=24, frame_policy="exact",
                source_directory="", model_directory="", device="cpu", precision="float32",
                reference_prefix_latents=0, normalization="comfy_normalized"):
        from comfy.model_management import throw_exception_if_processing_interrupted
        check = throw_exception_if_processing_interrupted
        check()
        grid = timeline(source_frames, source_fps, frame_policy)
        extract_video(h3_latent, expected_frames=grid["h3_latent_frames"],
                      reference_prefix_latents=reference_prefix_latents)
        with owned_adapter(source_directory, model_directory, device=device, precision=precision, check_cancel=check) as adapter:
            result, original, report = convert_video_latent(h3_latent, adapter=adapter,
                source_frames=source_frames, fps=source_fps, frame_policy=frame_policy,
                reference_prefix_latents=reference_prefix_latents, normalization=normalization, check_cancel=check)
        report.update(model_sha256=MODEL_SHA, model_revision=MODEL_REVISION,
                      source_revision=SOURCE_REVISION, adapter_device=device, adapter_precision=precision)
        return io.NodeOutput(result, original, report["output_frames"], float(report["output_fps"]),
                             json.dumps(report, ensure_ascii=False, sort_keys=True))
