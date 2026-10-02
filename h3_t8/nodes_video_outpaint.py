"""Draft V3 outpaint nodes. Not registered until integration/release gates pass.

The four stages are explicit; saving does not hide a deferred sampling generator.
Old node registrations and workflows are deliberately unchanged by this module.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import folder_paths
import comfy.model_management
from comfy_api.latest import io

from .video_outpaint_audio_runtime import prepare_outpaint_audio_cache
from .video_outpaint_conditioning import prepare_outpaint_conditioning
from .video_outpaint_compose import compose_sampled_outpaint
from .video_outpaint_execution import sample_verified_outpaint
from .video_outpaint_media import inspect_outpaint_source, validate_outpaint_source
from .video_outpaint_plan import ASPECTS, WINDOW_FRAMES, build_outpaint_plan, canonical
from .video_outpaint_source_runtime import prepare_outpaint_source_cache


PlanIO = io.Custom("T8_H3_OUTPAINT_PLAN")
PreparedIO = io.Custom("T8_H3_OUTPAINT_PREPARED")
SampledIO = io.Custom("T8_H3_OUTPAINT_SAMPLED")
CATEGORY = "T8/MiniMax H3/Video Outpaint EXP"


def _name(value):
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value)
            or value.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}):
        raise ValueError("run/output name must be 1..64 letters, digits, underscores or hyphens, not a device name")
    return value


def _root(handle, run_name):
    plan = handle["plan"]
    validate_outpaint_source(handle["inspection"], plan)
    base = Path(folder_paths.get_output_directory()).resolve() / "T8_H3_Outpaint_Cache"
    return base / f"{_name(run_name)}-{plan['plan_sha256'][:16]}"


def _interrupt():
    comfy.model_management.throw_exception_if_processing_interrupted()


class MiniMaxH3VideoOutpaintPlanT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VideoOutpaintPlanT8", display_name="H3 Video Outpaint · Plan (EXP)",
            category=CATEGORY, is_experimental=True,
            inputs=[io.Video.Input("source_video"), io.Combo.Input("aspect", options=list(ASPECTS), default="9:16"),
                io.Int.Input("left", default=0, min=0, max=32768), io.Int.Input("top", default=0, min=0, max=32768),
                io.Int.Input("right", default=0, min=0, max=32768), io.Int.Input("bottom", default=0, min=0, max=32768),
                io.Float.Input("anchor_x", default=0.5, min=0, max=1), io.Float.Input("anchor_y", default=0.5, min=0, max=1),
                io.Float.Input("generation_megapixels", default=0.5, min=0.01, max=64),
                io.Combo.Input("window_frames", options=[str(v) for v in WINDOW_FRAMES], default="73"),
                io.String.Input("cut_frames_json", default="[]", multiline=True)],
            outputs=[PlanIO.Output(display_name="plan"), io.String.Output(display_name="plan_report")])

    @classmethod
    def execute(cls, source_video, aspect, left, top, right, bottom, anchor_x, anchor_y,
                generation_megapixels, window_frames, cut_frames_json):
        inspection = inspect_outpaint_source(source_video)
        cuts = json.loads(cut_frames_json)
        if not isinstance(cuts, list):
            raise ValueError("cut_frames_json must be an array of absolute source frame indices")
        plan = build_outpaint_plan(source_sha256=inspection["sha256"], width=inspection["width"],
            height=inspection["height"], frame_count=inspection["frames"], source_fps=inspection["fps"],
            aspect=aspect, left=left, top=top, right=right, bottom=bottom, anchor_x=anchor_x, anchor_y=anchor_y,
            generation_megapixels=generation_megapixels, window_frames=int(window_frames), cut_frames=tuple(cuts))
        return io.NodeOutput({"plan": plan, "inspection": inspection}, canonical(plan))


class MiniMaxH3VideoOutpaintPrepareT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VideoOutpaintPrepareT8", display_name="H3 Video Outpaint · Prepare (EXP)",
            category=CATEGORY, is_experimental=True,
            inputs=[PlanIO.Input("plan"), io.Clip.Input("clip"), io.Vae.Input("video_vae"),
                io.String.Input("prompt", default="", multiline=True),
                io.String.Input("shot_prompts_json", default="[]", multiline=True),
                io.String.Input("run_name", default="outpaint_01"),
                io.Int.Input("audio_track", default=0, min=0, max=1000),
                io.Int.Input("audio_block_tokens", default=64, min=1, max=128),
                io.Boolean.Input("resume_audio", default=False), io.Vae.Input("audio_vae", optional=True)],
            outputs=[PreparedIO.Output(display_name="prepared"), io.String.Output(display_name="preparation_report")])

    @classmethod
    def execute(cls, plan, clip, video_vae, prompt, shot_prompts_json, run_name, audio_track,
                audio_block_tokens, resume_audio, audio_vae=None):
        root = _root(plan, run_name)
        if plan["inspection"]["audio_pcm"] and audio_vae is None:
            raise ValueError("this source has audio: connect the native H3 audio VAE; audio will not be silently discarded")
        prompts = json.loads(shot_prompts_json)
        if not isinstance(prompts, list):
            raise ValueError("shot_prompts_json must be an array, or [] to reuse prompt for all shots")
        if prompts and (len(prompts) != len(plan["plan"]["shots"]) or any(not isinstance(p, str) for p in prompts)):
            raise ValueError("shot_prompts_json must contain one string per planned shot")
        tracks = len(plan["inspection"]["audio_pcm"])
        if isinstance(audio_track, bool) or not isinstance(audio_track, int) or not 0 <= audio_track < max(1, tracks):
            raise ValueError("selected audio track does not exist in this source")
        source, source_report = prepare_outpaint_source_cache(video_vae, plan["inspection"], plan["plan"],
            root / "source", interrupt_check=_interrupt)
        audio, audio_report = prepare_outpaint_audio_cache(audio_vae, plan["inspection"], plan["plan"],
            root / "audio", stream_position=audio_track, block_tokens=audio_block_tokens,
            resume=resume_audio, interrupt_check=_interrupt)
        text = prepare_outpaint_conditioning(clip, prompts or prompt, plan["plan"], root / "text", interrupt_check=_interrupt)
        handle = {**plan, "root": root, "source": source, "audio": audio, "conditioning": text}
        report = {"source": source_report, "audio": audio_report, "conditioning_sha256": text.verify(),
                  "generated_video_complete": False}
        return io.NodeOutput(handle, canonical(report))


class MiniMaxH3VideoOutpaintSampleT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VideoOutpaintSampleT8", display_name="H3 Video Outpaint · Serial Sample (EXP)",
            category=CATEGORY, is_experimental=True,
            description="Draft native MODEL path, including audited KJ memory patches. Other acceleration adapters remain pending.",
            inputs=[io.Model.Input("model"), PreparedIO.Input("prepared"),
                io.Int.Input("seed", default=20260808, min=0, max=2**64-1, control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=100), io.Boolean.Input("resume", default=False),
                io.Combo.Input("noise_algorithm", options=["t8.outpaint.coordinate_noise/v1", "t8.outpaint.native_cpu_noise/v1"],
                    default="t8.outpaint.coordinate_noise/v1", optional=True)],
            outputs=[SampledIO.Output(display_name="sampled"), io.String.Output(display_name="sampling_report")])

    @classmethod
    def execute(cls, model, prepared, seed, steps, resume, noise_algorithm="t8.outpaint.coordinate_noise/v1"):
        validate_outpaint_source(prepared["inspection"], prepared["plan"])
        store, report = sample_verified_outpaint(model=model, conditioning=prepared["conditioning"],
            source_store=prepared["source"], audio=prepared["audio"], cache_root=prepared["root"] / "sampling",
            seed=seed, steps=steps, resume=resume, interrupt_check=_interrupt, noise_algorithm=noise_algorithm)
        return io.NodeOutput({"prepared": prepared, "windows": store}, canonical(report))


class MiniMaxH3VideoOutpaintComposeT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VideoOutpaintComposeT8", display_name="H3 Video Outpaint · Compose (EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            inputs=[SampledIO.Input("sampled"), io.Vae.Input("video_vae"),
                io.String.Input("output_name", default="outpaint"), io.Boolean.Input("color_match", default=True),
                io.Boolean.Input("geometry_align", default=False, optional=True,
                    tooltip="实验：校正扩画侧轮廓，原片像素不变；需OpenCV，未人审前不默认开启。"),
                io.Combo.Input("source_mode", options=["joint_decode", "preserve_source"], default="joint_decode", optional=True,
                    tooltip="默认联合解码：原片也经过VAE重建，跳过接缝修色/几何校正；原音频不变。preserve_source可选精确保留原片像素，但接缝可能更明显。")],
            outputs=[io.Video.Output(display_name="video"), io.String.Output(display_name="delivery_report")])

    @classmethod
    def execute(cls, sampled, video_vae, output_name, color_match, geometry_align=False, source_mode="joint_decode"):
        prepared = sampled["prepared"]
        name = _name(output_name)
        root = Path(folder_paths.get_output_directory()).resolve() / "T8_H3_Outpaint"
        counter = 1
        while ((root / f"{name}_{counter:05d}.mp4").exists()
               or (root / f"{name}_{counter:05d}.mp4.outpaint.json").exists()):
            counter += 1
        target = root / f"{name}_{counter:05d}.mp4"
        video, report = compose_sampled_outpaint(vae=video_vae, inspection=prepared["inspection"],
            source_store=prepared["source"], window_store=sampled["windows"], output_path=target,
            color_match=color_match, geometry_align=geometry_align, source_mode=source_mode, interrupt_check=_interrupt)
        return io.NodeOutput(video, canonical(report))


VIDEO_OUTPAINT_DRAFT_NODE_CLASSES = [MiniMaxH3VideoOutpaintPlanT8, MiniMaxH3VideoOutpaintPrepareT8,
                                    MiniMaxH3VideoOutpaintSampleT8, MiniMaxH3VideoOutpaintComposeT8]
