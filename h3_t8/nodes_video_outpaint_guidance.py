"""Unregistered H3-specific regional guidance and source-person audit nodes."""
from __future__ import annotations

import json

from comfy_api.latest import io

from .nodes_video_outpaint import CATEGORY, PlanIO, PreparedIO, _interrupt, _root
from .video_outpaint_audio_runtime import prepare_outpaint_audio_cache
from .video_outpaint_compatibility_report import outpaint_model_compatibility_report
from .video_outpaint_guidance import build_outpaint_guidance, validate_outpaint_guidance
from .video_outpaint_plan import canonical
from .video_outpaint_regional import patch_outpaint_regional_model
from .video_outpaint_regional_conditioning import (
    OutpaintRegionalConditioningProvider,
    prepare_outpaint_regional_conditioning,
)
from .video_outpaint_source_runtime import prepare_outpaint_source_cache


GuidanceIO = io.Custom("T8_H3_OUTPAINT_GUIDANCE")


class MiniMaxH3VideoOutpaintGuidanceT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VideoOutpaintGuidanceT8",
            display_name="H3 Video Outpaint · 区域提示/人物保护计划 (EXP)",
            category=CATEGORY,
            is_experimental=True,
            description=(
                "区域提示按H3原生32px空间token路由；人物框使用原片坐标，只审计已锁定原片中的人物及边界风险，"
                "不声称扩出区域能绝对阻止人物幻觉。"
            ),
            inputs=[
                PlanIO.Input("plan"),
                io.String.Input("regions_json", default="[]", multiline=True),
                io.String.Input("person_boxes_json", default="[]", multiline=True),
            ],
            outputs=[GuidanceIO.Output(display_name="guidance"),
                     io.String.Output(display_name="guidance_report")],
        )

    @classmethod
    def execute(cls, plan, regions_json, person_boxes_json):
        guidance = build_outpaint_guidance(plan["plan"], regions_json, person_boxes_json)
        summary = {
            "schema": guidance["schema"],
            "plan_sha256": guidance["plan_sha256"],
            "guidance_sha256": guidance["guidance_sha256"],
            "regions_per_shot": [len(item["regions"]) for item in guidance["shots"]],
            "people_per_shot": [len(item["people"]) for item in guidance["shots"]],
            "person_boundary_risks": [
                {"shot": shot["shot_index"], "label": person["label"],
                 "boundaries": person["touches_source_boundary"]}
                for shot in guidance["shots"] for person in shot["people"]
                if person["touches_source_boundary"]
            ],
            "source_pixels_locked": True,
            "expanded_area_person_pixels_guaranteed": False,
            "human_review_required": True,
        }
        return io.NodeOutput(guidance, canonical(summary))


class MiniMaxH3VideoOutpaintPrepareGuidedT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VideoOutpaintPrepareGuidedT8",
            display_name="H3 Video Outpaint · 准备区域引导缓存 (EXP)",
            category=CATEGORY,
            is_experimental=True,
            description="独立的区域引导准备路径；不会改变普通Prepare节点或已有工作流。请使用新的run_name。",
            inputs=[
                PlanIO.Input("plan"), GuidanceIO.Input("guidance"), io.Clip.Input("clip"),
                io.Vae.Input("video_vae"), io.String.Input("prompt", default="", multiline=True),
                io.String.Input("shot_prompts_json", default="[]", multiline=True),
                io.String.Input("run_name", default="outpaint_guided_01"),
                io.Int.Input("audio_track", default=0, min=0, max=1000),
                io.Int.Input("audio_block_tokens", default=64, min=1, max=128),
                io.Boolean.Input("resume_audio", default=False),
                io.Vae.Input("audio_vae", optional=True),
            ],
            outputs=[PreparedIO.Output(display_name="prepared"),
                     io.String.Output(display_name="preparation_report")],
        )

    @classmethod
    def execute(cls, plan, guidance, clip, video_vae, prompt, shot_prompts_json, run_name,
                audio_track, audio_block_tokens, resume_audio, audio_vae=None):
        guidance = validate_outpaint_guidance(guidance, plan["plan"])
        root = _root(plan, run_name)
        if plan["inspection"]["audio_pcm"] and audio_vae is None:
            raise ValueError("this source has audio: connect the native H3 audio VAE")
        prompts = json.loads(shot_prompts_json)
        if not isinstance(prompts, list):
            raise ValueError("shot_prompts_json must be an array")
        if prompts and (len(prompts) != len(plan["plan"]["shots"])
                        or any(not isinstance(item, str) for item in prompts)):
            raise ValueError("shot_prompts_json must contain one string per planned shot")
        tracks = len(plan["inspection"]["audio_pcm"])
        if isinstance(audio_track, bool) or not isinstance(audio_track, int) or not 0 <= audio_track < max(1, tracks):
            raise ValueError("selected audio track does not exist in this source")
        source, source_report = prepare_outpaint_source_cache(
            video_vae, plan["inspection"], plan["plan"], root / "source", interrupt_check=_interrupt)
        audio, audio_report = prepare_outpaint_audio_cache(
            audio_vae, plan["inspection"], plan["plan"], root / "audio",
            stream_position=audio_track, block_tokens=audio_block_tokens,
            resume=resume_audio, interrupt_check=_interrupt)
        text = prepare_outpaint_regional_conditioning(
            clip, prompts or prompt, guidance, plan["plan"], root / "text", interrupt_check=_interrupt)
        handle = {**plan, "root": root, "source": source, "audio": audio,
                  "conditioning": text, "guidance": guidance}
        report = {
            "source": source_report,
            "audio": audio_report,
            "regional_conditioning_sha256": text.verify(),
            "binding_sha256": text.binding["binding_sha256"],
            "generated_video_complete": False,
            "ordinary_prepare_modified": False,
        }
        return io.NodeOutput(handle, canonical(report))


class MiniMaxH3VideoOutpaintRegionalModelT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VideoOutpaintRegionalModelT8",
            display_name="H3 Video Outpaint · 区域引导模型 (EXP)",
            category=CATEGORY,
            is_experimental=True,
            description=(
                "把已准备的区域提示绑定到原生H3 MODEL；兼容已审计KJ低显存Attention/FFN，"
                "拒绝LoRA、其他Attention所有者和未知wrapper静默叠加。"
            ),
            inputs=[io.Model.Input("model"), PreparedIO.Input("prepared"),
                    io.Int.Input("query_chunk_rows", default=256, min=32, max=2048)],
            outputs=[io.Model.Output(display_name="model"),
                     PreparedIO.Output(display_name="prepared"),
                     io.String.Output(display_name="regional_model_report")],
        )

    @classmethod
    def execute(cls, model, prepared, query_chunk_rows):
        conditioning = prepared.get("conditioning")
        if not isinstance(conditioning, OutpaintRegionalConditioningProvider):
            raise ValueError("connect the output of Prepare Regional Guidance")
        patched, contract = patch_outpaint_regional_model(model, conditioning, query_chunk_rows)
        report = {
            **contract,
            "directly_routed_queries": "target_video_only",
            "audio_directly_routed": False,
            "dense_s_by_s_mask_created": False,
            "semantics_cannot_leak_indirectly_guaranteed": False,
            "human_review_required": True,
        }
        return io.NodeOutput(patched, prepared, canonical(report))


class MiniMaxH3VideoOutpaintCompatibilityAuditT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VideoOutpaintCompatibilityAuditT8",
            display_name="H3 Video Outpaint · 模型组合审计 (EXP)",
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            description=(
                "读取真实MODEL内容与patch/wrapper，不按文件名猜。Stock20与已锁定KJ低显存补丁可用；"
                "Turbo/SPEED/SLA/VDN/Fast H3/Prompt Relay尚无扩画组合验收时明确标为不支持。"
            ),
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output(display_name="model"), io.Boolean.Output(display_name="ready"),
                     io.String.Output(display_name="compatibility_report")],
        )

    @classmethod
    def execute(cls, model):
        report = outpaint_model_compatibility_report(model)
        return io.NodeOutput(model, report["ready"], canonical(report))


VIDEO_OUTPAINT_GUIDANCE_DRAFT_NODE_CLASSES = [
    MiniMaxH3VideoOutpaintGuidanceT8,
    MiniMaxH3VideoOutpaintPrepareGuidedT8,
    MiniMaxH3VideoOutpaintRegionalModelT8,
    MiniMaxH3VideoOutpaintCompatibilityAuditT8,
]
