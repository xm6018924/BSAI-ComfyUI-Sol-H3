"""Unregistered native candidate/selection/continuation drafts.

Candidate is an output node so it can be run independently. Connect selection
and continuation only after inspecting its image; selection defaults unapproved.
No existing registered nodes or workflows are changed by importing this module.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import folder_paths
from comfy_api.latest import io, ui

from .nodes_video_outpaint import PreparedIO, CATEGORY, _interrupt, _name
from .video_outpaint_candidate_execution import sample_verified_first_candidate, continue_verified_candidate
from .video_outpaint_candidate_preview import render_candidate_first_frame
from .video_outpaint_candidates import select_candidate
from .video_outpaint_candidate_archive import (validate_archived_preview, save_candidate_archive,
    load_candidate_archive, save_selection_archive, load_selection_archive)
from .video_outpaint_compose import compose_sampled_outpaint
from .video_outpaint_media import validate_outpaint_source
from .video_outpaint_noise import NATIVE_NOISE
from .video_outpaint_plan import canonical
from .video_outpaint_pixel_receipt import validate_source_mode


CandidateIO = io.Custom("T8_H3_OUTPAINT_CANDIDATE")
SelectedIO = io.Custom("T8_H3_OUTPAINT_SELECTED_CANDIDATE")
SelectedSampledIO = io.Custom("T8_H3_OUTPAINT_SELECTED_SAMPLED")


def _preview_contract(handle):
    return validate_archived_preview(handle)


class MiniMaxH3VideoOutpaintCandidateT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VideoOutpaintCandidateT8",
            display_name="H3 Video Outpaint · 生成首帧候选 (EXP)", category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description="只生成首个窗口并显示真实首帧。先单独运行本节点审图，再连接选择/接续。候选名或生成设置改变需使用新缓存。",
            inputs=[io.Model.Input("model"), PreparedIO.Input("prepared"), io.Vae.Input("video_vae"),
                io.String.Input("candidate_name", default="candidate_01"),
                io.Int.Input("seed", default=20260808, min=0, max=2**64-1),
                io.Int.Input("steps", default=20, min=1, max=100),
                io.Boolean.Input("resume", default=False), io.Boolean.Input("color_match", default=True),
                io.Boolean.Input("geometry_align", default=False, optional=True,
                    tooltip="实验：只校正扩区接缝几何，原片像素不变；需OpenCV，关闭不加载。"),
                io.Combo.Input("source_mode", options=["joint_decode", "preserve_source"], default="joint_decode", optional=True,
                    tooltip="默认联合解码会重建原片像素，跳过接缝修色/几何；可选preserve_source保留原片。选择后保存沿用候选模式。")],
            outputs=[CandidateIO.Output(display_name="candidate"), io.Image.Output(display_name="first_frame"),
                     io.String.Output(display_name="candidate_report"), io.String.Output(display_name="candidate_id")])

    @classmethod
    def execute(cls, model, prepared, video_vae, candidate_name, seed, steps, resume, color_match, geometry_align=False, source_mode="joint_decode"):
        validate_source_mode(source_mode)
        validate_outpaint_source(prepared["inspection"], prepared["plan"])
        root = prepared["root"] / "candidates" / _name(candidate_name)
        settings = {"seed": seed, "steps": steps, "noise_algorithm": NATIVE_NOISE}
        store, candidate, sampled = sample_verified_first_candidate(model=model,
            conditioning=prepared["conditioning"], source_store=prepared["source"], audio=prepared["audio"],
            cache_root=root, resume=resume, interrupt_check=_interrupt, **settings)
        image, report = render_candidate_first_frame(vae=video_vae, candidate=candidate,
            inspection=prepared["inspection"], source_store=prepared["source"], window_store=store,
            color_match=color_match, geometry_align=geometry_align, source_mode=source_mode, interrupt_check=_interrupt)
        handle = {"prepared": prepared, "windows": store, "candidate": candidate,
                  "preview_report": report, "settings": settings, "cache_root": root}
        archive_id = save_candidate_archive(handle, image, interrupt_check=_interrupt)
        handle["archive_id"] = archive_id
        return io.NodeOutput(handle, image, canonical({"sampling": sampled, "preview": report,
                             "candidate_id": archive_id}), archive_id,
                             ui=ui.PreviewImage(image, cls=cls))


class MiniMaxH3VideoOutpaintSelectCandidateT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VideoOutpaintSelectCandidateT8",
            display_name="H3 Video Outpaint · 确认所连候选 (EXP)", category=CATEGORY, is_experimental=True,
            description="将喜欢的候选接入。只有已看过首帧并开启确认，才允许接续；不会自动挑选候选。",
            inputs=[CandidateIO.Input("candidate"), io.Boolean.Input("confirm_selection", default=False)],
            outputs=[SelectedIO.Output(display_name="selected"), io.String.Output(display_name="selection_report"),
                     io.String.Output(display_name="selection_id")])

    @classmethod
    def execute(cls, candidate, confirm_selection):
        if confirm_selection is not True:
            raise ValueError("请先单独运行候选节点查看首帧，再开启 confirm_selection 确认所连接的候选。")
        report = _preview_contract(candidate)
        selection = select_candidate(candidate["candidate"], candidate["windows"])
        selected = {**candidate, "candidate": deepcopy(candidate["candidate"]),
                    "preview_report": deepcopy(report), "settings": dict(candidate["settings"]),
                    "selection": selection}
        selection_id = save_selection_archive(selected, interrupt_check=_interrupt)
        selected["selection_id"] = selection_id
        return io.NodeOutput(selected, canonical({"selection": selection, "preview_sha256": report["sha256"],
                                                  "selection_id": selection_id}), selection_id)


class MiniMaxH3VideoOutpaintContinueCandidateT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VideoOutpaintContinueCandidateT8",
            display_name="H3 Video Outpaint · 接续所选候选 (EXP)", category=CATEGORY, is_experimental=True,
            inputs=[io.Model.Input("model"), SelectedIO.Input("selected")],
            outputs=[SelectedSampledIO.Output(display_name="sampled"), io.String.Output(display_name="sampling_report")])

    @classmethod
    def execute(cls, model, selected):
        _preview_contract(selected)
        prepared = selected["prepared"]
        validate_outpaint_source(prepared["inspection"], prepared["plan"])
        store, report = continue_verified_candidate(model=model, selection=selected["selection"],
            conditioning=prepared["conditioning"], source_store=prepared["source"], audio=prepared["audio"],
            cache_root=selected["cache_root"], interrupt_check=_interrupt, **selected["settings"])
        return io.NodeOutput({**selected, "windows": store}, canonical(report))


class MiniMaxH3VideoOutpaintComposeCandidateT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VideoOutpaintComposeCandidateT8",
            display_name="H3 Video Outpaint · 保存所选扩画 (EXP)", category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description="沿用候选的颜色设置，首帧RGB与所选预览不符时拒绝发布。原声音保留。",
            inputs=[SelectedSampledIO.Input("sampled"), io.Vae.Input("video_vae"),
                    io.String.Input("output_name", default="selected_outpaint")],
            outputs=[io.Video.Output(display_name="video"), io.String.Output(display_name="delivery_report")])

    @classmethod
    def execute(cls, sampled, video_vae, output_name):
        preview = _preview_contract(sampled)
        prepared = sampled["prepared"]
        root = Path(folder_paths.get_output_directory()).resolve() / "T8_H3_Outpaint"
        name, counter = _name(output_name), 1
        while ((root / f"{name}_{counter:05d}.mp4").exists()
               or (root / f"{name}_{counter:05d}.mp4.outpaint.json").exists()):
            counter += 1
        settings = dict(preview["color_settings"])
        enabled = settings.pop("enabled")
        video, report = compose_sampled_outpaint(vae=video_vae, inspection=prepared["inspection"],
            source_store=prepared["source"], window_store=sampled["windows"],
            output_path=root / f"{name}_{counter:05d}.mp4", color_match=enabled, color_settings=settings,
            **preview.get("geometry_settings", {}),
            source_mode=preview.get("source_mode", "preserve_source"),
            expected_first_frame_sha256=preview["rgb8_sha256"], interrupt_check=_interrupt)
        return io.NodeOutput(video, canonical(report))


class MiniMaxH3VideoOutpaintLoadCandidateT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VideoOutpaintLoadCandidateT8",
            display_name="H3 Video Outpaint · 读取已存候选 (EXP)", category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description="按生成时返回的candidate_id读取原候选PNG，不采样、不解码VAE、不自动确认。",
            inputs=[PreparedIO.Input("prepared"), io.String.Input("candidate_name", default="candidate_01"),
                    io.String.Input("candidate_id", default="")],
            outputs=[CandidateIO.Output(display_name="candidate"), io.Image.Output(display_name="first_frame"),
                     io.String.Output(display_name="preview_report")])

    @classmethod
    def execute(cls, prepared, candidate_name, candidate_id):
        root = prepared["root"] / "candidates" / _name(candidate_name)
        handle, image = load_candidate_archive(prepared, root, candidate_id, interrupt_check=_interrupt)
        return io.NodeOutput(handle, image, canonical(handle["preview_report"]), ui=ui.PreviewImage(image, cls=cls))


class MiniMaxH3VideoOutpaintLoadSelectionT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VideoOutpaintLoadSelectionT8",
            display_name="H3 Video Outpaint · 恢复已确认候选 (EXP)", category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description="明确输入之前确认时返回的selection_id。读取原候选和确认记录，可接Continue；不会自动生成或改选。",
            inputs=[PreparedIO.Input("prepared"), io.String.Input("candidate_name", default="candidate_01"),
                    io.String.Input("selection_id", default="")],
            outputs=[SelectedIO.Output(display_name="selected"), io.Image.Output(display_name="first_frame"),
                     io.String.Output(display_name="selection_report")])

    @classmethod
    def execute(cls, prepared, candidate_name, selection_id):
        root = prepared["root"] / "candidates" / _name(candidate_name)
        handle, image = load_selection_archive(prepared, root, selection_id, interrupt_check=_interrupt)
        return io.NodeOutput(handle, image, canonical({"selection_id": selection_id,
            "candidate_id": handle["archive_id"], "selection": handle["selection"]}), ui=ui.PreviewImage(image, cls=cls))


class MiniMaxH3VideoOutpaintLoadCompletedSelectionT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VideoOutpaintLoadCompletedSelectionT8",
            display_name="H3 Video Outpaint · 读取已完成选择 (EXP)", category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description="读取已确认且全部窗口已完成的缓存，可直接接保存节点；不加载生成模型、不采样。未完成缓存会拒绝。",
            inputs=[PreparedIO.Input("prepared"), io.String.Input("candidate_name", default="candidate_01"),
                    io.String.Input("selection_id", default="")],
            outputs=[SelectedSampledIO.Output(display_name="sampled"), io.Image.Output(display_name="first_frame"),
                     io.String.Output(display_name="completed_report")])

    @classmethod
    def execute(cls, prepared, candidate_name, selection_id):
        root = prepared["root"] / "candidates" / _name(candidate_name)
        handle, image = load_selection_archive(prepared, root, selection_id, interrupt_check=_interrupt)
        preview = _preview_contract(handle)
        state = handle["windows"].snapshot()
        expected = sum(len(shot["windows"]) for shot in prepared["plan"]["shots"])
        if state["status"] != "sampled" or len(state["committed"]) != expected:
            raise ValueError("所选候选尚未完成全部窗口；请先使用接续节点完成生成。")
        report = {"selection_id": selection_id, "candidate_id": handle["archive_id"],
                  "committed_windows": len(state["committed"]), "sampling_called": False,
                  "preview_sha256": preview["sha256"]}
        return io.NodeOutput(handle, image, canonical(report), ui=ui.PreviewImage(image, cls=cls))


VIDEO_OUTPAINT_CANDIDATE_DRAFT_NODE_CLASSES = [MiniMaxH3VideoOutpaintCandidateT8,
    MiniMaxH3VideoOutpaintSelectCandidateT8, MiniMaxH3VideoOutpaintContinueCandidateT8,
    MiniMaxH3VideoOutpaintComposeCandidateT8, MiniMaxH3VideoOutpaintLoadCandidateT8,
    MiniMaxH3VideoOutpaintLoadSelectionT8, MiniMaxH3VideoOutpaintLoadCompletedSelectionT8]
