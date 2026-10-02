"""Unregistered geometry-preview node; independent of the running GPU draft."""
from __future__ import annotations

from comfy_api.latest import io, ui

from .video_outpaint_plan import canonical
from .video_outpaint_preview import render_source_geometry_preview


PlanIO = io.Custom("T8_H3_OUTPAINT_PLAN")


class MiniMaxH3VideoOutpaintGeometryPreviewT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VideoOutpaintGeometryPreviewT8",
            display_name="H3 Video Outpaint · 扩画范围预览 (EXP)",
            category="T8/MiniMax H3/Video Outpaint EXP", is_experimental=True, is_output_node=True,
            description="只预览原片位置和待补区域，不运行生成模型，也不代表扩画效果。请用独立预览工作流；连着采样分支不会暂停生成。",
            inputs=[PlanIO.Input("plan"), io.Int.Input("frame_index", default=0, min=0, max=None),
                    io.Int.Input("preview_max_edge", default=768, min=64, max=1536)],
            outputs=[PlanIO.Output(display_name="plan"), io.Image.Output(display_name="geometry_preview"),
                     io.String.Output(display_name="preview_report")])

    @classmethod
    def execute(cls, plan, frame_index, preview_max_edge):
        import comfy.model_management
        import numpy as np
        import torch

        image, report = render_source_geometry_preview(plan["inspection"], plan["plan"], frame_index, preview_max_edge,
            interrupt_check=comfy.model_management.throw_exception_if_processing_interrupted)
        tensor = torch.from_numpy(np.asarray(image).copy()).to(dtype=torch.float32).div_(255).unsqueeze(0)
        return io.NodeOutput(plan, tensor, canonical(report), ui=ui.PreviewImage(tensor, cls=cls))
