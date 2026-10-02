"""Unregistered preparation-cache reload stage for independent restore workflows."""
from comfy_api.latest import io

from .nodes_video_outpaint import PlanIO, PreparedIO, CATEGORY, _root, _interrupt
from .video_outpaint_plan import canonical
from .video_outpaint_prepared_reload import load_prepared_outpaint


class MiniMaxH3VideoOutpaintLoadPreparedT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VideoOutpaintLoadPreparedT8",
            display_name="H3 Video Outpaint · 读取已准备缓存 (EXP)", category=CATEGORY, is_experimental=True,
            description="用相同原片、Plan参数和run_name读取已完成的源视频/音频/提示词缓存。不加载CLIP/VAE，不重新编码提示词；缺失或损坏会报错。",
            inputs=[PlanIO.Input("plan"), io.String.Input("run_name", default="outpaint_01")],
            outputs=[PreparedIO.Output(display_name="prepared"), io.String.Output(display_name="reload_report")])

    @classmethod
    def execute(cls, plan, run_name):
        handle, report = load_prepared_outpaint(plan, _root(plan, run_name), interrupt_check=_interrupt)
        return io.NodeOutput(handle, canonical(report))
