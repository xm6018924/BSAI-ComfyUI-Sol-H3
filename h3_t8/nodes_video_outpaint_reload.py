"""Unregistered preparation-cache reload stage for independent restore workflows."""
from comfy_api.latest import io

from .nodes_video_outpaint import PlanIO, PreparedIO, CATEGORY, _root, _interrupt
from .video_outpaint_plan import canonical
from .video_outpaint_prepared_reload import load_prepared_outpaint


class MiniMaxH3VideoOutpaintLoadPreparedT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3VideoOutpaintLoadPreparedT8",
            display_name="H3 Video Outpaint · Read Prepared Cache (EXP)", category=CATEGORY, is_experimental=True,
            description="Reads completed source video/audio/prompt cache using the same source, Plan parameters, and run_name. Does not load CLIP/VAE or re-encode prompts; missing or corrupt data raises an error.",
            inputs=[PlanIO.Input("plan"), io.String.Input("run_name", default="outpaint_01")],
            outputs=[PreparedIO.Output(display_name="prepared"), io.String.Output(display_name="reload_report")])

    @classmethod
    def execute(cls, plan, run_name):
        handle, report = load_prepared_outpaint(plan, _root(plan, run_name), interrupt_check=_interrupt)
        return io.NodeOutput(handle, canonical(report))
