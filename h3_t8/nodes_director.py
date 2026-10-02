from __future__ import annotations

import json
from comfy_api.latest import io
from .director_project import compile_project
from .director_routes import get_store


class MiniMaxH3DirectorProjectT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DirectorProjectT8",
            display_name="曜石导演台 / Obsidian Director D1 (T8)",
            category="T8/MiniMax H3/Studio",
            is_experimental=True,
            is_output_node=True,
            description="Open the real director UI using the node button. This D1 node only validates saved projects and compiles contracts on CPU; it never queues generation.",
            inputs=[
                io.String.Input("project_json", default="", multiline=True),
                io.String.Input("shot_id", default=""),
            ],
            outputs=[
                io.String.Output("compiled_prompt"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Int.Output("length"),
                io.String.Output("media_map_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, project_json, shot_id):
        project = json.loads(project_json)
        report = compile_project(project, get_store())
        if not report["ready"]:
            raise ValueError(
                "导演台预检未通过：" + json.dumps(report["errors"], ensure_ascii=False)
            )
        shot = next((s for s in report["shots"] if s["id"] == shot_id), None)
        if shot is None:
            raise ValueError("请选择项目内有效的镜头 UUID")
        return io.NodeOutput(
            shot["prompt"],
            shot["canvas"]["width"],
            shot["canvas"]["height"],
            shot["time"]["aligned_frames"],
            json.dumps(shot["media_map"], ensure_ascii=False),
            json.dumps(report, ensure_ascii=False),
        )
