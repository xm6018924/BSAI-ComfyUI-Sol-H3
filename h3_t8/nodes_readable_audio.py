from __future__ import annotations

from comfy_api.latest import io, ui

from .audio_opening_fade import mute_fade_audio, mute_fade_video_components
from .readable_diagnostics import diagnose_node_source, explain_audio_report


class MiniMaxH3NodeSourceDiagnosticT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="T8 节点来源／运行版本诊断",
                         category="T8/MiniMax H3/Diagnostics", is_output_node=True,
                         inputs=[io.String.Input("node_id", default="MiniMaxH3AudioConditioningT8")],
                         outputs=[io.String.Output("说明"), io.String.Output("report_json")])

    @classmethod
    async def execute(cls, node_id):
        from .nodes import comfy_entrypoint
        classes = await comfy_entrypoint().get_node_list()
        values = diagnose_node_source(node_id, classes)
        return io.NodeOutput(*values, ui=ui.PreviewText(values[0]))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")


class MiniMaxH3AudioSourceExplanationT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="T8 音频来源／缓存说明（只读）",
                         category="T8/MiniMax H3/Diagnostics", is_output_node=True,
                         inputs=[io.String.Input("report_json", default="", multiline=True,
                                 tooltip="连接已有条件／采样／放大报告，不是提示词，不改变声音。")],
                         outputs=[io.String.Output("说明"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, report_json):
        values = explain_audio_report(report_json)
        return io.NodeOutput(*values, ui=ui.PreviewText(values[0]))


def _envelope_inputs():
    return [io.Boolean.Input("enabled", default=True),
            io.Int.Input("mute_first_frames", default=1, min=0, max=None,
                         tooltip="对应开头N帧的声音置零，保留时长；太大会削掉首字。"),
            io.Float.Input("fade_in_ms", default=10.0, min=0.0, max=10000.0, step=1.0,
                           tooltip="静音后半余弦淡入；N=0时仅淡入。不做全片降噪或归一化。")]


class MiniMaxH3AudioOpeningMuteFadeT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="T8 开头音频静音＋淡入（AUDIO）",
                         category="T8/MiniMax H3/Audio", is_experimental=True,
                         inputs=[io.Audio.Input("audio"), *_envelope_inputs(),
                                 io.String.Input("fps", default="24", tooltip="必须与视频一致；支持30000/1001。")],
                         outputs=[io.Audio.Output("audio"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*mute_fade_audio(**kwargs))


class MiniMaxH3VideoOpeningMuteFadeT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="T8 末端音频静音＋淡入（VIDEO EXP）",
                         category="T8/MiniMax H3/Audio", is_experimental=True,
                         description="只改开头音频，不缩短或重采样。tensor VIDEO帧数据直通；文件VIDEO按实际帧PTS定位，视频流复制。"
                                     "原有trim/crop/非MP4视图由Core流式导出后处理，不加载全片RGB。AAC有损编码可能有尾部padding。",
                         inputs=[io.Video.Input("video"), *_envelope_inputs()],
                         outputs=[io.Video.Output("video"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*mute_fade_video_components(**kwargs))


READABLE_AUDIO_NODE_CLASSES = [MiniMaxH3NodeSourceDiagnosticT8, MiniMaxH3AudioSourceExplanationT8,
                              MiniMaxH3AudioOpeningMuteFadeT8, MiniMaxH3VideoOpeningMuteFadeT8]
