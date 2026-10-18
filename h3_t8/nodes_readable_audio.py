from __future__ import annotations

from comfy_api.latest import io, ui

from .audio_opening_fade import mute_fade_audio, mute_fade_video_components
from .readable_diagnostics import diagnose_node_source, explain_audio_report


class MiniMaxH3NodeSourceDiagnosticT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="T8 Node Source / Runtime Version Diagnostics",
                         category="T8/MiniMax H3/Diagnostics", is_output_node=True,
                         inputs=[io.String.Input("node_id", default="MiniMaxH3AudioConditioningT8")],
                         outputs=[io.String.Output("Notes"), io.String.Output("report_json")])

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
        return io.Schema(node_id=cls.__name__, display_name="T8 Audio Source / Cache Notes (read-only)",
                         category="T8/MiniMax H3/Diagnostics", is_output_node=True,
                         inputs=[io.String.Input("report_json", default="", multiline=True,
                                 tooltip="Connect an existing conditioning / sampling / upscale report; this is not a prompt and does not alter audio.")],
                         outputs=[io.String.Output("Notes"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, report_json):
        values = explain_audio_report(report_json)
        return io.NodeOutput(*values, ui=ui.PreviewText(values[0]))


def _envelope_inputs():
    return [io.Boolean.Input("enabled", default=True),
            io.Int.Input("mute_first_frames", default=1, min=0, max=None,
                         tooltip="Zeroes the audio for the first N frames while preserving duration; too large a value cuts off the first word."),
            io.Float.Input("fade_in_ms", default=10.0, min=0.0, max=10000.0, step=1.0,
                           tooltip="Half-cosine fade-in after silence; N=0 fades in only. No film-wide denoising or normalization.")]


class MiniMaxH3AudioOpeningMuteFadeT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="T8 Head Audio Silence + Fade-In (AUDIO)",
                         category="T8/MiniMax H3/Audio", is_experimental=True,
                         inputs=[io.Audio.Input("audio"), *_envelope_inputs(),
                                 io.String.Input("fps", default="24", tooltip="1001。")],
                         outputs=[io.Audio.Output("audio"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*mute_fade_audio(**kwargs))


class MiniMaxH3VideoOpeningMuteFadeT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="T8 Tail Audio Silence + Fade-In (VIDEO EXP)",
                         category="T8/MiniMax H3/Audio", is_experimental=True,
                         description="Changes only the head audio; does not shorten or resample. tensor VIDEO frames pass through; file VIDEO is located by real frame PTS and the video stream is copied."
                                     "Existing trim/crop/non-MP4 views are post-processed by Core streaming export; no full-film RGB is loaded. AAC lossy encoding may add tail padding.",
                         inputs=[io.Video.Input("video"), *_envelope_inputs()],
                         outputs=[io.Video.Output("video"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*mute_fade_video_components(**kwargs))


READABLE_AUDIO_NODE_CLASSES = [MiniMaxH3NodeSourceDiagnosticT8, MiniMaxH3AudioSourceExplanationT8,
                              MiniMaxH3AudioOpeningMuteFadeT8, MiniMaxH3VideoOpeningMuteFadeT8]
