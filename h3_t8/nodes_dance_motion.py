"""Append-only source-motion input for the existing serial native/dual loops."""
import json
from fractions import Fraction
import math

from comfy_api.latest import io

from .dance_motion import DANCE_MOTION_TYPE, DanceMotionSource, dance_audio_at_source_start


DanceMotionIO = io.Custom(DANCE_MOTION_TYPE)


class MiniMaxH3DanceMotionSourceEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DanceMotionSourceEXPT8",
            display_name="MiniMax H3 舞蹈动作参考 / Dance Motion Source (EXP/T8)",
            category="T8/MiniMax H3/Long Video/Experimental", is_experimental=True,
            description="Connect source dance frames in original CFR order. Each long-video segment reads its own "
                "source interval including overlap; generated continuity stays separate. Connect the target character "
                "to ref_images and original music to final_audio on the loop. This is soft RGB reference, not strict "
                "pose control. Normalize variable-frame-rate media first; no automatic speed change.",
            inputs=[io.Image.Input("source_frames"),
                io.Int.Input("source_fps_numerator", default=24, min=1, max=240000),
                io.Int.Input("source_fps_denominator", default=1, min=1, max=10000),
                io.Float.Input("source_start_seconds", default=0., min=0., max=86400., step=.001,
                    tooltip="Offset into source frames and source_audio; aligned original music is output below."),
                io.Float.Input("source_fps", optional=True, force_input=True,
                    tooltip="Connect Get Video Components fps. Overrides the manual rational frame rate."),
                io.Audio.Input("source_audio", optional=True)],
            outputs=[DanceMotionIO.Output("source_motion"), io.String.Output("report_json"),
                     io.Audio.Output("aligned_original_audio")])

    @classmethod
    def execute(cls, source_frames, source_fps_numerator=24, source_fps_denominator=1,
                source_start_seconds=0., source_fps=None, source_audio=None):
        if source_fps is not None:
            if not math.isfinite(source_fps) or not 1 <= source_fps <= 240:
                raise ValueError("Connected source_fps must be finite and between 1 and 240")
            rate = Fraction(str(source_fps)).limit_denominator(10000)
            if abs(float(rate) - source_fps) > 1e-7:
                raise ValueError("Cannot represent connected FPS precisely; supply its rational frame rate instead")
            source_fps_numerator, source_fps_denominator = rate.numerator, rate.denominator
        source = DanceMotionSource(source_frames, source_fps_numerator, source_fps_denominator, source_start_seconds)
        audio, audio_report = dance_audio_at_source_start(source_audio, source_start_seconds)
        return io.NodeOutput(source, json.dumps({"status": "prepared_not_sampled",
            "frame_count": len(source.frames), "source_fps": [source.fps.numerator, source.fps.denominator],
            "source_start_seconds": source.start_seconds, "role": "editable_reference",
            "identity": "full RGB content hashed by loop before sampling/resume",
            "original_audio": audio_report}, ensure_ascii=False), audio)
