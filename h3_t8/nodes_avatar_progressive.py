"""Append-only phase1 Avatar entry; the released progressive schema is unchanged."""
from comfy_api.latest import io

from .nodes_progressive_sampling import MiniMaxH3ProgressiveSamplerEXPT8
from .avatar_progressive_entry import sample_avatar_progressive


class MiniMaxH3AvatarProgressiveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        original = MiniMaxH3ProgressiveSamplerEXPT8.define_schema()
        for field in original.inputs:
            if field.id == "low_evaluations":
                field.default = 4
            elif field.id == "task":
                field.default = "i2va"
        return io.Schema(node_id=cls.__name__, display_name="H3 Avatar Recording-Driven Progressive Sampling (EXP/T8)",
                         category="T8/MiniMax H3/Audio/Experimental", is_experimental=True,
                         inputs=original.inputs, outputs=original.outputs,
                         description="Character first frame + existing recording; connect after the original Audio Conditioning lock_source."
                         "Recording participates in sampling via native audio mask=0; it is not just muxed onto the track at save. Reuses the original learned3D and clean-anchor."
                         "Default LOW4 + HIGH4; optional independent HIGH MODEL / LoRA. Older progressive nodes still require empty AV."
                         "Phase-1 standalone single segment only; no spatial chunking; not same-voice new-dialogue generation. The specified short film is validated; unknown assets/combos must be verified by the user.")

    @classmethod
    def execute(cls, **kwargs):
        import comfy.utils
        progress = comfy.utils.ProgressBar(max(1, len(kwargs["sigmas"]) - 1))

        def notify(step, prediction, state, total):
            progress.update_absolute(step + 1, total)

        return io.NodeOutput(*sample_avatar_progressive(**kwargs, callback=notify))
