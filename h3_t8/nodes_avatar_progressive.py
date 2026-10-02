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
        return io.Schema(node_id=cls.__name__, display_name="H3 Avatar 录音驱动渐进采样（EXP/T8）",
                         category="T8/MiniMax H3/Audio/Experimental", is_experimental=True,
                         inputs=original.inputs, outputs=original.outputs,
                         description="人物首帧＋现成录音，经原 Audio Conditioning 的 lock_source 后接此节点。"
                         "录音通过原生audio mask=0参与采样，不是仅保存时贴音轨；复用原learned3D与clean-anchor。"
                         "默认LOW4＋HIGH4，支持可选独立HIGH MODEL／LoRA，旧渐进节点仍要求空AV。"
                         "仅phase1独立单段；不含空间分块，不是同音色生成新台词。指定短片已验收，未知素材／组合需自行验证。")

    @classmethod
    def execute(cls, **kwargs):
        import comfy.utils
        progress = comfy.utils.ProgressBar(max(1, len(kwargs["sigmas"]) - 1))

        def notify(step, prediction, state, total):
            progress.update_absolute(step + 1, total)

        return io.NodeOutput(*sample_avatar_progressive(**kwargs, callback=notify))
