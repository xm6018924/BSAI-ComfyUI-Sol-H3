from comfy_api.latest import io

from .prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from .prompt_relay_long_video_advanced import configure_long_video_window_text, WINDOW_TEXT_POLICIES


PlanIO = io.Custom(PROMPT_RELAY_PLAN_TYPE)


class MiniMaxH3PromptRelayWindowTextEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptRelayWindowTextEXPT8",
            display_name="MiniMax H3 Long Relay · 窗口文本策略 (T8 EXP)",
            category="T8/MiniMax H3/Conditioning/Experimental",
            is_experimental=True,
            description=(
                "Optional global-Plan policy for native long-video projection. Old all-key "
                "behavior stays the default. The experiment omits completed/future event text "
                "without changing accepted AV context, crossing-event sigma or sampling. "
                "It is not paper-faithful all-key Relay or a no-repeat guarantee."
            ),
            inputs=[
                PlanIO.Input("prompt_relay_plan", tooltip="接全局Plan／Query Route后，再接长视频内循环；不要接已投影窗口。"),
                io.Combo.Input("text_policy", options=list(WINDOW_TEXT_POLICIES), default="preserve_all",
                    tooltip=("preserve_all保持旧图。accepted_window_text_exp只保留与当前交付窗口相交的局部文本；"
                             "已结束／未来／只在已知上下文的事件不重复送入CLIP。跨段事件仍保留原sigma。"
                             "dialogue_start_owner_exp另外只让事件起始交付窗口保留<d>台词；"
                             "跨段视觉描述及AV上下文不删。Global不可放<d>，未标记的对白无法识别。"
                             "一次性台词尽量完整放入一个物理段；不保证逐字／不重复，组合需新chain_id。")),
            ],
            outputs=[PlanIO.Output("prompt_relay_plan"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*configure_long_video_window_text(**kwargs))
