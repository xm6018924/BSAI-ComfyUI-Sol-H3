from comfy_api.latest import io

from .prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from .prompt_relay_long_video_advanced import configure_long_video_window_text, WINDOW_TEXT_POLICIES


PlanIO = io.Custom(PROMPT_RELAY_PLAN_TYPE)


class MiniMaxH3PromptRelayWindowTextEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptRelayWindowTextEXPT8",
            display_name="MiniMax H3 Long Relay Window Text Policy (T8 EXP)",
            category="T8/MiniMax H3/Conditioning/Experimental",
            is_experimental=True,
            description=(
                "Optional global-Plan policy for native long-video projection. Old all-key "
                "behavior stays the default. The experiment omits completed/future event text "
                "without changing accepted AV context, crossing-event sigma or sampling. "
                "It is not paper-faithful all-key Relay or a no-repeat guarantee."
            ),
            inputs=[
                PlanIO.Input("prompt_relay_plan", tooltip="Connect after the global Plan / Query Route and before the long-video in-node loop; do not connect a projected window."),
                io.Combo.Input("text_policy", options=list(WINDOW_TEXT_POLICIES), default="preserve_all",
                    tooltip=("preserve_all keeps the old graph. accepted_window_text_exp keeps only local text intersecting the current delivery window; "
                             "ended / future / context-only events are not re-fed to CLIP. Cross-segment events keep their original sigma."
                             "dialogue_start_owner_exp additionally lets only the event-start delivery window keep <d> dialogue; "
                             "cross-segment visual descriptions and AV context are not deleted. Global cannot contain <d>; unmarked dialogue is not recognized."
                             "Put one-off dialogue fully into a single physical segment; verbatim / no-duplicate is not guaranteed; use a new chain_id for combinations.")),
            ],
            outputs=[PlanIO.Output("prompt_relay_plan"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*configure_long_video_window_text(**kwargs))
