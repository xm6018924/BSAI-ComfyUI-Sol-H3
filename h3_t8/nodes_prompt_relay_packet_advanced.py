from __future__ import annotations

from comfy_api.latest import io

from .prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from .prompt_relay_packet_advanced import (
    PROMPT_RELAY_EVENTS_TYPE,
    build_prompt_relay_event,
    build_prompt_relay_plan_from_packet,
)


CATEGORY = "T8/MiniMax H3/Conditioning/Experimental"
PromptPacketIO = io.Custom("H3_T8_PROMPT_PACKET")
PromptRelayPlanIO = io.Custom(PROMPT_RELAY_PLAN_TYPE)
PromptRelayEventsIO = io.Custom(PROMPT_RELAY_EVENTS_TYPE)


class MiniMaxH3PromptPacketRelayPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptPacketRelayPlanT8Advanced",
            display_name=(
                "MiniMax H3 Prompt Packet to Relay Plan "
                "(Advanced)"
            ),
            description=(
                "Converts an authenticated MiniMax H3 Prompt Compiler packet into an "
                "auditable Prompt Relay timeline. The packet's compiled visual/audio "
                "prompt remains global; local events are never inferred or rewritten."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                PromptPacketIO.Input(
                    "prompt_packet",
                    tooltip=(
                        "Connect directly to the T8 Video Prompt Compiler; accepts only backend=minimax_h3 "
                        "and a hash-complete packet."
                    ),
                ),
                io.String.Input(
                    "events_json",
                    multiline=True,
                    dynamic_prompts=True,
                    default=(
                        '[\n  {"prompt":"The subject raises one hand."},\n'
                        '  {"prompt":"The subject turns and walks forward."},\n'
                        '  {"prompt":"The camera pulls back into a wide shot."}\n]'
                    ),
                    tooltip=(
                        "Explicit event list. auto_equal writes prompt only; other modes also write start/end."
                        "Use [] to keep only the global Prompt Packet without installing a Relay patch."
                    ),
                ),
                io.Combo.Input(
                    "timing_mode",
                    options=["auto_equal", "frames", "seconds", "percent"],
                    default="auto_equal",
                    tooltip=(
                        "frames end is inclusive; seconds/percent end is a time boundary."
                    ),
                ),
                io.Combo.Input(
                    "math_profile",
                    options=["paper_v1", "legacy_repo_compat"],
                    default="paper_v1",
                    tooltip="Default uses the Prompt Relay paper formula.",
                ),
                io.Float.Input(
                    "epsilon",
                    default=0.1,
                    min=0.000001,
                    max=0.999999,
                    step=0.01,
                    advanced=True,
                ),
                io.Boolean.Input("allow_gaps", default=False, advanced=True),
                io.Boolean.Input("allow_overlaps", default=False, advanced=True),
                PromptRelayEventsIO.Input(
                    "prompt_relay_events",
                    optional=True,
                    tooltip=(
                        "Optional: connect a chained Event node as the authoritative event source; when unconnected, reads  events_json。"
                    ),
                ),
            ],
            outputs=[
                PromptRelayPlanIO.Output("prompt_relay_plan"),
                io.String.Output("compiled_prompt"),
                io.Int.Output("length"),
                io.String.Output("timeline_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_prompt_relay_plan_from_packet(**kwargs))


class MiniMaxH3PromptRelayEventT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptRelayEventT8Advanced",
            display_name="MiniMax H3 Prompt Relay Event (Advanced)",
            description=(
                "Adds one explicit local action to a chainable Prompt Relay event list. "
                "Duplicate this node for each event; start/end are interpreted by the "
                "Packet bridge timing mode."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="The subject performs one clearly observable action.",
                    tooltip="Write only local actions for this time range; do not repeat global character/scene/audio settings.",
                ),
                io.Float.Input(
                    "start",
                    default=0.0,
                    min=0.0,
                    max=100000.0,
                    step=0.01,
                    tooltip=(
                        "Ignored in auto_equal; frames is integer start frame, seconds is seconds, percent is 0..100; "
                        "Multiple Events must be sorted by start time."
                    ),
                ),
                io.Float.Input(
                    "end",
                    default=100.0,
                    min=0.0,
                    max=100000.0,
                    step=0.01,
                    tooltip=(
                        "frames mode accepts integers and includes the end frame; seconds/percent mode treats end as a boundary."
                    ),
                ),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip=(
                        "When off, this event is not added to the chain; if the whole chain is off, the Plan becomes "
                        "a global-prompt passthrough; no need to disconnect wires."
                    ),
                ),
                PromptRelayEventsIO.Input(
                    "previous_events",
                    optional=True,
                    tooltip="Connect the previous Event node; leave the first Event empty.",
                ),
            ],
            outputs=[
                PromptRelayEventsIO.Output("prompt_relay_events"),
                io.String.Output("preview_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_prompt_relay_event(**kwargs))


PROMPT_RELAY_PACKET_ADVANCED_NODE_CLASSES = [
    MiniMaxH3PromptPacketRelayPlanT8Advanced,
    MiniMaxH3PromptRelayEventT8Advanced,
]
