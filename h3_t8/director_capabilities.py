"""Capability inventory used by the Director D3 hand-off panel.

This is deliberately an inventory and dependency gate, not a generic graph
builder.  D3 features have different state contracts (long-video resume,
Bridge/Relay projection, post-processing, or Meridian geometry), so the
Director must show the real registered entry points and send users to the
corresponding native workflow until each recipe has its own end-to-end receipt.
"""

from __future__ import annotations

from typing import Iterable, Mapping


CAPABILITY_SPECS: tuple[Mapping[str, object], ...] = (
    {
        "id": "long_video",
        "label": "Continuous long film / second pass",
        "entry_nodes": (
            "MiniMaxH3LongVideoPlannerT8",
            "MiniMaxH3LongVideoConditioningT8",
            "MiniMaxH3LongVideoOrchestratorT8",
            "MiniMaxH3LongVideoComposeAcceptedT8",
        ),
        "route": "native_long_video_workflow",
        "note": "Preserves context, cancel, resume, and accepted seams; does not stitch single-segment generations into a continuous long film.",
    },
    {
        "id": "semantic_bridge",
        "label": "Semantic Bridge",
        "entry_nodes": (
            "MiniMaxH3SemanticBridgeConfigT8",
            "MiniMaxH3SemanticBridgeApplyT8",
        ),
        "route": "native_semantic_bridge_workflow",
        "note": "Bridge changes only the conditioning chain; existing sampling, audio, time bias, and seam defaults are untouched.",
    },
    {
        "id": "prompt_relay",
        "label": "Prompt Relay",
        "entry_nodes": (
            "MiniMaxH3PromptRelayPlanT8Advanced",
            "MiniMaxH3PromptRelayConditioningT8Advanced",
        ),
        "route": "native_prompt_relay_workflow",
        "note": "Compiled per the formal Relay Plan -> Conditioning contract; source blanks and role mapping are auditable.",
    },
    {
        "id": "fast_h3_v2",
        "label": "FastH3 V2／Tao",
        "entry_nodes": (
            "MiniMaxH3FastH3V2SetupEXPT8",
            "MiniMaxH3FastH3V2RuntimeAuditEXPT8",
            "MiniMaxH3FastH3V2DualModelLongVideoEXPT8",
        ),
        "route": "native_fast_h3_v2_workflow",
        "note": "Available only when the actual installed FastH3 V2 model and corresponding LoRA/audit nodes are present.",
    },
    {
        "id": "eav",
        "label": "EAV / enhanced AV",
        "entry_nodes": (
            "MiniMaxH3EnhanceAVideoPromptRelayComposerT8Advanced",
            "MiniMaxH3EnhanceAVideoLongVideoComposerT8Advanced",
        ),
        "route": "native_eav_workflow",
        "note": "EAV and Relay/long-film have independent wrapper contracts; unvalidated combinations are not force-stacked by the director.",
    },
    {
        "id": "low_vram",
        "label": "LowVRAM／ChunkFFN",
        "entry_nodes": (
            "MiniMaxH3LowVRAMAttentionT8Advanced",
            "MiniMaxH3ChunkFeedForwardT8Advanced",
        ),
        "route": "native_memory_workflow",
        "note": "VRAM policy is an explicit auditable node; it does not silently change the normal H3 default path.",
    },
    {
        "id": "taeh3_preview",
        "label": "TAEH3 live sampling preview",
        "entry_nodes": (
            "MiniMaxH3TAEH3PreviewCapabilityT8Advanced",
            "MiniMaxH3TAEH3SamplingPreviewEXPT8",
        ),
        "route": "native_taeh3_preview_workflow",
        "note": "The preview is an intermediate x0 approximation, not the final VAE output; cancellation must return to the native sampling entry.",
    },
    {
        "id": "topaz",
        "label": "Topaz upscale",
        "entry_nodes": (
            "MiniMaxH3TopazEnvironmentEXPT8",
            "MiniMaxH3TopazVideoEXPT8",
        ),
        "route": "native_topaz_workflow",
        "note": "Input can be a regular downloaded video or H3 output; environment check, cancel, and safe H.264 delivery each must pass separately.",
    },
    {
        "id": "meridian",
        "label": "Meridian",
        "entry_nodes": (
            "MiniMaxH3MeridianConfigEXPT8",
            "MiniMaxH3MeridianMaterialEXPT8",
            "MiniMaxH3MeridianCameraEXPT8",
            "MiniMaxH3MeridianGenerateEXPT8",
        ),
        "route": "native_meridian_workflow",
        "note": "Geometry weights, ConvRot INT8, and the full P2/P3/P4 quality gates are recorded independently; they are not faked from D2 generation results.",
    },
)


def inspect_director_capabilities(
    node_ids: Iterable[str] | None = None,
    *,
    model_names: Iterable[str] | None = None,
) -> dict[str, object]:
    """Return a truthful, JSON-safe D3 inventory for the current Core.

    ``model_names`` is intentionally only reported as context.  Each feature
    owns different model/weight checks, so a filename-only global gate would be
    misleading; its native workflow performs the final dependency validation.
    """

    if node_ids is None:
        try:
            import nodes

            node_ids = nodes.NODE_CLASS_MAPPINGS.keys()
        except Exception:
            node_ids = ()
    available = {str(value) for value in node_ids}
    models = sorted({str(value) for value in (model_names or ())})
    rows = []
    for spec in CAPABILITY_SPECS:
        required = [str(value) for value in spec["entry_nodes"]]
        missing = [value for value in required if value not in available]
        rows.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "state": "ready" if not missing else "missing_entry_nodes",
                "entry_nodes": required,
                "missing_entry_nodes": missing,
                "route": spec["route"],
                "note": spec["note"],
                "execution": "native_workflow_only_until_gpu_receipt",
            }
        )
    return {
        "schema": "t8.minimax_h3.director_capabilities.v1",
        "capabilities": rows,
        "model_inventory_context": models,
        "warning": "ready means registered entry nodes only; it is not a GPU or human quality acceptance.",
    }
