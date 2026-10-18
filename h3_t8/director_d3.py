"""Truthful D3 route registry and per-shot dependency preflight.

The Director D3 surface is intentionally a router, not a second implementation
of nine unrelated runtimes.  Each capability keeps its native workflow and
entry nodes; this module makes the current project/shot, installed nodes and
model inventory visible before the user leaves the Director for that route.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from .director_capabilities import CAPABILITY_SPECS
from .director_project import ProjectStore, canonical, compile_project, validate_project


ROUTE_WORKFLOWS: Mapping[str, tuple[str, ...]] = {
    "long_video": (
        "examples/workflows/04-long-video/2026-09-12_H3_Dual_Model_Long_Video_4plus4_Joint_AV_Dialogue_Aligned_24s_EXP.json",
        "examples/workflows/04-long-video/2026-09-15_H3_Dual_4plus4_Accepted_Picture_T8_LowVRAM_EXP.json",
    ),
    "semantic_bridge": (
        "examples/workflows/34-semantic-bridge/2026-09-17_H3_SemanticBridge_DualIndependent_8s_Advanced.json",
        "examples/workflows/34-semantic-bridge/README.md",
    ),
    "prompt_relay": (
        "examples/workflows/14-prompt-relay/2026-08-20_H3_Prompt_Relay_I2VA_Stock20_Advanced_EXP.json",
        "examples/workflows/14-prompt-relay/2026-08-20_H3_Prompt_Relay_Ref2VA_Stock20_Advanced_EXP.json",
    ),
    "fast_h3_v2": (
        "examples/workflows/10-speed/FastH3_V2_Trained_VSA_73f_h1c1_EXP.json",
        "examples/workflows/10-speed/FAST_H3_V2_README.md",
    ),
    "eav": (
        "examples/workflows/04-long-video/2026-09-11_H3_In_Node_Long_Video_Prompt_Relay_EAV_Dialogue_Segment_Aligned_Stock20_Advanced_EXP.json",
        "examples/workflows/36-avatar-voice/2026-09-18_T8_voice_eav_on_EXP.json",
    ),
    "low_vram": (
        "examples/workflows/04-long-video/2026-09-15_H3_Dual_4plus4_Accepted_Picture_T8_LowVRAM_EXP.json",
        "docs/H3_MEMORY_NODES_EXP.md",
    ),
    "taeh3_preview": (
        "docs/TAEH3_SAMPLING_PREVIEW_EXP.md",
        "examples/workflows/04-long-video/2026-09-12_H3_Dual_Model_Long_Video_4plus4_Joint_AV_Dialogue_Aligned_24s_EXP.json",
    ),
    "topaz": (
        "examples/workflows/31-topaz/2026-09-11_H3_Topaz_Video_EXP.json",
        "examples/workflows/31-topaz/2026-09-11_H3_Topaz_Environment_EXP.json",
    ),
    "meridian": (
        "examples/workflows/37-meridian/2026-09-18_T8_Meridian_video_source_camera_EXP.json",
        "examples/workflows/37-meridian/README.md",
    ),
}

# These contracts are intentionally short and operational.  They are shown in
# the hand-off dialog and copied into the route package so a user can continue
# in the native workflow without guessing which part the Director did (and did
# not) compile.
ROUTE_CONTRACTS: Mapping[str, Mapping[str, object]] = {
    "long_video": {
        "mode": "native_only",
        "steps": ["Open the continuous long-film workflow", "Replace source video / segment inputs", "Keep planning, seam, cancel, and resume nodes"],
        "requires": ["Continuous source video", "Segment or context configuration", "Available H3 model"],
        "boundary": "The director does not stitch single shots into a continuous long film; the native workflow owns context and second-pass state.",
    },
    "semantic_bridge": {
        "mode": "native_only",
        "steps": ["Choose the transform weights under models/semantic_bridge", "Connect the Bridge conditioning chain", "Run along the existing sampling and audio chain"],
        "requires": ["MiniMaxH3_SemanticBridge_v1_T8_Compat.safetensors or equivalent compatible weights"],
        "boundary": "Bridge changes only the conditioning chain; it does not replace the sampler, audio, or seam policy.",
    },
    "prompt_relay": {
        "mode": "native_only",
        "steps": ["Fill in the Relay Plan time ranges", "Connect the Plan to Conditioning", "Run along the native H3 sampling chain"],
        "requires": ["Prompt Relay Plan / Conditioning node", "H3 model matching the input type"],
        "boundary": "Blank segments and role mapping must be confirmed in the native Plan; the director does not guess for the user.",
    },
    "fast_h3_v2": {
        "mode": "native_only",
        "steps": ["Use the FastH3 V2 dedicated model", "Configure steps per V2 Setup", "Sample after running Runtime Audit"],
        "requires": ["fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors", "FastH3 V2 entry node"],
        "boundary": "Do not stack a normal H3 Turbo LoRA when FastH3 is enabled; the director already enforces this gate on the real compile path.",
    },
    "eav": {
        "mode": "native_only",
        "steps": ["Prepare video and audio inputs", "Choose the EAV/Relay combination entry", "Do a short segment test first before extending duration"],
        "requires": ["EAV Composer node", "Clear driving audio or reference audio", "Input video / first frame"],
        "boundary": "The EAV wrapper has its own AV timing contract; it is not replaced by the single-shot generator.",
    },
    "low_vram": {
        "mode": "native_only",
        "steps": ["Connect LowVRAM on the MODEL link", "Continue with ChunkFFN if needed", "Keep order and record the VRAM policy"],
        "requires": ["LowVRAM Attention node", "ChunkFFN node (optional)"],
        "boundary": "VRAM optimization is an explicit MODEL patch; it does not silently change the default H3 path.",
    },
    "taeh3_preview": {
        "mode": "native_only",
        "steps": ["Enable the TAEH3 preview entry", "Observe the x0 approximation while sampling", "Return to native queue state after cancel"],
        "requires": ["TAEH3 Preview Capability / Sampling Preview node", "Corresponding TAEH3 resources"],
        "boundary": "The preview is an intermediate approximation, not the final VAE output; it cannot replace delivery acceptance.",
    },
    "topaz": {
        "mode": "native_only",
        "steps": ["Run environment check first", "Feed the H3 or regular downloaded video into upscale", "Confirm cancel and H.264 delivery"],
        "requires": ["Official Topaz Video AI and models directory (auto-discoverable by the environment node)", "Readable video input"],
        "boundary": "Input is not limited to H3 output; environment, model, and cancel capability are finally verified by the native Topaz node.",
    },
    "meridian": {
        "mode": "native_only",
        "steps": ["Prepare image / camera / material inputs", "Choose ConvRot INT8 weights", "Accept per the P2/P3/P4 quality gates"],
        "requires": ["Meridian ConvRot INT8 ComfyUI weights", "VGGT-Omega geometry weights (if the route requires)", "Meridian native entry node"],
        "boundary": "Without geometry weights only structural preflight is possible; do not claim the Meridian output is complete.",
    },
}


def _route_file(path: str) -> Path:
    """Resolve one allow-listed native workflow below the formal package root."""

    root = Path(__file__).resolve().parents[1]
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError("Native route file path is out of bounds")
    return candidate


def handoff_d3_route(capability: str, file_path: str | None = None) -> dict[str, object]:
    """Return a concrete, download-ready native workflow handoff.

    This deliberately does not rewrite the workflow into a fake Director graph or
    queue it.  The allow-list is the contract: the user receives the exact checked-
    in native JSON/README, plus existence and format metadata, so missing optional
    runtimes remain visible instead of becoming a dead button.
    """

    if capability not in ROUTE_WORKFLOWS:
        raise ValueError("Unknown D3 native route")
    allowed = tuple(ROUTE_WORKFLOWS[capability])
    selected = file_path or allowed[0]
    if selected not in allowed:
        raise ValueError("This file does not belong to the selected D3 native route")
    rows: list[dict[str, object]] = []
    for relative in allowed:
        path = _route_file(relative)
        exists = path.is_file()
        row: dict[str, object] = {
            "path": relative,
            "name": path.name,
            "kind": "workflow" if path.suffix.lower() == ".json" else "readme",
            "exists": exists,
        }
        if exists:
            row["bytes"] = path.stat().st_size
        rows.append(row)
    chosen_path = _route_file(selected)
    if not chosen_path.is_file():
        raise FileNotFoundError(f"原生路线文件不存在：{selected}")
    raw = chosen_path.read_bytes()
    result: dict[str, object] = {
        "schema": "t8.minimax_h3.director_d3_handoff.v1",
        "capability": capability,
        "files": rows,
        "selected": selected,
        "name": chosen_path.name,
        "kind": "workflow" if chosen_path.suffix.lower() == ".json" else "readme",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "warning": "This is a verified copy of a native workflow/guide from the repo; it does not auto-map the current director shot or queue. After opening, follow that route's input and model gates.",
    }
    if chosen_path.suffix.lower() == ".json":
        try:
            text = raw.decode("utf-8")
            result["content"] = json.loads(text)
            # Keep the original bytes available to the browser download so a
            # handoff does not silently reformat or reorder a native workflow.
            result["text"] = text
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"原生工作流 JSON 无法读取：{chosen_path.name}") from error
    else:
        result["text"] = raw.decode("utf-8")
    return result


def export_d3_package(
    capability: str,
    project: Mapping[str, object] | None = None,
    shot_id: str | None = None,
    store: ProjectStore | None = None,
) -> dict[str, object]:
    """Export a truthful, self-contained hand-off manifest for native routes.

    The package is JSON rather than a zip on purpose: it is inspectable, diffable
    and safe to download from the Director.  Native workflow text is preserved
    byte-for-byte; local media are represented by stable IDs and server paths,
    never copied or silently re-encoded.
    """

    if capability not in ROUTE_WORKFLOWS:
        raise ValueError("Unknown D3 native route")
    checked = validate_project(project) if project is not None else None
    preflight = inspect_d3_routes(
        checked, shot_id, store, capability=capability
    ) if checked is not None else inspect_d3_routes(capability=capability)
    files: list[dict[str, object]] = []
    for relative in ROUTE_WORKFLOWS[capability]:
        path = _route_file(relative)
        if not path.is_file():
            continue
        raw = path.read_bytes()
        item: dict[str, object] = {
            "path": relative,
            "name": path.name,
            "kind": "workflow" if path.suffix.lower() == ".json" else "readme",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "text": raw.decode("utf-8"),
        }
        if path.suffix.lower() == ".json":
            item["content"] = json.loads(item["text"])
        files.append(item)
    assets: list[dict[str, object]] = []
    if checked is not None:
        for asset in checked.get("assets", []):
            if not isinstance(asset, Mapping):
                continue
            assets.append({
                key: asset.get(key)
                for key in ("id", "kind", "name", "server_path", "duration", "width", "height")
                if key in asset
            })
    contract = dict(ROUTE_CONTRACTS.get(capability, {}))
    package: dict[str, object] = {
        "schema": "t8.minimax_h3.director_d3_route_package.v1",
        "capability": capability,
        "contract": contract,
        "preflight": preflight,
        "files": files,
        "assets": assets,
        "project": json.loads(canonical(checked)) if checked is not None else None,
        "selected_shot": shot_id,
        "warning": "The route package saves only the current project snapshot, stable asset identities, and native workflow copies; it does not upload media, rewrite workflows, or auto-queue.",
    }
    return package


def _model_names() -> dict[str, list[str]]:
    try:
        import folder_paths

        result = {}
        for folder in (
            "diffusion_models",
            "loras",
            "vae",
            "semantic_bridge",
            "meridian",
        ):
            try:
                result[folder] = sorted(str(value) for value in folder_paths.get_filename_list(folder))
            except (KeyError, TypeError, AttributeError):
                result[folder] = []
        # Semantic Bridge recursively resolves its own nested model roots and
        # therefore is not required to register a folder_paths category.
        if not result.get("semantic_bridge"):
            try:
                from .nodes_semantic_bridge import model_paths

                result["semantic_bridge"] = sorted(model_paths())
            except (ImportError, OSError, RuntimeError):
                pass
        return result
    except Exception:
        return {}


def _dependency_rows(capability_id: str, inventory: Mapping[str, Iterable[str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_names = [name.lower() for values in inventory.values() for name in values]

    def add(label: str, ok: bool, detail: str) -> None:
        rows.append({"id": label, "ok": bool(ok), "detail": detail})

    if capability_id == "semantic_bridge":
        matches = [name for name in inventory.get("semantic_bridge", ()) if name.lower().endswith(".safetensors")]
        add("bridge_weights", bool(matches), "Found " + ", ".join(matches[:3]) if matches else "no .safetensors under models/semantic_bridge")
    elif capability_id == "fast_h3_v2":
        matches = [name for name in all_names if "fasth3" in name or "fast_h3" in name]
        add("fasth3_assets", bool(matches), "Found FastH3-related model/LoRA" if matches else "No FastH3 V2 model or LoRA found; the recipe can be opened but not queued")
    elif capability_id == "meridian":
        matches = [name for name in inventory.get("meridian", ()) if "meridian" in name.lower()]
        add("meridian_weights", bool(matches), "Found Meridian resources" if matches else "No weights under models/meridian; Omega/ConvRot require separate licensing and installation")
    elif capability_id == "topaz":
        matches = [name for name in all_names if "topaz" in name or "iris" in name or "apollo" in name]
        discovered = None
        try:
            # Read-only filesystem metadata only.  This deliberately does not
            # launch Topaz, inspect a license, or claim that a model loaded.
            from .topaz_runtime import discover_official_installation
            discovered = discover_official_installation()
        except (ImportError, OSError, RuntimeError, ValueError):
            discovered = None
        if discovered:
            add("topaz_runtime", True,
                "Found official Topaz program, models definitions, and candidate weights; the environment node still verifies signature/filter/actual load")
        else:
            add("topaz_runtime", bool(matches),
                "The node handles runtime environment checks; the Topaz node re-verifies models/executables" if matches
                else "No complete Topaz program and models directory found; clear the three environment-node paths to enable auto-discover, or fill manually")
    else:
        add("native_runtime", True, "The entry node continues its own dependency, VRAM, and input checks at execution")
    return rows


def inspect_d3_routes(
    project: Mapping[str, object] | None = None,
    shot_id: str | None = None,
    store: ProjectStore | None = None,
    *,
    node_ids: Iterable[str] | None = None,
    model_inventory: Mapping[str, Iterable[str]] | None = None,
    capability: str | None = None,
) -> dict[str, object]:
    """Return a per-shot D3 hand-off report without queuing or mutating state."""

    if node_ids is None:
        try:
            import nodes

            node_ids = nodes.NODE_CLASS_MAPPINGS.keys()
        except Exception:
            node_ids = ()
    available = {str(value) for value in node_ids}
    inventory = dict(model_inventory or _model_names())
    compile_report = None
    selected_shot = None
    if project is not None:
        checked = validate_project(project)
        if store is not None:
            compile_report = compile_project(checked, store)
        selected_shot = next(
            (shot for shot in checked["doc"]["shots"] if shot["id"] == shot_id), None
        )
        if shot_id and selected_shot is None:
            raise ValueError("Please select a valid shot UUID in the project")

    rows = []
    specs = [spec for spec in CAPABILITY_SPECS if capability in (None, spec["id"])]
    for spec in specs:
        capability_id = str(spec["id"])
        required = [str(value) for value in spec["entry_nodes"]]
        missing = [value for value in required if value not in available]
        deps = _dependency_rows(capability_id, inventory)
        deps_ok = all(bool(row["ok"]) for row in deps)
        compile_ok = bool(compile_report is None or compile_report.get("ready"))
        state = (
            "ready_for_native_workflow"
            if not missing and deps_ok and compile_ok
            else "blocked_missing_entry_or_dependency"
        )
        rows.append(
            {
                "id": capability_id,
                "label": spec["label"],
                "state": state,
                "entry_nodes": required,
                "missing_entry_nodes": missing,
                "dependencies": deps,
                "route": spec["route"],
                "workflows": list(ROUTE_WORKFLOWS.get(capability_id, ())),
                "contract": dict(ROUTE_CONTRACTS.get(capability_id, {})),
                "shot_id": selected_shot.get("id") if selected_shot else None,
                "shot_name": selected_shot.get("name") if selected_shot else None,
                "compile_ready": compile_ok,
                "execution_boundary": "handoff_to_native_workflow; no implicit queue or quality claim",
                "next_action": "Open the corresponding native workflow and save it as a project copy, then execute per that node's real input/model gates",
            }
        )
    return {
        "schema": "t8.minimax_h3.director_d3_preflight.v1",
        "capabilities": rows,
        "project_id": project.get("id") if project else None,
        "shot_id": shot_id,
        "compile": compile_report,
        "warning": "ready_for_native_workflow means the current shot can be safely handed off to the native route; it does not imply GPU, human-review, or publication approval.",
    }
