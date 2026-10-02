from __future__ import annotations

import json
import copy
import re
from collections.abc import Mapping

import node_helpers

from .conditioning import build_packed_layout
from .core import FPS, align_frame_count, nested_av_parts
from .long_video import (
    CONTEXT_FRAME_STEPS,
    LONG_VIDEO_PATCH_VERSION,
    LONG_VIDEO_SCHEMA,
    build_long_video_conditioning,
    patch_long_video_model,
    repair_long_video_layout,
)
from .prompt_relay_advanced import (
    APPLY_EXP_TASKS,
    EXECUTION_MODES,
    PROMPT_RELAY_BINDING_KEY,
    _assert_core_contract,
    _attach_binding_model_cond,
    _bind_layout_contract,
    _install_prompt_relay_model,
    _sha256_json,
    _validate_plan,
    build_prompt_relay_binding,
)


PROMPT_RELAY_LONG_VIDEO_PROJECTION_SCHEMA = 1
PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY = "t8_prompt_relay_long_video_v1"
WINDOW_TEXT_POLICY_KEY = "long_video_window_text_policy"
WINDOW_TEXT_POLICIES = ("preserve_all", "accepted_window_text_exp", "dialogue_start_owner_exp")


def _dialogue_block_spans(text):
    """Strict native <d> pairs, used only by the explicitly selected experiment."""
    spans, opened = [], None
    for match in re.finditer(r"</?d(?=[\s>/]|$)", text, re.IGNORECASE):
        end = text.find(">", match.start())
        token = text[match.start():end + 1] if end >= 0 else ""
        if token not in ("<d>", "</d>"):
            raise ValueError("Dialogue owner experiment requires exact paired <d>...</d> tags")
        if token == "<d>":
            if opened is not None:
                raise ValueError("Dialogue owner experiment rejects nested <d> tags")
            opened = match.start()
        else:
            if opened is None:
                raise ValueError("Dialogue owner experiment rejects unpaired </d> tags")
            spans.append((opened, end + 1))
            opened = None
    if opened is not None:
        raise ValueError("Dialogue owner experiment rejects unclosed <d> tags")
    return spans


def _validate_dialogue_owner_source(source):
    if _dialogue_block_spans(source["global_prompt"]):
        raise ValueError("Move once-only <d> dialogue from Global into its timed Local event")
    for event in source["events"]:
        _dialogue_block_spans(event["local_prompt"])


def configure_long_video_window_text(prompt_relay_plan, text_policy="preserve_all"):
    """Opt in to window-local text without changing the old Plan or its defaults."""
    validated = _validate_plan(prompt_relay_plan)
    if text_policy not in WINDOW_TEXT_POLICIES:
        raise ValueError("Unknown Long Video window text policy")
    if "long_video_projection" in validated:
        raise ValueError("Configure the global Plan before projecting a window")
    previous = validated.get(WINDOW_TEXT_POLICY_KEY, "preserve_all")
    if previous not in WINDOW_TEXT_POLICIES:
        raise ValueError("Unknown Long Video window text policy in the Plan")
    if text_policy == "dialogue_start_owner_exp":
        _validate_dialogue_owner_source(validated)
    if text_policy == previous:
        output = prompt_relay_plan
    else:
        output = copy.deepcopy(validated)
        output.pop("plan_hash")
        if text_policy == "preserve_all":
            output.pop(WINDOW_TEXT_POLICY_KEY, None)
        else:
            output[WINDOW_TEXT_POLICY_KEY] = text_policy
        output["plan_hash"] = _sha256_json(output)
    return output, json.dumps({
        "status": "window_text_policy_configured",
        "text_policy": text_policy,
        "plan_hash": output["plan_hash"],
        "experimental": text_policy != "preserve_all",
        "notes": [
            "Only long-video window projection consumes this policy; ordinary Relay is unchanged",
            "Opt-in omits finished and future event text, not the accepted AV context or crossing-event sigma",
            "This changes text conditioning and is not the original paper all-key method or a no-repeat guarantee",
            "Cross-boundary events remain; place once-only dialogue wholly within a physical segment when possible",
            "dialogue_start_owner_exp keeps tagged dialogue only in the accepted window owning its event start; untagged speech is not recognized",
        ],
    }, ensure_ascii=False, indent=2)


def _seconds_to_exact_frame(value: float, name: str) -> int:
    frame = round(float(value) * FPS)
    if abs(float(value) * FPS - frame) > 1e-4:
        raise ValueError(f"{name} must lie on the native 24fps frame grid")
    return int(frame)


def project_prompt_relay_plan_to_long_video_window(
    prompt_relay_plan: Mapping,
    segment_index: int,
    length: int,
    context_frames: int,
    timeline_start_seconds: float,
    timeline_end_seconds: float,
) -> tuple[dict, str, str]:
    source = _validate_plan(prompt_relay_plan)
    text_policy = source.get(WINDOW_TEXT_POLICY_KEY, "preserve_all")
    if text_policy not in WINDOW_TEXT_POLICIES:
        raise ValueError("Unknown Long Video window text policy in the Plan")
    if text_policy == "dialogue_start_owner_exp":
        _validate_dialogue_owner_source(source)
    segment_index = int(segment_index)
    render_frames = int(length)
    context_frames = int(context_frames)
    if segment_index < 0:
        raise ValueError("Long Video Prompt Relay segment_index cannot be negative")
    if align_frame_count(render_frames) != render_frames:
        raise ValueError("Long Video Prompt Relay length must already satisfy the H3 17n+5 grid")
    if segment_index == 0:
        if context_frames != 0:
            raise ValueError("Long Video Prompt Relay segment 0 must use context_frames=0")
    elif context_frames not in CONTEXT_FRAME_STEPS:
        raise ValueError("Continuation context_frames must be 5, 22, or 39")

    accepted_start_frame = _seconds_to_exact_frame(
        timeline_start_seconds,
        "timeline_start_seconds",
    )
    accepted_end_frame = _seconds_to_exact_frame(
        timeline_end_seconds,
        "timeline_end_seconds",
    )
    if accepted_end_frame <= accepted_start_frame:
        raise ValueError("Long Video Prompt Relay timeline window must have positive duration")
    render_start_frame = accepted_start_frame - context_frames
    render_end_frame = render_start_frame + render_frames
    if render_start_frame < 0:
        raise ValueError("Long Video Prompt Relay context begins before global frame 0")
    if accepted_start_frame < render_start_frame or accepted_end_frame > render_end_frame:
        raise ValueError("Accepted long-video timeline is outside the rendered segment window")
    global_frame_count = int(source["frame_count"])
    if accepted_end_frame > global_frame_count:
        raise ValueError(
            "Long Video accepted timeline exceeds the global Prompt Relay plan; "
            "increase the global Plan length"
        )

    coordinate_shift = (5.0 / 3.0) * render_start_frame
    projected_events = []
    render_active = []
    accepted_active = []
    for event in source["events"]:
        projected = dict(event)
        global_start = int(event["start_frame"])
        global_end = int(event["end_frame_exclusive"])
        projected.update(
            {
                "global_start_frame": global_start,
                "global_end_frame_exclusive": global_end,
                "start_frame": global_start - render_start_frame,
                "end_frame_exclusive": global_end - render_start_frame,
                "start_seconds": (global_start - render_start_frame) / FPS,
                "end_seconds": (global_end - render_start_frame) / FPS,
                "start_coord": float(event["start_coord"]) - coordinate_shift,
                "end_coord": float(event["end_coord"]) - coordinate_shift,
                "midpoint": float(event["midpoint"]) - coordinate_shift,
            }
        )
        projected_events.append(projected)
        event_index = int(event["event_index"])
        if global_start < render_end_frame and global_end > render_start_frame:
            render_active.append(event_index)
        if global_start < accepted_end_frame and global_end > accepted_start_frame:
            accepted_active.append(event_index)

    # The legacy all-key path below remains byte-for-byte identical by default.
    # The explicit experiment changes text only, retaining absolute coordinates
    # and original widths/sigma for every event crossing the accepted window.
    text_details = None
    compiled_prompt = source["compiled_prompt"]
    if text_policy in ("accepted_window_text_exp", "dialogue_start_owner_exp"):
        kept = set(accepted_active)
        omitted = [e["event_index"] for e in projected_events if e["event_index"] not in kept]
        projected_events = [e for e in projected_events if e["event_index"] in kept]
        dialogue_owners, dialogue_omitted = [], []
        if text_policy == "dialogue_start_owner_exp":
            for event in projected_events:
                spans = _dialogue_block_spans(event["local_prompt"])
                if not spans:
                    continue
                if accepted_start_frame <= event["global_start_frame"] < accepted_end_frame:
                    dialogue_owners.append(event["event_index"])
                    continue
                # Keep the visual/emotion text and the original event timing;
                # do not trim, mute, replace or retime any accepted AV context.
                text = event["local_prompt"]
                for start, end in reversed(spans):
                    text = text[:start] + text[end:]
                event["local_prompt"] = text + (
                    " [The dialogue already started in an earlier segment; "
                    "continue the performance without restarting or repeating that line.]"
                )
                dialogue_omitted.append(event["event_index"])
        compiled_prompt = f"Global scene: {source['global_prompt']}"
        for event in projected_events:
            compiled_prompt += "\n"
            event["prompt_char_start"] = len(compiled_prompt)
            compiled_prompt += f"Event {event['event_index']}: {event['local_prompt']}"
            event["prompt_char_end"] = len(compiled_prompt)
        text_details = {"text_policy": text_policy,
                        "text_event_indices": [e["event_index"] for e in projected_events],
                        "omitted_text_event_indices": omitted}
        if text_policy == "dialogue_start_owner_exp":
            text_details.update(dialogue_owner_event_indices=dialogue_owners,
                                omitted_dialogue_event_indices=dialogue_omitted)

    projected_plan = dict(source)
    projected_plan.pop("plan_hash", None)
    projected_plan.update(
        {
            "timing_mode": "long_video_absolute_projection",
            "frame_count": render_frames,
            "events": projected_events,
            "global_plan_hash": source["plan_hash"],
            "global_frame_count": global_frame_count,
            "long_video_projection": {
                "schema": PROMPT_RELAY_LONG_VIDEO_PROJECTION_SCHEMA,
                "segment_index": segment_index,
                "render_start_frame": render_start_frame,
                "render_end_frame_exclusive": render_end_frame,
                "accepted_start_frame": accepted_start_frame,
                "accepted_end_frame_exclusive": accepted_end_frame,
                "context_frames": context_frames,
                "render_active_event_indices": render_active,
                "accepted_active_event_indices": accepted_active,
            },
        }
    )
    if text_details is not None:
        projected_plan["compiled_prompt"] = compiled_prompt
        projected_plan["long_video_projection"].update(text_details)
    projected_plan["plan_hash"] = _sha256_json(projected_plan)
    report = {
        "status": "long_video_window_projected",
        "experimental": True,
        "global_plan_hash": source["plan_hash"],
        "projected_plan_hash": projected_plan["plan_hash"],
        "segment_index": segment_index,
        "global_frame_count": global_frame_count,
        "render_window_frames": [render_start_frame, render_end_frame],
        "accepted_window_frames": [accepted_start_frame, accepted_end_frame],
        "context_frames": context_frames,
        "render_active_event_indices": render_active,
        "accepted_active_event_indices": accepted_active,
        "event_count_kept": len(projected_events),
        "notes": [
            "local frame 0 maps to accepted timeline start minus context overlap",
            "all event keys are retained and translated, so a cross-boundary event keeps its original width and sigma",
            "events outside this render window receive their original global-time decay rather than restarting from event 1",
        ],
    }
    if text_details is not None:
        report.update(text_details)
        report["notes"] = [
            "Explicit non-paper window-text experiment; old all-key projection is unchanged",
            "Only events intersecting accepted output are tokenized; context-only text is omitted",
            "Crossing events retain original global widths and sigma, not shortened or restarted",
            "No AV context, mask, sampler, time coordinate, reference media or seam processing is changed",
            "One or zero text events bypass Relay attention only, not native long-video conditioning",
        ]
        if text_policy == "dialogue_start_owner_exp":
            report["notes"].append(
                "Only an event-start owner keeps its paired <d> literals; crossing visual text "
                "remains. Native audio carry-over is neither muted nor shortened. No-repeat "
                "and utterance completion are unverified; untagged speech is not recognized."
            )
    return (
        projected_plan,
        projected_plan["compiled_prompt"],
        json.dumps(report, ensure_ascii=False, indent=2),
    )


def _validate_projected_window(
    prompt_relay_plan: Mapping,
    *,
    segment_index: int,
    length: int,
    context_frames: int,
) -> dict:
    plan = _validate_plan(prompt_relay_plan)
    projection = plan.get("long_video_projection")
    if not isinstance(projection, Mapping) or int(projection.get("schema", 0)) != 1:
        raise ValueError(
            "Long Video Prompt Relay Conditioning requires the paired projected Plan output"
        )
    expected = {
        "segment_index": int(segment_index),
        "context_frames": int(context_frames),
    }
    mismatches = [
        key for key, value in expected.items() if int(projection.get(key, -1)) != value
    ]
    if int(plan["frame_count"]) != int(length):
        mismatches.append("length")
    if mismatches:
        raise ValueError(
            "Long Video Prompt Relay projected Plan does not match Conditioning inputs: "
            + ", ".join(mismatches)
        )
    return plan


def build_prompt_relay_long_video_conditioning(
    model,
    clip,
    video_vae,
    audio_vae,
    context,
    prompt_relay_plan,
    segment_index,
    context_frames,
    context_audio,
    width,
    height,
    length,
    task_type,
    audio_mode,
    audio_denoise_strength,
    add_source_as_reference,
    prompt_primary_audio_ordinal,
    strict_prompt_tags,
    ref_image_size,
    reference_video_policy,
    execution_mode,
    query_chunk_rows,
    drive_audio=None,
    final_audio=None,
    first_frame=None,
    last_frame=None,
    ref_images=None,
    ref_videos=None,
    ref_video_audios=None,
    ref_audios=None,
    first_frame_reuse="segment0_only",
    persistent_identity_image=None,
    persistent_identity_strategy="single_reference",
    persistent_identity_interval=1,
    semantic_bridge=None,
):
    plan = _validate_projected_window(
        prompt_relay_plan,
        segment_index=segment_index,
        length=length,
        context_frames=context_frames,
    )
    if execution_mode not in EXECUTION_MODES:
        raise ValueError(f"Unknown Prompt Relay execution mode {execution_mode!r}")
    result = build_long_video_conditioning(
        clip=clip,
        video_vae=video_vae,
        audio_vae=audio_vae,
        context=context,
        segment_index=segment_index,
        context_frames=context_frames,
        context_audio=context_audio,
        prompt=plan["compiled_prompt"],
        width=width,
        height=height,
        length=length,
        task_type=task_type,
        audio_mode=audio_mode,
        audio_denoise_strength=audio_denoise_strength,
        add_source_as_reference=add_source_as_reference,
        prompt_primary_audio_ordinal=prompt_primary_audio_ordinal,
        strict_prompt_tags=strict_prompt_tags,
        ref_image_size=ref_image_size,
        reference_video_policy=reference_video_policy,
        drive_audio=drive_audio,
        final_audio=final_audio,
        first_frame=first_frame,
        last_frame=last_frame,
        ref_images=ref_images,
        ref_videos=ref_videos,
        ref_video_audios=ref_video_audios,
        ref_audios=ref_audios,
        first_frame_reuse=first_frame_reuse,
        persistent_identity_image=persistent_identity_image,
        persistent_identity_strategy=persistent_identity_strategy,
        persistent_identity_interval=persistent_identity_interval,
        return_details=True,
        semantic_bridge=semantic_bridge,
    )
    conditioning, latent, output_audio, conditioned_prompt, media_map, long_report, details = result
    binding = build_prompt_relay_binding(
        clip,
        plan,
        conditioned_prompt,
        conditioning,
        details["tokens"],
    )
    query_route = str(binding["query_route"])
    if (
        execution_mode == "apply_exp"
        and len(binding["events"]) > 1
        and query_route == "joint_av_exp"
        and details["audio_mode"] == "lock_source"
    ):
        raise ValueError(
            "Long Video Prompt Relay joint_av_exp cannot be combined with lock_source; "
            "use video_only_paper and mux_audio instead"
        )

    video, audio = nested_av_parts(latent)
    layout = build_packed_layout(
        binding["text_len"],
        int(video.shape[2]),
        int(video.shape[3]),
        int(video.shape[4]),
        int(audio.shape[-1]),
        keyframes=details["keyframes"],
        refs=details["refs"],
        frame_count=details["frame_count"],
    )
    layout = repair_long_video_layout(
        layout,
        list(details["keyframes"]),
        list(details["refs"]),
        int(details["frame_count"]),
    )
    binding = _bind_layout_contract(
        binding,
        layout,
        resolved_task=details["resolved_task"],
        keyframes=details["keyframes"],
        refs=details["refs"],
    )

    patched_model = model
    core_hashes = {}
    status = "report_only"
    if execution_mode == "apply_exp":
        resolved_task = str(details["resolved_task"]).lower()
        base_task = resolved_task.removesuffix("-motion")
        if base_task not in APPLY_EXP_TASKS:
            raise RuntimeError(
                f"Long Video Prompt Relay does not support task {resolved_task!r}"
            )
        core_hashes = _assert_core_contract(
            model,
            allowed_live_extra_conds_patch_versions=(LONG_VIDEO_PATCH_VERSION,),
        )
        long_video_model = patch_long_video_model(model)
        if len(binding["events"]) <= 1:
            patched_model = long_video_model
            status = (
                "passthrough_single_event_long_video_only"
                if binding["events"]
                else "passthrough_no_events_long_video_only"
            )
        else:
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {PROMPT_RELAY_BINDING_KEY: binding},
            )
            conditioning = _attach_binding_model_cond(
                conditioning,
                binding["binding_hash"],
            )
            patched_model, core_hashes = _install_prompt_relay_model(
                long_video_model,
                binding,
                int(query_chunk_rows),
                core_hashes,
            )
            if hasattr(patched_model, "set_attachments"):
                patched_model.set_attachments(
                    PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY,
                    {
                        "schema": PROMPT_RELAY_LONG_VIDEO_PROJECTION_SCHEMA,
                        "global_plan_hash": plan["global_plan_hash"],
                        "projected_plan_hash": plan["plan_hash"],
                        "binding_hash": binding["binding_hash"],
                        "segment_index": int(segment_index),
                    },
                )
            status = "applied_exp"

    projection = plan["long_video_projection"]
    report = {
        "status": status,
        "experimental": True,
        "long_video_schema": LONG_VIDEO_SCHEMA,
        "segment_index": int(segment_index),
        "resolved_task": details["resolved_task"],
        "audio_mode": details["audio_mode"],
        "query_route": query_route,
        "global_plan_hash": plan["global_plan_hash"],
        "projected_plan_hash": plan["plan_hash"],
        "binding_hash": binding["binding_hash"],
        "render_window_frames": [
            int(projection["render_start_frame"]),
            int(projection["render_end_frame_exclusive"]),
        ],
        "accepted_window_frames": [
            int(projection["accepted_start_frame"]),
            int(projection["accepted_end_frame_exclusive"]),
        ],
        "context_frames": int(context_frames),
        "event_count": len(binding["events"]),
        "render_active_event_indices": projection["render_active_event_indices"],
        "accepted_active_event_indices": projection["accepted_active_event_indices"],
        "packed_layout_hash": binding["layout_contract"]["contract_hash"],
        "query_chunk_rows": int(query_chunk_rows),
        "dense_s_by_s_mask_created": False,
        "attention_patch_installed": status == "applied_exp",
        "core_hashes": core_hashes,
        "long_video_report": json.loads(long_report),
        "stable_conditioning_report": long_report,
        "warnings": [
            "Long Video Prompt Relay is an isolated Advanced composition; old Long Video nodes are unchanged",
            "video_only_paper remains the default; joint_av_exp perceptual advantage remains unverified",
            "all global event keys remain tokenized per segment so cross-boundary sigma and absolute timing are preserved",
            (
                "zero or one active event bypasses only the Relay attention patch; the "
                "required scoped Long Video layout patch remains active"
            ),
        ],
    }
    if plan.get(WINDOW_TEXT_POLICY_KEY) in ("accepted_window_text_exp", "dialogue_start_owner_exp"):
        report["window_text_policy"] = plan[WINDOW_TEXT_POLICY_KEY]
        report["text_event_indices"] = projection["text_event_indices"]
        report["omitted_text_event_indices"] = projection["omitted_text_event_indices"]
        report["warnings"][2] = (
            "Explicit window-text experiment omits finished/future keys; crossing-event sigma "
            "and native accepted AV context stay unchanged. No dialogue-repeat guarantee."
        )
        if plan[WINDOW_TEXT_POLICY_KEY] == "dialogue_start_owner_exp":
            for key in ("dialogue_owner_event_indices", "omitted_dialogue_event_indices"):
                report[key] = projection[key]
            report["warnings"].append(
                "Tagged dialogue belongs to its event-start accepted window only; visual crossing "
                "events and native AV context remain. Untagged speech is not recognized. "
                "Experimental conditioning, not a no-repeat or utterance-completion guarantee."
            )
    return (
        patched_model,
        conditioning,
        latent,
        output_audio,
        conditioned_prompt,
        media_map,
        json.dumps(report, ensure_ascii=False, indent=2),
    )
