from __future__ import annotations

from pathlib import Path

import folder_paths
from comfy_api.latest import InputImpl, io, ui

from .mv_lipsync_advanced import (
    MV_PROMPT_PLAN_TYPE,
    MV_SCENE_PLAN_TYPE,
    MV_VOCAL_LOCK_PROMPT_PLAN_TYPE,
    build_mv_prompt_plan,
    build_mv_scene_plan,
    build_mv_vocal_lock_scene_plan,
    build_mv_vocal_lock_prompt_plan,
    build_mv_vocal_lock_visual_prompt_plan,
    run_local_mv_in_node_loop,
    run_local_mv_vocal_lock_in_node_loop,
    run_local_mv_vocal_lock_visual_in_node_loop,
)
from .prompt_relay_events_advanced import PROMPT_RELAY_EVENTS_TYPE
from .sampling import (
    DEFAULT_SAMPLER_NAME,
    DEFAULT_SCHEDULER_NAME,
    SAMPLER_OPTIONS,
    SCHEDULER_OPTIONS,
)


CATEGORY = "T8/MiniMax H3/MV & Lip Sync/Experimental"
MVScenePlanIO = io.Custom(MV_SCENE_PLAN_TYPE)
MVPromptPlanIO = io.Custom(MV_PROMPT_PLAN_TYPE)
MVVocalLockPromptPlanIO = io.Custom(MV_VOCAL_LOCK_PROMPT_PLAN_TYPE)
PromptRelayEventsIO = io.Custom(PROMPT_RELAY_EVENTS_TYPE)


def _preview_video(path_value: str):
    path = Path(path_value).resolve()
    output_root = Path(folder_paths.get_output_directory()).resolve()
    if output_root not in path.parents:
        raise ValueError("Local MV preview is outside the ComfyUI output directory")
    relative = path.relative_to(output_root)
    saved = ui.SavedResult(relative.name, relative.parent.as_posix(), io.FolderType.output)
    return InputImpl.VideoFromFile(str(path)), ui.PreviewVideo([saved])


class MiniMaxH3MVVocalScenePlannerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MVVocalScenePlannerT8Advanced",
            display_name=(
                "MiniMax H3 MV Vocal Scene Planner "
                "(Advanced EXP/T8)"
            ),
            description=(
                "Analyzes local AUDIO with deterministic CPU tensor math and chooses 5–10s "
                "scene boundaries before generation. A local vocal stem is optional. No remote "
                "LLM, TTS, music service or external API is called."
            ),
            category=CATEGORY,
            inputs=[
                io.Audio.Input("full_song", tooltip="Full original song used in the final output."),
                io.Float.Input(
                    "min_scene_seconds", default=5.0, min=1.0, max=None, step=0.1
                ),
                io.Float.Input(
                    "target_scene_seconds", default=7.0, min=1.0, max=None, step=0.1
                ),
                io.Float.Input(
                    "max_scene_seconds", default=10.0, min=1.0, max=None, step=0.1
                ),
                io.Int.Input(
                    "analysis_hop_ms",
                    default=100,
                    min=20,
                    max=500,
                    step=10,
                    advanced=True,
                ),
                io.Combo.Input(
                    "vocal_policy",
                    options=["assume_vocal", "energy_proxy", "vocal_stem_required"],
                    default="assume_vocal",
                    tooltip=(
                        "assume_vocal is recommended when no vocal stem is available; energy_proxy estimates by energy only; "
                        "vocal_stem_required requires a connected local vocal stem."
                    ),
                ),
                io.String.Input(
                    "manual_boundaries_json",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip='Optional list of seconds, e.g. [5.2, 11.8]; leave empty for auto-analysis.',
                ),
                io.Audio.Input("vocal_stem", optional=True, tooltip="Optional local vocal stem."),
            ],
            outputs=[
                MVScenePlanIO.Output("scene_plan"),
                io.Int.Output("scene_count"),
                io.Float.Output("duration_seconds"),
                io.String.Output("timeline_json"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_mv_scene_plan(**kwargs))


class MiniMaxH3MVRef2VAPromptCompilerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MVRef2VAPromptCompilerT8Advanced",
            display_name=(
                "MiniMax H3 MV Ref2VA Prompt Compiler "
                "(Advanced EXP/T8)"
            ),
            description=(
                "Deterministically compiles one six-section Ref2VA prompt per local scene. "
                "It uses <Picture 1>/<Audio 1>, never guesses lyrics and also exposes typed "
                "Prompt Relay events. No LLM or external API is used."
            ),
            category=CATEGORY,
            inputs=[
                MVScenePlanIO.Input("scene_plan"),
                io.String.Input(
                    "global_creative_prompt",
                    default=(
                        "A singer performs through a coherent cinematic music video with "
                        "natural expression and intentional scene changes."
                    ),
                    multiline=True,
                    dynamic_prompts=True,
                ),
                io.String.Input(
                    "performer_description",
                    default="the same lead performer shown in the reference picture",
                    multiline=True,
                ),
                io.String.Input(
                    "visual_style",
                    default="cinematic music video, natural skin, realistic light and texture",
                    multiline=True,
                ),
                io.String.Input(
                    "camera_pattern",
                    default=(
                        "stable medium close-up with subtle handheld movement\n"
                        "smooth lateral tracking medium shot\n"
                        "restrained slow push-in with a stable background"
                    ),
                    multiline=True,
                    tooltip="One shot per line or per | separator; cycles by scene.",
                ),
                io.String.Input(
                    "non_vocal_action",
                    default="keeps the mouth naturally closed and moves with the rhythm",
                    multiline=True,
                ),
                io.String.Input(
                    "exact_lyrics_json",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip=(
                        "Optional: provide exact lyric strings per scene. When empty, lyrics are never guessed and no subtitles are generated."
                    ),
                ),
            ],
            outputs=[
                MVPromptPlanIO.Output("mv_prompt_plan"),
                io.String.Output("segment_prompts_json"),
                PromptRelayEventsIO.Output("prompt_relay_events"),
                io.String.Output("prompt_preview"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_mv_prompt_plan(**kwargs))


class MiniMaxH3LocalMVInNodeRendererT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LocalMVInNodeRendererT8Advanced",
            display_name=(
                "MiniMax H3 Local MV In-Node Renderer "
                "(Advanced EXP/T8)"
            ),
            description=(
                "Generates every Ref2VA scene serially through the connected local H3 MODEL, "
                "atomically resumes accepted scenes, assembles video with bounded memory and "
                "muxes the original full song once. It never submits ComfyUI /prompt HTTP jobs "
                "and never calls a remote video, LLM, TTS or music API."
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model", tooltip="Local MiniMax H3 MODEL."),
                io.Clip.Input("clip", tooltip="Local MiniMax H3 Qwen3-VL CLIP."),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.Image.Input("reference_image", tooltip="Singer/character identity reference image."),
                io.Audio.Input("full_song", tooltip="Original song that drives the performance and is mixed into the final output once."),
                MVPromptPlanIO.Input("mv_prompt_plan"),
                io.String.Input("chain_id", default="my_h3_local_mv"),
                io.Int.Input(
                    "width",
                    default=1024,
                    min=32,
                    max=16384,
                    step=32,
                    tooltip="Official Ref2V Turbo v0.1 validated dimensions; override explicitly if another aspect is needed.",
                ),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("base_seed", default=123456789, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("steps", default=8, min=1, max=1000),
                io.Float.Input(
                    "shift_video",
                    default=6.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "shift_audio",
                    default=3.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=SAMPLER_OPTIONS,
                    default=DEFAULT_SAMPLER_NAME,
                ),
                io.Combo.Input(
                    "scheduler",
                    options=SCHEDULER_OPTIONS,
                    default=DEFAULT_SCHEDULER_NAME,
                ),
                io.Boolean.Input("resume_existing", default=True),
                io.String.Input("filename_prefix", default="H3_Local_MV"),
                io.Combo.Input("bit_depth", options=[8, 10], default=8, advanced=True),
                io.Int.Input("crf", default=18, min=0, max=51, advanced=True),
                io.String.Input(
                    "model_id",
                    default="user-selected-local-h3",
                    advanced=True,
                    tooltip="Writes only to the audit report; does not verify filename, size, or hash.",
                ),
            ],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("video_path"),
                io.String.Output("manifest_path"),
                io.Int.Output("completed_scenes"),
                io.String.Output("status"),
                io.String.Output("report_json"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        video_path, manifest_path, completed, status, report = run_local_mv_in_node_loop(
            **kwargs
        )
        video, preview = _preview_video(video_path)
        return io.NodeOutput(
            video,
            video_path,
            manifest_path,
            completed,
            status,
            report,
            ui=preview,
        )


MV_LIPSYNC_ADVANCED_NODE_CLASSES = [
    MiniMaxH3MVVocalScenePlannerT8Advanced,
    MiniMaxH3MVRef2VAPromptCompilerT8Advanced,
    MiniMaxH3LocalMVInNodeRendererT8Advanced,
]


class MiniMaxH3MVVocalLockScenePlannerV2T8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MVVocalLockScenePlannerV2T8Advanced",
            display_name=(
                "MiniMax H3 MV Vocal Lock Scene Planner V2 "
                "(Advanced EXP/T8)"
            ),
            description=(
                "Requires full_song and a timeline-aligned isolated vocal_lock_audio. It uses "
                "local CPU energy analysis for boundaries and vocal-active intervals, without "
                "transcription, remote separation, an LLM, or any external API."
            ),
            category=CATEGORY,
            inputs=[
                io.Audio.Input("full_song", tooltip="Full original song used in the final output."),
                io.Audio.Input(
                    "vocal_lock_audio",
                    tooltip="Required: a local isolated vocal or clear dialogue aligned to the same start time and timeline as the original song.",
                ),
                io.Float.Input(
                    "min_scene_seconds", default=5.0, min=1.0, max=None, step=0.1
                ),
                io.Float.Input(
                    "target_scene_seconds", default=7.0, min=1.0, max=None, step=0.1
                ),
                io.Float.Input(
                    "max_scene_seconds", default=10.0, min=1.0, max=None, step=0.1
                ),
                io.Int.Input(
                    "analysis_hop_ms",
                    default=50,
                    min=20,
                    max=500,
                    step=10,
                    advanced=True,
                ),
                io.Float.Input(
                    "vocal_active_ratio",
                    default=0.12,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Minimum fraction of time within a scene above the local vocal energy threshold.",
                ),
                io.String.Input(
                    "manual_boundaries_json",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip='Optional list of seconds, e.g. [5.2, 11.8]; leave empty for auto-analysis.',
                ),
            ],
            outputs=[
                MVScenePlanIO.Output("scene_plan"),
                io.Int.Output("scene_count"),
                io.Float.Output("duration_seconds"),
                io.String.Output("timeline_json"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_mv_vocal_lock_scene_plan(**kwargs))


class MiniMaxH3MVVocalLockPromptCompilerV2T8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MVVocalLockPromptCompilerV2T8Advanced",
            display_name=(
                "MiniMax H3 MV Vocal Lock Prompt Compiler V2 "
                "(Advanced EXP/T8)"
            ),
            description=(
                "Compiles the official six-section Ref2VA structure with <Subject 1> from "
                "<Picture 1>, isolated <Audio 1> reuse, fully_preserved/fully_copy markers, "
                "and mandatory visible-mouth performance framing. It is deterministic and local."
            ),
            category=CATEGORY,
            inputs=[
                MVScenePlanIO.Input("scene_plan"),
                io.String.Input(
                    "global_creative_prompt",
                    default=(
                        "A coherent cinematic performance focused on the same lead performer, "
                        "with natural expression and restrained motion."
                    ),
                    multiline=True,
                    dynamic_prompts=True,
                ),
                io.String.Input(
                    "performer_description",
                    default="the same lead performer shown in the reference picture",
                    multiline=True,
                ),
                io.String.Input(
                    "visual_style",
                    default="cinematic realism, natural skin, realistic light and texture",
                    multiline=True,
                ),
                io.String.Input(
                    "camera_pattern",
                    default=(
                        "locked-off static camera with stable framing, no camera movement, "
                        "and continuous mouth visibility"
                    ),
                    multiline=True,
                    tooltip=(
                        "One shot plan per line or per | separator. Camera is locked by default to reduce character contour ghosting; explicitly switch to push/pull, "
                        "handheld, or lateral movement increases temporal ghosting risk. V2 still enforces medium close-up, front or 3/4 face."
                    ),
                ),
                io.Combo.Input(
                    "vocal_content_type",
                    options=["singing", "spoken_dialogue"],
                    default="singing",
                ),
                io.String.Input("vocal_language", default="English", advanced=True),
                io.String.Input(
                    "exact_vocal_text_json",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip=(
                        "Optional: provide exact lyric/dialogue strings per scene; only the text here enters <d>. When empty, no words are guessed."
                    ),
                ),
                io.String.Input(
                    "non_vocal_action",
                    default="keeps the mouth naturally closed and breathes with the rhythm",
                    multiline=True,
                    advanced=True,
                ),
            ],
            outputs=[
                MVVocalLockPromptPlanIO.Output("mv_vocal_lock_prompt_plan"),
                io.String.Output("segment_prompts_json"),
                PromptRelayEventsIO.Output("prompt_relay_events"),
                io.String.Output("prompt_preview"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_mv_vocal_lock_prompt_plan(**kwargs))


class MiniMaxH3LocalMVVocalLockRendererV2T8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LocalMVVocalLockRendererV2T8Advanced",
            display_name=(
                "MiniMax H3 Local MV Vocal Lock Renderer V2 "
                "(Advanced EXP/T8)"
            ),
            description=(
                "Uses the required isolated vocal_lock_audio for each local H3 Ref2VA lock_source "
                "window, samples strictly serially through the connected MODEL, and reserves "
                "full_song for one final delivery mux. No HTTP prompt queue or external API is used."
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model", tooltip="Local MiniMax H3 MODEL."),
                io.Clip.Input("clip", tooltip="Local MiniMax H3 Qwen3-VL CLIP."),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.Image.Input("reference_image", tooltip="Identity reference image of the singer/speaking character."),
                io.Audio.Input("full_song", tooltip="Full original song mixed into the final delivery once."),
                io.Audio.Input(
                    "vocal_lock_audio",
                    tooltip="Required: local isolated vocal/clear dialogue on the same timeline as full_song; drives H3 per scene.",
                ),
                MVVocalLockPromptPlanIO.Input("mv_vocal_lock_prompt_plan"),
                io.String.Input("chain_id", default="my_h3_local_mv_vocal_lock_v2"),
                io.Int.Input("width", default=1056, min=32, max=16384, step=32),
                io.Int.Input("height", default=608, min=32, max=16384, step=32),
                io.Int.Input("base_seed", default=123456789, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("steps", default=8, min=1, max=1000),
                io.Float.Input(
                    "shift_video",
                    default=6.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "shift_audio",
                    default=3.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=SAMPLER_OPTIONS,
                    default=DEFAULT_SAMPLER_NAME,
                ),
                io.Combo.Input(
                    "scheduler",
                    options=SCHEDULER_OPTIONS,
                    default=DEFAULT_SCHEDULER_NAME,
                ),
                io.Boolean.Input("resume_existing", default=True),
                io.String.Input("filename_prefix", default="H3_Local_MV_VocalLock_V2"),
                io.Combo.Input("bit_depth", options=[8, 10], default=8, advanced=True),
                io.Int.Input("crf", default=18, min=0, max=51, advanced=True),
                io.String.Input(
                    "model_id",
                    default="user-selected-local-h3-ref2va-vocal-lock",
                    advanced=True,
                    tooltip="Writes only to the audit report; does not verify model filename, size, or hash.",
                ),
            ],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("video_path"),
                io.String.Output("manifest_path"),
                io.Int.Output("completed_scenes"),
                io.String.Output("status"),
                io.String.Output("report_json"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        video_path, manifest_path, completed, status, report = (
            run_local_mv_vocal_lock_in_node_loop(**kwargs)
        )
        video, preview = _preview_video(video_path)
        return io.NodeOutput(
            video,
            video_path,
            manifest_path,
            completed,
            status,
            report,
            ui=preview,
        )


MV_LIPSYNC_V2_ADVANCED_NODE_CLASSES = [
    MiniMaxH3MVVocalLockScenePlannerV2T8Advanced,
    MiniMaxH3MVVocalLockPromptCompilerV2T8Advanced,
    MiniMaxH3LocalMVVocalLockRendererV2T8Advanced,
]


class MiniMaxH3MVVocalLockVisualDirectorV3T8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MVVocalLockVisualDirectorV3T8Advanced",
            display_name=(
                "MiniMax H3 MV Vocal Lock Visual Director V3 "
                "(Advanced EXP/T8)"
            ),
            description=(
                "Compiles one explicit six-section Ref2VA shot per scene with an exact one-person, "
                "one-face contract that forbids mirrors, projections, screens, posters, duplicate "
                "faces, background people and visible props. It is append-only and leaves V2 unchanged."
            ),
            category=CATEGORY,
            inputs=[
                MVScenePlanIO.Input("scene_plan"),
                io.String.Input(
                    "global_creative_prompt",
                    default=(
                        "A coherent cinematic performance focused on the same lead performer, "
                        "with natural expression and restrained motion."
                    ),
                    multiline=True,
                    dynamic_prompts=True,
                ),
                io.String.Input(
                    "performer_description",
                    default="the same lead performer shown in the reference picture",
                    multiline=True,
                ),
                io.String.Input(
                    "visual_style",
                    default="cinematic realism, natural skin, realistic light and texture",
                    multiline=True,
                ),
                io.String.Input(
                    "scene_directions_json",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Optional: must be the same length as scene_count; each item is a camera/lighting/performance/emotion object."
                        "Leave empty for a safe studio arc; conflicting mirror, projection, poster, screen, background-character, and prop words are rejected."
                    ),
                ),
                io.Combo.Input(
                    "vocal_content_type",
                    options=["singing", "spoken_dialogue"],
                    default="singing",
                ),
                io.String.Input("vocal_language", default="English", advanced=True),
                io.String.Input(
                    "exact_vocal_text_json",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip="Optional: provide exact lyrics/dialogue per scene; when empty, no words are guessed.",
                ),
                io.String.Input(
                    "non_vocal_action",
                    default="keeps the mouth naturally closed and breathes with the rhythm",
                    multiline=True,
                    advanced=True,
                ),
            ],
            outputs=[
                MVVocalLockPromptPlanIO.Output("mv_vocal_lock_prompt_plan"),
                io.String.Output("segment_prompts_json"),
                PromptRelayEventsIO.Output("prompt_relay_events"),
                io.String.Output("prompt_preview"),
                io.String.Output("report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_mv_vocal_lock_visual_prompt_plan(**kwargs))


class MiniMaxH3LocalMVVocalLockVisualRendererV3T8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LocalMVVocalLockVisualRendererV3T8Advanced",
            display_name=(
                "MiniMax H3 Local MV Vocal Lock Visual Renderer V3 "
                "(Advanced EXP/T8)"
            ),
            description=(
                "Runs only V3 visual-director prompt plans through the connected local H3 MODEL, "
                "using an independent resume contract and preserving full_song for one final mux."
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model", tooltip="Local MiniMax H3 MODEL."),
                io.Clip.Input("clip", tooltip="Local MiniMax H3 Qwen3-VL CLIP; a prefix-cache wrapper may be connected."),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.Image.Input("reference_image", tooltip="Clear character identity reference image, front or 3/4 face."),
                io.Audio.Input("full_song", tooltip="Full original song mixed into the final delivery once."),
                io.Audio.Input(
                    "vocal_lock_audio",
                    tooltip="Vocal/dialogue on the same timeline as full_song; drives H3 per scene.",
                ),
                MVVocalLockPromptPlanIO.Input("mv_vocal_lock_prompt_plan"),
                io.String.Input("chain_id", default="my_h3_local_mv_vocal_lock_v3"),
                io.Int.Input("width", default=1056, min=32, max=16384, step=32),
                io.Int.Input("height", default=608, min=32, max=16384, step=32),
                io.Int.Input("base_seed", default=123456789, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input(
                    "steps",
                    default=4,
                    min=1,
                    max=1000,
                    tooltip="Official Ref2V Turbo v0.1 uses 4 NFE; do not run this distilled LoRA as an 8-step schedule.",
                ),
                io.Float.Input(
                    "shift_video",
                    default=12.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Official Ref2VA Turbo4 video shift.",
                ),
                io.Float.Input(
                    "shift_audio", default=3.0, min=0.01, max=100.0, step=0.01, advanced=True
                ),
                io.Combo.Input(
                    "sampler_name", options=SAMPLER_OPTIONS, default="euler"
                ),
                io.Combo.Input(
                    "scheduler", options=SCHEDULER_OPTIONS, default="simple"
                ),
                io.Boolean.Input("resume_existing", default=True),
                io.String.Input("filename_prefix", default="H3_Local_MV_VocalLock_V3"),
                io.Combo.Input("bit_depth", options=[8, 10], default=8, advanced=True),
                io.Int.Input("crf", default=18, min=0, max=51, advanced=True),
                io.String.Input(
                    "model_id",
                    default="minimax_h3_ref2va+official_ref2v_turbo4_v0.1",
                    advanced=True,
                    tooltip="Writes only to the audit report; does not verify model filename, size, or hash.",
                ),
            ],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("video_path"),
                io.String.Output("manifest_path"),
                io.Int.Output("completed_scenes"),
                io.String.Output("status"),
                io.String.Output("report_json"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs):
        video_path, manifest_path, completed, status, report = (
            run_local_mv_vocal_lock_visual_in_node_loop(**kwargs)
        )
        video, preview = _preview_video(video_path)
        return io.NodeOutput(
            video,
            video_path,
            manifest_path,
            completed,
            status,
            report,
            ui=preview,
        )


MV_LIPSYNC_V3_ADVANCED_NODE_CLASSES = [
    MiniMaxH3MVVocalLockVisualDirectorV3T8Advanced,
    MiniMaxH3LocalMVVocalLockVisualRendererV3T8Advanced,
]
