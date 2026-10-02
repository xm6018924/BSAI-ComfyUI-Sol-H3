"""Append-only two-MODEL learned latent-upscale long-video node."""
from pathlib import Path
import json
import time

import folder_paths
import torch
from comfy_api.latest import io

from .learned_latent_upscale_advanced import learned_upscale_geometry, PIXELS_PER_H3_LATENT
from .long_video_dual_identity import content_identity, stage_model_identity, _implementation
from .long_video_dual_model_runner import DualModelSegmentRunner
from .long_video_delivery import _sha256_file
from .long_video_in_node_loop_effects_advanced import run_long_video_in_node_loop_effects
from .nodes_long_video_in_node_loop_effects_advanced import MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced, _preview_video
from .patch_stack_policy import UnverifiedModelStack, nonportable_component_identity


def _component_identity(component):
    try:
        return _audited_component_identity(component)
    except UnverifiedModelStack as error:
        return nonportable_component_identity(component, str(error), schema="h3_component_user_stack_v1")


def _audited_component_identity(component):
    patcher = getattr(component, "patcher", None)
    if patcher is None or not callable(getattr(patcher, "model_state_dict", None)):
        raise ValueError("CLIP/VAE has no verifiable state identity; use a native H3 component")
    object_patches = dict(getattr(patcher, "object_patches", {}))
    # Core CLIP always installs its FP32 compute cast; this is not a foreign
    # model hook. Include it in the identity rather than rejecting every CLIP.
    cast = object_patches.pop("manual_cast_dtype", None)
    if cast is not None and not isinstance(cast, torch.dtype):
        raise ValueError("CLIP/VAE manual cast must be a dtype, not an execution object")
    if getattr(patcher, "patches", {}) or object_patches:
        raise UnverifiedModelStack("CLIP/VAE component patches have no portable identity adapter")
    for name in ("wrappers", "callbacks", "injections", "hook_patches", "forced_hooks", "current_hooks"):
        if getattr(patcher, name, None):
            raise UnverifiedModelStack(f"CLIP/VAE contains unverified runtime {name}")
    result = {"implementation": _implementation(type(component)), "manual_cast_dtype": str(cast),
              "state": content_identity(patcher.model_state_dict())}
    if hasattr(component, "tokenizer"):
        from .long_video_dual_tokenizer import tokenizer_identity
        result["tokenizer"] = tokenizer_identity(component)
    return result


class MiniMaxH3DualModelLongVideoEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        original = MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced.define_schema()
        removed = {"model", "steps", "shift_video", "shift_audio", "sampler_name", "scheduler",
                   "model_id", "long_video_sampling_plan", "semantic_bridge"}
        inherited = [item for item in original.inputs if item.id not in removed]
        for item in inherited:
            if item.id == "chain_id":
                item.default = "h3_dual_model_long_video"
            elif item.id == "width":
                item.default = 1024
            elif item.id == "height":
                item.default = 576
            elif item.id == "total_duration_seconds":
                item.default = 24.
        return io.Schema(node_id="MiniMaxH3DualModelLongVideoEXPT8",
            display_name="MiniMax H3 双MODEL潜空间放大长视频 / Dual Model Long Video (EXP/T8)",
            category=original.category, is_experimental=True, is_output_node=True,
            description="Serial low-resolution first MODEL → learned latent upscale → independent high-resolution second MODEL. "
                        "Each MODEL can have its own LoRA. Width/height are final resolution; low_width/low_height are first pass. "
                        "Authenticated T8 LowVRAM Attention and ChunkFFN nodes may be connected independently to either MODEL. "
                        "Auto audio continues joint AV for partial4+4; only complete Stock20 first-pass audio is locked. "
                        "Both spatial continuations use completed output audio. Older partial first_pass zero-lock settings are migrated. "
                        "EAV requires coarse_steps20; disable it for4+4. Resume verifies content, not filenames.",
            inputs=[io.Model.Input("model_pass1"), io.Model.Input("model_pass2"),
                    io.Int.Input("low_width", default=512, min=32, max=16384, step=32),
                    io.Int.Input("low_height", default=288, min=32, max=16384, step=32),
                    io.Combo.Input("upscaler_model", options=folder_paths.get_filename_list("latent_upscale_models")),
                    io.Combo.Input("coarse_steps", options=[4, 20], default=4),
                    io.Combo.Input("refine_steps", options=[3, 4, 5], default=4),
                    io.Float.Input("first_shift_video", default=12., min=.01, max=100.),
                    io.Float.Input("first_shift_audio", default=3., min=.01, max=100.),
                    io.Float.Input("second_shift_video", default=12., min=.01, max=100.),
                    io.Float.Input("second_shift_audio", default=3., min=.01, max=100.),
                    io.Combo.Input("second_audio_source", options=["auto", "legacy_policy", "first_pass", "highres_template"], default="auto"),
                    io.Float.Input("second_audio_strength", default=0., min=0., max=1., step=.05),
                    *inherited,
                    io.Boolean.Input('color_match', default=True, optional=True,
                        tooltip='Match each continuation to the accepted RGB tail; bounded color correction only, not geometry repair.'),
                    io.Combo.Input('video_context_mode',
                        options=['reference_only', 'high_native_mask_exp', 'high_native_mask_ramp_exp'],
                        default='reference_only', optional=True,
                        tooltip='EXP: constrain high-pass overlap to the accepted final tail. Ramp mode releases three latent cells at 0.25/0.5/0.75. Audio unchanged; inspect the full continuation.'),
                    io.Combo.Input('low_context_source',
                        options=['independent_low_x0', 'accepted_picture_low_context_v1'],
                        default='independent_low_x0', optional=True,
                        tooltip='Accepted picture: re-encode the previous accepted movie tail for LOW video guidance only. Adds a short VAE encode, no sampling steps. New chain_id when switching. Example reviewed at 0.4MP/8s/22 context/4+4.'),
                    io.Combo.Input('color_match_mode',
                        options=['bounded_spatial_v2', 'bounded_spatial_temporal_exp', 'bounded_motion_color_exp'],
                        default='bounded_spatial_v2', optional=True,
                        tooltip='Temporal mode suppresses short RGB flicker in the first12 continuation frames. Motion Color EXP additionally corrects confident, bracketed local color outliers; requires OpenCV. No frame blending, geometry or audio changes; new chain_id required.'),
                    io.Custom("T8_SEMANTIC_BRIDGE").Input("semantic_bridge", optional=True),
                    io.Custom("T8_SEMANTIC_BRIDGE").Input("semantic_bridge_pass1", optional=True),
                    io.Custom("T8_SEMANTIC_BRIDGE").Input("semantic_bridge_pass2", optional=True)], outputs=original.outputs)

    @classmethod
    def execute(cls, model_pass1, model_pass2, low_width, low_height, upscaler_model,
                coarse_steps, refine_steps, first_shift_video, first_shift_audio,
                second_shift_video, second_shift_audio, second_audio_source, second_audio_strength, color_match=True,
                video_context_mode='reference_only', low_context_source='independent_low_x0',
                color_match_mode='bounded_spatial_v2', _fast_h3_v2_profile=None,
                semantic_bridge_pass1=None, semantic_bridge_pass2=None, **kwargs):
        from .semantic_bridge import preflight_bridge
        common_bridge = kwargs.get("semantic_bridge")
        bridge_configs = tuple(config if config is not None else common_bridge
                               for config in (semantic_bridge_pass1, semantic_bridge_pass2))
        bridge_identities = [preflight_bridge(config) for config in bridge_configs]
        geometry = learned_upscale_geometry(low_width // PIXELS_PER_H3_LATENT, low_height // PIXELS_PER_H3_LATENT,
            "target_dimensions", 2., 1., kwargs["width"], kwargs["height"], "honor_dimensions_exp", 1.05)
        if geometry["output_width"] != kwargs["width"] or geometry["output_height"] != kwargs["height"]:
            raise ValueError("Final dimensions are not representable by the learned latent-upscale geometry")
        if kwargs["width"] <= low_width or kwargs["height"] <= low_height:
            raise ValueError("Both final dimensions must be larger than the first-pass dimensions")
        eav_config = {"mode": kwargs["eav_mode"], "tau": kwargs["eav_tau"],
                      "start_video_progress": kwargs["eav_start_video_progress"],
                      "end_video_progress": kwargs["eav_end_video_progress"],
                      "max_workspace_mib": kwargs["eav_max_workspace_mib"], "g_hard_limit": kwargs["eav_g_hard_limit"]}
        settings = dict(low_width=low_width, low_height=low_height, upscaler_model=upscaler_model,
            coarse_steps=coarse_steps, refine_steps=refine_steps, first_shift_video=first_shift_video,
            first_shift_audio=first_shift_audio, second_shift_video=second_shift_video, second_shift_audio=second_shift_audio,
            prompt_relay_mode=kwargs["prompt_relay_mode"], query_chunk_rows=kwargs["query_chunk_rows"],
            second_audio_source=second_audio_source, second_audio_strength=second_audio_strength, eav_config=eav_config,
            color_match=bool(color_match))
        # Validate settings before expensive full-content hashing or any GPU work.
        if video_context_mode != 'reference_only':
            settings['video_context_mode'] = video_context_mode
        if low_context_source != 'independent_low_x0':
            settings['low_context_source'] = low_context_source
        if color_match_mode != 'bounded_spatial_v2':
            settings['color_match_mode'] = color_match_mode
        if _fast_h3_v2_profile is not None:
            settings['fast_h3_v2_profile'] = _fast_h3_v2_profile
        engine = DualModelSegmentRunner(model_pass1, model_pass2, contract={}, **settings)
        engine.bridge_configs = bridge_configs
        engine.bridge_identities = tuple(bridge_identities)
        if any(identity is not None for identity in bridge_identities):
            settings["semantic_bridges"] = bridge_identities
        started = time.perf_counter()
        path = folder_paths.get_full_path_or_raise("latent_upscale_models", upscaler_model)
        first_identity = stage_model_identity(model_pass1)
        second_identity = stage_model_identity(model_pass2)
        identities = {"first_model": first_identity, "second_model": second_identity,
                      "upscaler_sha256": _sha256_file(Path(path)), "geometry": geometry, "settings": settings,
                      "media": content_identity({key: kwargs.get(key) for key in
                          ("first_frame", "last_frame", "drive_audio", "final_audio", "persistent_identity_image",
                           "ref_images", "ref_videos", "ref_video_audios", "ref_audios")}),
                      "components": {key: _component_identity(kwargs[key]) for key in ("clip", "video_vae", "audio_vae")},
                      "implementation_sha256s": {path.name: _sha256_file(path)
                          for path in sorted(Path(__file__).parent.glob('*.py'))}}
        if engine.motion_color_runtime_identity is not None:
            identities['motion_color_runtime_identity'] = engine.motion_color_runtime_identity
        engine.contract = identities
        engine.identity_seconds = time.perf_counter() - started
        # Descriptor fields remain readable, but are not used as identity proof.
        model_id = first_identity["sha256"][:16] + ":" + second_identity["sha256"][:16]
        video_path, manifest_path, completed, status, report = run_long_video_in_node_loop_effects(
            model_pass2, steps=20 if coarse_steps == 20 else 8, shift_video=second_shift_video,
            shift_audio=second_shift_audio, sampler_name="dual_clock_euler", scheduler="native_flow",
            model_id=model_id, _stage_runner=engine, **kwargs)
        payload = json.loads(report)
        payload['dual_model_identity_scan_seconds'] = engine.identity_seconds
        payload['dual_model_identity_scope'] = 'actual model/LoRA/components/media content; complete local runtime source files'
        report = json.dumps(payload, ensure_ascii=False)
        video, preview = _preview_video(video_path)
        return io.NodeOutput(video, video_path, manifest_path, completed, status, report, ui=preview)
