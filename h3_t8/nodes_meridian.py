"""Four append-only Meridian V3 entries. Lazy external dependencies and no server GPU thread."""

import json

from comfy_api.latest import io

from .meridian_plan import (
    PRESETS,
    canonical_plan,
    plan_json,
    preset_plan,
    source_map,
    identity,
)

RuntimeIO = io.Custom("T8_MERIDIAN_RUNTIME")
MaterialIO = io.Custom("T8_MERIDIAN_MATERIAL")
PlanIO = io.Custom("T8_MERIDIAN_PREPARED")
CATEGORY = "T8/MiniMax H3/Meridian (EXP)"


def schema(cls, name, inputs, outputs, **kwargs):
    return io.Schema(
        node_id=cls.__name__,
        display_name=name,
        category=CATEGORY,
        is_experimental=True,
        inputs=inputs,
        outputs=outputs,
        **kwargs,
    )


class MiniMaxH3MeridianConfigEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Meridian 1 · Models & Resources (INT8/T8)",
            [
                io.String.Input(
                    "model_path",
                    default="",
                    tooltip="Leave empty to auto-find models",
                ),
                io.String.Input(
                    "vae_path",
                    default="",
                    tooltip="Default models/vae/minimax_h3_video_vae_fp16.safetensors; can be replaced with a standard video_vae input",
                ),
                io.String.Input(
                    "source_directory",
                    default="",
                    tooltip="Default models/meridian/source; pins the 2083d059 source and assets; MERIDIAN_SOURCE_DIR is supported",
                ),
                io.String.Input(
                    "omega_source_directory",
                    default="",
                    tooltip="meridian/vggt-omega；VGGT_OMEGA_DIR",
                ),
                io.String.Input(
                    "omega_checkpoint",
                    default="",
                    tooltip="Correct Omega1B512 original PT, not generic VGGT or INT8. Default models",
                ),
                io.Float.Input(
                    "cache_gib",
                    default=20.0,
                    min=0.0,
                    step=1.0,
                    tooltip="Dedicated disk cache budget; this is not a VRAM gate. 0 disables disk cache",
                ),
                io.String.Input("cache_directory", default=""),
                io.Model.Input("model", optional=True),
                io.Vae.Input("video_vae", optional=True),
            ],
            [RuntimeIO.Output("runtime"), io.String.Output("report_json")],
            description="Standalone Meridian merged-DMD native ConvRot INT8. Lazy loading; does not alter existing two-pass or user global optimizations. External MODEL",
        )

    @classmethod
    def execute(
        cls,
        model_path,
        vae_path,
        source_directory,
        omega_source_directory,
        omega_checkpoint,
        cache_gib,
        cache_directory,
        model=None,
        video_vae=None,
    ):
        from .meridian_runtime import config

        value = config(
            model_path,
            vae_path,
            source_directory,
            omega_source_directory,
            omega_checkpoint,
            cache_directory,
            cache_gib,
            model,
            video_vae,
        )
        report = {k: v for k, v in value.items() if k not in ("model", "video_vae")}
        return io.NodeOutput(value, json.dumps(report, ensure_ascii=False))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float(
            "nan"
        )  # Validate actual asset bytes; file edits cannot inherit an old Core cache.


class MiniMaxH3MeridianMaterialEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Meridian 2 · Media Window & Geometry (T8)",
            [
                RuntimeIO.Input("runtime"),
                io.Int.Input("start_frame", default=0, min=0, max=None),
                io.Int.Input(
                    "window_frames",
                    default=73,
                    min=1,
                    max=None,
                    tooltip="Source window, normalized to 24fps; IMAGE is reconstructed once, not duplicated into 73 frames and reconstructed",
                ),
                io.Image.Input("image", optional=True),
                io.Video.Input("video", optional=True),
            ],
            [
                MaterialIO.Output("material"),
                io.Image.Output("source_preview"),
                io.String.Output("report_json"),
            ],
            description="Connect only IMAGE or VIDEO. Preserves VIDEO trim/crop/aspect; the selected window is taken by real PTS. Omega joint geometry runs once; after 1280 letterbox the training ROI is used without stretching. Large windows may OOM; no auto-shrink or re-run.",
        )

    @classmethod
    def execute(cls, runtime, start_frame, window_frames, image=None, video=None):
        from .meridian_runtime import prepare, editor_payload

        material = prepare(runtime, image, video, start_frame, window_frames)
        payload = editor_payload(material)
        report = dict(
            identity=material["identity"],
            kind=material["kind"],
            start=material["start"],
            end=material["end"],
            canvas=material["canvas"],
            box=material["box"],
            geometry_stage=material["receipt"],
            clock=material["clock"],
        )
        return io.NodeOutput(
            material,
            material["full"][:1].float().div(255),
            json.dumps(report, ensure_ascii=False),
            ui={"meridian_material": [payload]},
        )


class MiniMaxH3MeridianCameraEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Meridian 3 · Camera / Timeline Editor (T8)",
            [
                MaterialIO.Input("material"),
                io.Int.Input("frames", default=73, min=22, max=None),
                io.Combo.Input("preset", options=list(PRESETS), default="slide"),
                io.Float.Input(
                    "strength",
                    default=0.08,
                    step=0.01,
                    tooltip="Displacement relative to the fixed pivot depth; orbit is strength x 100 degrees. Larger displacement requires more hole-filling; not a performance/quality guarantee",
                ),
                io.String.Input(
                    "camera_plan",
                    default="",
                    multiline=True,
                    tooltip="The advanced editor writes canonical JSON automatically; leave empty to use presets. Spatial track and source time track are independent.",
                ),
            ],
            [
                PlanIO.Output("prepared"),
                io.String.Output("canonical_plan"),
                io.Image.Output("warp_preview"),
                io.String.Output("report_json"),
            ],
            description="Run a preset first to get a real geometry preview, then open the full editor. Pins the first-source camera coordinates, depth units, roll=0; no reverse source time or automatic character tracking. Explicitly reset old paths after changing source/window. The preview is a gray-hole point-cloud warp, not a generated video.",
        )

    @classmethod
    def execute(cls, material, frames, preset, strength, camera_plan):
        from .meridian_runtime import editor_payload, warp_material
        from .meridian_generate import fixed_assets

        plan = (
            canonical_plan(camera_plan, material)
            if camera_plan.strip()
            else preset_plan(material, frames, preset, strength)
        )
        if plan["frames"] != frames:
            raise ValueError(
                "Output frames changed while a saved camera plan exists: reset or explicitly update both tracks in the editor"
            )
        # Validate paired length assets before expensive warping.
        fixed_assets(material["runtime"], plan["frames"])
        value = warp_material(material, plan)
        payload = editor_payload(material)
        payload.update(
            plan=plan,
            source_map=source_map(plan),
            coverage=value["coverage"],
            revision=identity(plan),
        )
        report = dict(
            warp_stage=value["receipt"],
            coverage_min=min(value["coverage"]),
            coverage_mean=sum(value["coverage"]) / len(value["coverage"]),
            canvas=value["canvas"],
            final_native_generation_ran=False,
        )
        return io.NodeOutput(
            value,
            plan_json(plan),
            value["render"].float().div(255),
            json.dumps(report),
            ui={"meridian_editor": [payload]},
        )


class MiniMaxH3MeridianGenerateEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(
            cls,
            "Meridian 4 · Native 3-Pass Generation (T8)",
            [
                PlanIO.Input("prepared"),
                io.Int.Input("seed", default=1234, min=0, max=2**64 - 1),
                io.Combo.Input(
                    "audio_mode",
                    options=["silent", "source_1to1"],
                    default="silent",
                    tooltip="Choose silent for freeze/speed-change; source_1to1 delivers only the original audio and does not certify newly generated lip-sync",
                ),
                io.String.Input("output_directory", default=""),
            ],
            [
                io.Video.Output("video"),
                io.String.Output("saved_path"),
                io.String.Output("report_json"),
            ],
            is_output_node=True,
            description="Native INT8 merged-DMD + two 480-level video token references + 768-level training bucket. CFG1, euler, shift3",
        )

    @classmethod
    def execute(cls, prepared, seed, audio_mode, output_directory):
        from .meridian_generate import generate

        return io.NodeOutput(*generate(prepared, seed, audio_mode, output_directory))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float(
            "nan"
        )  # A returned filename cannot mask a deleted/changed final delivery.


MERIDIAN_NODE_CLASSES = [
    MiniMaxH3MeridianConfigEXPT8,
    MiniMaxH3MeridianMaterialEXPT8,
    MiniMaxH3MeridianCameraEXPT8,
    MiniMaxH3MeridianGenerateEXPT8,
]
