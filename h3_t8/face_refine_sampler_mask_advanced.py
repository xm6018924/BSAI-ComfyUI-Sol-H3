from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

import comfy.ldm.minimax.model as minimax_model
import comfy.nested_tensor
import comfy.utils

from .core import nested_av_parts, split_noise_masks
from .face_refine_advanced import canonical_json


PATCH_SCHEMA = "h3_t8_face_refine_sampler_mask_patch/v1"
DENOISE_REPORT_SCHEMA = "h3_t8_face_refine_per_frame_denoise/v1"
UPSTREAM_REPOSITORY = "https://github.com/Carasibana/ComfyUI-H3-FaceRefine"
UPSTREAM_VERSION = "1.1.1"
UPSTREAM_COMMIT = "d7ae3ee1ec445ea29fff7fc7366fa6fe85bdc2f5"
PATCH_PATHS = ("_denoise_mask_conds", "scale_latent_inpaint")


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(part) for part in tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _parse_denoise_report(report_json: str) -> tuple[dict[str, Any], str]:
    try:
        report = json.loads(report_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("denoise_report_json must be valid JSON") from exc
    if not isinstance(report, dict):
        raise ValueError("denoise_report_json must contain a JSON object")
    if report.get("schema") != DENOISE_REPORT_SCHEMA:
        raise ValueError("denoise_report_json has an unsupported schema")
    if report.get("status") != "parity_per_frame_video_mask_applied":
        raise ValueError("denoise_report_json is not a completed per-frame denoise report")
    if report.get("require_locked_audio") is not True:
        raise ValueError("sampler-mask correction requires the locked-audio denoise route")
    if report.get("audio_mask_all_zero") is not True:
        raise ValueError("sampler-mask correction refuses a report with a nonzero audio mask")
    if report.get("audio_samples_modified") is not False:
        raise ValueError("sampler-mask correction refuses a report that modified audio samples")
    if report.get("video_mask_mode") not in {"replace_video_parity", "cap_existing"}:
        raise ValueError("denoise_report_json has an unsupported video_mask_mode")
    canonical = canonical_json(report)
    return report, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_live_latent(
    av_latent: dict[str, Any], report: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(av_latent, dict):
        raise ValueError("av_latent must be a MiniMax H3 latent mapping")
    samples = av_latent.get("samples")
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        raise ValueError("Expected a MiniMax H3 joint AV NestedTensor latent")
    video, audio = nested_av_parts(av_latent)
    video_mask, audio_mask = split_noise_masks(av_latent, video, audio)
    if video_mask is None or audio_mask is None:
        raise ValueError("sampler-mask correction requires explicit video and audio masks")
    if not torch.isfinite(video_mask).all() or not torch.isfinite(audio_mask).all():
        raise ValueError("sampler-mask correction refuses non-finite noise masks")
    video_min = float(video_mask.amin().item())
    video_max = float(video_mask.amax().item())
    if video_min < 0.0 or video_max > 1.0:
        raise ValueError("video noise mask values must stay within [0, 1]")
    if int(torch.count_nonzero(audio_mask).item()) != 0:
        raise ValueError("sampler-mask correction requires an all-zero live audio mask")
    if int(report.get("latent_time", -1)) != int(video.shape[2]):
        raise ValueError("denoise report latent_time does not match the connected AV latent")
    expected_video_sha = str(report.get("video_mask_sha256", ""))
    expected_audio_sha = str(report.get("audio_mask_sha256", ""))
    if not expected_video_sha or not expected_audio_sha:
        raise ValueError("denoise report is missing live mask integrity hashes")
    if tensor_sha256(video_mask) != expected_video_sha:
        raise ValueError("connected video mask does not match denoise_report_json")
    if tensor_sha256(audio_mask) != expected_audio_sha:
        raise ValueError("connected audio mask does not match denoise_report_json")
    return video, audio, video_mask, audio_mask


def _validate_model(model: Any) -> tuple[Any, dict[str, Any]]:
    if not callable(getattr(model, "clone", None)) or not callable(
        getattr(model, "add_object_patch", None)
    ):
        raise ValueError("model must be a ComfyUI MODEL / ModelPatcher")
    base = getattr(model, "model", None)
    missing = [
        name
        for name in (
            "_denoise_mask_conds",
            "scale_latent_inpaint",
            "audio_scale",
            "model_sampling",
        )
        if base is None or not hasattr(base, name)
    ]
    if missing:
        raise ValueError(
            "sampler-mask correction requires a native MiniMax H3 base model; missing "
            + ", ".join(missing)
        )
    object_patches = getattr(model, "object_patches", {})
    if not isinstance(object_patches, dict):
        object_patches = {}
    conflicts = sorted(path for path in PATCH_PATHS if path in object_patches)
    for path in conflicts:
        if not callable(object_patches[path]):
            raise TypeError(f'sampler-mask correction: existing {path} must be callable')
    if conflicts:
        from .patch_stack_policy import warn_patch_stack
        warn_patch_stack("sampler-mask correction composes existing model object patches: " + ", ".join(conflicts))
    return base, {
        "base_model_class": type(base).__name__,
        "base_model_module": type(base).__module__,
        "preexisting_object_patch_keys": sorted(str(key) for key in object_patches),
    }


def apply_face_refine_sampler_mask_patch(
    model: Any,
    av_latent: dict[str, Any],
    denoise_report_json: str,
    enabled: bool = False,
):
    if not enabled:
        return model, av_latent, canonical_json({
            "schema": PATCH_SCHEMA,
            "status": "disabled_passthrough",
            "enabled": False,
            "source_model_mutated": False,
            "source_latent_mutated": False,
            "object_patch_paths": [],
        })
    report, report_sha256 = _parse_denoise_report(denoise_report_json)
    video, audio, video_mask, audio_mask = _validate_live_latent(av_latent, report)
    base, model_audit = _validate_model(model)
    source_patch_keys = sorted(str(key) for key in getattr(model, "object_patches", {}))
    original_denoise_mask_conds = model.get_model_object("_denoise_mask_conds")

    def audio_only_denoise_mask_conds(denoise_mask, latent_shapes):
        output = dict(original_denoise_mask_conds(denoise_mask, latent_shapes))
        output.pop("denoise_mask", None)
        return output

    audio_only_denoise_mask_conds._t8_face_refine_sampler_mask_patch = (
        f"{UPSTREAM_VERSION}@{UPSTREAM_COMMIT}"
    )

    def renoised_scale_latent_inpaint(
        sigma,
        noise,
        latent_image,
        x=None,
        denoise_mask=None,
        **kwargs,
    ):
        del x, denoise_mask, kwargs
        latent_shapes = getattr(base, "latent_shapes", None)
        if latent_shapes is None or len(latent_shapes) < 2:
            raise RuntimeError(
                "MiniMax H3 latent_shapes are unavailable while applying the sampler-mask "
                "correction"
            )
        clean_streams = comfy.utils.unpack_latents(latent_image, latent_shapes)
        noise_streams = comfy.utils.unpack_latents(noise, latent_shapes)
        model_sampling = base.model_sampling
        sigma_video = sigma.reshape(
            [sigma.shape[0]] + [1] * (clean_streams[0].ndim - 1)
        )
        clean_streams[0] = model_sampling.noise_scaling(
            sigma_video,
            noise_streams[0],
            clean_streams[0],
        )
        audio_scale = base.audio_scale()
        if audio_scale != 1.0:
            sigma_video_flat = sigma.clamp(min=1e-6)
            sigma_audio = minimax_model.time_shift_sigma(
                sigma_video_flat,
                model_sampling.shift,
                model_sampling.audio_shift,
            )
            factor = (sigma_video_flat / sigma_audio) / audio_scale
            clean_streams[1] = clean_streams[1] * factor.view(
                factor.shape[:1] + (1,) * (clean_streams[1].ndim - 1)
            ).to(clean_streams[1].dtype)
        return comfy.utils.pack_latents(clean_streams)[0]

    renoised_scale_latent_inpaint._t8_face_refine_sampler_mask_patch = (
        f"{UPSTREAM_VERSION}@{UPSTREAM_COMMIT}"
    )

    patched = model.clone()
    patched.add_object_patch("_denoise_mask_conds", audio_only_denoise_mask_conds)
    foreign_scale = "scale_latent_inpaint" in model.object_patches
    if foreign_scale:
        from .patch_stack_policy import warn_patch_stack
        warn_patch_stack("Face sampler-mask correction retains the existing scale_latent_inpaint; T8 re-noise correction may be bypassed")
    else:
        patched.add_object_patch("scale_latent_inpaint", renoised_scale_latent_inpaint)
    if sorted(str(key) for key in getattr(model, "object_patches", {})) != source_patch_keys:
        raise RuntimeError("source MODEL was mutated while applying sampler-mask correction")

    patch_report = {
        "schema": PATCH_SCHEMA,
        "status": "executed_user_stack_unverified" if foreign_scale else "sampler_only_video_mask_and_current_sigma_renoise_applied",
        "enabled": True,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_version": UPSTREAM_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "denoise_report_sha256": report_sha256,
        "plan_sha256": report.get("plan_sha256"),
        "video_mask_sha256": tensor_sha256(video_mask),
        "audio_mask_sha256": tensor_sha256(audio_mask),
        "video_latent_shape": [int(part) for part in video.shape],
        "audio_latent_shape": [int(part) for part in audio.shape],
        "video_mask_range": [
            float(video_mask.amin().item()),
            float(video_mask.amax().item()),
        ],
        "audio_mask_all_zero": True,
        "audio_mask_condition_preserved": True,
        "video_mask_model_condition_removed": True,
        "video_mask_sampler_path_preserved": True,
        "held_video_renoise_clock": "user_selected_unverified" if foreign_scale else "current_sampler_sigma",
        "existing_scale_owner_preserved": foreign_scale,
        "composition_verified": not foreign_scale,
        "audio_rescale_policy": "native_minimax_h3_time_shift_preserved",
        "source_model_mutated": False,
        "object_patch_paths": list(PATCH_PATHS),
        **model_audit,
    }
    return patched, av_latent, canonical_json(patch_report)
