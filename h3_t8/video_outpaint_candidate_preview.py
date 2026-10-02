"""Decode the selected trajectory's actual first frame using delivery math.

Only the first seven-token VAE decode is requested; no remaining sampling or
full-clip decode occurs. The returned IMAGE is RGB8-quantized like encoder input,
not a claim that a lossy MP4 or the remaining clip will look exactly identical.
"""
from __future__ import annotations

from contextlib import closing
import hashlib

import torch

from .video_outpaint_candidates import _validate_candidate
from .video_outpaint_finish import finish_outpaint_frames, geometry_settings
from .video_outpaint_decode import iter_decode_outpaint_shot
from .video_outpaint_media import validate_outpaint_source
from .video_outpaint_plan import canonical
from .video_outpaint_prepare import sampling_to_output_canvas, SequentialOutpaintFrameReader
from .video_outpaint_source_runtime import loaded_video_vae_identity, outpaint_gpu_lease
from .video_outpaint_pixel_receipt import validate_source_mode
from .video_outpaint_source_runtime import source_vae_identity_matches
from .patch_stack_policy import model_identity_matches


def render_candidate_first_frame(*, vae, candidate, inspection, source_store, window_store,
                                 color_match=True, color_settings=None, interrupt_check=None,
                                 geometry_align=False, alignment_band_pixels=64, alignment_max_displacement=8.0,
                                 source_mode="preserve_source"):
    if not isinstance(color_match, bool):
        raise ValueError("color_match must be boolean")
    settings = dict(color_settings or {})
    mode = validate_source_mode(source_mode)
    preserve_source = mode == "preserve_source"
    geometry = geometry_settings(geometry_align, alignment_band_pixels, alignment_max_displacement)
    if not set(settings) <= {"strength", "strip_pixels", "fade_pixels", "temporal_alpha", "max_offset"}:
        raise ValueError("unsupported candidate color setting")

    def checkpoint():
        if interrupt_check:
            interrupt_check()

    def verify_source():
        if source_store.plan != window_store.plan or source_store.position() is not None:
            raise ValueError("candidate preview requires the complete matching source cache")
        if hashlib.sha256(source_store.path.read_bytes()).hexdigest() != window_store.identity["source_cache_sha256"]:
            raise ValueError("candidate source manifest changed")
        return validate_outpaint_source(inspection, window_store.plan)

    with outpaint_gpu_lease():
        checkpoint()
        _validate_candidate(candidate, window_store)
        source_path, plan = verify_source()
        identity = loaded_video_vae_identity(vae, interrupt_check=interrupt_check)
        if not source_vae_identity_matches(source_store, identity):
            raise ValueError("candidate decode VAE differs from the source preparation VAE")
        manifest_sha = hashlib.sha256(window_store.path.read_bytes()).hexdigest()
        with closing(iter_decode_outpaint_shot(
            vae, lambda a, b: window_store.read_video_range(0, a, b), plan,
            shot_index=0, interrupt_check=interrupt_check,
        )) as decoder:
            start, pixels, decode_report = next(decoder)
            if start != 0 or pixels.shape[0] < 1:
                raise ValueError("candidate decoder did not return the source first frame")
            pixels = pixels[:1].clone()
        checkpoint()
        with SequentialOutpaintFrameReader(source_path, plan, interrupt_check=interrupt_check) as reader:
            original = reader(0, 1)
        canvas = sampling_to_output_canvas(pixels, plan)
        composed, _, preservation = finish_outpaint_frames(
            original.float()/255, canvas, plan, start_frame=0, state=None,
            enabled=color_match, source_mode=mode, **settings, **geometry)
        encoded_rgb = (composed.clamp(0, 1)*255).round().to(torch.uint8)
        x0, y0, x1, y1 = plan["output"]["source_rect"]
        if (preservation["source_exact_before_encoding"] is not preserve_source or
                (preserve_source and not torch.equal(encoded_rgb[:, y0:y1, x0:x1], original))):
            raise ValueError("candidate preview changed the original RGB region")
        checkpoint()
        verify_source()
        _validate_candidate(candidate, window_store)
        if not model_identity_matches(identity, loaded_video_vae_identity(vae, interrupt_check=interrupt_check)):
            raise ValueError("candidate decode VAE changed during preview")
        if hashlib.sha256(window_store.path.read_bytes()).hexdigest() != manifest_sha:
            raise ValueError("candidate sampling manifest changed during preview")
        report = {
            "schema": "t8.h3.video_outpaint.candidate_first_frame/v1",
            "candidate_sha256": candidate["sha256"], "plan_sha256": plan["plan_sha256"],
            "source_sha256": inspection["sha256"], "video_vae_sha256": identity["sha256"],
            "window_manifest_sha256": manifest_sha, "first_frame_index": 0,
            "width": encoded_rgb.shape[2], "height": encoded_rgb.shape[1],
            "rgb8_sha256": hashlib.sha256(encoded_rgb.contiguous().numpy().tobytes()).hexdigest(),
            "source_exact_before_encoding": preserve_source, "decode": decode_report,
            "source_mode": mode, "source_reconstructed": not preserve_source,
            "color_settings": preservation["settings"], "color_report": preservation,
            "geometry_settings": geometry,
            "vae_decode_calls": 1, "vae_max_tokens": 7, "full_video_materialized": False,
            "sampling_called_by_preview": False, "generated_prefix_used": True,
            "lossy_encoded_equality_claimed": False, "perceptual_acceptance": False,
        }
        report["sha256"] = hashlib.sha256(canonical(report).encode()).hexdigest()
        return encoded_rgb.float()/255, report
