"""Distinct evidence contracts for exact source pasteback and joint VAE decode."""
from __future__ import annotations

import hashlib

from .video_outpaint_plan import canonical


SOURCE_MODES = ("preserve_source", "joint_decode")


def validate_source_mode(value):
    if not isinstance(value, str) or value not in SOURCE_MODES:
        raise ValueError("source_mode must be preserve_source or joint_decode")
    return value


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_pixel_receipt(receipt, *, plan, source_sha256, candidate_sha256, source_mode="preserve_source"):
    """Validate integrity/declared provenance, not authenticate a malicious producer."""
    mode = validate_source_mode(source_mode)
    if not isinstance(receipt, dict):
        raise ValueError("pre-encode pixel receipt must be a dictionary")
    body = dict(receipt)
    digest = body.pop("receipt_sha256", None)
    exact = mode == "preserve_source"
    expected = {
        "schema": "t8.h3.video_outpaint.pre_encode_pixels/v1" if exact else "t8.h3.video_outpaint.reconstructed_pixels/v1",
        "plan_sha256": plan["plan_sha256"], "source_sha256": source_sha256,
        "candidate_sha256": candidate_sha256, "frame_count": plan["output"]["frames"],
        "width": plan["output"]["width"], "height": plan["output"]["height"],
        "source_exact_before_encoding": exact, "lossy_encoded_equality_claimed": False,
    }
    output_key = "pasted_rgb_sha256" if exact else "reconstructed_source_rgb_sha256"
    if not exact:
        expected.update(source_mode="joint_decode", source_reconstructed=True)
    valid = (digest == hashlib.sha256(canonical(body).encode()).hexdigest()
             and all(body.get(k) == v and (not isinstance(v, bool) or body.get(k) is v) for k, v in expected.items())
             and _hex(body.get("source_rgb_sha256")) and _hex(body.get(output_key)))
    if exact:
        valid = valid and body.get(output_key) == body.get("source_rgb_sha256")
        valid = valid and body.get("source_mode", "preserve_source") == "preserve_source" and not body.get("source_reconstructed", False)
    else:
        valid = valid and "pasted_rgb_sha256" not in body
    if not valid:
        raise ValueError("pre-encode pixel receipt does not match the source/plan/encoded candidate/source mode")
    return body
