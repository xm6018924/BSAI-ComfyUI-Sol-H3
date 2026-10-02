"""Validated spatial guidance and source-person protection plans for H3 outpaint.

Coordinates supplied by users are always output-pixel coordinates, except
``person_boxes_json`` which deliberately uses original source coordinates.  The
runtime route is compiled onto the native H3 32-pixel spatial token grid so it
cannot be confused with Comfy's generic post-pack conditioning areas.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

from .video_outpaint_plan import canonical, validate_outpaint_plan


GUIDANCE_SCHEMA = "t8.h3.video_outpaint.spatial_guidance/v1"
REGION_KINDS = ("expanded", "top", "bottom", "left", "right", "bbox")
MAX_REGIONS_PER_SHOT = 8
MAX_PROMPT_CHARS = 4096


def _integer(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _json_array(value, label):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} must be valid JSON") from exc
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _shots(selector, count, label):
    if selector == "all":
        return range(count)
    return (_integer(selector, label, 0, count - 1),)


def _box(value, label, width, height):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} must be [x, y, width, height]")
    x, y, w, h = (_number(item, label) for item in value)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError(f"{label} is outside its coordinate space")
    return [x, y, x + w, y + h]


def _intersects(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _sampling_box(output_box, plan):
    scale = plan["sampling"]["isotropic_scale"]
    ox, oy = plan["sampling"]["offset"]
    x0, y0, x1, y1 = output_box
    return [ox + x0 * scale, oy + y0 * scale, ox + x1 * scale, oy + y1 * scale]


def _side_box(kind, plan):
    width, height = plan["sampling"]["width"], plan["sampling"]["height"]
    x0, y0, x1, y1 = plan["sampling"]["source_rect"]
    return {
        "top": [0.0, 0.0, float(width), y0],
        "bottom": [0.0, y1, float(width), float(height)],
        "left": [0.0, y0, x0, y1],
        "right": [x1, y0, float(width), y1],
    }[kind]


def _spatial_rows(kind, box, plan):
    """Return H3 2x2-latent patch rows whose pixel centres belong to a region."""
    height = plan["sampling"]["height"] // 32
    width = plan["sampling"]["width"] // 32
    source = plan["sampling"]["source_rect"]
    rows = []
    for y in range(height):
        py = y * 32.0 + 16.0
        for x in range(width):
            px = x * 32.0 + 16.0
            inside = (
                not (source[0] <= px < source[2] and source[1] <= py < source[3])
                if kind == "expanded"
                else box[0] <= px < box[2] and box[1] <= py < box[3]
            )
            if inside:
                rows.append(y * width + x)
    if not rows:
        raise ValueError("a spatial guidance region covers no H3 token centres at this sampling budget")
    return rows


def validate_outpaint_guidance(guidance: Mapping, plan=None):
    if not isinstance(guidance, Mapping):
        raise ValueError("outpaint guidance must be an object")
    data = json.loads(canonical(dict(guidance)))
    digest = data.pop("guidance_sha256", None)
    if data.get("schema") != GUIDANCE_SCHEMA or digest != hashlib.sha256(canonical(data).encode()).hexdigest():
        raise ValueError("outpaint guidance schema or integrity hash mismatch")
    data["guidance_sha256"] = digest
    if plan is not None:
        checked = validate_outpaint_plan(plan)
        if data.get("plan_sha256") != checked["plan_sha256"]:
            raise ValueError("outpaint guidance belongs to another plan")
        rebuilt = build_outpaint_guidance(
            checked,
            data["request"]["regions"],
            data["request"]["person_boxes"],
        )
        if rebuilt != data:
            raise ValueError("outpaint guidance coordinates do not match its request")
    return data


def build_outpaint_guidance(plan, regions_json="[]", person_boxes_json="[]"):
    checked = validate_outpaint_plan(plan)
    regions = _json_array(regions_json, "regions_json")
    people = _json_array(person_boxes_json, "person_boxes_json")
    shot_count = len(checked["shots"])
    compiled_regions = [[] for _ in range(shot_count)]
    normalized_regions = []
    output_w, output_h = checked["output"]["width"], checked["output"]["height"]
    source_output = checked["output"]["source_rect"]

    for index, item in enumerate(regions):
        if not isinstance(item, dict) or set(item) - {"shot", "region", "prompt", "box"}:
            raise ValueError(f"region {index} contains unknown fields")
        if not {"shot", "region", "prompt"} <= set(item):
            raise ValueError(f"region {index} requires shot, region and prompt")
        kind = item["region"]
        prompt = item["prompt"]
        if kind not in REGION_KINDS:
            raise ValueError(f"region {index} has an unsupported region kind")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"region {index} prompt must contain 1..{MAX_PROMPT_CHARS} characters")
        if kind == "bbox":
            output_box = _box(item.get("box"), f"region {index} box", output_w, output_h)
            if _intersects(output_box, source_output):
                raise ValueError(f"region {index} bbox overlaps the source-owned rectangle")
            sample_box = _sampling_box(output_box, checked)
            normalized = {"shot": item["shot"], "region": kind, "prompt": prompt,
                          "box": [output_box[0], output_box[1],
                                  output_box[2] - output_box[0], output_box[3] - output_box[1]]}
        else:
            if "box" in item:
                raise ValueError(f"region {index} {kind} must not supply box")
            output_box = None
            sample_box = None if kind == "expanded" else _side_box(kind, checked)
            normalized = {"shot": item["shot"], "region": kind, "prompt": prompt}
        selected = list(_shots(item["shot"], shot_count, f"region {index} shot"))
        normalized["shot"] = item["shot"]
        normalized_regions.append(normalized)
        spatial_rows = _spatial_rows(kind, sample_box, checked)
        for shot in selected:
            if len(compiled_regions[shot]) >= MAX_REGIONS_PER_SHOT:
                raise ValueError(f"shot {shot} exceeds the {MAX_REGIONS_PER_SHOT}-region limit")
            compiled_regions[shot].append({
                "region_index": index,
                "kind": kind,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "output_box": output_box,
                "sampling_box": sample_box,
                "spatial_rows": spatial_rows,
            })

    normalized_people = []
    compiled_people = [[] for _ in range(shot_count)]
    source_w, source_h = checked["source"]["width"], checked["source"]["height"]
    source_x, source_y = source_output[:2]
    for index, item in enumerate(people):
        if not isinstance(item, dict) or set(item) - {"shot", "label", "box"}:
            raise ValueError(f"person box {index} contains unknown fields")
        if not {"shot", "label", "box"} <= set(item):
            raise ValueError(f"person box {index} requires shot, label and box")
        label = item["label"]
        if not isinstance(label, str) or not label.strip() or len(label) > 128:
            raise ValueError(f"person box {index} label must contain 1..128 characters")
        source_box = _box(item["box"], f"person box {index}", source_w, source_h)
        output_box = [source_x + source_box[0], source_y + source_box[1],
                      source_x + source_box[2], source_y + source_box[3]]
        touching = [name for value, edge, name in (
            (source_box[0], 0, "left"), (source_box[1], 0, "top"),
            (source_box[2], source_w, "right"), (source_box[3], source_h, "bottom"),
        ) if value == edge]
        normalized = {"shot": item["shot"], "label": label,
                      "box": [source_box[0], source_box[1],
                              source_box[2] - source_box[0], source_box[3] - source_box[1]]}
        normalized_people.append(normalized)
        for shot in _shots(item["shot"], shot_count, f"person box {index} shot"):
            compiled_people[shot].append({
                "person_index": index, "label": label, "source_box": source_box,
                "output_box": output_box, "sampling_box": _sampling_box(output_box, checked),
                "source_pixels_locked": True, "touches_source_boundary": touching,
                "expanded_person_continuation_review_required": bool(touching),
            })

    data = {
        "schema": GUIDANCE_SCHEMA,
        "plan_sha256": checked["plan_sha256"],
        "request": {"regions": normalized_regions, "person_boxes": normalized_people},
        "sampling_grid": {
            "patch_pixels": 32,
            "rows": checked["sampling"]["height"] // 32,
            "columns": checked["sampling"]["width"] // 32,
            "source_rect": checked["sampling"]["source_rect"],
        },
        "shots": [
            {"shot_index": index, "regions": compiled_regions[index], "people": compiled_people[index]}
            for index in range(shot_count)
        ],
        "policies": {
            "source_rectangle_latent_locked": True,
            "source_pixels_exact_before_lossy_encoding": True,
            "person_detection_performed": False,
            "person_boxes_are_manual_audit_annotations": True,
            "expanded_area_person_pixels_guaranteed": False,
            "human_review_required": True,
        },
    }
    data["guidance_sha256"] = hashlib.sha256(canonical(data).encode()).hexdigest()
    return data
