"""Canonical Meridian space/time tracks. Pure CPU math; no model or server imports."""

from copy import deepcopy
import json
import math

import numpy as np

from .meridian_checkpoint_io import canonical_identity

SCHEMA = "t8.meridian.camera.v1"
PRESETS = (
    "slide",
    "push_in",
    "pull_out",
    "crane",
    "orbit",
    "freeze_orbit",
    "source_camera",
)
TARGETS = (
    (672, 1536),
    (704, 1408),
    (768, 1344),
    (864, 1184),
    (1024, 1024),
    (1184, 864),
    (1344, 768),
    (1408, 704),
    (1536, 672),
)
CONDITIONS = (
    (416, 960),
    (448, 896),
    (480, 832),
    (544, 736),
    (640, 640),
    (736, 544),
    (832, 480),
    (896, 448),
    (960, 416),
)


def integer(value, name):
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer, not a float or boolean")
    return value


def finite(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def vector(value, name):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three coordinates")
    return [finite(v, name) for v in value]


def bucket(width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Positive source dimensions required")
    index = min(
        range(len(TARGETS)),
        key=lambda i: abs(math.log(TARGETS[i][0] / TARGETS[i][1] * height / width)),
    )
    return TARGETS[index], CONDITIONS[index]


def preset_plan(material, frames=73, preset="slide", strength=0.08, pivot=None):
    if preset not in PRESETS:
        raise ValueError("Unknown Meridian camera preset")
    frames = integer(frames, "frames")
    strength = finite(strength, "strength")
    if frames < 2:
        raise ValueError("At least two output frames required")
    start, end = material["start"], material["end"]
    pivot = vector(pivot or material.get("pivot", [0.0, 0.0, 1.0]), "pivot")
    frozen = material["kind"] == "image" or preset == "freeze_orbit"
    last_src = start if frozen else min(end, start + frames - 1)

    def key(t, position):
        return dict(t=t, pos=position, look=pivot.copy(), focal=1.0, ease=True)

    first, last = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    if preset == "slide":
        last[0] = strength
    elif preset == "push_in":
        last[2] = strength
    elif preset == "pull_out":
        last[2] = -strength
    elif preset == "crane":
        last[1] = -strength
    elif preset in ("orbit", "freeze_orbit"):
        angle = math.radians(strength * 100)
        p = np.asarray(pivot)
        rotation = np.array(
            [
                [math.cos(angle), 0, math.sin(angle)],
                [0, 1, 0],
                [-math.sin(angle), 0, math.cos(angle)],
            ]
        )
        last = (p - rotation @ p).tolist()
    # Source-camera mode uses every estimated pose, not a two-key replay claim.
    mode = "source_camera" if preset == "source_camera" else "authored"
    first_key, last_key = key(0, first), key(frames - 1, last)
    if preset == "slide":
        # A truck/slide translates both camera and its look point. Holding the
        # look point fixed instead introduces a compensating pan which keeps
        # the pivot subject centred and hides small lateral camera movements.
        # Orbit presets deliberately retain their fixed target.
        last_key["look"] = [p + delta for p, delta in zip(pivot, last)]
    return canonical_plan(
        dict(
            schema=SCHEMA,
            geometry_id=material["identity"],
            origin_frame=start,
            window_end=end,
            frames=frames,
            space="source_window_first_camera_x_right_y_down_z_forward",
            units="fixed_pivot_depth",
            roll=0,
            mode=mode,
            pivot=pivot,
            camera_keys=[first_key, last_key],
            time_keys=[dict(t=0, src=start), dict(t=frames - 1, src=last_src)],
        ),
        material,
    )


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate camera plan JSON key")
        result[key] = value
    return result


def canonical_plan(value, material):
    if isinstance(value, str):
        if len(value) > 1024 * 1024:
            raise ValueError("Camera plan exceeds1MiB")
        value = json.loads(value, object_pairs_hook=_unique)
    value = deepcopy(value)
    required = {
        "schema",
        "geometry_id",
        "origin_frame",
        "window_end",
        "frames",
        "space",
        "units",
        "roll",
        "mode",
        "pivot",
        "camera_keys",
        "time_keys",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema"] != SCHEMA
    ):
        raise ValueError("Unsupported canonical Meridian camera plan")
    integer(value["origin_frame"], "origin_frame")
    integer(value["window_end"], "window_end")
    integer(value["roll"], "roll")
    if (
        value["geometry_id"] != material["identity"]
        or value["origin_frame"] != material["start"]
        or value["window_end"] != material["end"]
    ):
        raise ValueError(
            "Material geometry/window changed: reset or explicitly reconfirm the camera plan"
        )
    if (
        value["space"] != "source_window_first_camera_x_right_y_down_z_forward"
        or value["units"] != "fixed_pivot_depth"
        or value["roll"] != 0
    ):
        raise ValueError("Unsupported coordinate frame/unit/roll")
    if value["mode"] not in ("authored", "source_camera"):
        raise ValueError("Unsupported camera mode")
    frames = integer(value["frames"], "frames")
    if frames < 2:
        raise ValueError("At least two output frames required")
    value["pivot"] = vector(value["pivot"], "pivot")
    for track in ("camera_keys", "time_keys"):
        keys = value[track]
        if not isinstance(keys, list) or not 2 <= len(keys) <= 4096:
            raise ValueError("Each camera/time track requires2–4096 keys")
        if any(not isinstance(k, dict) or "t" not in k for k in keys):
            raise ValueError("Each key must be an object with an output time")
        times = [integer(k["t"], "key t") for k in keys]
        if (
            times[0] != 0
            or times[-1] != frames - 1
            or any(a >= b for a, b in zip(times, times[1:]))
        ):
            raise ValueError("Output key times must cover0..N-1 and strictly increase")
    for key in value["camera_keys"]:
        if (
            set(key) != {"t", "pos", "look", "focal", "ease"}
            or type(key["ease"]) is not bool
        ):
            raise ValueError("Invalid camera key fields/ease")
        key["pos"], key["look"] = vector(key["pos"], "pos"), vector(key["look"], "look")
        if np.linalg.norm(np.asarray(key["look"]) - key["pos"]) < 1e-6:
            raise ValueError("Camera position and target cannot coincide")
        key["focal"] = finite(key["focal"], "focal")
        if key["focal"] <= 0:
            raise ValueError("Focal multiplier must be positive; it is not millimetres")
    sources = []
    for key in value["time_keys"]:
        if set(key) != {"t", "src"}:
            raise ValueError("Invalid time key fields")
        src = integer(key["src"], "source index")
        if not material["start"] <= src <= material["end"]:
            raise ValueError("Source index outside reconstructed window")
        sources.append(src)
    if any(a > b for a, b in zip(sources, sources[1:])):
        raise ValueError("Reverse source time is not supported")
    if material["kind"] == "image" and any(s != material["start"] for s in sources):
        raise ValueError("Still-image source time must hold its only frame")
    return value


def plan_json(plan):
    return json.dumps(
        plan, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def source_map(plan):
    keys = plan["time_keys"]
    # Backend alone defines integer source sampling. NumPy ties-even, not JS Math.round.
    return (
        np.rint(
            np.interp(
                np.arange(plan["frames"]),
                [k["t"] for k in keys],
                [k["src"] for k in keys],
            )
        )
        .astype(int)
        .tolist()
    )


def audio_policy(plan, requested):
    if requested not in ("silent", "source_1to1"):
        raise ValueError("Unknown Meridian soundtrack policy")
    mapping = source_map(plan)
    one_to_one = all(b == a + 1 for a, b in zip(mapping, mapping[1:]))
    if requested == "source_1to1" and not one_to_one:
        raise ValueError(
            "Original audio only supports1:1 source time; freeze/retiming must explicitly be silent"
        )
    return dict(
        mode=requested,
        one_to_one=one_to_one,
        duration=plan["frames"] / 24.0,
        start_seconds=mapping[0] / 24.0,
        original_sample_sync_not_lipsync_certification=True,
    )


def identity(plan):
    return canonical_identity(plan)


def horizontal_fov(fx, width, multiplier=1.0):
    fx, width, multiplier = (finite(v, "FOV input") for v in (fx, width, multiplier))
    if min(fx, width, multiplier) <= 0:
        raise ValueError("Positive FOV dimensions required")
    return math.degrees(2 * math.atan(width / (2 * fx * multiplier)))


def fov_multiplier(degrees, fx, width):
    degrees = finite(degrees, "horizontal FOV")
    if not 0 < degrees < 180:
        raise ValueError("Horizontal FOV must be between0 and180degrees")
    fx, width = finite(fx, "fx"), finite(width, "width")
    if min(fx, width) <= 0:
        raise ValueError("Positive FOV dimensions required")
    return width / (2 * fx * math.tan(math.radians(degrees) / 2))
