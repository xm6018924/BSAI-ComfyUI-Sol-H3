"""D1 persistent project and pure-CPU compiler. No model imports or queue access."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import uuid

SCHEMA = "t8.minimax_h3.director_project"
VERSION = 1
ALIAS = re.compile(r"@(image|video|audio)([0-9]+)\b")
NATIVE_MEDIA = re.compile(r"<(Picture|Video|Audio) ([0-9]+)>")
RATIOS = {"16:9", "9:16", "2:3", "1:1", "original"}
_LOCK = threading.RLock()


class ProjectConflict(ValueError):
    pass


def identity(value):
    try:
        parsed = uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("Identity must be a stable UUID") from error
    if str(parsed) != str(value):
        raise ValueError("UUID must be in canonical format")
    return str(parsed)


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError("Asset or project path is out of bounds")
    return candidate


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + str(uuid.uuid4()) + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def new_project():
    shot_id = str(uuid.uuid4())
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "id": str(uuid.uuid4()),
        "title": "My first short film",
        "revision": 0,
        "current": shot_id,
        "assets": [],
        "doc": {
            "global": "",
            "sharedRefs": [],
            "sharedRatio": True,
            "ratio": "16:9",
            "d3": {
                "semantic_bridge": {"enabled": False},
                "prompt_relay": {"enabled": False},
                "fast_h3_v2": {"enabled": False},
                "memory": {"low_vram": False, "chunk_ffn": False},
            },
            "generation": {
                "unet": "auto",
                "clip": "auto",
                "video_vae": "auto",
                "audio_vae": "auto",
                "lora_mode": "auto",
                "loras": [],
                "resolution_mp": "auto",
            },
            "shots": [
                {
                    "id": shot_id,
                    "name": "Opening",
                    "mode": "text",
                    "sound": "native",
                    "writingMode": "simple",
                    "simplePrompt": "",
                    "prompt": "",
                    "events": [],
                    "refs": [],
                    "tray": [],
                    "first": None,
                    "last": None,
                    "audio": None,
                    "start": 0,
                    "end": 0,
                    "duration": 4,
                    "manualDuration": 4,
                    "autoDuration": True,
                    "ratio": "16:9",
                    "ownRatio": "16:9",
                    "d3": {
                        "semantic_bridge": {"enabled": False},
                        "prompt_relay": {"enabled": False},
                        "fast_h3_v2": {"enabled": False},
                        "memory": {"low_vram": False, "chunk_ffn": False},
                    },
                    "d3Inherit": True,
                    "rev": 1,
                }
            ],
        },
    }


def _number(value, label, minimum=0):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise ValueError(f"{label} 必须是有效数字且 ≥ {minimum}")
    return value


def generation_settings(doc):
    """Normalize a saved selection without mutating old project snapshots."""
    raw = doc.get("generation") or {}
    if not isinstance(raw, dict):
        raise ValueError("Model and LoRA config must be an object")
    result = {key: raw.get(key) or "auto" for key in ("unet", "clip", "video_vae", "audio_vae")}
    for key, value in result.items():
        if not isinstance(value, str):
            raise ValueError(f"模型配置 {key} 必须是文字")
    legacy = raw.get("lora", "auto")
    legacy = [legacy] if isinstance(legacy, str) else legacy
    if not isinstance(legacy, list) or not all(isinstance(v, str) for v in legacy):
        raise ValueError("Legacy LoRA config is invalid")
    mode = raw.get("lora_mode", "manual" if any(v not in ("auto", "none") for v in legacy) else "none" if legacy == ["none"] else "auto")
    if mode not in ("auto", "none", "manual"):
        raise ValueError("Invalid LoRA mode")
    rows = raw.get("loras", [{"name": v, "strength": raw.get("lora_strength", 1.0), "enabled": True} for v in legacy if v not in ("auto", "none")])
    if not isinstance(rows, list):
        raise ValueError("LoRA must be a list")
    result.update(lora_mode=mode, loras=[])
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("enabled", True), bool):
            raise ValueError("LoRA entries require a model name, strength, and enabled toggle")
        strength = _number(row.get("strength", 1.0), "LoRA strength", -2)
        if strength > 2:
            raise ValueError("LoRA strength must be between -2 and 2")
        if mode == "manual" and row.get("enabled", True) and not row["name"]:
            raise ValueError("Please choose a LoRA file for an enabled entry")
        result["loras"].append({"name": row["name"], "strength": strength, "enabled": row.get("enabled", True)})
    mp = raw.get("resolution_mp", "auto")
    if mp != "auto":
        try:
            mp = float(mp)
        except (TypeError, ValueError) as error:
            raise ValueError("Total megapixels must be a number") from error
        _number(mp, "Total megapixels", 0.001024)
    result["resolution_mp"] = mp
    return result


def director_canvas(fraction, megapixels="auto"):
    if not math.isfinite(fraction) or fraction <= 0:
        raise ValueError("Aspect ratio must be positive")
    if megapixels == "auto":
        height = 768 if fraction <= 1 else max(32, round(768 / fraction / 32) * 32)
        width = max(32, round(height * fraction / 32) * 32)
    else:
        area = _number(megapixels, "Total megapixels", 0.001024) * 1_000_000
        # Same 32-pixel spatial grid as H3 conditioning/empty joint AV latent.
        width = max(32, math.floor(math.sqrt(area * fraction) / 32 + 0.5) * 32)
        height = max(32, math.floor(math.sqrt(area / fraction) / 32 + 0.5) * 32)
    if max(width, height) > 8192:
        raise ValueError("Computed size exceeds the director image preprocessing range (max 8192 per side); reduce MP or adjust aspect")
    return width, height


def validate_project(value):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Not a director project.json; for unknown workflows, keep the original and return to canvas")
    project = deepcopy(value)
    version = project.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("Project version must be an integer")
    # Explicit, lossless migration of the first development envelope; no prose parsing.
    if version == 0:
        project["version"] = VERSION
        project.setdefault("title", "My short film")
        project.setdefault("assets", [])
        project.setdefault("revision", 0)
    elif version != VERSION:
        raise ValueError(f"不支持项目版本 {version}，请保留原文件")
    identity(project.get("id"))
    revision = project.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("revision must be a non-negative integer")
    if not isinstance(project.get("title"), str) or len(project["title"]) > 200:
        raise ValueError("Project name must be at most 200 characters")
    doc = project.get("doc")
    if (
        not isinstance(doc, dict)
        or not isinstance(doc.get("shots"), list)
        or not doc["shots"]
    ):
        raise ValueError("At least one shot must remain")
    if (
        not isinstance(doc.get("global"), str)
        or not isinstance(doc.get("sharedRatio"), bool)
        or doc.get("ratio") not in RATIOS
    ):
        raise ValueError("Film-wide text/aspect config is invalid")
    if doc.get("d3") is not None and not isinstance(doc.get("d3"), dict):
        raise ValueError("Film-wide D3 config must be an object")
    generation = doc.get("generation")
    if generation is not None and not isinstance(generation, dict):
        raise ValueError("Model and LoRA config must be an object")
    if generation:
        for field in ("unet", "clip", "video_vae", "audio_vae"):
            if field in generation and not isinstance(generation[field], str):
                raise ValueError(f"模型配置 {field} 必须是文字")
        if "lora" in generation:
            value = generation["lora"]
            if not isinstance(value, (str, list)) or (isinstance(value, list) and not all(isinstance(item, str) for item in value)):
                raise ValueError("Model config lora must be a string or list of strings")
        if "lora_strength" in generation:
            _number(generation["lora_strength"], "lora_strength", 0)
            if generation["lora_strength"] > 2:
                raise ValueError("lora_strength must be <= 2")
        generation_settings(doc)
    ids = []
    for shot in doc["shots"]:
        ids.append(identity(shot.get("id")))
        if shot.get("mode") not in {"text", "first", "ends", "refs"} or shot.get(
            "sound"
        ) not in {"native", "voice", "record"}:
            raise ValueError("Shot visual or audio intent is invalid")
        if shot.get("writingMode") not in {"simple", "advanced"}:
            raise ValueError("Prompt format is invalid")
        for field in ("name", "prompt", "simplePrompt"):
            if not isinstance(shot.get(field), str):
                raise ValueError(f"{field} 必须是文字")
        for field in ("duration", "manualDuration"):
            _number(shot.get(field), field, 0.001)
        for field in ("start", "end"):
            _number(shot.get(field), field)
        if (
            shot.get("ratio") not in RATIOS
            or shot.get("ownRatio") not in RATIOS
            or not isinstance(shot.get("autoDuration"), bool)
        ):
            raise ValueError("Shot aspect or auto-duration is invalid")
        if "d3Inherit" in shot and not isinstance(shot["d3Inherit"], bool):
            raise ValueError("Shot D3 global-inherit toggle is invalid")
        if shot.get("d3") is not None and not isinstance(shot.get("d3"), dict):
            raise ValueError("Shot D3 config must be an object")
        for field in ("first", "last", "audio", "selected"):
            if shot.get(field) is not None:
                identity(shot[field])
        for field in ("tray", "refs"):
            if not isinstance(shot.get(field), list):
                raise ValueError(f"{field} 必须是资产列表")
            for asset_id in shot[field]:
                identity(asset_id)
        if not isinstance(shot.get("events"), list):
            raise ValueError("events must be a list")
        event_ids = []
        for event in shot["events"]:
            event_ids.append(identity(event.get("id")))
            _number(event.get("start"), "event.start")
            _number(event.get("end"), "event.end")
            if not isinstance(event.get("text"), str):
                raise ValueError("Event text is invalid")
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Duplicate time-event UUID")
    if len(ids) != len(set(ids)) or project.get("current") not in ids:
        raise ValueError("Duplicate shot UUID or current shot does not exist")
    if not isinstance(doc.get("sharedRefs"), list):
        raise ValueError("sharedRefs must be a list")
    for asset_id in doc["sharedRefs"]:
        identity(asset_id)
    assets = project.get("assets")
    if not isinstance(assets, list):
        raise ValueError("assets must be an asset list")
    asset_ids = [identity(a.get("id")) for a in assets if isinstance(a, dict)]
    if len(asset_ids) != len(assets) or len(asset_ids) != len(set(asset_ids)):
        raise ValueError("Asset UUID is invalid or duplicate")
    canonical(
        project
    )  # Reject NaN even in extra fields; unknown fields otherwise preserved.
    return project


def referenced_assets(project):
    result = set(project["doc"]["sharedRefs"])
    for shot in project["doc"]["shots"]:
        result.update(shot["tray"] + shot["refs"])
        result.update(
            shot[key] for key in ("first", "last", "audio", "selected") if shot.get(key)
        )
    return result


class ProjectStore:
    def __init__(self, user_root, input_root):
        self.root = Path(user_root).resolve() / "t8_director"
        self.input_root = Path(input_root).resolve()

    def _path(self, project_id):
        return contained(self.root, f"projects/{identity(project_id)}.json")

    def load(self, project_id):
        with _LOCK:
            return validate_project(
                json.loads(self._path(project_id).read_text(encoding="utf-8"))
            )

    def list(self):
        result = []
        with _LOCK:
            for path in sorted(
                (self.root / "projects").glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                try:
                    project = self.load(path.stem)
                    result.append(
                        {key: project[key] for key in ("id", "title", "revision")}
                    )
                except (ValueError, OSError):
                    result.append(
                        {"id": path.stem, "error": "Project is corrupt; keep the file and restore from backup"}
                    )
        return result

    def asset(self, asset_id, verify=False, allow_missing=False):
        asset_id = identity(asset_id)
        manifest = contained(self.root, f"assets/{asset_id}.json")
        asset = json.loads(manifest.read_text(encoding="utf-8"))
        if asset.get("id") != asset_id:
            raise ValueError("Asset identity is corrupt")
        path = contained(self.input_root, asset["server_path"])
        if not path.is_file():
            if allow_missing:
                return {**asset, "missing": True}
            raise ValueError("Server asset is missing; please re-upload and explicitly reconnect")
        if path.stat().st_size != asset["size"] or (
            verify and file_sha(path) != asset["sha256"]
        ):
            if allow_missing:
                return {**asset, "missing": True, "unavailable_reason": "bytes_changed"}
            raise ValueError("Server asset bytes changed; please re-upload and explicitly reconnect")
        return asset

    def register_asset(self, path, asset_id, original_name):
        """Only server-generated paths are accepted; metadata derived from actual bytes."""
        asset_id = identity(asset_id)
        expected = contained(self.input_root, f"t8_director/{asset_id}")
        path = Path(path).resolve()
        if path.parent != expected or not path.is_file():
            raise ValueError("Upload path does not belong to this asset")
        name = Path(str(original_name).replace("\\", "/")).name[:200]
        asset = {
            "id": asset_id,
            "name": name,
            "server_path": path.relative_to(self.input_root).as_posix(),
            "sha256": file_sha(path),
            "size": path.stat().st_size,
            "width": 0,
            "height": 0,
            "duration": 0,
            "has_audio": False,
        }
        from PIL import Image, ImageOps

        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                upright = ImageOps.exif_transpose(image)
                asset.update(kind="image", width=upright.width, height=upright.height)
        except (OSError, ValueError):
            import av

            with av.open(str(path)) as media:
                video = list(media.streams.video)
                audio = list(media.streams.audio)
                if not video and not audio:
                    raise ValueError("Asset is not a readable image, video, or audio")
                asset["kind"] = "video" if video else "audio"
                asset["has_audio"] = bool(audio)
                if video:
                    asset.update(width=video[0].width, height=video[0].height)
                stream = (video or audio)[0]
                duration = (
                    float(stream.duration * stream.time_base)
                    if stream.duration
                    else float(media.duration or 0) / av.time_base
                )
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("Asset duration is unreadable; convert to standard MP4/WAV before uploading")
                asset["duration"] = duration
        with _LOCK:
            manifest = contained(self.root, f"assets/{asset_id}.json")
            if manifest.exists():
                raise ProjectConflict("Asset UUID already exists; overwriting is not allowed")
            atomic_json(manifest, asset)
        return asset

    def prepare_image(self, asset_id, width, height):
        from PIL import Image, ImageOps

        if (
            width % 32
            or height % 32
            or not 32 <= width <= 8192
            or not 32 <= height <= 8192
        ):
            raise ValueError("Input preview size must be 32-aligned and between 32 and 8192 px")
        asset = self.asset(asset_id, verify=True)
        if asset["kind"] != "image":
            raise ValueError("Only images support first/last-frame input preprocessing")
        target = contained(
            self.input_root, f"t8_director/{asset_id}/fit_{width}_{height}.png"
        )
        receipt_path = target.with_suffix(".json")
        policy = "contain_letterbox_lanczos_rgb_exif"
        # Derived CPU file only, never overwrites the uploaded original. Deterministic letterbox.
        with _LOCK:
            if target.exists() and receipt_path.exists():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (
                    receipt.get("source_sha256") != asset["sha256"]
                    or receipt.get("policy") != policy
                    or receipt.get("width") != width
                    or receipt.get("height") != height
                    or receipt.get("sha256") != file_sha(target)
                ):
                    raise ValueError("Preprocess cache identity or bytes are corrupt; explicitly reconnect the asset")
            else:
                with Image.open(
                    contained(self.input_root, asset["server_path"])
                ) as image:
                    upright = ImageOps.exif_transpose(image).convert("RGB")
                    fitted = ImageOps.contain(
                        upright, (width, height), Image.Resampling.LANCZOS
                    )
                    canvas = Image.new("RGB", (width, height), (0, 0, 0))
                    canvas.paste(
                        fitted,
                        ((width - fitted.width) // 2, (height - fitted.height) // 2),
                    )
                    temporary = target.with_name(
                        target.name + "." + str(uuid.uuid4()) + ".tmp"
                    )
                    try:
                        canvas.save(temporary, format="PNG")
                        os.replace(temporary, target)
                    finally:
                        temporary.unlink(missing_ok=True)
                atomic_json(
                    receipt_path,
                    {
                        "source_sha256": asset["sha256"],
                        "policy": policy,
                        "width": width,
                        "height": height,
                        "sha256": file_sha(target),
                    },
                )
            with Image.open(target) as image:
                if image.size != (width, height):
                    raise ValueError("Preprocess cache dimensions are corrupt")
        return {
            "server_path": target.relative_to(self.input_root).as_posix(),
            "sha256": file_sha(target),
            "width": width,
            "height": height,
            "policy": policy,
            "source_sha256": asset["sha256"],
        }

    def save(self, project, expected_revision):
        project = validate_project(project)
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer")
        if expected_revision != project["revision"]:
            raise ProjectConflict("Submitted revision does not match the read version")
        with _LOCK:
            path = self._path(project["id"])
            current = self.load(project["id"])["revision"] if path.exists() else 0
            if current != expected_revision:
                raise ProjectConflict(
                    f"另一标签已保存（服务端版本 {current}）；保留本地草稿，另存副本或重新载入"
                )
            # Client metadata is never trusted as a filesystem or media authority.
            referenced = referenced_assets(project)

            def saved_asset(asset):
                try:
                    return self.asset(
                        asset["id"], allow_missing=asset["id"] not in referenced
                    )
                except FileNotFoundError:
                    if asset["id"] in referenced:
                        raise ValueError("Server asset manifest is missing; explicitly reconnect") from None
                    # Imported/retired library metadata is non-executable: never trust its path.
                    return {
                        "id": asset["id"],
                        "name": str(asset.get("name", "Missing asset"))[:200],
                        "kind": asset.get("kind"),
                        "width": 0,
                        "height": 0,
                        "duration": 0,
                        "server_path": None,
                        "sha256": None,
                        "size": 0,
                        "missing": True,
                        "retired": True,
                        "unavailable_reason": "manifest_missing",
                    }

            project["assets"] = [saved_asset(a) for a in project["assets"]]
            project["revision"] = current + 1
            atomic_json(path, project)
        return project


def compile_project(value, store=None):
    project = validate_project(value)
    assets = {a["id"]: a for a in project["assets"]}
    errors, warnings, shots = [], [], []
    referenced = referenced_assets(project)
    for asset_id in assets:
        if store:
            try:
                assets[asset_id] = store.asset(asset_id, verify=True)
            except (ValueError, OSError) as error:
                (errors if asset_id in referenced else warnings).append(
                    {"asset_id": asset_id, "message": str(error)}
                )
        asset = assets[asset_id]
        if asset.get("kind") not in {"image", "video", "audio"}:
            errors.append({"asset_id": asset_id, "message": "Invalid asset kind"})
    doc = project["doc"]
    timeline_start = 0
    for shot in doc["shots"]:
        sid = shot["id"]

        def fail(message):
            errors.append({"shot_id": sid, "message": message})

        shared = list(dict.fromkeys(doc["sharedRefs"]))
        tray = list(
            dict.fromkeys(
                shared
                + shot["tray"]
                + [x for x in (shot.get("first"), shot.get("last")) if x]
                + shot["refs"]
                + ([shot["audio"]] if shot.get("audio") else [])
            )
        )
        if any(x not in assets for x in tray):
            fail("Some asset identities are missing; re-upload and use reconnect; do not silently swap images")
        alias_counts = {"image": 0, "video": 0, "audio": 0}
        alias_to_id = {}
        for aid in tray:
            if aid not in assets or assets[aid].get("kind") not in alias_counts:
                continue
            kind = assets[aid]["kind"]
            alias_counts[kind] += 1
            alias_to_id[f"@{kind}{alias_counts[kind]}"] = aid
        saved_aliases = project.get("aliasMaps", {}).get(sid)
        if saved_aliases is not None and saved_aliases != alias_to_id:
            fail("Asset alias does not match the stable identity; redirect reference is refused")
        refs = list(
            dict.fromkeys(shared + (shot["refs"] if shot["mode"] != "text" else []))
        )
        first = shot.get("first") if shot["mode"] in {"first", "ends"} else None
        last = shot.get("last") if shot["mode"] == "ends" else None
        if shot["mode"] == "first" and not first:
            fail("Please add a first frame")
        if shot["mode"] == "ends" and not last:
            fail("Please add a last frame (first frame may be empty for last-frame-only)")
        if shot["mode"] == "refs" and not refs and shot["sound"] != "voice":
            fail("Please assign a reference asset")
        selected_audio = shot.get("audio") if shot["sound"] != "native" else None
        if shot["sound"] != "native" and (
            not selected_audio or assets.get(selected_audio, {}).get("kind") != "audio"
        ):
            fail("Please choose an uploaded audio asset")
        if selected_audio and (
            shot["end"] <= shot["start"]
            or shot["end"] > assets.get(selected_audio, {}).get("duration", 0) + 0.001
        ):
            fail("Recording selection is invalid")
        if shot["sound"] == "record" and selected_audio in refs:
            refs.remove(
                selected_audio
            )  # Explicit drive slot, not a duplicate independent ref.
        if shot["sound"] == "voice" and selected_audio and selected_audio not in refs:
            refs.append(selected_audio)
        if shot["sound"] == "record":
            refs = [aid for aid in refs if aid != first]
            if not first or last or refs:
                fail(
                    "Original-audio drive uses the validated Avatar single-segment first-frame recipe; one first frame is required, no last frame or other reference. For complex combos, return to the advanced canvas"
                )
        typed = {"image": [], "video": [], "audio": []}
        for aid in refs:
            kind = assets.get(aid, {}).get("kind")
            if kind in typed:
                typed[kind].append(aid)
        for kind, limit in (("image", 9), ("video", 3), ("audio", 3)):
            if len(typed[kind]) > limit:
                fail(f"参考{kind}超过原生槽 {limit}；请调整用途，素材不截断")
        for aid in typed["video"]:
            if assets[aid].get("duration", 0) < 2:
                fail("Reference video must be at least 2 seconds; the current recipe prepares at 24fps; over 15 seconds is a warning only, no hard cap")
            elif assets[aid]["duration"] > 15:
                warnings.append(
                    {
                        "shot_id": sid,
                        "asset_id": aid,
                        "message": "Reference video exceeds the 15-second training recommendation; truncation is not forced; memory and quality must be tested",
                    }
                )
        media, id_to_native = [], {}
        pictures = [x for x in (first, last) if x] + typed["image"]
        for index, aid in enumerate(pictures, 1):
            if assets.get(aid, {}).get("kind") != "image":
                fail("First/last-frame or image reference is not an image")
            role = (
                "first_frame"
                if aid == first and index == 1
                else "last_frame"
                if aid == last and index <= int(bool(first)) + 1
                else "ref_image"
            )
            native = f"<Picture {index}>"
            id_to_native.setdefault(aid, native)
            media.append({"asset_id": aid, "role": role, "native": native})
        audio_ordinal = 0
        for index, aid in enumerate(typed["video"], 1):
            native = f"<Video {index}>"
            id_to_native[aid] = native
            media.append({"asset_id": aid, "role": "ref_video", "native": native})
            if assets[aid].get("has_audio"):
                audio_ordinal += 1
                media.append(
                    {
                        "asset_id": aid,
                        "role": "ref_video_audio",
                        "native": f"<Audio {audio_ordinal}>",
                    }
                )
        if shot["sound"] == "record" and selected_audio:
            audio_ordinal += 1
            id_to_native[selected_audio] = f"<Audio {audio_ordinal}>"
            media.append(
                {
                    "asset_id": selected_audio,
                    "role": "drive_audio",
                    "native": id_to_native[selected_audio],
                }
            )
        for aid in typed["audio"]:
            audio_ordinal += 1
            native = f"<Audio {audio_ordinal}>"
            id_to_native[aid] = native
            media.append({"asset_id": aid, "role": "ref_audio", "native": native})
        for item in media:
            item["server_path"] = assets.get(item["asset_id"], {}).get("server_path")
            item["sha256"] = assets.get(item["asset_id"], {}).get("sha256")

        def translate(text):
            if "@missing_" in text:
                fail("Text-referenced asset has been removed; explicitly fix the missing reference")

            def replacement(match):
                token = match.group(0)
                aid = alias_to_id.get(token)
                native = id_to_native.get(aid)
                if native is None:
                    fail(f"{token} 没有生效用途，请先分配首尾／参考／声音用途")
                    return token
                return native

            translated = ALIAS.sub(replacement, text)
            available = {item["native"] for item in media}
            for match in NATIVE_MEDIA.finditer(translated):
                if match.group(0) not in available:
                    fail(f"{match.group(0)} 没有对应的生效素材槽，请核对引用")
            return translated

        duration = shot["manualDuration"]
        active_events = shot["events"] if shot["writingMode"] == "advanced" else []
        if shot["autoDuration"]:
            if active_events:
                duration = max(e["end"] for e in active_events)
            elif shot["sound"] == "record" and selected_audio:
                duration = shot["end"] - shot["start"]
        if duration <= 0:
            fail("Duration must be greater than 0")
            duration = 0.001
        requested = max(5, math.ceil(duration * 24))
        frames = requested + ((5 - requested) % 17)
        events = []
        for event in active_events:
            if event["end"] <= event["start"] or event["end"] > duration:
                fail("Event ends before it starts or extends beyond this shot")
            start, end = event["start"] * 24, event["end"] * 24
            if not float(start).is_integer() or not float(end).is_integer():
                fail("Advanced events must fall on the 24fps frame grid; time is not silently rounded")
            events.append(
                {
                    **event,
                    "text": translate(event["text"]),
                    "start_frame": start,
                    "end_frame": end,
                }
            )
        local = translate(
            shot["simplePrompt"] if shot["writingMode"] == "simple" else shot["prompt"]
        )
        global_text = translate(doc["global"])
        for token in ALIAS.finditer(doc["global"]):
            if alias_to_id.get(token.group(0)) not in shared:
                fail("Film-wide text may only reference film-wide shared assets")
        if (
            not local.strip()
            and not any(e["text"].strip() for e in events)
            and shot["sound"] != "record"
        ):
            fail("Please enter the prompt for this shot")
        prompt = "\n\n".join(
            x
            for x in (
                global_text,
                local,
                "\n\n".join(f"[{e['start']}–{e['end']}秒] {e['text']}" for e in events),
            )
            if x
        )
        # Text budget is informative here, not an invented limit of local H3 or vision tokens.
        if len(prompt) > 7000:
            warnings.append(
                {
                    "shot_id": sid,
                    "message": "Exceeds the official 7000-character submission recommendation; local generation has no unified hard cap; not truncated",
                }
            )
        ratio = doc["ratio"] if doc["sharedRatio"] else shot["ownRatio"]
        source = assets.get(first or last or (pictures[0] if pictures else None), {})
        if ratio == "original":
            if not source.get("width") or not source.get("height"):
                fail("The original-aspect mode requires an active image")
            fraction = source.get("width", 1) / source.get("height", 1)
        else:
            a, b = map(int, ratio.split(":"))
            fraction = a / b
        resolution_mp = generation_settings(doc)["resolution_mp"]
        width, height = director_canvas(fraction, resolution_mp)
        preprocess = []
        for aid in (first, last):
            if not aid:
                continue
            a = assets.get(aid, {})
            same_ratio = a.get("width", 0) * height == a.get("height", 0) * width
            policy = "disabled" if aid == first else "center"
            preprocess.append(
                {
                    "asset_id": aid,
                    "source": [a.get("width"), a.get("height")],
                    "target": [width, height],
                    "native_crop": policy,
                    "aspect_preserved": True,
                    "padding_needed": not same_ratio,
                    "policy": "contain_letterbox_lanczos_rgb_exif",
                    "state": "planned_not_processed",
                }
            )
            if store and a.get("kind") == "image":
                try:
                    prepared = store.prepare_image(aid, width, height)
                    preprocess[-1].update(state="cpu_processed", actual_input=prepared)
                    for item in media:
                        if item["asset_id"] == aid and item["role"] in {
                            "first_frame",
                            "last_frame",
                        }:
                            item["source_server_path"] = item["server_path"]
                            item["server_path"] = prepared["server_path"]
                            item["sha256"] = prepared["sha256"]
                except (ValueError, OSError) as error:
                    fail(str(error))
        # A lock-source recording is a real audio reference once the
        # conditioning node receives ``add_source_as_reference=True``.  With
        # a first frame that makes the native task Hybrid, not I2VA: I2VA is
        # intentionally fail-closed when any reference media is present.
        source_audio_ref = shot["sound"] == "record" and bool(selected_audio)
        task = (
            "hybrid"
            if (refs or source_audio_ref) and (first or last)
            else "ref2va"
            if (refs or source_audio_ref)
            else "fl2va"
            if first and last
            else "i2va"
            if first
            else "l2va"
            if last
            else "t2va"
        )
        recipe = (
            "avatar_single_segment_progressive_lock_source"
            if shot["sound"] == "record"
            else "native_ref_voice_stock20"
            if shot["sound"] == "voice"
            else "native_h3"
        )
        shots.append(
            {
                "id": sid,
                "name": shot["name"],
                "writing_mode": shot["writingMode"],
                "prompt": prompt,
                "source_drafts": {
                    "global": doc["global"],
                    "simple": shot["simplePrompt"],
                    "local": shot["prompt"],
                    "events": deepcopy(shot["events"]),
                },
                "global_prompt": global_text,
                "local_prompt": local,
                "events": events,
                "timeline_start": timeline_start,
                "timeline_end": timeline_start + duration,
                "time": {
                    "fps": 24,
                    "requested_seconds": duration,
                    "requested_frames": requested,
                    "aligned_frames": frames,
                    "generated_seconds": frames / 24,
                    "delivery_trim_frames": math.ceil(duration * 24),
                },
                "canvas": {
                    "ratio": ratio,
                    "requested_megapixels": resolution_mp,
                    "actual_megapixels": width * height / 1_000_000,
                    "width": width,
                    "height": height,
                    "preprocessing": preprocess,
                },
                "task_type": task,
                "recipe": recipe,
                "audio_mode": "lock_source" if shot["sound"] == "record" else "native",
                "drive_audio": selected_audio if shot["sound"] == "record" else None,
                "final_audio": selected_audio if shot["sound"] == "record" else None,
                "audio_selection": {
                    "asset_id": selected_audio,
                    "start": shot["start"],
                    "end": shot["end"],
                }
                if selected_audio
                else None,
                "delivery_audio": "original_selected_recording"
                if shot["sound"] == "record"
                else "generated",
                "media_map": media,
                "aliases": [
                    {"alias": token, "asset_id": aid, "native": id_to_native.get(aid)}
                    for token, aid in alias_to_id.items()
                ],
            }
        )
        timeline_start += duration
    report = {
        "schema": "t8.minimax_h3.director_compilation.v1",
        "project_id": project["id"],
        "project_sha256": sha(project),
        "ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "shots": shots,
        "total_seconds": timeline_start,
        "gpu_queued": False,
        "scope": "Project compilation contract; generation is submitted only by the explicit D2 route",
    }
    report["compilation_sha256"] = sha(report)
    return report
