"""File-backed snapshots of actual native CLIP conditioning, one shot at a time.

Calling preparation again re-encodes prompts; it never assumes a different loaded
CLIP produces the same conditioning because a model filename matches. Sampling
reads frozen output bytes, so later changes to CLIP cannot alter a resumed shot.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import torch
from safetensors.torch import save, load

from .long_video_delivery import _atomic_write_bytes, _manifest_lock
from .video_outpaint_identity import value_identity
from .video_outpaint_plan import canonical, validate_outpaint_plan, _integer
from .video_outpaint_source_runtime import outpaint_gpu_lease
from .patch_stack_policy import warn_patch_stack


MAX_CONDITION_BYTES = 256*1024*1024


def _pack(value, tensors, live_objects=None):
    if isinstance(value, torch.Tensor):
        if value.numel()*value.element_size()+sum(v.numel()*v.element_size() for v in tensors.values()) > MAX_CONDITION_BYTES:
            raise ValueError("shot conditioning exceeds the 256MiB tensor budget")
        tensor = value.detach().to("cpu").contiguous().clone()
        value_identity(tensor)
        key = str(len(tensors))
        tensors[key] = tensor
        return {"tensor": key}
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {"dict": {k: _pack(v, tensors, live_objects) for k, v in sorted(value.items())}}
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [_pack(v, tensors, live_objects) for v in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        canonical(value)
        return {"scalar": value}
    if live_objects is None:
        raise TypeError("Opaque conditioning requires its live process-local object binding; it cannot be serialized")
    token = uuid.uuid4().hex
    live_objects[token] = value
    warn_patch_stack("Opaque conditioning retained in this provider only; cross-process resume disabled; no pickle or executable object stored")
    return {"live_object": token}


def _unpack(value, tensors, live_objects=None):
    if set(value) == {"tensor"}:
        return tensors[value["tensor"]]
    if set(value) == {"dict"}:
        return {k: _unpack(v, tensors, live_objects) for k, v in value["dict"].items()}
    if set(value) == {"list"}:
        return [_unpack(v, tensors, live_objects) for v in value["list"]]
    if set(value) == {"tuple"}:
        return tuple(_unpack(v, tensors, live_objects) for v in value["tuple"])
    if set(value) == {"scalar"}:
        return value["scalar"]
    if set(value) == {"live_object"}:
        token = value["live_object"]
        if live_objects is None or token not in live_objects:
            raise FileNotFoundError("Live conditioning object is unavailable after process restart; encode into a new run instead of reusing this snapshot")
        return live_objects[token]
    raise ValueError("invalid stored conditioning structure")


def _validate(conditioning):
    if not isinstance(conditioning, list) or not conditioning:
        raise ValueError("native CLIP must return nonempty CONDITIONING")
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("conditioning entry must be embeddings plus metadata")
        tensor, metadata = entry
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 1
                or min(tensor.shape) < 1 or not tensor.is_floating_point() or not isinstance(metadata, dict)):
            raise ValueError("outpaint conditioning requires [1,tokens,channels] embeddings and metadata")


class OutpaintConditioningProvider:
    def __init__(self, root, plan, *, live_objects=None):
        self.root = Path(root).resolve()
        self.live_objects = dict(live_objects or {})
        self.portable_cache_reuse = not bool(self.live_objects)
        self.plan = validate_outpaint_plan(plan)
        self.path = self.root / "outpaint_conditioning.json"
        self.manifest_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        digest = data.pop("sha256", None)
        if (digest != hashlib.sha256(canonical(data).encode()).hexdigest()
                or data.get("schema") != "t8.h3.outpaint.conditioning/v1"
                or data.get("plan_sha256") != self.plan["plan_sha256"]
                or len(data.get("shots", [])) != len(self.plan["shots"])):
            raise ValueError("conditioning manifest integrity or plan/shot mismatch")
        self.data = data
        self.verify()
        for shot in range(len(self.plan["shots"])):
            self(shot, 0)

    def verify(self):
        implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if (self.data.get("implementation_sha256") != implementation
                or hashlib.sha256(self.path.read_bytes()).hexdigest() != self.manifest_sha256):
            raise ValueError("conditioning snapshot or implementation changed")
        return self.manifest_sha256

    def __call__(self, shot, window):
        self.verify()
        _integer(shot, "conditioning shot", 0, len(self.plan["shots"])-1)
        _integer(window, "conditioning window", 0, len(self.plan["shots"][shot]["windows"])-1)
        record = self.data["shots"][shot]
        digest = record["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid conditioning asset hash")
        path = self.root / f"conditioning-{digest}.safetensors"
        if not path.is_file() or path.stat().st_size != record["bytes"] or path.stat().st_size > MAX_CONDITION_BYTES+1048576:
            raise ValueError("conditioning asset missing, truncated or oversized")
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError("conditioning asset integrity mismatch")
        result = _unpack(record["tree"], load(blob), self.live_objects)
        _validate(result)
        return result


def prepare_outpaint_conditioning(clip, prompts, plan, cache_root, *, interrupt_check=None, progress=None):
    checked = validate_outpaint_plan(plan)
    if isinstance(prompts, str):
        prompts = [prompts]*len(checked["shots"])
    if (not isinstance(prompts, list) or len(prompts) != len(checked["shots"])
            or any(not isinstance(prompt, str) for prompt in prompts)):
        raise ValueError("provide one prompt or one prompt per shot; an empty prompt is allowed")
    root = Path(cache_root).resolve()
    path = root / "outpaint_conditioning.json"
    if (root / "outpaint_regional_conditioning.json").exists():
        raise ValueError("regional conditioning already exists in this run; choose a new run_name for plain prompts")
    with outpaint_gpu_lease(), _manifest_lock(root / "conditioning-worker", timeout_seconds=0.1):
        records = []
        live_objects = {}
        for shot, prompt in enumerate(prompts):
            if interrupt_check:
                interrupt_check()
            conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
            _validate(conditioning)
            tensors = {}
            tree = _pack(conditioning, tensors, live_objects)
            blob = save(tensors)
            digest = hashlib.sha256(blob).hexdigest()
            target = root / f"conditioning-{digest}.safetensors"
            if target.exists():
                if target.stat().st_size != len(blob) or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError("existing conditioning asset corrupted; refusing overwrite")
            else:
                _atomic_write_bytes(target, blob)
            records.append({"prompt": prompt, "tree": tree, "sha256": digest, "bytes": len(blob)})
            del blob, tensors, conditioning
            if progress:
                progress({"stage": "encode_outpaint_prompt", "shot": shot})
        data = {"schema": "t8.h3.outpaint.conditioning/v1", "plan_sha256": checked["plan_sha256"],
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "shots": records}
        blob = canonical({**data, "sha256": hashlib.sha256(canonical(data).encode()).hexdigest()}).encode()
        if path.exists() and path.read_bytes() != blob:
            raise ValueError("new CLIP outputs/prompts differ from saved conditioning; choose a new cache directory")
        if not path.exists():
            _atomic_write_bytes(path, blob)
    return OutpaintConditioningProvider(root, checked, live_objects=live_objects)
