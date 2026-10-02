"""Immutable CLIP snapshots carrying authenticated H3 spatial text routing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load, save

from .long_video_delivery import _atomic_write_bytes, _manifest_lock
from .video_outpaint_conditioning import MAX_CONDITION_BYTES, _pack, _unpack, _validate
from .video_outpaint_guidance import validate_outpaint_guidance
from .video_outpaint_identity import value_identity
from .video_outpaint_plan import canonical, validate_outpaint_plan, _integer
from .video_outpaint_source_runtime import outpaint_gpu_lease
from .patch_stack_policy import UnverifiedModelStack, warn_patch_stack


REGIONAL_MANIFEST = "outpaint_regional_conditioning.json"
REGIONAL_SCHEMA = "t8.h3.outpaint.regional_conditioning/v1"
REGIONAL_BINDING_SCHEMA = "t8.h3.outpaint.regional_binding/v1"
REGIONAL_PAYLOAD_KEY = "t8_outpaint_regional_binding"
MAX_COMBINED_TOKENS = 4096


def regional_claim(binding_hash, shot):
    return f"{binding_hash}:{int(shot)}"


def _same_metadata(left, right):
    try:
        return value_identity(left) == value_identity(right)
    except UnverifiedModelStack:
        warn_patch_stack("regional CLIP retains opaque metadata; equal live objects required")
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(_same_metadata(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
            return len(left) == len(right) and all(_same_metadata(a, b) for a, b in zip(left, right))
        return left is right


def _merge_encoded(encoded):
    if not encoded:
        raise ValueError("regional conditioning needs a base prompt encoding")
    for result in encoded:
        _validate(result)
        if len(result) != 1:
            raise ValueError("regional conditioning currently requires one CLIP schedule entry per prompt")
    tensors = [result[0][0] for result in encoded]
    if len({(item.shape[0], item.shape[2], item.dtype) for item in tensors}) != 1:
        raise ValueError("base and regional prompt embeddings are incompatible")
    lengths = [int(item.shape[1]) for item in tensors]
    if sum(lengths) > MAX_COMBINED_TOKENS:
        raise ValueError(f"combined regional text exceeds the {MAX_COMBINED_TOKENS}-token limit")
    metadata = [result[0][1] for result in encoded]
    if any("model_conds" in item for item in metadata):
        warn_patch_stack("regional conditioning retains pre-existing runtime model conditions")
    keys = set(metadata[0])
    if any(set(item) != keys for item in metadata):
        raise ValueError("base and regional CLIP metadata fields differ")
    tags = []
    for index, (item, length) in enumerate(zip(metadata, lengths)):
        tag = item.get("minimax_token_tags")
        if not isinstance(tag, torch.Tensor) or tag.numel() != length or tag.ndim not in (1, 2):
            raise ValueError(f"regional prompt {index} lacks exact MiniMax token tags")
        tags.append(tag.reshape(-1))
    for key in keys - {"minimax_token_tags"}:
        if any(not _same_metadata(metadata[0][key], item[key]) for item in metadata[1:]):
            raise ValueError(f"regional CLIP metadata {key!r} differs between prompt blocks")
    merged = dict(metadata[0])
    merged["minimax_token_tags"] = torch.cat(tags, dim=0)
    result = [[torch.cat(tensors, dim=1), merged]]
    _validate(result)
    return result, lengths


def _binding(guidance, lengths_by_shot):
    shots = []
    for shot, (item, lengths) in enumerate(zip(guidance["shots"], lengths_by_shot)):
        if len(lengths) != len(item["regions"]) + 1:
            raise ValueError("regional prompt count differs from its guidance plan")
        cursor = lengths[0]
        regions = []
        for region, length in zip(item["regions"], lengths[1:]):
            regions.append({
                "region_index": region["region_index"],
                "kind": region["kind"],
                "prompt_sha256": region["prompt_sha256"],
                "text_key_start": cursor,
                "text_key_end": cursor + length,
                "spatial_rows": region["spatial_rows"],
            })
            cursor += length
        shots.append({"shot_index": shot, "text_len": cursor, "regions": regions})
    data = {
        "schema": REGIONAL_BINDING_SCHEMA,
        "plan_sha256": guidance["plan_sha256"],
        "guidance_sha256": guidance["guidance_sha256"],
        "sampling_grid": guidance["sampling_grid"],
        "shots": shots,
    }
    data["binding_sha256"] = hashlib.sha256(canonical(data).encode()).hexdigest()
    return data


def validate_regional_binding(binding, plan=None):
    if not isinstance(binding, dict):
        raise ValueError("regional binding must be an object")
    data = json.loads(canonical(binding))
    digest = data.pop("binding_sha256", None)
    if data.get("schema") != REGIONAL_BINDING_SCHEMA or digest != hashlib.sha256(canonical(data).encode()).hexdigest():
        raise ValueError("regional binding schema or integrity hash mismatch")
    data["binding_sha256"] = digest
    if plan is not None and data.get("plan_sha256") != validate_outpaint_plan(plan)["plan_sha256"]:
        raise ValueError("regional binding belongs to another outpaint plan")
    return data


class OutpaintRegionalConditioningProvider:
    def __init__(self, root, plan, *, interrupt_check=None, live_objects=None):
        self.root = Path(root).resolve()
        self.plan = validate_outpaint_plan(plan)
        self.path = self.root / REGIONAL_MANIFEST
        self.interrupt_check = interrupt_check
        self.live_objects = dict(live_objects or {})
        self.portable_cache_reuse = not bool(self.live_objects)
        self.manifest_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        digest = data.pop("sha256", None)
        if (digest != hashlib.sha256(canonical(data).encode()).hexdigest()
                or data.get("schema") != REGIONAL_SCHEMA
                or data.get("plan_sha256") != self.plan["plan_sha256"]
                or len(data.get("shots", [])) != len(self.plan["shots"])):
            raise ValueError("regional conditioning manifest integrity or plan/shot mismatch")
        self.data = data
        self.guidance = validate_outpaint_guidance(data["guidance"], self.plan)
        self.binding = validate_regional_binding(data["binding"], self.plan)
        if self.binding["guidance_sha256"] != self.guidance["guidance_sha256"]:
            raise ValueError("regional conditioning binding and guidance differ")
        self.verify()
        for shot in range(len(self.plan["shots"])):
            self(shot, 0)

    def verify(self):
        implementations = {
            "conditioning": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "guidance": hashlib.sha256(Path(validate_outpaint_guidance.__code__.co_filename).read_bytes()).hexdigest(),
        }
        if (self.data.get("implementation_sha256") != implementations
                or hashlib.sha256(self.path.read_bytes()).hexdigest() != self.manifest_sha256):
            raise ValueError("regional conditioning snapshot or implementation changed")
        return self.manifest_sha256

    def __call__(self, shot, window):
        if self.interrupt_check:
            self.interrupt_check()
        self.verify()
        _integer(shot, "conditioning shot", 0, len(self.plan["shots"]) - 1)
        _integer(window, "conditioning window", 0, len(self.plan["shots"][shot]["windows"]) - 1)
        record = self.data["shots"][shot]
        digest = record["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid regional conditioning asset hash")
        path = self.root / f"conditioning-{digest}.safetensors"
        if not path.is_file() or path.stat().st_size != record["bytes"] or path.stat().st_size > MAX_CONDITION_BYTES + 1048576:
            raise ValueError("regional conditioning asset missing, truncated or oversized")
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError("regional conditioning asset integrity mismatch")
        result = _unpack(record["tree"], load(blob), self.live_objects)
        _validate(result)
        if int(result[0][0].shape[1]) != int(self.binding["shots"][shot]["text_len"]):
            raise ValueError("regional conditioning asset token count differs from its binding")
        from comfy.conds import CONDConstant
        output = []
        for cross, metadata in result:
            updated = dict(metadata)
            model_conds = dict(updated.get("model_conds", {}))
            if REGIONAL_PAYLOAD_KEY in model_conds:
                raise ValueError("regional conditioning contains a duplicate runtime binding")
            model_conds[REGIONAL_PAYLOAD_KEY] = CONDConstant(regional_claim(self.binding["binding_sha256"], shot))
            updated["model_conds"] = model_conds
            output.append([cross, updated])
        return output


def prepare_outpaint_regional_conditioning(clip, prompts, guidance, plan, cache_root, *, interrupt_check=None, progress=None):
    checked = validate_outpaint_plan(plan)
    guidance = validate_outpaint_guidance(guidance, checked)
    if isinstance(prompts, str):
        prompts = [prompts] * len(checked["shots"])
    if (not isinstance(prompts, list) or len(prompts) != len(checked["shots"])
            or any(not isinstance(prompt, str) for prompt in prompts)):
        raise ValueError("provide one base prompt or one base prompt per shot")
    root = Path(cache_root).resolve()
    path = root / REGIONAL_MANIFEST
    if (root / "outpaint_conditioning.json").exists():
        raise ValueError("plain conditioning already exists in this run; choose a new run_name for regional guidance")
    with outpaint_gpu_lease(), _manifest_lock(root / "conditioning-worker", timeout_seconds=0.1):
        records, lengths_by_shot = [], []
        live_objects = {}
        for shot, prompt in enumerate(prompts):
            if interrupt_check:
                interrupt_check()
            regional_prompts = [item["prompt"] for item in guidance["shots"][shot]["regions"]]
            encoded = [clip.encode_from_tokens_scheduled(clip.tokenize(item)) for item in [prompt, *regional_prompts]]
            conditioning, lengths = _merge_encoded(encoded)
            tensors = {}
            tree = _pack(conditioning, tensors, live_objects)
            blob = save(tensors)
            digest = hashlib.sha256(blob).hexdigest()
            target = root / f"conditioning-{digest}.safetensors"
            if target.exists():
                if target.stat().st_size != len(blob) or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError("existing regional conditioning asset corrupted; refusing overwrite")
            else:
                _atomic_write_bytes(target, blob)
            records.append({"base_prompt": prompt, "regional_prompts": regional_prompts,
                            "tree": tree, "sha256": digest, "bytes": len(blob)})
            lengths_by_shot.append(lengths)
            del blob, tensors, conditioning, encoded
            if progress:
                progress({"stage": "encode_outpaint_regional_prompts", "shot": shot})
        binding = _binding(guidance, lengths_by_shot)
        implementations = {
            "conditioning": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "guidance": hashlib.sha256(Path(validate_outpaint_guidance.__code__.co_filename).read_bytes()).hexdigest(),
        }
        data = {"schema": REGIONAL_SCHEMA, "plan_sha256": checked["plan_sha256"],
                "implementation_sha256": implementations, "guidance": guidance,
                "binding": binding, "shots": records}
        blob = canonical({**data, "sha256": hashlib.sha256(canonical(data).encode()).hexdigest()}).encode()
        if path.exists() and path.read_bytes() != blob:
            raise ValueError("regional prompts or CLIP outputs differ from saved conditioning; choose a new run_name")
        if not path.exists():
            _atomic_write_bytes(path, blob)
    return OutpaintRegionalConditioningProvider(root, checked, interrupt_check=interrupt_check, live_objects=live_objects)
