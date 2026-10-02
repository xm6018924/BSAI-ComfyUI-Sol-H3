"""File-backed immutable AV stage checkpoints, separate from accepted delivery.

Caller holds the chain's exclusive loop lock. Its contract must include verified
execution identities; this serializer does not authenticate a user model label.
An orphan tensor file after power loss is never considered a completed stage.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

from comfy.nested_tensor import NestedTensor
from safetensors.torch import load_file, save_file

from .long_video_delivery import _atomic_write_json, _sha256_file
from .long_video_dual_model_stages import _validate_av_samples

_STAGES = {"low_x0", "high_input", "high_output"}
_RECEIPT_FIELDS = {
    "schema",
    "stage",
    "contract",
    "tensor_file",
    "tensor_sha256",
    "shapes",
    "has_mask",
    "metadata",
    "report",
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class AVStageCache:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def _key(self, stage, contract):
        if stage not in _STAGES:
            raise ValueError("Unknown dual-model stage checkpoint")
        digest = hashlib.sha256(_canonical(contract).encode()).hexdigest()
        return f"{stage}-{digest}"

    def load(self, stage, contract):
        key = self._key(stage, contract)
        receipt_path = self.root / (key + ".json")
        if not receipt_path.exists():
            return None
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (type(receipt) is not dict or set(receipt) != _RECEIPT_FIELDS
                or receipt.get("schema") != 1 or receipt.get("stage") != stage
                or _canonical(receipt.get("contract")) != _canonical(contract)):
            raise ValueError("Stage checkpoint contract is corrupt or mismatched")
        filename = receipt["tensor_file"]
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.startswith(key + "-"):
            raise ValueError("Stage checkpoint tensor path is invalid")
        path = (self.root / filename).resolve()
        if path.parent != self.root or _sha256_file(path) != receipt["tensor_sha256"]:
            raise ValueError("Stage checkpoint tensor integrity failed")
        tensors = load_file(str(path), device="cpu")
        expected = {"samples_video", "samples_audio"}
        if receipt["has_mask"]:
            expected |= {"mask_video", "mask_audio"}
        if set(tensors) != expected:
            raise ValueError("Stage checkpoint tensor fields changed")
        output = dict(receipt["metadata"])
        if "samples" in output or "noise_mask" in output:
            raise ValueError("Stage checkpoint metadata contains reserved tensor keys")
        output["samples"] = NestedTensor((tensors["samples_video"], tensors["samples_audio"]))
        shapes = _validate_av_samples(output["samples"])
        if list(map(list, shapes)) != receipt["shapes"]:
            raise ValueError("Stage checkpoint AV geometry changed")
        if receipt["has_mask"]:
            output["noise_mask"] = NestedTensor((tensors["mask_video"], tensors["mask_audio"]))
            if _validate_av_samples(output["noise_mask"]) != shapes:
                raise ValueError("Stage checkpoint mask geometry changed")
        return output, receipt

    def save(self, stage, contract, latent, report):
        key = self._key(stage, contract)
        if (self.root / (key + ".json")).exists():
            # Existing accepted stage must be loaded, not overwritten by a
            # second result with a potentially different random/runtime state.
            raise FileExistsError("Immutable stage checkpoint already exists")
        shapes = _validate_av_samples(latent["samples"])
        tensors = {"samples_" + name: value.detach().cpu().contiguous().clone()
                   for name, value in zip(("video", "audio"), latent["samples"].unbind())}
        mask = latent.get("noise_mask")
        if mask is not None:
            if _validate_av_samples(mask) != shapes:
                raise ValueError("Stage noise mask must match the full AV samples")
            tensors.update({"mask_" + name: value.detach().cpu().contiguous().clone()
                            for name, value in zip(("video", "audio"), mask.unbind())})
        metadata = {name: value for name, value in latent.items() if name not in {"samples", "noise_mask"}}
        _canonical(metadata)
        _canonical(report)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (key + "-" + uuid.uuid4().hex + ".safetensors")
        # Tensor filename is private and immutable; the atomic receipt is the
        # only completion marker. Partial/unreferenced files are never resumed.
        save_file(tensors, str(path))
        with path.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        receipt = {"schema": 1, "stage": stage, "contract": contract, "tensor_file": path.name,
                   "tensor_sha256": _sha256_file(path), "shapes": list(map(list, shapes)),
                   "has_mask": mask is not None, "metadata": metadata, "report": report}
        _atomic_write_json(self.root / (key + ".json"), receipt)
        return receipt
