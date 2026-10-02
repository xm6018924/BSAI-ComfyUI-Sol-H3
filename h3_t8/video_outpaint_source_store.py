"""Immutable, checksummed source-video latent chunks shared by sample windows.

This is source preparation storage, not a claim of finished resumable sampling.
Writes are serialized with an OS-owned lock; interrupted uncommitted assets can
be reused only if their bytes match. Existing committed chunks are never replaced.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load, save

from .long_video_delivery import _atomic_write_bytes, _manifest_lock
from .video_outpaint_plan import canonical, _integer, validate_outpaint_plan


SCHEMA = "t8.h3.video_outpaint.source_latents/v1"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class OutpaintSourceStore:
    def __init__(self, root, plan, *, video_vae_sha256):
        self.root = Path(root).resolve()
        self.plan = validate_outpaint_plan(plan)
        if not isinstance(video_vae_sha256, str) or len(video_vae_sha256) != 64 or any(c not in "0123456789abcdef" for c in video_vae_sha256):
            raise ValueError("video_vae_sha256 must be the verified loaded-VAE state identity")
        self.identity = {"plan_sha256": self.plan["plan_sha256"], "video_vae_sha256": video_vae_sha256,
                         "encoder_contract": "native_17_5_public22_v1"}
        self.path = self.root / "outpaint_source_latents.json"
        self.expected = []
        for shot_index, shot in enumerate(self.plan["shots"]):
            total = (shot["aligned_frames"] - 5) // 17 * 5 + 2
            self.expected.extend((shot_index, start, min(start+5, total)) for start in range(0, total, 5))
        with _manifest_lock(self.root):
            if self.path.exists():
                self._read()
            else:
                self._write({"schema": SCHEMA, "identity": self.identity, "chunks": []})

    def _read(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        digest = data.pop("manifest_sha256", None)
        if (data.get("schema") != SCHEMA or data.get("identity") != self.identity
                or digest != _sha(canonical(data).encode())):
            raise ValueError("source latent store integrity or source/VAE/plan mismatch")
        records = data.get("chunks")
        if not isinstance(records, list) or len(records) > len(self.expected):
            raise ValueError("source latent store has an invalid chunk table")
        for record, expected in zip(records, self.expected):
            if [record.get(k) for k in ("shot", "start", "stop")] != list(expected):
                raise ValueError("source latent store contains a gap or reordered chunk")
            digest = record.get("sha256", "")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid latent asset hash")
            _integer(record.get("bytes"), "asset bytes", 1, 2**31)
        return data

    def _write(self, data):
        signed = {**data, "manifest_sha256": _sha(canonical(data).encode())}
        _atomic_write_bytes(self.path, canonical(signed).encode())

    def position(self):
        with _manifest_lock(self.root):
            count = len(self._read()["chunks"])
        return None if count == len(self.expected) else self.expected[count]

    def _shape(self, count):
        return (1, 24, count, self.plan["sampling"]["height"] // 16, self.plan["sampling"]["width"] // 16)

    def append(self, shot_index, token_start, tensor):
        """Commit exactly the next chunk; caller must have checked current source/VAE identity."""
        with _manifest_lock(self.root):
            data = self._read()
            count = len(data["chunks"])
            if count == len(self.expected):
                raise ValueError("all source chunks are already committed")
            shot, start, stop = self.expected[count]
            if (shot_index, token_start) != (shot, start):
                raise ValueError("source chunks must commit in serial shot/token order")
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32
                    or tuple(tensor.shape) != self._shape(stop-start) or not torch.isfinite(tensor).all()):
                raise ValueError("source chunk must have the planned finite CPU float32 shape")
            blob = save({"video": tensor.contiguous()})
            digest = _sha(blob)
            asset = self.root / f"video-{digest}.safetensors"
            if asset.exists():
                if _sha(asset.read_bytes()) != digest:
                    raise ValueError("existing source asset was corrupted; refusing overwrite")
            else:
                _atomic_write_bytes(asset, blob)
            data["chunks"].append({"shot": shot, "start": start, "stop": stop, "sha256": digest, "bytes": len(blob)})
            self._write(data)

    def read_range(self, shot_index, token_start, token_stop):
        _integer(shot_index, "shot_index", 0, len(self.plan["shots"])-1)
        total = (self.plan["shots"][shot_index]["aligned_frames"] - 5) // 17 * 5 + 2
        _integer(token_start, "token_start", 0, total-1)
        _integer(token_stop, "token_stop", token_start+1, total)
        # At most the largest supported sampling window; never allocate a full long video here.
        if token_stop-token_start > 57:
            raise ValueError("source read exceeds a 192-frame H3 sampling window")
        with _manifest_lock(self.root):
            data = self._read()
        result = torch.empty(self._shape(token_stop-token_start))
        cursor = token_start
        for item in data["chunks"]:
            if item["shot"] != shot_index or item["stop"] <= cursor or item["start"] >= token_stop:
                continue
            if item["start"] > cursor:
                raise ValueError("source latent range contains an uncommitted gap")
            asset = self.root / f"video-{item['sha256']}.safetensors"
            if not asset.is_file() or asset.stat().st_size != item["bytes"]:
                raise ValueError("source latent asset is missing or truncated")
            blob = asset.read_bytes()
            if _sha(blob) != item["sha256"]:
                raise ValueError("source latent asset integrity mismatch")
            tensors = load(blob)
            tensor = tensors.get("video")
            if (set(tensors) != {"video"} or tensor.dtype != torch.float32
                    or tuple(tensor.shape) != self._shape(item["stop"]-item["start"]) or not torch.isfinite(tensor).all()):
                raise ValueError("source latent asset has invalid tensor contents")
            stop = min(item["stop"], token_stop)
            result[:, :, cursor-token_start:stop-token_start] = tensor[:, :, cursor-item["start"]:stop-item["start"]]
            cursor = stop
        if cursor != token_stop:
            raise ValueError("source latent range has not been completely prepared")
        return result
