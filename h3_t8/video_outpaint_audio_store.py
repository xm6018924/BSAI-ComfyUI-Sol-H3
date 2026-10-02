"""Durable observed-audio posterior chunks and bounded sampler windows.

Source gaps are encoded silence; source-free/padded audio cells remain unobserved.
These latents never replace the original delivery soundtrack.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load, save

from .long_video_delivery import _atomic_write_bytes, _manifest_lock
from .video_outpaint_plan import canonical, validate_outpaint_plan, _integer


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _hash(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


class OutpaintAudioStore:
    def __init__(self, root, plan, *, audio_vae_sha256, pcm_sha256, stream_position,
                 encoding_device, block_tokens=64):
        self.root = Path(root).resolve()
        self.plan = validate_outpaint_plan(plan)
        _integer(block_tokens, "audio block_tokens", 1, 128)
        self.has_audio = stream_position is not None
        if self.has_audio:
            _integer(stream_position, "audio stream_position", 0, 1000)
            _hash(audio_vae_sha256, "audio VAE identity")
            _hash(pcm_sha256, "canonical PCM identity")
            if not isinstance(encoding_device, str) or not encoding_device:
                raise ValueError("audio encoding device is required")
        elif any(v is not None for v in (audio_vae_sha256, pcm_sha256, encoding_device)):
            raise ValueError("audio-free cache must not claim a VAE/PCM/device identity")
        implementation = {name: _sha(Path(__file__).with_name(name).read_bytes()) for name in (
            "video_outpaint_audio.py", "video_outpaint_audio_file.py", "video_outpaint_audio_store.py",
            "video_outpaint_audio_runtime.py", "video_outpaint_audio_decode.py")}
        self.identity = {"plan_sha256": self.plan["plan_sha256"], "audio_vae_sha256": audio_vae_sha256,
                         "pcm_sha256": pcm_sha256, "stream_position": stream_position,
                         "encoding_device": encoding_device, "block_tokens": block_tokens,
                         "implementation_sha256": _sha(canonical(implementation).encode())}
        self.totals = []
        self.expected = []
        for shot_index, shot in enumerate(self.plan["shots"]):
            samples = round(Fraction(shot["stop"]*32000, 24))-round(Fraction(shot["start"]*32000, 24))
            total = (samples+799)//800 if self.has_audio else 0
            self.totals.append(total)
            self.expected.extend((shot_index, start, min(total, start+block_tokens))
                                 for start in range(0, total, block_tokens))
        self.path = self.root / "outpaint_source_audio.json"
        self.poison = self.root / "invalid_audio_preparation.json"
        with _manifest_lock(self.root):
            if self.poison.exists():
                raise ValueError("audio cache was invalidated; inspect it and choose a new directory")
            if self.path.exists():
                self._read()
            else:
                self._write({"schema": "t8.h3.outpaint.audio_store/v1", "identity": self.identity,
                             "status": "ready" if self.expected else "prepared", "chunks": []})

    def _write(self, data):
        _atomic_write_bytes(self.path, canonical({**data, "sha256": _sha(canonical(data).encode())}).encode())

    def _read(self):
        if self.poison.exists():
            raise ValueError("audio cache was invalidated during preparation")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        digest = data.pop("sha256", None)
        if (digest != _sha(canonical(data).encode()) or data.get("identity") != self.identity
                or data.get("schema") != "t8.h3.outpaint.audio_store/v1"):
            raise ValueError("audio store integrity or source/VAE/settings identity mismatch")
        records = data.get("chunks")
        if (not isinstance(records, list) or len(records) > len(self.expected)
                or data.get("status") not in {"ready", "running", "interrupted", "prepared"}):
            raise ValueError("invalid audio cache state")
        if (data["status"] == "prepared") != (len(records) == len(self.expected)):
            raise ValueError("audio preparation status disagrees with coverage")
        for record, expected in zip(records, self.expected):
            if not isinstance(record, dict) or tuple(record.get(k) for k in ("shot", "start", "stop")) != expected:
                raise ValueError("audio chunks contain a gap or reordered position")
            _hash(record.get("sha256"), "audio asset identity")
            _integer(record.get("bytes"), "audio asset bytes", 1, 65536)
        return data

    def snapshot(self):
        with _manifest_lock(self.root):
            return self._read()

    def begin(self, *, resume=False):
        with _manifest_lock(self.root):
            data = self._read()
            if data["status"] in {"running", "interrupted"} and not resume:
                raise ValueError("interrupted audio preparation requires explicit resume")
            if data["status"] != "prepared":
                data["status"] = "running"
                self._write(data)
            return len(data["chunks"])

    def mark_interrupted(self):
        with _manifest_lock(self.root):
            data = self._read()
            if data["status"] != "prepared":
                data["status"] = "interrupted"
                self._write(data)

    def append(self, shot, start, tensor):
        with _manifest_lock(self.root):
            data = self._read()
            index = len(data["chunks"])
            if data["status"] != "running" or index >= len(self.expected):
                raise ValueError("audio chunks must be committed during serial preparation")
            expected_shot, expected_start, stop = self.expected[index]
            if (shot, start) != (expected_shot, expected_start):
                raise ValueError("audio chunks must commit once in serial order")
            self._validate_tensor(tensor, stop-start)
            blob = save({"audio": tensor.contiguous()})
            digest = _sha(blob)
            path = self.root / f"audio-{digest}.safetensors"
            if path.exists():
                if path.stat().st_size != len(blob) or _sha(path.read_bytes()) != digest:
                    raise ValueError("existing audio asset is corrupted; refusing overwrite")
            else:
                _atomic_write_bytes(path, blob)
            data["chunks"].append({"shot": shot, "start": start, "stop": stop,
                                    "sha256": digest, "bytes": len(blob)})
            if len(data["chunks"]) == len(self.expected):
                data["status"] = "prepared"
            self._write(data)

    @staticmethod
    def _validate_tensor(tensor, count):
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32
                or tuple(tensor.shape) != (1, 32, 2, count) or not torch.isfinite(tensor).all()):
            raise ValueError("audio chunk must have the planned finite CPU float32 shape")

    def _load(self, record):
        path = self.root / f"audio-{record['sha256']}.safetensors"
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError("audio asset is missing or truncated")
        blob = path.read_bytes()
        if _sha(blob) != record["sha256"]:
            raise ValueError("audio asset integrity mismatch")
        data = load(blob)
        if set(data) != {"audio"}:
            raise ValueError("invalid audio asset contents")
        self._validate_tensor(data["audio"], record["stop"]-record["start"])
        return data["audio"]

    def verify_assets(self):
        for record in self.snapshot()["chunks"]:
            self._load(record)

    def read_range(self, shot, start, stop):
        _integer(shot, "audio shot", 0, len(self.totals)-1)
        _integer(start, "audio start", 0, self.totals[shot]-1)
        _integer(stop, "audio stop", start+1, self.totals[shot])
        if stop-start > 320:
            raise ValueError("audio read exceeds a 192-frame sampling window")
        result = torch.empty((1, 32, 2, stop-start))
        cursor = start
        for record in self.snapshot()["chunks"]:
            if record["shot"] != shot or record["stop"] <= cursor or record["start"] >= stop:
                continue
            if record["start"] > cursor:
                raise ValueError("audio range contains an uncommitted gap")
            tensor = self._load(record)
            end = min(stop, record["stop"])
            result[..., cursor-start:end-start] = tensor[..., cursor-record["start"]:end-record["start"]]
            cursor = end
        if cursor != stop:
            raise ValueError("audio range is not completely prepared")
        return result

    def window(self, shot, window):
        _integer(shot, "audio shot", 0, len(self.plan["shots"])-1)
        _integer(window, "audio window", 0, len(self.plan["shots"][shot]["windows"])-1)
        if self.snapshot()["status"] != "prepared":
            raise ValueError("sampler audio requires completed source preparation")
        spec = self.plan["shots"][shot]["windows"][window]
        count = round(spec["render_frames"]/24*40)
        start = spec["audio_start"]
        observed = min(count, max(0, self.totals[shot]-start))
        samples = torch.zeros((1, 32, 2, count))
        mask = torch.ones((1, 1, 2, count))
        if observed:
            samples[..., :observed] = self.read_range(shot, start, start+observed)
            mask[..., :observed] = 0
        return {"samples": samples, "noise_mask": mask}
