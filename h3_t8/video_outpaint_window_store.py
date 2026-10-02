"""Durable serial sampling checkpoints; no video decoding or automatic acceptance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load, save

from .long_video_delivery import _atomic_write_bytes, _manifest_lock
from .video_outpaint_plan import canonical, validate_outpaint_plan, _integer
from .video_outpaint_noise import NOISE_ALGORITHMS


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class OutpaintWindowStore:
    def __init__(self, root, plan, *, execution_identity):
        self.root = Path(root).resolve()
        self.plan = validate_outpaint_plan(plan)
        required = {"model_sha256", "conditioning_sha256", "source_cache_sha256", "audio_source_sha256",
                    "seed", "steps", "sampler_name", "scheduler", "noise_algorithm"}
        if not isinstance(execution_identity, dict) or set(execution_identity) != required:
            raise ValueError("execution identity must include model/conditioning/source/audio/settings bindings")
        for key in required:
            value = execution_identity[key]
            if key.endswith("_sha256") and (not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError(f"invalid execution {key}")
        _integer(execution_identity["seed"], "seed", 0, 2**64-1)
        _integer(execution_identity["steps"], "steps", 1, 100)
        if execution_identity["noise_algorithm"] not in NOISE_ALGORITHMS:
            raise ValueError("unrecognized outpaint noise algorithm")
        if any(not isinstance(execution_identity[k], str) or not execution_identity[k] for k in ("sampler_name", "scheduler")):
            raise ValueError("sampler and scheduler names are required")
        implementation = {name: _sha(Path(__file__).with_name(name).read_bytes()) for name in (
            "video_outpaint_sampling.py", "video_outpaint_sampling_runtime.py", "video_outpaint_noise.py",
            "video_outpaint_execution.py")}
        self.identity = json.loads(canonical({"plan_sha256": self.plan["plan_sha256"], **execution_identity,
                                             "implementation_sha256": _sha(canonical(implementation).encode())}))
        self.windows = [(s, w) for s, shot in enumerate(self.plan["shots"]) for w in range(len(shot["windows"]))]
        self.path = self.root / "outpaint_windows.json"
        with _manifest_lock(self.root):
            if self.path.exists():
                self._read()
            else:
                self._write({"schema": "t8.h3.video_outpaint.window_store/v1", "identity": self.identity,
                             "status": "ready", "committed": []})

    def _read(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        digest = data.pop("sha256", None)
        if (digest != _sha(canonical(data).encode()) or data.get("identity") != self.identity
                or data.get("schema") != "t8.h3.video_outpaint.window_store/v1"):
            raise ValueError("sampling store integrity or execution identity mismatch")
        records = data.get("committed")
        if (not isinstance(records, list) or len(records) > len(self.windows)
                or data.get("status") not in {"ready", "running", "interrupted", "paused", "sampled"}):
            raise ValueError("invalid sampling checkpoint state")
        if data["status"] == "paused" and not 0 < len(records) < len(self.windows):
            raise ValueError("paused sampling requires a nonempty incomplete prefix")
        if (data["status"] == "sampled") != (len(records) == len(self.windows)):
            raise ValueError("sampling completion status disagrees with window coverage")
        for index, record in enumerate(records):
            digest = record.get("sha256", "")
            if (record.get("shot"), record.get("window")) != self.windows[index] or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("sampling checkpoints contain a gap or invalid asset hash")
            _integer(record.get("bytes"), "window asset bytes", 1, 2**31)
        return data

    def _write(self, data):
        _atomic_write_bytes(self.path, canonical({**data, "sha256": _sha(canonical(data).encode())}).encode())

    def snapshot(self):
        with _manifest_lock(self.root):
            return self._read()

    def begin(self, *, resume=False):
        """Caller must hold the OS-owned execution lease for the whole sampling run."""
        with _manifest_lock(self.root):
            data = self._read()
            if data["status"] in {"running", "interrupted", "paused"} and not resume:
                raise ValueError("interrupted sampling requires an explicit resume request")
            if data["status"] != "sampled":
                data["status"] = "running"
                self._write(data)
            return len(data["committed"])

    def pause(self):
        """Normal bounded-run boundary, while the caller still holds the GPU lease."""
        with _manifest_lock(self.root):
            data = self._read()
            if data["status"] == "sampled":
                return
            if data["status"] != "running" or not data["committed"]:
                raise ValueError("only a running committed prefix can pause")
            data["status"] = "paused"
            self._write(data)

    def mark_interrupted(self):
        with _manifest_lock(self.root):
            data = self._read()
            if data["status"] != "sampled":
                data["status"] = "interrupted"
                self._write(data)

    def _shapes(self, index):
        shot, window = self.windows[index]
        frames = self.plan["shots"][shot]["windows"][window]["render_frames"]
        return ((1, 24, (frames-5)//17*5+2, self.plan["sampling"]["height"]//16, self.plan["sampling"]["width"]//16),
                (1, 32, 2, round(frames/24*40)))

    def commit(self, index, video, audio):
        with _manifest_lock(self.root):
            data = self._read()
            if data["status"] != "running" or index != len(data["committed"]) or index >= len(self.windows):
                raise ValueError("sampling windows must commit once in serial order")
            for tensor, shape in zip((video, audio), self._shapes(index)):
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.device.type != "cpu" or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                    raise ValueError("sample checkpoint requires planned finite CPU float32 AV latents")
            blob = save({"video": video.contiguous(), "audio": audio.contiguous()})
            digest = _sha(blob)
            path = self.root / f"window-{digest}.safetensors"
            if path.exists():
                if _sha(path.read_bytes()) != digest:
                    raise ValueError("existing window asset is corrupted; refusing overwrite")
            else:
                _atomic_write_bytes(path, blob)
            shot, window = self.windows[index]
            data["committed"].append({"shot": shot, "window": window, "sha256": digest, "bytes": len(blob)})
            if len(data["committed"]) == len(self.windows):
                data["status"] = "sampled"
            self._write(data)

    def load(self, index):
        data = self.snapshot()
        _integer(index, "checkpoint index", 0, len(data["committed"])-1)
        item = data["committed"][index]
        path = self.root / f"window-{item['sha256']}.safetensors"
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("sample checkpoint asset is missing or truncated")
        blob = path.read_bytes()
        if _sha(blob) != item["sha256"]:
            raise ValueError("sample checkpoint asset hash mismatch")
        tensors = load(blob)
        if set(tensors) != {"video", "audio"}:
            raise ValueError("sample checkpoint has unexpected tensor keys")
        for key, shape in zip(("video", "audio"), self._shapes(index)):
            if tuple(tensors[key].shape) != shape or tensors[key].dtype != torch.float32 or not torch.isfinite(tensors[key]).all():
                raise ValueError("sample checkpoint has invalid tensor contents")
        return tensors["video"], tensors["audio"]

    def context_for(self, index):
        _integer(index, "window index", 0, len(self.windows)-1)
        shot_index, window_index = self.windows[index]
        if window_index == 0:
            return None
        video, audio = self.load(index-1)
        shot = self.plan["shots"][shot_index]
        previous, current = shot["windows"][window_index-1], shot["windows"][window_index]
        return {"plan_sha256": self.plan["plan_sha256"], "shot_index": shot_index, "target_window_index": window_index,
                "video_tail": video[:, :, current["video_start"]-previous["video_start"]:].clone(),
                "audio_tail": audio[..., current["audio_start"]-previous["audio_start"]:].clone()}

    def read_video_range(self, shot_index, token_start, token_stop):
        """Read the committed global trajectory with earliest-window ownership."""
        _integer(shot_index, "shot_index", 0, len(self.plan["shots"])-1)
        shot = self.plan["shots"][shot_index]
        total = (shot["aligned_frames"]-5)//17*5+2
        _integer(token_start, "token_start", 0, total-1)
        _integer(token_stop, "token_stop", token_start+1, total)
        if token_stop-token_start > 57:
            raise ValueError("global latent read exceeds the bounded window contract")
        snapshot = self.snapshot()
        result = torch.empty((1, 24, token_stop-token_start, self.plan["sampling"]["height"]//16,
                              self.plan["sampling"]["width"]//16))
        cursor = token_start
        for index, record in enumerate(snapshot["committed"]):
            if record["shot"] != shot_index:
                continue
            window = shot["windows"][record["window"]]
            start = window["video_start"]
            stop = start + (window["render_frames"]-5)//17*5+2
            if stop <= cursor or start >= token_stop:
                continue
            if start > cursor:
                raise ValueError("global sampled trajectory has an uncommitted gap")
            video, _ = self.load(index)
            end = min(stop, token_stop)
            result[:, :, cursor-token_start:end-token_start] = video[:, :, cursor-start:end-start]
            cursor = end
            if cursor == token_stop:
                return result
        raise ValueError("global sampled trajectory is not complete for this range")
