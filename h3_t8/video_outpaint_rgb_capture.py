"""Explicit diagnostic-only, streaming capture of encoder-input RGB24 bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


class EncoderRGBCapture:
    """Keep raw evidence even on failure; never treats it as a deliverable video."""

    def __init__(self, path, *, width, height, frames, provenance):
        for value in (width, height, frames):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("RGB capture dimensions and frame count must be positive integers")
        self.path = Path(path).resolve()
        self.metadata_path = self.path.with_suffix(self.path.suffix + ".json")
        self.frame_bytes = width * height * 3
        self.expected_frames = frames
        self.record = dict(schema="t8.outpaint.encoder_rgb_capture/v1", width=width, height=height,
                           fps="24/1", pixel_format="rgb24", expected_frames=frames,
                           expected_bytes=self.frame_bytes * frames, provenance=provenance,
                           scope="diagnostic_only_not_generation_or_video_acceptance")
        self.handle = self.metadata = None
        self.sha = hashlib.sha256()
        self.bytes_written = self.frames_written = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.metadata_path.exists():
            raise FileExistsError("RGB diagnostic capture must use new paths")
        if shutil.disk_usage(self.path.parent).free < self.record["expected_bytes"] + 256 * 1024**2:
            raise OSError("insufficient disk headroom for explicitly requested raw RGB capture")
        self.handle = self.path.open("xb", buffering=0)
        try:
            self.metadata = self.metadata_path.open("x", encoding="utf-8")
            self._save("capturing")
        except BaseException:
            self.handle.close()
            if self.metadata is not None:
                self.metadata.close()
            raise
        return self

    def _save(self, status, error=None):
        value = {**self.record, "status": status, "frames_written": self.frames_written,
                 "bytes_written": self.bytes_written, "rgb_sha256": self.sha.hexdigest(),
                 "complete_rgb_stream": self.frames_written == self.expected_frames,
                 "video_acceptance_claimed": False, "error": error}
        self.metadata.seek(0)
        json.dump(value, self.metadata, indent=2)
        self.metadata.truncate()
        self.metadata.flush()

    def write_frame(self, payload):
        data = memoryview(payload).cast("B")
        if len(data) != self.frame_bytes or self.frames_written >= self.expected_frames:
            raise ValueError("RGB diagnostic frame size/count mismatch")
        while data:
            written = self.handle.write(data)
            if not written:
                raise OSError("RGB diagnostic capture stopped accepting bytes")
            self.sha.update(data[:written])
            self.bytes_written += written
            data = data[written:]
        self.frames_written += 1

    def __exit__(self, exc_type, exc, traceback):
        try:
            complete = self.frames_written == self.expected_frames
            self._save("captured" if complete else "partial", None if exc is None else f"{exc_type.__name__}: {exc}")
        finally:
            self.handle.close()
            self.metadata.close()
        if exc is None and not complete:
            raise ValueError("RGB diagnostic capture ended before all frames")
        return False
