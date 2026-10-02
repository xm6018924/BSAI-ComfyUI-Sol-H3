"""Independent, bounded DLSSG binary transport under development.

No node registration or automatic runtime lookup. The low-level session also
accepts CPU protocol fixtures. Transport success is NOT device/model/quality
qualification; a separate runtime/media adapter must establish those facts.
"""
from __future__ import annotations

from collections import deque
from fractions import Fraction
import hashlib
import math
import os
from pathlib import Path
import struct
import subprocess
import threading
import time

from ..dlss_fi_contract import setup_packet, validate_setup_response, validate_frame_response  # noqa: E402

WORKER_SHA256 = "8a747f9ed613842d5b8b34a811ad43bc1a9466540e2e5a0c8ef4005f0db9e384"
RUNTIME_SHA256 = "135eaf0733c1e37381a8c28abcf7a862404a54132b81787c04e35d09efc5e36f"
FRAME_MAGIC = 0x31464746


def runtime_identity(directory):
    """Recognize only the researched pair, not a general binary trust decision."""
    root = Path(directory).resolve(strict=True)
    result = {}
    for name, digest in (("dlssg-worker.exe", WORKER_SHA256), ("nvngx_dlssg.dll", RUNTIME_SHA256)):
        path = (root / name).resolve(strict=True)
        if path.parent != root or not path.is_file() or path.stat().st_size > 32 * 1024**2:
            raise ValueError("Runtime file leaves the bounded runtime directory")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("DLSSG runtime differs from the pinned research identity: " + name)
        result[name] = {"path": str(path), "sha256": digest, "bytes": len(data)}
    return result


def frame_packet(index, timestamp, reset):
    if type(index) is not int or not 0 <= index < 1_000_000 or type(reset) is not bool:
        raise ValueError("Invalid frame index/reset")
    if type(timestamp) not in (int, Fraction):
        raise ValueError("Exact timestamp required")
    stamp = Fraction(timestamp)
    if not -(2**63) <= stamp.numerator < 2**63 or not 0 < stamp.denominator < 2**63:
        raise ValueError("Timestamp exceeds signed wire range")
    return struct.pack("<4I2q", FRAME_MAGIC, index, int(reset), 0, stamp.numerator, stamp.denominator)


class BinarySession:
    """One Popen owner, one outstanding request, bounded pipes and deadlines.

Only command lines constructed by the caller are executed, never through a shell.
Native runtime callers must call runtime_identity and verify device/feature
separately. A CPU fixture can exercise failure/cleanup without loading a GPU.
"""
    def __init__(self, command, *, cwd, width, height, frame_count, cancel=None, timeout=30, on_start=None):
        packet = setup_packet(width, height, frame_count)
        if width * height > 4096 * 2160:
            raise ValueError("First transport limits each frame to 4096x2160 pixels")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError("Operation timeout must be finite, positive and at most 120 seconds")
        if not isinstance(command, (list, tuple)) or not command or any(not isinstance(x, str) for x in command):
            raise ValueError("Explicit argument vector required")
        self.width, self.height, self.frame_count = width, height, frame_count
        self.color_bytes = width * height * 4
        self.motion_bytes = width * height * 4
        self.timeout, self.cancel = timeout, cancel if cancel is not None else threading.Event()
        self.closed, self.failed, self.next_index = False, False, 0
        self.last_timestamp = None
        self.logs = deque(maxlen=16)  # Each chunk <=4096 bytes, no unbounded readline.
        self.io_lock, self.close_lock = threading.Lock(), threading.Lock()
        self.frame_lock = threading.Lock()
        self.io_thread = None
        self.stop_receipt = None
        if self.cancel.is_set():
            raise RuntimeError("Interpolation cancelled before starting a worker")
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0, creationflags=flags)
        self.log_thread = threading.Thread(target=self._drain_logs, daemon=True, name="t8-fi-stderr")
        self.log_thread.start()
        try:
            if on_start is not None:
                on_start(self.process)
            self.setup = self._exchange((packet,), lambda: validate_setup_response(self._read_exact(16)))
        except BaseException:
            self.close()
            raise

    def _drain_logs(self):
        try:
            while block := self.process.stderr.read(4096):
                self.logs.append(block)
        except (OSError, ValueError):
            pass

    def _read_exact(self, size):
        data = bytearray()
        while len(data) < size:
            block = self.process.stdout.read(min(size-len(data), 65536))
            if not block:
                raise RuntimeError("Worker closed stdout before completing a bounded response")
            data.extend(block)
        return bytes(data)

    def _write_all(self, data):
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = self.process.stdin.write(view[offset:offset+65536])
            if not written:
                raise RuntimeError("Worker stopped consuming its bounded request")
            offset += written

    def _exchange(self, parts, decode):
        if self.closed or self.failed:
            raise RuntimeError("Interpolation transport is closed or failed")
        if not self.io_lock.acquire(blocking=False):
            raise RuntimeError("Only one serial worker request may be active")
        done, outcome = threading.Event(), {}
        def operation():
            try:
                for part in parts:
                    self._write_all(part)
                outcome["result"] = decode()
            except BaseException as error:
                outcome["error"] = error
            finally:
                done.set()
        try:
            deadline = time.perf_counter() + self.timeout
            if self.cancel.is_set():
                raise RuntimeError("Interpolation cancelled")
            self.io_thread = threading.Thread(target=operation, daemon=True, name="t8-fi-pipe")
            self.io_thread.start()
            while not done.wait(.02):
                if self.cancel.is_set():
                    raise RuntimeError("Interpolation cancelled")
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Bounded worker request timed out")
            if self.cancel.is_set():
                raise RuntimeError("Interpolation cancelled")
            if time.perf_counter() > deadline:
                raise TimeoutError("Worker completed outside its deadline")
            if "error" in outcome:
                raise outcome["error"]
            return outcome["result"]
        except BaseException:
            self.failed = True
            self.close()
            raise
        finally:
            self.io_lock.release()

    def frame(self, rgba, motion, timestamp, *, reset=False):
        if not self.frame_lock.acquire(blocking=False):
            raise RuntimeError("Only one serial frame may be active")
        try:
            return self._frame(rgba, motion, timestamp, reset=reset)
        finally:
            self.frame_lock.release()

    def _frame(self, rgba, motion, timestamp, *, reset=False):
        # Bytes are copied/validated by the later ndarray adapter; no dtype cast
        # or optical-flow warp is performed here. Validate before writing any data.
        if self.closed or self.failed or self.next_index >= self.frame_count:
            raise RuntimeError("Frame submitted after session end/failure or beyond declared count")
        if not isinstance(rgba, bytes) or len(rgba) != self.color_bytes:
            raise ValueError("RGBA must have exact uint8 payload size")
        if not isinstance(motion, bytes) or len(motion) != self.motion_bytes:
            raise ValueError("Motion must have exact 2-channel float16 payload size")
        packet = frame_packet(self.next_index, timestamp, reset)
        if self.next_index == 0 and not reset:
            raise ValueError("First frame must reset worker history")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Source timestamps must strictly increase")
        def response():
            info = validate_frame_response(self._read_exact(16), reset=reset)
            payload = self._read_exact(self.color_bytes) if info["payload_frames"] else None
            return {**info, "rgba": payload if info["usable_generated_frames"] else None}
        result = self._exchange((packet, rgba, motion), response)
        self.next_index += 1
        self.last_timestamp = Fraction(timestamp)
        return result

    def close(self):
        import psutil
        with self.close_lock:
            if self.stop_receipt is not None:
                return
            self.closed = True
            children = []
            if self.process.poll() is None:
                try:
                    children = psutil.Process(self.process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    pass
                # Kill the owned process before closing blocked pipes; this also
                # releases a writer stuck on an unresponsive reader.
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            for child in reversed(children):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(children, timeout=2)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=2)
            threads = [self.log_thread, self.io_thread]
            for thread in threads:
                if thread and thread is not threading.current_thread():
                    thread.join(timeout=2)
            unfinished = [t.name for t in threads if t and t.is_alive()]
            if not unfinished:
                for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
                    pipe.close()
            self.stop_receipt = {"pid": self.process.pid, "exit_code": self.process.poll(),
                "owned_children_remaining": [p.pid for p in alive], "unfinished_threads": unfinished,
                "completed_inputs": self.next_index, "runtime_qualified": False}
            if alive or unfinished:
                raise RuntimeError("Owned transport cleanup incomplete; do not start another worker")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
