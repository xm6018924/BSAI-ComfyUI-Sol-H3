"""Independent CPU contracts for the planned DLSSG 2x file route.

Wire format verified against the pinned upstream Python protocol at c755e274.
This module does not import the external plugin, start a worker or qualify a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import struct


SETUP_MAGIC = 0x31534746
SETUP_OUT_MAGIC = 0x31524746
FRAME_OUT_MAGIC = 0x314F4746


def _positive_int(value, name, maximum):
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return value


def _rational(value, name):
    if type(value) not in (int, Fraction):
        raise ValueError(f"{name} requires exact integer/Fraction values, not float FPS")
    return Fraction(value)


def setup_packet(width, height, frame_count):
    """One generated frame per interval, never implicit multi-frame/cascade."""
    _positive_int(width, "width", 8192)
    _positive_int(height, "height", 8192)
    _positive_int(frame_count, "frame_count", 1_000_000)
    if frame_count < 2:
        raise ValueError("Interpolation requires at least two source frames")
    return struct.pack("<5I", SETUP_MAGIC, width, height, frame_count, 1)


def _header(data, magic):
    if not isinstance(data, bytes) or len(data) != 16:
        raise ValueError("Worker response header must contain exactly 16 bytes")
    got_magic, status, count, flags = struct.unpack("<4I", data)
    if got_magic != magic or status != 0:
        raise ValueError(f"Worker protocol/status failure: magic={got_magic:#x}, status={status}")
    return count, flags


def validate_setup_response(data):
    maximum, reserved = _header(data, SETUP_OUT_MAGIC)
    if reserved or maximum < 1:
        raise ValueError("Worker cannot provide the requested native 2x route")
    return {"requested_multiplier": 2, "maximum_generated_per_interval": maximum}


def validate_frame_response(data, *, reset):
    """Validate BEFORE reading or allocating the worker-declared payload.

At reset the worker may return zero or one frame; any payload must be consumed
but is never used to synthesize across a cut. Disabled is always a failure.
"""
    if type(reset) is not bool:
        raise ValueError("reset must be boolean")
    generated, disabled = _header(data, FRAME_OUT_MAGIC)
    if disabled:
        raise ValueError("DLSSG disabled: source duplication cannot qualify as interpolation")
    if generated > 1 or (not reset and generated != 1):
        raise ValueError("Worker did not produce exactly one requested intermediate frame")
    return {"payload_frames": generated, "usable_generated_frames": 0 if reset else 1}


@dataclass(frozen=True)
class OutputSlot:
    index: int
    pts: Fraction
    kind: str
    source_index: int
    right_source_index: int | None = None


@dataclass(frozen=True)
class TwoXTimeline:
    source_count: int
    source_rate: Fraction
    origin: Fraction
    cuts: frozenset[int]

    def __post_init__(self):
        _positive_int(self.source_count, "source_count", 1_000_000)
        if self.source_count < 2 or not 0 < _rational(self.source_rate, "source_rate") <= 240:
            raise ValueError("Invalid first-route source count/rate")
        _rational(self.origin, "origin")
        if type(self.cuts) is not frozenset or any(type(i) is not int or not 1 <= i < self.source_count for i in self.cuts):
            raise ValueError("Invalid cut indices")

    @property
    def target_rate(self):
        return 2 * self.source_rate

    @property
    def output_count(self):
        return 2 * self.source_count

    @property
    def duration(self):
        return Fraction(self.source_count, 1) / self.source_rate

    @property
    def generated_count(self):
        return self.source_count - 1 - len(self.cuts)

    def slots(self):
        """Constant extra memory; source origin and total frame duration preserved."""
        for index in range(self.output_count):
            source_index, half = divmod(index, 2)
            right = None
            if not half:
                kind = "source"
            elif source_index == self.source_count - 1:
                kind = "tail_hold"
            elif source_index + 1 in self.cuts:
                kind = "cut_hold"
            else:
                kind, right = "generated", source_index + 1
            yield OutputSlot(index, self.origin + Fraction(index, 1) / self.target_rate,
                             kind, source_index, right)


def build_timeline(timestamps, source_rate, *, cuts=()):
    """Reject VFR rather than silently treating it as CFR or changing audio offset.

Cuts are explicit right-source indices, not a claim of automatic cut detection.
Actual container tick rounding must be resolved by the media adapter separately.
"""
    rate = _rational(source_rate, "source_rate")
    if not 0 < rate <= 240:
        raise ValueError("Source rate outside the first-route 0..240fps contract")
    count = _positive_int(len(timestamps), "source_count", 1_000_000)
    if count < 2:
        raise ValueError("Interpolation requires at least two source frames")
    origin = _rational(timestamps[0], "source timestamp")
    for i, stamp in enumerate(timestamps):
        if _rational(stamp, "source timestamp") != origin + Fraction(i, 1) / rate:
            raise ValueError("Source is not exact CFR: explicit timeline conversion required")
    cut_list = tuple(cuts)
    if any(type(i) is not int or not 1 <= i < count for i in cut_list) or len(set(cut_list)) != len(cut_list):
        raise ValueError("Cuts must be unique interior right-source indices")
    return TwoXTimeline(count, rate, origin, frozenset(cut_list))


class FrameLedger:
    """Streaming output accounting. Counts cannot replace actual worker/device proof."""
    def __init__(self, plan):
        if not isinstance(plan, TwoXTimeline):
            raise ValueError("Expected validated 2x timeline")
        self.plan = plan
        self._slots = iter(plan.slots())
        self.counts = dict.fromkeys(("source", "generated", "cut_hold", "tail_hold"), 0)
        self.index = 0
        self.failed = False
        self.finished = False

    def observe(self, slot, *, payload_sha256, worker_generated=False, endpoint_hashes=()):
        if self.failed or self.finished:
            raise ValueError("Ledger is closed or failed")
        try:
            if type(endpoint_hashes) not in (tuple, list) or len(endpoint_hashes) not in (1, 2):
                raise ValueError("Supply one source digest or two interpolation endpoint digests")
            wanted = next(self._slots)
            if slot != wanted:
                raise ValueError("Output order, PTS or provenance does not match the 2x plan")
            hashes = (payload_sha256, *endpoint_hashes)
            if any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes):
                raise ValueError("Frame identity requires a lowercase SHA-256 digest")
            if type(worker_generated) is not bool or worker_generated != (slot.kind == "generated"):
                raise ValueError("Frame provenance disagrees with actual worker output")
            if slot.kind == "generated":
                if len(endpoint_hashes) != 2:
                    raise ValueError("Both endpoint identities are required")
                # Equal static inputs may legitimately remain unchanged. A motion
                # interval copied byte-for-byte from either endpoint must fail.
                if endpoint_hashes[0] != endpoint_hashes[1] and payload_sha256 in endpoint_hashes:
                    raise ValueError("Moving interval is an exact endpoint copy, not qualified interpolation")
            elif len(endpoint_hashes) != 1 or payload_sha256 != endpoint_hashes[0]:
                raise ValueError("Source/cut/tail hold changed source pixels")
            self.counts[slot.kind] += 1
            self.index += 1
        except (ValueError, StopIteration) as error:
            self.failed = True
            raise ValueError(str(error) or "Extra output frame") from error

    def finish(self):
        if self.failed or self.finished or self.index != self.plan.output_count:
            raise ValueError("Incomplete, failed or already finished output ledger")
        self.finished = True
        return {"status": "frame_accounting_only_not_runtime_or_quality_qualification",
                "output_frames": self.index, "counts": dict(self.counts),
                "expected_generated_frames": self.plan.generated_count,
                "has_generated_intervals": self.plan.generated_count > 0,
                "origin": str(self.plan.origin), "duration": str(self.plan.duration),
                "target_rate": str(self.plan.target_rate),
                "runtime_qualified": False, "audio_qualified": False, "quality_qualified": False}
