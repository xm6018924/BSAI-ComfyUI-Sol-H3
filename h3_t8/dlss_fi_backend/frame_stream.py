"""Bounded-memory RGB frame sequencing for the independent fixed-2x route.

Caller owns the worker and encoder lifetime. All output comes from unchanged
source frames, explicitly accounted holds, or validated worker payloads; there is
no nearest-time/source-copy fallback for a missing intermediate frame.
"""
from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np

from .guides import MotionGuide
from ..dlss_fi_contract import FrameLedger


def rgb_hash(payload, width, height):
    if not isinstance(payload, bytes) or len(payload) != width*height*4:
        raise ValueError("Worker/source RGBA payload has incorrect dimensions")
    return hashlib.sha256(np.frombuffer(payload, np.uint8).reshape(height,width,4)[...,:3].tobytes()).hexdigest()


class FrameStream:
    def __init__(self, plan, *, width, height, session, observer=None, check=None):
        self.plan, self.width, self.height, self.session = plan, width, height, session
        self.observer = observer if observer is not None else lambda row: None
        self.check = check if check is not None else lambda: None
        self.guide = MotionGuide(width, height)
        self.ledger, self.slots = FrameLedger(plan), iter(plan.slots())
        self.report, self.used = None, False

    def outputs(self, frames):
        if self.used:
            raise RuntimeError("A frame stream can only be consumed once")
        self.used = True
        previous, previous_hash, consumed = None, None, 0
        for index, frame in enumerate(frames):
            self.check()
            if index >= self.plan.source_count:
                raise ValueError("Decoder produced more source frames than the qualified timeline")
            vectors, reset = self.guide.process(frame, reset=index in self.plan.cuts)
            source = frame.tobytes()
            source_hash = rgb_hash(source, self.width, self.height)
            response = self.session.frame(source, vectors.tobytes(), self.plan.origin+Fraction(index)/Fraction(self.plan.source_rate), reset=reset)
            if response.get("usable_generated_frames") != (0 if reset else 1):
                raise ValueError("Worker frame count does not match the declared reset/normal interval")
            if index:
                slot = next(self.slots)
                intermediate = previous if reset else response["rgba"]
                digest = rgb_hash(intermediate, self.width, self.height)
                endpoints = (previous_hash,) if reset else (previous_hash, source_hash)
                self.ledger.observe(slot, payload_sha256=digest, worker_generated=not reset, endpoint_hashes=endpoints)
                self.observer({"slot": slot.index, "pts": str(slot.pts), "kind": slot.kind,
                    "rgb_sha256": digest, "input_index": index, "reset": reset})
                self.check()
                yield slot, intermediate
            slot = next(self.slots)
            self.ledger.observe(slot, payload_sha256=source_hash, endpoint_hashes=(source_hash,))
            self.observer({"slot": slot.index, "pts": str(slot.pts), "kind": slot.kind,
                "rgb_sha256": source_hash, "input_index": index, "reset": reset})
            yield slot, source
            previous, previous_hash, consumed = source, source_hash, index+1
        if consumed != self.plan.source_count:
            raise ValueError("Decoder returned fewer frames than the qualified timeline")
        self.check()
        slot = next(self.slots)
        self.ledger.observe(slot, payload_sha256=previous_hash, endpoint_hashes=(previous_hash,))
        self.observer({"slot": slot.index, "pts": str(slot.pts), "kind": slot.kind,
            "rgb_sha256": previous_hash, "input_index": consumed-1, "reset": False})
        yield slot, previous
        self.report = self.ledger.finish()
