"""Byte-budgeted immutable stage payloads, OS ownership, interrupted-state recovery."""

import os
import re
from pathlib import Path
import time
import uuid

from .meridian_checkpoint_io import (
    atomic_json,
    canonical_identity,
    conversion_lock,
    file_sha,
)


class StageStore:
    def __init__(self, root, budget_bytes=20 * 1024**3, cancelled=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget = int(budget_bytes)
        if self.budget < 0:
            raise ValueError("Nonnegative cache budget required")
        self.cancelled = cancelled or (lambda: False)

    def check(self):
        if self.cancelled():
            raise InterruptedError("Meridian job cancelled before next stage")

    def paths(self, stage, contract):
        if stage not in ("geometry", "warp", "encode", "sample", "delivery"):
            raise ValueError("Unknown stage")
        key = canonical_identity(dict(stage=stage, contract=contract))
        return key, self.root / f"{stage}-{key}.pt", self.root / f"{stage}-{key}.json"

    def load(self, stage, contract):
        import json
        import torch

        self.check()
        key, data, receipt = self.paths(stage, contract)
        if not receipt.is_file():
            return None
        saved = json.loads(receipt.read_text(encoding="utf8"))
        if saved.get("status") != "complete":
            return None
        if (
            saved.get("identity") != key
            or saved.get("contract") != contract
            or not data.is_file()
        ):
            raise ValueError("Meridian stage cache contract/payload mismatch")
        if file_sha(data, cancelled=self.cancelled) != saved["sha256"]:
            raise ValueError("Meridian stage cache bytes changed")
        self.check()
        return torch.load(data, map_location="cpu", weights_only=True)

    def run(self, stage, contract, operation, *, portable=True):
        import json
        import torch

        # Store and compare the same JSON representation (tuples otherwise roundtrip as lists).
        contract = json.loads(json.dumps(contract, allow_nan=False))
        key, data, receipt = self.paths(stage, contract)
        # No cache identity is inferred for opaque mutable MODEL/VAE patches.
        if not portable or self.budget == 0:
            # A unique execution receipt is durable but is NEVER a reusable cache.
            # Concurrent opaque objects may share a descriptive contract, not an identity.
            receipt = self.root / f"{stage}-{key}.uncached-{uuid.uuid4().hex}.json"
            started = time.monotonic()
            record = dict(
                stage=stage,
                identity=key,
                contract=contract,
                status="running",
                portable=False,
                pid=os.getpid(),
                reused=False,
                reason="opaque_mutable_inputs" if not portable else "cache_budget_zero",
            )
            atomic_json(receipt, record)
            try:
                self.check()
                value = self.observed_operation(operation, record)
                self.check()
                record.update(
                    status="complete_not_cached", seconds=time.monotonic() - started
                )
                atomic_json(receipt, record, replace_existing=True)
                return value, dict(
                    stage=stage,
                    status=record["status"],
                    identity=key,
                    reused=False,
                    portable=False,
                    receipt_path=str(receipt),
                    reason=record["reason"],
                    resources=record["resources"],
                )
            except BaseException as error:
                record.update(
                    status="interrupted"
                    if type(error).__name__
                    in ("InterruptedError", "InterruptProcessingException")
                    else "failed",
                    error=f"{type(error).__name__}: {error}",
                    seconds=time.monotonic() - started,
                )
                atomic_json(receipt, record, replace_existing=True)
                raise
        self.check()
        with conversion_lock(self.root / f"{stage}-{key}.lock"):
            self.check()
            value = self.load(stage, contract)
            if value is not None:
                return value, dict(
                    stage=stage,
                    status="complete",
                    identity=key,
                    reused=True,
                    resources={
                        "status": "operation_skipped_warm_cache_not_new_peak_observation"
                    },
                )
            if receipt.exists():
                saved = json.loads(receipt.read_text(encoding="utf8"))
                if saved.get("status") == "running":
                    # Lock acquired: no live writer remains. Preserve this interrupted receipt.
                    atomic_json(
                        self.root
                        / f"{stage}-{key}.interrupted-{uuid.uuid4().hex}.json",
                        dict(saved, status="interrupted"),
                    )
            started = time.monotonic()
            record = dict(
                stage=stage,
                identity=key,
                contract=contract,
                status="running",
                pid=os.getpid(),
            )
            atomic_json(receipt, record, replace_existing=True)
            partial = data.with_name(data.name + ".partial-" + uuid.uuid4().hex)
            try:
                value = self.observed_operation(operation, record)
                self.check()
                with partial.open("xb") as stream:
                    torch.save(value, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                sha = file_sha(partial, cancelled=self.cancelled)
                self.check()
                if partial.stat().st_size > self.budget:
                    atomic_json(
                        receipt,
                        dict(
                            record,
                            status="complete_not_cached",
                            seconds=time.monotonic() - started,
                        ),
                        replace_existing=True,
                    )
                    return value, dict(
                        stage=stage,
                        status="complete_not_cached",
                        identity=key,
                        reused=False,
                        resources=record["resources"],
                    )
                if not self._admit(partial, data):
                    atomic_json(
                        receipt,
                        dict(
                            record,
                            status="complete_not_cached",
                            reason="budget_locked_or_unverified",
                            seconds=time.monotonic() - started,
                        ),
                        replace_existing=True,
                    )
                    return value, dict(
                        stage=stage,
                        status="complete_not_cached",
                        identity=key,
                        reused=False,
                        resources=record["resources"],
                    )
                atomic_json(
                    receipt,
                    dict(
                        record,
                        status="complete",
                        sha256=sha,
                        bytes=data.stat().st_size,
                        seconds=time.monotonic() - started,
                    ),
                    replace_existing=True,
                )
                return value, dict(
                    stage=stage,
                    status="complete",
                    identity=key,
                    reused=False,
                    resources=record["resources"],
                )
            except BaseException as error:
                atomic_json(
                    receipt,
                    dict(
                        record,
                        status="interrupted"
                        if type(error).__name__
                        in ("InterruptedError", "InterruptProcessingException")
                        else "failed",
                        error=f"{type(error).__name__}: {error}",
                        seconds=time.monotonic() - started,
                    ),
                    replace_existing=True,
                )
                raise
            finally:
                if partial.exists():
                    partial.unlink()  # only this call's UUID incomplete payload

    @staticmethod
    def observed_operation(operation, record):
        from .meridian_resources import ResourceTrace

        trace = ResourceTrace(record["stage"])
        try:
            with trace:
                return operation()
        finally:
            record["resources"] = trace.report()

    def _evict(self, incoming, exclude):
        import json

        # Evict only complete owned payloads; locks prevent removing active readers/writers.
        candidates = sorted(self.root.glob("*.pt"), key=lambda p: p.stat().st_mtime_ns)
        total = sum(p.stat().st_size for p in candidates if p != exclude)
        for path in candidates:
            if total + incoming <= self.budget:
                break
            if path == exclude:
                continue
            receipt = path.with_suffix(".json")
            if not receipt.is_file():
                continue
            saved = json.loads(receipt.read_text(encoding="utf8"))
            match = re.fullmatch(
                r"(geometry|warp|encode|sample|delivery)-([0-9a-f]{64})\.pt", path.name
            )
            if (
                not match
                or saved.get("identity") != match.group(2)
                or saved.get("stage") != match.group(1)
            ):
                continue  # Never delete unrelated files, even if they have a JSON sidecar.
            if canonical_identity(
                dict(stage=match.group(1), contract=saved.get("contract"))
            ) != match.group(2):
                continue
            if (
                saved.get("status") != "complete"
                or saved.get("bytes") != path.stat().st_size
            ):
                continue
            if path.resolve().parent != self.root:
                raise ValueError("Cache payload escaped owned directory")
            try:
                with conversion_lock(path.with_suffix(".lock")):
                    saved = json.loads(receipt.read_text(encoding="utf8"))
                    if (
                        saved.get("status") != "complete"
                        or saved.get("bytes") != path.stat().st_size
                    ):
                        continue
                    size = path.stat().st_size
                    path.unlink()
                    atomic_json(
                        receipt, dict(saved, status="evicted"), replace_existing=True
                    )
                    total -= size
            except RuntimeError:
                continue
        return total + incoming <= self.budget

    def _admit(self, partial, data):
        try:
            with conversion_lock(self.root / "budget.lock"):
                if not self._evict(partial.stat().st_size, data):
                    return False
                self.check()
                os.replace(
                    partial, data
                )  # UUID payload, locked identity and locked byte admission.
                return True
        except RuntimeError:
            return False  # A competing admission is not a reason to fail generation.
