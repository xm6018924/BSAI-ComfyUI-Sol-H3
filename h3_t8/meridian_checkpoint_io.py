"""Bounded-memory, non-overwriting safetensors assembly for CPU conversion.

Only complete, hashed layer parts are accepted. No model/framework imports.
"""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import uuid

DTYPE_BYTES = {"F32": 4, "BF16": 2, "I8": 1, "U8": 1}
MAX_HEADER = 16 * 1024**2


def file_sha(path, *, cancelled=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            if cancelled is not None and cancelled():
                raise InterruptedError("Meridian file verification cancelled")
            digest.update(block)
    return digest.hexdigest()


def canonical_identity(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path, value, *, replace_existing=False):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        if replace_existing:
            os.replace(temporary, path)
        else:
            # Atomic NOREPLACE on Windows and POSIX; no exists→rename race.
            os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()  # only the UUID file created by this call


@contextmanager
def conversion_lock(path):
    """OS-owned lock: process death releases it; the durable lock file is harmless."""
    path = Path(path)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(
                    "Conversion directory is owned by another process"
                ) from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError(
                    "Conversion directory is owned by another process"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate safetensors JSON key")
        result[key] = value
    return result


def part_header(path):
    """Validate actual contiguous payload spans, types, dimensions and file size."""
    path = Path(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors prefix")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= MAX_HEADER:
            raise ValueError("Invalid safetensors header size")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("Truncated safetensors header")
        header = json.loads(raw, object_pairs_hook=_unique_pairs)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(v, str) for v in metadata.values()
    ):
        raise ValueError("Invalid safetensors metadata")
    entries = []
    for name, spec in header.items():
        if (
            not name
            or not isinstance(spec, dict)
            or set(spec) != {"dtype", "shape", "data_offsets"}
        ):
            raise ValueError("Invalid tensor entry")
        shape, span, dtype = spec["shape"], spec["data_offsets"], spec["dtype"]
        if not isinstance(shape, list) or any(
            type(n) is not int or n < 0 for n in shape
        ):
            raise ValueError("Invalid tensor dimensions")
        if (
            dtype not in DTYPE_BYTES
            or not isinstance(span, list)
            or len(span) != 2
            or any(type(n) is not int or n < 0 for n in span)
        ):
            raise ValueError("Invalid tensor dtype/offsets")
        if span[1] - span[0] != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError("Tensor size and offsets differ")
        entries.append((name, spec))
    cursor = 0
    for _, spec in sorted(
        entries,
        key=lambda item: (item[1]["data_offsets"][0], item[1]["data_offsets"][1]),
    ):
        if spec["data_offsets"][0] != cursor:
            raise ValueError("Tensor payload has holes or overlapping ranges")
        cursor = spec["data_offsets"][1]
    if not entries or path.stat().st_size != 8 + length + cursor:
        raise ValueError("Part is empty, truncated or has unclaimed trailing data")
    return dict(
        header=header,
        metadata=metadata,
        prefix=prefix + raw,
        data_bytes=cursor,
        data_start=8 + length,
    )


def assemble_checkpoint(
    parts, destination, metadata, *, cancelled=None, progress=None, before_publish=None
):
    """Stream complete layer files into one native checkpoint, never load all weights.

    Each part is {path, sha256}. Its entire copied bytes are hashed, not just the
    header or file stats. A concurrent change cannot promote an unverified file.
    Destination never overwrites another file. Cancellation removes only this
    call's private incomplete output, preserving all verified resumable parts.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if not isinstance(metadata, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()
    ):
        raise ValueError("Checkpoint metadata must be string pairs")
    combined = {"__metadata__": metadata}
    plans, offset = [], 0
    for part in parts:
        if (
            set(part) != {"path", "sha256"}
            or not isinstance(part["sha256"], str)
            or len(part["sha256"]) != 64
        ):
            raise ValueError("Each part needs an exact path and full SHA256")
        path = Path(part["path"]).resolve(strict=True)
        info = part_header(path)
        for name, spec in info["header"].items():
            if name in combined:
                raise ValueError(f"Duplicate output tensor: {name}")
            combined[name] = dict(
                spec, data_offsets=[offset + n for n in spec["data_offsets"]]
            )
        plans.append((path, part["sha256"], info))
        offset += info["data_bytes"]
    if not plans:
        raise ValueError("No verified parts to assemble")
    encoded = json.dumps(combined, separators=(",", ":"), allow_nan=False).encode(
        "utf8"
    )
    encoded += b" " * ((-len(encoded)) % 8)
    if len(encoded) > MAX_HEADER:
        raise ValueError("Assembled checkpoint header too large")
    temporary = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    digest = hashlib.sha256()

    def write(stream, block):
        stream.write(block)
        digest.update(block)

    try:
        with temporary.open("xb") as output:
            write(output, struct.pack("<Q", len(encoded)) + encoded)
            for i, (path, expected, info) in enumerate(plans):
                if cancelled is not None and cancelled():
                    raise InterruptedError("Meridian checkpoint assembly cancelled")
                part_digest = hashlib.sha256()
                with path.open("rb") as source:
                    prefix = source.read(info["data_start"])
                    if prefix != info["prefix"]:
                        raise ValueError(
                            f"Part header changed during assembly: {path.name}"
                        )
                    part_digest.update(prefix)
                    remaining = info["data_bytes"]
                    while remaining:
                        if cancelled is not None and cancelled():
                            raise InterruptedError(
                                "Meridian checkpoint assembly cancelled"
                            )
                        block = source.read(min(remaining, 8 * 1024**2))
                        if not block:
                            raise ValueError(
                                f"Part truncated during assembly: {path.name}"
                            )
                        part_digest.update(block)
                        write(output, block)
                        remaining -= len(block)
                    if source.read(1):
                        raise ValueError(f"Part grew during assembly: {path.name}")
                if part_digest.hexdigest() != expected:
                    raise ValueError(f"Part SHA mismatch during assembly: {path.name}")
                if progress is not None:
                    progress(i + 1, len(plans), path.name)
            output.flush()
            os.fsync(output.fileno())
        # Parse our completed structure before publishing. Data were hashed as
        # written, and every contributing part was verified during the copy.
        part_header(temporary)
        if cancelled is not None and cancelled():
            raise InterruptedError("Meridian checkpoint assembly cancelled")
        receipt = dict(
            path=str(destination.resolve()),
            sha256=digest.hexdigest(),
            bytes=temporary.stat().st_size,
            tensor_count=len(combined) - 1,
        )
        if before_publish is not None:
            before_publish(receipt)
        if cancelled is not None and cancelled():
            raise InterruptedError("Meridian checkpoint assembly cancelled")
        os.link(temporary, destination)
        return receipt
    finally:
        if temporary.exists():
            temporary.unlink()
