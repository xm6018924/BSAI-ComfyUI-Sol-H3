"""Resumable CPU-only layer conversion with complete source/content identities."""

import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import uuid

from .meridian_checkpoint_io import (
    assemble_checkpoint,
    atomic_json,
    canonical_identity,
    conversion_lock,
    file_sha,
    part_header,
)
from .meridian_conversion import (
    derived_rope,
    encode_native_tensor,
    expected_lora_shapes,
    materialize_rule,
    validate_adapter,
    validate_index,
)


class SourcePending(FileNotFoundError):
    """A declared source has not finished downloading; verified parts remain."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema",
        "repo_id",
        "revision",
        "files",
    }:
        raise ValueError("Invalid Meridian source manifest")
    if (
        manifest["schema"] != "t8-meridian-source-v1"
        or manifest["repo_id"] != "Viggle/Meridian"
    ):
        raise ValueError("Expected Meridian source identity")
    if not isinstance(manifest["revision"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", manifest["revision"]
    ):
        raise ValueError("Full source revision required")
    if not isinstance(manifest["files"], list) or not manifest["files"]:
        raise ValueError("Complete source inventory required")
    files = {}
    for item in manifest["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("Complete file identities required")
        name = item["path"]
        if (
            not isinstance(name, str)
            or name in ("", ".")
            or "\\" in name
            or ":" in name
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or PurePosixPath(name).as_posix() != name
        ):
            raise ValueError("Source paths must be canonical relative paths")
        if (
            name in files
            or type(item["bytes"]) is not int
            or item["bytes"] <= 0
            or not isinstance(item["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise ValueError("Invalid or duplicate source identity")
        files[name] = item
    return files


def _stamp(path):
    st = path.stat()
    return st.st_size, st.st_mtime_ns, st.st_ino


def run_conversion(
    source, job, destination, manifest, *, row_chunk=256, progress=None, cancelled=None
):
    """CPU stage conversion; incomplete assets raise SourcePending, never finalize.

    Resume validates all saved layer file hashes and exact identity. A failed
    decode or model test does not make this conversion receipt human-qualified.
    The caller must set CUDA_VISIBLE_DEVICES=-1 before importing Torch.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise RuntimeError("Conversion requires a CPU-only process")
    if type(row_chunk) is not int or not 1 <= row_chunk <= 4096:
        raise ValueError("Conversion row chunk must be within1..4096")
    files = validate_manifest(manifest)
    source, job, destination = (
        Path(source).resolve(strict=True),
        Path(job).resolve(),
        Path(destination).resolve(),
    )
    if (
        destination.is_relative_to(source)
        or job.is_relative_to(source)
        or source.is_relative_to(job)
        or destination.is_relative_to(job)
    ):
        raise ValueError("Keep conversion outputs outside original sources")
    verified = {}

    def check_cancel():
        if cancelled is not None and cancelled():
            raise InterruptedError("Meridian conversion cancelled")

    def notify(stage, **info):
        if progress is not None:
            progress(dict(stage=stage, **info))

    def verify(name, *, again=False):
        check_cancel()
        if name not in files:
            raise ValueError(f"Undeclared source asset: {name}")
        path = (source / name).resolve()
        if not path.is_relative_to(source):
            raise ValueError("Source asset escapes root")
        if not path.is_file():
            raise SourcePending(name)
        stamp = _stamp(path)
        if stamp[0] != files[name]["bytes"]:
            raise ValueError(f"Source size mismatch: {name}")
        if not again and name in verified:
            if stamp != verified[name]:
                raise ValueError(f"Source changed after verification: {name}")
            return path
        notify("verifying_source", file=name, postflight=again)
        if (
            file_sha(path, cancelled=cancelled) != files[name]["sha256"]
            or _stamp(path) != stamp
        ):
            raise ValueError(f"Source SHA/stability mismatch: {name}")
        if name.startswith("transformer/") and name.endswith(".safetensors"):
            wanted = {
                k for k, v in index["weight_map"].items() if "transformer/" + v == name
            }
            with safe_open(path, framework="pt", device="cpu") as handle:
                if set(handle.keys()) != wanted:
                    raise ValueError(
                        f"Unconsumed/missing teacher shard tensors: {name}"
                    )
                for key in wanted:
                    spec = handle.get_slice(key)
                    if tuple(spec.get_shape()) != source_shapes[
                        key
                    ] or spec.get_dtype() not in ("F32", "BF16"):
                        raise ValueError(f"Teacher header shape/dtype mismatch: {key}")
            if _stamp(path) != stamp:
                raise ValueError("Teacher changed during header verification")
        verified[name] = stamp
        return path

    config = read_json(verify("transformer/config.json"))
    index = read_json(
        verify("transformer/diffusion_pytorch_model.safetensors.index.json")
    )
    rules = validate_index(config, index["weight_map"])
    source_shapes = {
        name: shape for rule in rules for name, shape in zip(rule.sources, rule.shapes)
    }
    shard_names = {"transformer/" + name for name in index["weight_map"].values()}
    required = shard_names | {
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "lora/pytorch_lora_weights.safetensors",
    }
    if set(files) != required:
        raise ValueError("Source manifest and actual teacher inventory differ")
    adapter_path = verify("lora/pytorch_lora_weights.safetensors")
    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()["lora_adapter_metadata"])
        specs = {
            key: dict(
                shape=handle.get_slice(key).get_shape(),
                dtype=handle.get_slice(key).get_dtype(),
            )
            for key in handle.keys()
        }
    scale = validate_adapter(config, metadata, specs)
    targets = expected_lora_shapes(config)
    implementation = {
        p.name: file_sha(p)
        for p in (
            Path(__file__),
            Path(__file__).with_name("meridian_conversion.py"),
            Path(__file__).with_name("meridian_checkpoint_io.py"),
        )
    }
    identity = dict(
        schema="t8-meridian-conversion-job-v1",
        sources=manifest,
        source_root=str(source),
        output=str(destination),
        implementation=implementation,
        rows=row_chunk,
        torch_threads=torch.get_num_threads(),
        environment={
            "python": platform.python_version(),
            "machine": platform.machine(),
            **{
                name: importlib.metadata.version(name)
                for name in ("torch", "safetensors", "comfy-kitchen")
            },
        },
    )
    fingerprint = canonical_identity(identity)
    job.mkdir(parents=True, exist_ok=True)
    with conversion_lock(job / "conversion.lock"):
        manifest_path = job / "job.json"
        if manifest_path.exists():
            if read_json(manifest_path) != identity:
                raise ValueError("Conversion identity changed; use a new job directory")
        else:
            if any(p.name != "conversion.lock" for p in job.iterdir()):
                raise ValueError("Nonempty unknown conversion directory")
            atomic_json(manifest_path, identity)
        state_path = job / "state.json"
        state = (
            read_json(state_path)
            if state_path.exists()
            else dict(
                schema="t8-meridian-conversion-state-v1",
                identity=fingerprint,
                completed={},
            )
        )
        if (
            state.get("identity") != fingerprint
            or state.get("schema") != "t8-meridian-conversion-state-v1"
            or not isinstance(state.get("completed"), dict)
        ):
            raise ValueError("Invalid conversion state identity")
        expected_names = [r.target for r in rules] + ["rope.inv_freq"]
        if set(state["completed"]) - set(expected_names):
            raise ValueError("Unrecognized completed conversion parts")
        publication_path = job / "publication.json"
        if destination.exists() and not publication_path.exists():
            raise FileExistsError(
                "Output already exists without this job's publication receipt"
            )
        if destination.exists() and set(state["completed"]) != set(expected_names):
            raise ValueError("Published checkpoint has incomplete layer state")

        def read_teacher(name):
            relative = "transformer/" + index["weight_map"][name]
            path = verify(relative)
            with safe_open(path, framework="pt", device="cpu") as handle:
                value = handle.get_tensor(name)
            if _stamp(path) != verified[relative]:
                raise ValueError("Teacher changed while reading")
            return value

        def read_adapter(name):
            path = verify("lora/pytorch_lora_weights.safetensors")
            with safe_open(path, framework="pt", device="cpu") as handle:
                value = handle.get_tensor(name)
            if _stamp(path) != verified["lora/pytorch_lora_weights.safetensors"]:
                raise ValueError("DMD changed while reading")
            return value

        for i, name in enumerate(expected_names):
            check_cancel()
            if name in state["completed"]:
                item = state["completed"][name]
                part = (job / item["file"]).resolve()
                if (
                    not part.is_relative_to(job)
                    or not part.is_file()
                    or file_sha(part, cancelled=cancelled) != item["sha256"]
                ):
                    raise ValueError(f"Saved part changed or missing: {name}")
                header = part_header(part)
                wanted = {name}
                if i < len(rules) and rules[i].quantize:
                    wanted |= {
                        name + "_scale",
                        name.removesuffix(".weight") + ".comfy_quant",
                    }
                if set(header["header"]) != wanted or set(item["keys"]) != wanted:
                    raise ValueError("Saved part tensor inventory changed")
                if header["metadata"] != {"t8_identity": fingerprint, "target": name}:
                    raise ValueError("Saved part belongs to another target/job")
                notify(
                    "reused_verified_layer",
                    index=i,
                    count=len(expected_names),
                    target=name,
                )
                continue
            notify("converting_layer", index=i, count=len(expected_names), target=name)
            if i == len(rules):
                tensors = {name: derived_rope(config)}
                statistics = dict(derived=True, dtype="torch.float32")
            else:
                rule = rules[i]
                value, dtype = materialize_rule(
                    rule, read_teacher, read_adapter, targets, scale
                )
                tensors, statistics = encode_native_tensor(
                    rule, value, dtype, row_chunk=row_chunk, cancelled=cancelled
                )
                del value
            filename = f"layer-{i:04d}-{uuid.uuid4().hex}.safetensors"
            part = job / filename
            temporary = part.with_name(part.name + ".partial")
            try:
                save_file(
                    tensors,
                    temporary,
                    metadata={"t8_identity": fingerprint, "target": name},
                )
                with temporary.open("r+b") as stream:
                    os.fsync(stream.fileno())
                check_cancel()
                os.link(temporary, part)
            finally:
                if temporary.exists():
                    temporary.unlink()
            del tensors
            state["completed"][name] = dict(
                file=filename,
                sha256=file_sha(part),
                keys=list(part_header(part)["header"]),
                statistics=statistics,
            )
            atomic_json(state_path, state, replace_existing=state_path.exists())

        # Reused layers still require complete original file verification in
        # this invocation. No final checkpoint from a partial downloaded set.
        for name in sorted(files):
            verify(name, again=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        parts = [
            dict(
                path=str(job / state["completed"][name]["file"]),
                sha256=state["completed"][name]["sha256"],
            )
            for name in expected_names
        ]
        output_metadata = dict(
            format="pt",
            t8_model_kind="meridian_dmd_int8_convrot",
            dmd_merged="true",
            dmd_strength="1.0",
            dmd_alpha_over_rank=str(scale),
            source_repo=manifest["repo_id"],
            source_revision=manifest["revision"],
            t8_conversion_identity=fingerprint,
            t8_recipe="dmd_grid4_forward3_shift3_3",
        )

        def record_publication(output):
            value = dict(identity=fingerprint, output=output)
            if publication_path.exists():
                if read_json(publication_path) != value:
                    raise ValueError(
                        "Reassembled checkpoint differs from publication journal"
                    )
            else:
                atomic_json(publication_path, value)

        if destination.exists():
            journal = read_json(publication_path)
            info = part_header(destination)
            output = journal["output"]
            if (
                journal.get("identity") != fingerprint
                or info["metadata"] != output_metadata
                or output.get("path") != str(destination)
                or output.get("bytes") != destination.stat().st_size
                or output.get("tensor_count") != len(info["header"])
                or output.get("sha256") != file_sha(destination, cancelled=cancelled)
            ):
                raise ValueError(
                    "Published checkpoint differs from verified publication journal"
                )
            notify("recovered_verified_publication")
        else:
            notify("assembling_checkpoint", parts=len(parts))
            output = assemble_checkpoint(
                parts,
                destination,
                output_metadata,
                cancelled=cancelled,
                before_publish=record_publication,
            )
        result = dict(
            status="CPU_converted_native_checkpoint_pending_load_and_GPU",
            identity=fingerprint,
            output=output,
            completed_layers=len(expected_names),
            quantized_layers=sum(r.quantize for r in rules),
            source_sha256={name: item["sha256"] for name, item in files.items()},
            cuda_initialized=torch.cuda.is_initialized(),
            full_model_GPU_ran=False,
            human_qualified=False,
        )
        result_path = job / "result.json"
        if result_path.exists():
            if read_json(result_path) != result:
                raise ValueError("Final result differs from verified conversion")
        else:
            atomic_json(result_path, result)
        return result
