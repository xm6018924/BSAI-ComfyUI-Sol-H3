"""Native ordered Tao requests in one owned model/runtime; normalized AV output.

New route only. Existing one-request worker and its manifests remain unchanged.
The controller owns the process tree and cancellation. No tokenizer or downloads.
"""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import time
import sys

from backend_files import sha


def generate(request, root, report, progress):
    source = Path(request["source"])
    if (
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        != request["source_revision"]
        or subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
    ):
        raise ValueError("Pinned Tao source revision changed")
    sys.path.insert(0, str(source / "src"))
    import psutil
    import torch
    from safetensors.torch import save_file
    from taomate_h3.model.dit import MiniMaxH3DiT
    from taomate_h3.model.weight_loading import load_minimax_h3_dit_weights
    from taomate_h3.model.layers import set_inference_packed_dense_flash_attn3
    from taomate_h3.inference.lora_checkpoint import (
        apply_h3_lora_checkpoint,
        materialize_h3_lora_bf16_buffers_,
    )
    from taomate_stream_inputs import load_stream_inputs
    from taomate_local_transport import local_transport
    from taomate_local_runtime import make_local_runtime
    from taomate_prepared_pipeline import prepared_pipeline
    from taomate_stream_execution import execute_stream

    torch.set_num_threads(4)
    teacher, prepared = load_stream_inputs(request, torch)
    if (
        str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-").lower()
        != request["gpu_uuid"].removeprefix("GPU-").lower()
    ):
        raise ValueError("Worker GPU differs from its controller binding")
    verified = json.loads(Path(request["download_receipt"]).read_text(encoding="utf8"))
    if verified["status"] != "download_and_transformer_checksums_pass":
        raise ValueError("Complete base download verification is required")
    if psutil.virtual_memory().available <= verified["total_bytes"] + 24 * 1024**3:
        raise RuntimeError(
            "Insufficient host RAM for the owned native model and streaming state"
        )
    # Complete base contents are already part of controller and worker identities.
    for item in verified["files"]:
        path = Path(request["base"]) / item["file"]
        if (
            path.stat().st_size != item["size"]
            or request["identities"].get(str(path)) != item["actual_sha256"]
        ):
            raise ValueError("Native base differs from its original download receipt")
    set_inference_packed_dense_flash_attn3(False)

    def interrupt():
        if (root / "cancel.request").exists():
            raise InterruptedError("Owned Tao stream cancelled")

    with local_transport() as context, torch.inference_mode():
        progress("loading_official_CPU_base")
        model = MiniMaxH3DiT.allocate("meta", parallel_context=context).eval()
        try:
            dtypes = {name: p.dtype for name, p in model.named_parameters()}
            model.to_empty(device="cpu")
            report["base_load"] = load_minimax_h3_dit_weights(
                model, Path(request["base"]) / "FL2VA/transformer"
            )
            if not all(
                p.device.type == "cpu" and p.dtype == dtypes[name]
                for name, p in model.named_parameters()
            ):
                raise ValueError("Native base parameter device/dtype changed")
            report["adapter"] = apply_h3_lora_checkpoint(
                model, Path(request["adapter"])
            )
            report["adapter_buffers"] = materialize_h3_lora_bf16_buffers_(model)
            pipeline = prepared_pipeline(model, context)
            runtime = make_local_runtime(
                teacher, minimum_free_bytes=2 * 1024**3, interrupt=interrupt
            )

            def save_request(index, video, audio):
                destination = root / f"request_{index:04d}.safetensors"
                save_file({"video": video, "audio": audio}, str(destination))
                return sha(destination)

            (video, audio), receipt = execute_stream(
                model,
                pipeline,
                runtime,
                prepared,
                interrupt=interrupt,
                progress=progress,
                save_request=save_request,
                torch=torch,
            )
            report.update(receipt)
            if runtime._cache is not None or runtime.request_owner.state != "closed":
                raise RuntimeError("Stream runtime did not release its retained state")
            report["owned_cache_released"] = True
            destination = root / "tao-stream-normalized-latents.safetensors"
            save_file(
                {"video": video, "audio": audio},
                str(destination),
                metadata={
                    "schema": "t8-taomate-stream-latents-v1",
                    "normalization": "upstream normalized latents",
                    "timing": json.dumps(receipt["timing"], sort_keys=True),
                    "width": "864",
                    "height": "480",
                },
            )
            report.update(
                output_sha256=sha(destination),
                prompts=[item["prompt"] for item in prepared],
                status="native_Tao_stream_latents_pass",
            )
        finally:
            # This worker owns the whole model. Dispose storage even if an error
            # traceback retains the Python object until report serialization.
            original = sys.exc_info()[1]
            try:
                model.to_empty(device="meta")
            except BaseException as error:
                if original is None:
                    raise
                report["model_cleanup_error"] = f"{type(error).__name__}: {error}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    path = parser.parse_args().request
    root = path.parent
    request = json.loads(path.read_text(encoding="utf8"))
    report = dict(status="incomplete", human_qualified=False)

    def progress(stage, **extra):
        value = dict(stage=stage, time=time.time(), **extra)
        (root / "worker-live.json").write_text(json.dumps(value), encoding="utf8")
        print(json.dumps(value), flush=True)

    try:
        if request.get("schema") != "t8-taomate-prepared-stream-v1":
            raise ValueError("Expected the controller-bound native stream request")
        for filename, expected in request["identities"].items():
            if sha(filename) != expected:
                raise ValueError(f"Prepared stream input changed: {filename}")
        generate(request, root, report, progress)
        gc.collect()
        progress("owned_model_released_postflight")
        for filename, expected in request["identities"].items():
            if sha(filename) != expected:
                raise ValueError(
                    f"Prepared stream input changed during execution: {filename}"
                )
        report["postflight_pass"] = True
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
