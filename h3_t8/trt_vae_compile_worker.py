"""Trusted worker: controller assigns a kill-on-close Windows Job before entry."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from trt_vae_build import (  # noqa: E402
    TRT_VERSION, digest_file, graph_spec, parsed_input_shape, loaded_windows_libraries, validate_request,
    set_fixed_profile, set_flex_profile, flex_output_shape, DECODER_FLEX_MIN_SHAPE,
    encoder_norm_precision, verify_sources, write_new_json,
    network_flags, configure_precision, is_w4,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf8"))
    request_hash = validate_request(request)
    input_name, output_name, input_shape, output_shape = graph_spec(request)
    flex = request["schema"] == "t8-trt-decoder-flex-build-v1"
    write_new_json(args.request.parent / "worker-started.json", {"pid": os.getpid(), "request_sha256": request_hash})
    verify_sources(request)
    staging = Path(request["staging_path"])
    if not staging.is_dir() or any(staging.iterdir()):
        raise ValueError("Controller must supply an empty private staging directory")
    runtime_site = Path(request["runtime_site"]).resolve()
    sys.path.insert(0, str(runtime_site))
    import torch
    import tensorrt as trt
    import tensorrt_bindings
    import tensorrt_libs

    for module in (trt, tensorrt_bindings, tensorrt_libs):
        if not Path(module.__file__).resolve().is_relative_to(runtime_site):
            raise RuntimeError("TRT module came from a different environment")
    if trt.__version__ != TRT_VERSION or torch.cuda.device_count() != 1:
        raise RuntimeError("TRT version or device count changed")
    mapped = loaded_windows_libraries()
    for name in ("nvinfer_10.dll", "nvonnxparser_10.dll"):
        if not Path(mapped[name]).is_relative_to(runtime_site):
            raise RuntimeError("TRT DLL loaded outside the isolated runtime")
    import pynvml
    pynvml.nvmlInit()
    try:
        if str(pynvml.nvmlSystemGetDriverVersion()) != request["driver_version"]:
            raise RuntimeError("GPU driver identity changed")
    finally:
        pynvml.nvmlShutdown()
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    if str(properties.uuid).removeprefix("GPU-").lower() != request["gpu_uuid"].removeprefix("GPU-").lower():
        raise RuntimeError(f"Torch device UUID mismatch: {properties.uuid}")
    print("T8_TRT_PHASE parsing", flush=True)
    start = time.perf_counter()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags(request, trt))
    config = builder.create_builder_config()
    precision = configure_precision(request, config, trt)
    if bool(network.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) != is_w4(request):
        raise RuntimeError("Actual network typing differs from pinned graph contract")
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, request["workspace_bytes"])
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(request["model_path"]):
        raise RuntimeError("ONNX parser failed: " + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError("Unexpected network I/O count")
    inp, out = network.get_input(0), network.get_output(0)
    if inp.name != input_name or tuple(inp.shape) != parsed_input_shape(request) or inp.dtype != trt.float16:
        raise RuntimeError("Unexpected parsed input declaration")
    if out.name != output_name or out.dtype != trt.float16:
        raise RuntimeError("Unexpected parsed output declaration")
    profile = builder.create_optimization_profile()
    if flex:
        set_flex_profile(profile, config)
    else:
        set_fixed_profile(profile, config, inp.name, input_shape)
    norm_constraints = encoder_norm_precision(network, config, trt, request.get("encoder_norm_precision", "default"))
    write_new_json(args.request.parent / "precision-constraints.json", {"policy": request.get("encoder_norm_precision", "default"), "layers": norm_constraints})
    print("T8_TRT_PHASE building", flush=True)
    build_start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT builder returned no engine")
    build_seconds = time.perf_counter() - build_start
    partial = staging / "model.engine.partial"
    with partial.open("xb") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    del serialized, parser, network, config, profile, builder, inp, out
    gc.collect()
    print("T8_TRT_PHASE inspecting", flush=True)
    rt = trt.Runtime(logger)
    with partial.open("rb") as stream:
        engine = rt.deserialize_cuda_engine(stream.read())
    if engine is None or engine.num_io_tensors != 2:
        raise RuntimeError("Built engine cannot be deserialized")
    if (engine.get_tensor_mode(input_name) != trt.TensorIOMode.INPUT
            or engine.get_tensor_mode(output_name) != trt.TensorIOMode.OUTPUT
            or engine.get_tensor_dtype(input_name) != trt.float16
            or engine.get_tensor_dtype(output_name) != trt.float16):
        raise RuntimeError("Unexpected engine I/O")
    context = engine.create_execution_context()
    if context is None or not context.set_input_shape(input_name, input_shape):
        raise RuntimeError("Engine context cannot accept required shape")
    actual_shape = tuple(context.get_tensor_shape(output_name))
    if actual_shape != output_shape:
        raise RuntimeError(f"Unexpected engine output: {actual_shape}")
    resolved = []
    if flex:
        expected_profile = (DECODER_FLEX_MIN_SHAPE, input_shape, input_shape)
        if tuple(map(tuple, engine.get_tensor_profile_shape(input_name, 0))) != expected_profile:
            raise RuntimeError("Built flex profile bounds changed")
        for shape in (DECODER_FLEX_MIN_SHAPE, (1,24,1,16,16), (1,24,7,8,13), input_shape):
            if not context.set_input_shape(input_name, shape) or tuple(context.get_tensor_shape(output_name)) != flex_output_shape(shape):
                raise RuntimeError("Actual flex context output geometry differs")
            resolved.append({"input": list(shape), "output": list(context.get_tensor_shape(output_name))})
    if is_w4(request):
        inspector = engine.create_engine_inspector()
        inspector.execution_context = context
        information = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        write_new_json(args.request.parent / "engine-layers.json", information)
        del inspector
    del context, engine, rt
    gc.collect()
    torch.cuda.synchronize()
    manifest = {"status": "compiled_not_execution_qualified", "request_sha256": request_hash,
                "runtime_version": trt.__version__, "gpu_uuid": request["gpu_uuid"],
                "driver_version": request["driver_version"], "loaded_libraries": mapped,
                "gpu_name": properties.name, "compute_capability": [properties.major, properties.minor],
                "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
                "input_shape": list(input_shape), "output_shape": list(output_shape),
                "input_name": input_name, "output_name": output_name, "io_dtype": "float16",
                **precision, "precision": request["precision"], "workspace_bytes": request["workspace_bytes"],
                "encoder_norm_precision": request.get("encoder_norm_precision", "default"),
                "precision_constraints": norm_constraints,
                "flex_resolved_shapes": resolved,
                "engine_sha256": digest_file(partial), "engine_bytes": partial.stat().st_size,
                "build_seconds": build_seconds, "worker_seconds": time.perf_counter() - start,
                "source_request": request}
    partial.rename(staging / "model.engine")
    write_new_json(staging / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "source_request"}), flush=True)


if __name__ == "__main__":
    main()
