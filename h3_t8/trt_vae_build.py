"""Engine build contracts and atomic bundle publication; no GPU/TRT import."""

import hashlib
import json
import os
from pathlib import Path
import re


MODEL_SHA = "d2420053272010f701c17cd295d6f89642f9728461c2b74124268de90670f1c1"
WEIGHTS_SHA = "8f3e8869794e896628a0b8e6f48c03522172d8715c4d06fe497769a78daadee4"
TRT_VERSION = "10.13.3.9.post1"
INPUT_SHAPE = (1, 24, 7, 16, 16)
OUTPUT_SHAPE = (1, 3, 28, 256, 256)
ENCODER_MODEL_SHA = "f1b8137f1f60e5a9829af0f50ad93932dbb12ebbdce853011093538886c8a069"
ENCODER_INPUT_SHAPE = (1, 3, 17, 256, 256)
ENCODER_OUTPUT_SHAPE = (1, 48, 5, 16, 16)
ENCODER_T1_MODEL_SHA = "2b224996d1e4ae65609586885f5453be78984d033f57ead2710cf4d61fc5eb10"
ENCODER_T1_INPUT_SHAPE = (1, 3, 1, 256, 256)
ENCODER_T1_OUTPUT_SHAPE = (1, 48, 1, 16, 16)
DECODER_FLEX_MODEL_SHA = "79010c17cf01592ec774ac6962c57178aedf6ae851e17b0de806649260bb3e11"
DECODER_FLEX_MIN_SHAPE = (1, 24, 1, 1, 1)
DECODER_W4_MODEL_SHA = "0bfcfdc31ea767cb8b0fb371ea1629f0803ee2e81058a2ca7f60ab52a8bc7a2e"


def is_w4(request):
    return request.get("schema") == "t8-trt-decoder-w4a16-build-v1"


def network_flags(request, trt):
    graph_spec(request)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if is_w4(request):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    return flags


def configure_precision(request, config, trt):
    # Explicit INT4 DQ weights define precision in the graph. FP16/INT8
    # builder selection flags are neither needed nor allowed in strong typing.
    if is_w4(request):
        config.clear_flag(trt.BuilderFlag.FP16)
        config.clear_flag(trt.BuilderFlag.INT8)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    else:
        config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    expected = not is_w4(request)
    if bool(config.get_flag(trt.BuilderFlag.FP16)) != expected or config.get_flag(trt.BuilderFlag.TF32):
        raise RuntimeError("Builder precision flag readback mismatch")
    if is_w4(request) and config.get_flag(trt.BuilderFlag.INT8):
        raise RuntimeError("Explicit W4 graph must not enable implicit INT8")
    return {"strongly_typed": is_w4(request), "fp16_enabled": expected, "tf32_enabled": False}


def flex_output_shape(shape):
    shape = tuple(shape)
    if (len(shape) != 5 or any(type(n) is not int for n in shape)
            or shape[:2] != (1, 24) or shape[2] not in (1, 7)
            or not 1 <= shape[3] <= 16 or not 1 <= shape[4] <= 16):
        raise ValueError("Flex decoder supports native T1/T7 tiles and spatial1..16 only")
    return 1, 3, shape[2]*4, shape[3]*16, shape[4]*16


def graph_spec(request):
    """Pinned graph family, not a user-controlled arbitrary engine shape."""
    if request.get("schema") == "t8-trt-build-v1" and request.get("kind", "decoder") == "decoder":
        return "latent_tile", "pixel_tile", INPUT_SHAPE, OUTPUT_SHAPE
    if request.get("schema") == "t8-trt-decoder-flex-build-v1" and request.get("kind") == "decoder":
        return "latent_tile", "pixel_tile", INPUT_SHAPE, OUTPUT_SHAPE
    if is_w4(request) and request.get("kind") == "decoder":
        return "latent_tile", "pixel_tile", INPUT_SHAPE, OUTPUT_SHAPE
    if request.get("schema") == "t8-trt-encoder-build-v1" and request.get("kind") == "encoder":
        return "pixel_tile", "moments_tile", ENCODER_INPUT_SHAPE, ENCODER_OUTPUT_SHAPE
    if request.get("schema") == "t8-trt-encoder-t1-build-v1" and request.get("kind") == "encoder":
        return "pixel_tile", "moments_tile", ENCODER_T1_INPUT_SHAPE, ENCODER_T1_OUTPUT_SHAPE
    raise ValueError("Unsupported engine graph family")


def parsed_input_shape(request):
    shape = graph_spec(request)[2]
    if request["schema"] == "t8-trt-decoder-flex-build-v1":
        return (1, 24, -1, -1, -1)
    return shape if request["schema"] == "t8-trt-encoder-t1-build-v1" else (-1,) + shape[1:]


def set_flex_profile(profile, config):
    shapes = (DECODER_FLEX_MIN_SHAPE, INPUT_SHAPE, INPUT_SHAPE)
    profile.set_shape("latent_tile", *shapes)
    if tuple(map(tuple, profile.get_shape("latent_tile"))) != shapes or not bool(profile):
        raise RuntimeError("Flex optimization profile readback mismatch")
    index = config.add_optimization_profile(profile)
    if type(index) is not int or index != 0:
        raise RuntimeError("Failed to add single flex profile")


def set_fixed_profile(profile, config, name="latent_tile", shape=INPUT_SHAPE):
    # Python set_shape returns None (raises ValueError on failure), unlike
    # IExecutionContext.set_input_shape. Verify stored dimensions explicitly.
    profile.set_shape(name, shape, shape, shape)
    shapes = tuple(tuple(shape) for shape in profile.get_shape(name))
    if shapes != (shape,) * 3 or not bool(profile):
        raise RuntimeError("Optimization profile readback mismatch")
    index = config.add_optimization_profile(profile)
    if type(index) is not int or index != 0:
        raise RuntimeError("Failed to add the single optimization profile")


def encoder_norm_precision(network, config, trt, policy):
    """Keep GroupNorm accumulation AND affine arithmetic in FP32, round once.

    The pinned ONNX expands each GroupNorm into InstanceNorm, Mul and Add.
    Native PyTorch applies its affine in the FP32 normalization calculation;
    separately rounding the intermediate unit-normalized/weighted outputs is
    a different calculation even if InstanceNorm itself accumulates in FP32.
    """
    if policy == "default":
        return []
    if policy != "fp32_norm_affine":
        raise ValueError("Unknown encoder normalization precision policy")
    rows, counts = [], {"InstanceNormalization": 0, "Mul_1": 0, "Add": 0}
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        match = re.search(r"/norm(?:[12](?:_\d+)?|_out)/(InstanceNormalization|Mul_1|Add)$", layer.name)
        if not match:
            continue
        operation = match[1]
        if layer.num_outputs != 1 or layer.get_output(0).is_shape_tensor:
            raise ValueError("Unexpected encoder normalization output")
        layer.precision = trt.float32
        output_type = trt.float16 if operation == "Add" else trt.float32
        layer.set_output_type(0, output_type)
        if layer.precision != trt.float32 or layer.get_output_type(0) != output_type:
            raise RuntimeError("Encoder precision constraint readback failed")
        counts[operation] += 1
        rows.append({"name": layer.name, "compute": "float32", "output": "float16" if operation == "Add" else "float32"})
    if counts != {"InstanceNormalization": 25, "Mul_1": 25, "Add": 25}:
        raise ValueError(f"Pinned encoder normalization layer coverage differs: {counts}")
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    return rows


def digest_file(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError(f"File changed during hashing: {path.name}")
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_new_json(path, value):
    """New research evidence only; no overwrite of earlier receipts."""
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def loaded_windows_libraries():
    """Actual process mappings, not DLL search-path predictions."""
    import ctypes

    if os.name != "nt":
        raise RuntimeError("Initial TRT isolation route requires Windows")
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    api.GetModuleHandleW.restype = ctypes.c_void_p
    api.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    api.GetModuleFileNameW.restype = ctypes.c_uint32
    paths = {}
    for name in ("nvinfer_10.dll", "nvonnxparser_10.dll", "cudart64_13.dll"):
        handle = api.GetModuleHandleW(name)
        if not handle:
            raise RuntimeError(f"Required DLL not mapped: {name}")
        buffer = ctypes.create_unicode_buffer(32768)
        count = api.GetModuleFileNameW(handle, buffer, len(buffer))
        if not count or count >= len(buffer):
            raise RuntimeError(f"Cannot resolve loaded DLL: {name}")
        paths[name] = str(Path(buffer.value).resolve())
    return paths


def validate_request(request):
    _, _, input_shape, _ = graph_spec(request)
    encoder = request.get("kind") == "encoder"
    norm_policy = request.get("encoder_norm_precision", "default")
    single_image = request.get("schema") == "t8-trt-encoder-t1-build-v1"
    flex = request.get("schema") == "t8-trt-decoder-flex-build-v1"
    embedded = encoder or is_w4(request)
    if single_image and norm_policy != "default":
        raise ValueError("T1 encoder requires its independently audited default graph policy")
    if norm_policy not in ("default", "fp32_norm_affine") or (not encoder and norm_policy != "default"):
        raise ValueError("Encoder precision policy is not valid for this graph")
    if request.get("precision") != ("w4a16" if is_w4(request) else "fp16"):
        raise ValueError("Unsupported engine build contract")
    if request.get("runtime_version") != TRT_VERSION:
        raise ValueError("TRT version is not qualified for this build route")
    if (tuple(request.get("input_shape", ())) != input_shape
            or any(type(dim) is not int for dim in request["input_shape"])):
        raise ValueError("This pinned graph requires its exact static profile")
    model_sha = ENCODER_T1_MODEL_SHA if single_image else (ENCODER_MODEL_SHA if encoder else MODEL_SHA)
    if flex:
        model_sha = DECODER_FLEX_MODEL_SHA
    if is_w4(request):
        model_sha = DECODER_W4_MODEL_SHA
    if request.get("model_sha256") != model_sha or request.get("weights_sha256") != (None if embedded else WEIGHTS_SHA):
        raise ValueError("Unexpected model asset identity")
    if type(request.get("workspace_bytes")) is not int or not 256 * 1024**2 <= request["workspace_bytes"] <= 4 * 1024**3:
        raise ValueError("Workspace limit must be 256MiB..4GiB; not a total VRAM limit")
    if type(request.get("device_index")) is not int or request["device_index"] != 0:
        raise ValueError("Initial build route requires the single verified GPU")
    if not isinstance(request.get("gpu_uuid"), str) or not request["gpu_uuid"].startswith("GPU-"):
        raise ValueError("A physical GPU UUID is required")
    if not isinstance(request.get("driver_version"), str) or not request["driver_version"]:
        raise ValueError("Driver identity required")
    for key in ("model_path", "runtime_site", "staging_path") + (() if embedded else ("weights_path",)):
        if not isinstance(request.get(key), str) or not Path(request[key]).is_absolute():
            raise ValueError(f"Absolute path required: {key}")
    if embedded and request.get("weights_path") is not None:
        raise ValueError("Pinned graph has embedded weights, not external data")
    expected_data = Path(request["model_path"]).resolve().with_suffix(".onnx.data")
    if flex:
        expected_data = expected_data.with_name("minimax_h3_vae_decoder.onnx.data")
    if not embedded and Path(request["weights_path"]).resolve() != expected_data:
        raise ValueError("Weights must be the expected neighboring external-data file")
    for key in ("runtime_sources", "worker_sources"):
        if not isinstance(request.get(key), dict) or not request[key]:
            raise ValueError(f"Missing source identities: {key}")
        for path, digest in request[key].items():
            if not Path(path).is_absolute() or not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("Invalid source identity entry")
    return canonical_hash(request)


def verify_sources(request):
    validate_request(request)
    pairs = {request["model_path"]: request["model_sha256"],
             **request["runtime_sources"], **request["worker_sources"]}
    if request.get("weights_path"):
        pairs[request["weights_path"]] = request["weights_sha256"]
    for name, expected in pairs.items():
        if digest_file(name) != expected:
            raise ValueError(f"Build source changed: {Path(name).name}")


def validate_quantized_manifest(manifest, request):
    if is_w4(request):
        for key, value in {"strongly_typed": True, "fp16_enabled": False,
                           "tf32_enabled": False, "precision": "w4a16"}.items():
            if type(manifest.get(key)) is not type(value) or manifest[key] != value:
                raise ValueError(f"Quantized engine compiler evidence mismatch: {key}")


def publish_bundle(staging, destination, request):
    """Publish only a complete, identity-bound bundle with one directory rename.

    Failed staging directories remain private evidence; no automatic deletion.
    This marks compilation complete, NEVER inference or visual qualification.
    """
    staging, destination = Path(staging).resolve(), Path(destination).resolve()
    if staging != Path(request["staging_path"]).resolve():
        raise ValueError("Staging identity mismatch")
    if not staging.name.startswith(".building-") or staging.parent != destination.parent or destination.exists():
        raise ValueError("Use a new destination next to the private staging directory")
    if {entry.name for entry in staging.iterdir()} != {"model.engine", "manifest.json"}:
        raise ValueError("Incomplete or unexpected bundle members")
    if any(entry.is_symlink() or not entry.is_file() or entry.resolve().parent != staging for entry in staging.iterdir()):
        raise ValueError("Bundle files must remain within the private staging directory")
    manifest = json.loads((staging / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("request_sha256") != validate_request(request) or manifest.get("status") != "compiled_not_execution_qualified":
        raise ValueError("Compiler manifest is not bound to this request")
    validate_quantized_manifest(manifest, request)
    if manifest.get("engine_sha256") != digest_file(staging / "model.engine"):
        raise ValueError("Engine digest does not match manifest")
    input_name, output_name, input_shape, output_shape = graph_spec(request)
    if request.get("kind") == "encoder" and (manifest.get("input_name") != input_name or manifest.get("output_name") != output_name):
        raise ValueError("Compiled encoder tensor names mismatch")
    if (manifest.get("input_shape") != list(input_shape) or manifest.get("output_shape") != list(output_shape)
            or manifest.get("gpu_uuid") != request["gpu_uuid"] or manifest.get("runtime_version") != TRT_VERSION):
        raise ValueError("Compiled engine identity or I/O mismatch")
    if (staging / "model.engine").stat().st_size <= 0:
        raise ValueError("Empty engine")
    # On Windows rename never replaces an existing destination directory.
    staging.rename(destination)
    return manifest
