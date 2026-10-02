"""Owned, GPU-guarded native T1 export with statically materialized causal taps."""
import argparse
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, write_new_json  # noqa: E402


def native_encoder_shell():
    import torch
    from comfy.ldm.minimax.vae import IMAGENET_MEAN, IMAGENET_STD, LATENTS_MEAN, LATENTS_STD, MiniMaxH3VideoVAE

    with torch.device("meta"):
        core = MiniMaxH3VideoVAE()
    del core.decoder, core.post_quant_conv
    core.latents_mean, core.latents_std = torch.tensor(LATENTS_MEAN), torch.tensor(LATENTS_STD)
    core.pixel_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
    core.pixel_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)
    return core


def static_t1_conv(source):
    """Same native causal_zero T1 kernel: LAST tap only, not sum/average.

    A registered, already-sliced parameter removes the dynamic weight Slice
    whose shape the legacy ONNX exporter cannot infer after spatial Pad.
    This export-only module is never installed into Core or a user's VAE.
    """
    import torch
    from torch import nn
    import torch.nn.functional as F
    if source.kernel_size[0] != 2 * source.causal_padding[0] + 1 or source.dilation != (1, 1, 1):
        raise ValueError("Unknown causal kernel; cannot specialize T1 safely")
    class StaticT1(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(source.weight[:, :, -1:].detach().clone().contiguous(), requires_grad=False)
            self.bias = None if source.bias is None else nn.Parameter(source.bias.detach().clone(), requires_grad=False)
            self.stride, self.groups = source.stride, source.groups
            self.spatial_padding = source.causal_padding[1:]
        def forward(self, x):
            if not torch.jit.is_tracing() and x.shape[2] != 1:
                raise ValueError("Static causal export accepts exactly one frame")
            ph, pw = self.spatial_padding
            if ph or pw:
                x = F.pad(x, (pw, pw, ph, ph, 0, 0), mode="reflect")
            return F.conv3d(x, self.weight, self.bias, self.stride, 0, 1, self.groups)
    return StaticT1()


def specialize(core):
    from comfy.ldm.minimax.vae import CausalConv3d
    import torch
    rows = []
    for name, module in list(core.named_modules()):
        if isinstance(module, CausalConv3d):
            replacement = static_t1_conv(module)
            if not torch.equal(replacement.weight, module.weight[:, :, -1:]) or not torch.equal(replacement.bias, module.bias):
                raise ValueError("T1 specialization changed active weight values")
            parent_name, _, field = name.rpartition(".")
            parent = core.get_submodule(parent_name) if parent_name else core
            setattr(parent, field, replacement)
            rows.append({"name": name, "source_shape": list(module.weight.shape), "active_shape": list(replacement.weight.shape),
                         "rule": "Exact last temporal tap and original bias, no arithmetic on weights"})
    if not rows:
        raise ValueError("No native causal convolution found")
    return rows


def bind_traced_output_shape(model, traced_shape):
    """Annotate symbolic output dimensions from the real static-input trace.

    No graph operator/weight is changed. The parser/runtime must still resolve
    and validate its own output shape before allocating any inference buffer.
    """
    import onnx
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Expected single-input/single-output export")
    if [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim] != [1, 3, 1, 256, 256] or tuple(traced_shape) != (1, 48, 1, 16, 16):
        raise ValueError("Only actually traced fixed256px T1 profile can be annotated")
    output = model.graph.output[0].type.tensor_type
    if output.elem_type != onnx.TensorProto.FLOAT16 or len(output.shape.dim) != 5:
        raise ValueError("Unexpected traced output dtype/rank")
    original = []
    for dim, actual in zip(output.shape.dim, traced_shape):
        original.append(dim.dim_param if dim.HasField("dim_param") else dim.dim_value)
        if dim.HasField("dim_value") and dim.dim_value != actual:
            raise ValueError("Fixed ONNX output contradicts actual trace")
        dim.ClearField("dim_param")
        dim.dim_value = actual
    return original


def trace_pixels():
    import torch
    # FP16 linspace with196608 steps can itself containNaNs. Build coordinates
    # inFP32 onCPU, then convert; do not mistake an invalid fixture for a VAE bug.
    pixels = torch.linspace(-1,1,3*256*256,dtype=torch.float32).half().reshape(1,3,1,256,256)
    if not bool(torch.isfinite(pixels).all()):
        raise ValueError('Nonfinite tracing fixture')
    return pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    root = args.request.parent
    request = json.loads(args.request.read_text(encoding="utf8"))
    for name, sha in request["sources"].items():
        if digest_file(name) != sha:
            raise ValueError("Export source identity changed")
    sys.path.insert(0,str(Path(request["core_directory"]).resolve(strict=True)))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    import onnx
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-").lower() != request["gpu_uuid"].removeprefix("GPU-").lower():
        raise ValueError("Export GPU identity differs from controller")
    core = native_encoder_shell()
    with safe_open(request["native_vae"], framework="pt", device="cpu") as file:
        state = {key: file.get_tensor(key) for key in file.keys() if key.startswith(("encoder.", "quant_conv."))}
    state.update(latents_mean=core.latents_mean, latents_std=core.latents_std)
    core.load_state_dict(state, strict=True, assign=True)
    del state
    core.encoder = core.encoder.to(device="cuda:0", dtype=torch.float16).eval()
    core.quant_conv = core.quant_conv.to(device="cuda:0", dtype=torch.float16).eval()
    # Deterministic tracing fixture, not generated media or a quality test.
    pixels = trace_pixels().to('cuda:0')
    started = time.perf_counter()
    with torch.inference_mode():
        native = core._encode_moments(pixels).cpu()
        rows = specialize(core)
        candidate = core._encode_moments(pixels).cpu()
        native_static_exact = torch.equal(native,candidate)
        difference = candidate.float()-native.float()
        trace_error = {'max_absolute':float(difference.abs().max()),
                       'relative_rmse':float(difference.square().mean().sqrt()/native.float().square().mean().sqrt().clamp_min(1e-12))}
        save_file({"normalized_pixels":pixels.cpu(),"native_moments":native,"static_moments":candidate},str(root/'reference.safetensors'))
        if tuple(candidate.shape) != (1, 48, 1, 16, 16) or not bool(torch.isfinite(candidate).all()) or not bool(torch.isfinite(native).all()):
            raise ValueError("Invalid actual single-frame moments")
        if not native_static_exact:
            raise ValueError('Native/static trace differs; reference tensors retained for diagnosis')
        class Encoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model._encode_moments(x)
        print("T8_TRT_T1_PHASE trace_checked_exporting_pinned_graph", flush=True)
        torch.onnx.export(Encoder(core).eval(), (pixels,), str(root / "encoder-t1.onnx"),
                          input_names=["pixel_tile"], output_names=["moments_tile"],
                          opset_version=18, dynamo=False, export_params=True, keep_initializers_as_inputs=False, do_constant_folding=True)
    model = onnx.load(root / "encoder-t1.onnx", load_external_data=False)
    original_output = bind_traced_output_shape(model, candidate.shape)
    onnx.checker.check_model(model)
    def shape(value):
        return [d.dim_value for d in value.type.tensor_type.shape.dim]
    if shape(model.graph.input[0]) != [1, 3, 1, 256, 256] or shape(model.graph.output[0]) != [1, 48, 1, 16, 16]:
        raise ValueError("Exported fixed T1 input/output geometry differs")
    onnx.save(model, root / "encoder-t1.onnx")
    report = {"status": "native_t1_static_onnx_exported_not_trt_validated", "native_static_bit_identical": native_static_exact,
              "trace_error":trace_error,
              "input": shape(model.graph.input[0]), "output": shape(model.graph.output[0]),
              "onnx_sha256": digest_file(root / "encoder-t1.onnx"), "bytes": (root / "encoder-t1.onnx").stat().st_size,
              "nodes": len(model.graph.node), "initializers": len(model.graph.initializer), "opset": 18,
              "exporter_output_before_trace_annotation": original_output,
              "output_shape_evidence": "Actual native and specialized GPU trace. Annotation only; independent TRT runtime shape check still required.",
              "specializations": rows, "seconds": time.perf_counter() - started,
              "reference_sha256": digest_file(root / "reference.safetensors"),
              "scope": "Same native FP16 active taps; synthetic tracing input only. Native/static equality and numerical difference reported explicitly, not a universal bit-exact claim. Controller requires exact previously validated graph SHA before publication. Not generated media or new quality/speed qualification."}
    write_new_json(root / "manifest.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
