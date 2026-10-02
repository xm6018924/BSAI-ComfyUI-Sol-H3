"""Explicit BF16 execution preparation from existing ConvRot INT8 base weights.

Reuses Comfy Kitchen's inverse layout and the pinned native LTX LoRA fuse rule.
This is NOT the original BF16 checkpoint or an INT8-activation performance path.
No on-disk weights are changed and no native source is patched.
"""
import math


def prepare_weight(key, weight, *, scale=None, convrot_groupsize=None,
                   lora_a=None, lora_b=None, strength=.8, fusion_device="cpu"):
    import torch
    from comfy_kitchen.tensor import TensorWiseINT8Layout
    from ltx_core.loader.fuse_loras import LoraProduct, aggregate_lora_products, bf16_fuse_rule

    device = torch.device(fusion_device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only explicit CPU or CUDA fusion is supported")

    if weight.is_meta or weight.device.type != "cpu" or not math.isfinite(strength):
        raise ValueError("Preparation requires real CPU weights and finite LoRA strength")
    if (lora_a is None) != (lora_b is None):
        raise ValueError("Partial LoRA pair")
    if weight.dtype == torch.int8:
        if (weight.ndim != 2 or scale is None or scale.dtype != torch.float32
                or scale.device.type != "cpu" or scale.is_meta
                or tuple(scale.shape) != (weight.shape[0], 1)
                or convrot_groupsize != 256 or weight.shape[1] % convrot_groupsize):
            raise ValueError("Missing or invalid row-scaled ConvRot INT8 contract")
        if not torch.isfinite(scale).all() or not torch.all(scale > 0):
            raise ValueError("Invalid ConvRot scale values")
        params = TensorWiseINT8Layout.Params(scale=scale, orig_dtype=torch.bfloat16,
            orig_shape=tuple(weight.shape), is_weight=True, convrot=True,
            convrot_groupsize=convrot_groupsize)
        # Dequantize includes inverse ConvRot. Multiplication by scale alone is wrong.
        value = TensorWiseINT8Layout.dequantize(weight, params)
    elif weight.dtype in (torch.bfloat16, torch.float32):
        if scale is not None or convrot_groupsize is not None:
            raise ValueError("Unexpected quantization metadata on floating-point weight")
        value = weight.clone()
    else:
        raise ValueError(f"Unsupported base dtype: {weight.dtype}")
    if lora_a is not None:
        if (lora_a.dtype != torch.bfloat16 or lora_b.dtype != torch.bfloat16
                or lora_a.device.type != "cpu" or lora_b.device.type != "cpu"
                or lora_a.is_meta or lora_b.is_meta or lora_a.ndim != 2 or lora_b.ndim != 2
                or value.ndim != 2 or lora_a.shape[0] != lora_b.shape[1]
                or (lora_b.shape[0], lora_a.shape[1]) != tuple(value.shape)):
            raise ValueError("LoRA pair dtype/device/dimensions do not match real base weight")
        with torch.inference_mode():
            if device.type == "cuda":
                free, _ = torch.cuda.mem_get_info(device)
                # One matrix/pair only, plus conservative GEMM workspace. No full
                # GPU state dict or connector is ever retained by this loader.
                needed = 4 * value.nbytes + 2 * (lora_a.nbytes + lora_b.nbytes) + 64 * 1024**2
                if free < 2 * 1024**3 + needed:
                    raise RuntimeError("Insufficient GPU margin for one native LoRA fusion")
            av = bv = base = delta = fused = None
            try:
                av, bv, base = lora_a.to(device), lora_b.to(device), value.to(device)
                delta = aggregate_lora_products([LoraProduct(av, bv, strength)], torch.bfloat16)
                # Same native BF16 aggregation/fuse rule, including rounding order.
                fused = bf16_fuse_rule(key, base, delta, None)[key]
                value = fused.to("cpu")
            finally:
                del av, bv, base, delta, fused
    if not torch.isfinite(value).all():
        raise ValueError("Prepared weight contains nonfinite values")
    return value.contiguous()


def load_joint_cpu(checkpoint, lora, *, progress, ops=None, fusion_device="cpu"):
    """Materialize only the actual joint model, one tensor/pair at a time on CPU.

    Native generic CPU+LoRA build may retain fused tensors on CUDA. This path
    deliberately prepares each weight on CPU, retaining neither connector nor
    a full duplicate state dict. The sampling worker will lease one GPU block.
    Caller must verify the full asset identity receipts before invoking this.
    """
    import time
    import torch
    from safetensors import safe_open
    from ltx_core.loader.single_gpu_model_builder import _cast_floating_sd
    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
    from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core.model.transformer.model import LTXModelType
    from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
    from runtime.cache_ops.int8_header import _read_header, read_dev_int8_convrot_specs, partition_dev_int8_convrot_specs
    from ltx_lora_contract import audit_pairs
    from pathlib import Path

    started = time.perf_counter()
    model = None
    try:
        metadata = SafetensorsModelStateDictLoader().metadata(str(checkpoint))
        with torch.device("meta"):
            model = (LTXModelConfigurator.from_metadata(metadata) if ops is None
                     else LTXModelConfigurator.from_metadata(metadata, ops=ops))
        if model.model_type != LTXModelType.AudioVideo or len(model.transformer_blocks) != 48:
            raise ValueError("Expected fixed native 48-layer joint AudioVideo model")
        specs, _ = partition_dev_int8_convrot_specs(read_dev_int8_convrot_specs(checkpoint))
        specs = {s.module_name: s for s in specs}
        shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        header, _, _ = _read_header(Path(lora))
        pairs = audit_pairs(header, shapes, LTXV_LORA_COMFY_RENAMING_MAP, set(specs))
        if len(pairs) != 1660 or len(specs) != 1344:
            raise ValueError("Fixed base/LoRA model cardinality changed")
        pair_names = {pair["target"] for pair in pairs}
        raw_lora = {LTXV_LORA_COMFY_RENAMING_MAP.apply_to_key(key): key
                    for key in header if key != "__metadata__"}
        counter = {"state_tensors": 0, "inverse_convrot": 0, "fused_pairs": 0}
        with safe_open(str(checkpoint), framework="pt", device="cpu") as base_file, \
                safe_open(str(lora), framework="pt", device="cpu") as lora_file, torch.inference_mode():
            for key, expected in shapes.items():
                base_key = "model.diffusion_model." + key
                weight = base_file.get_tensor(base_key)
                if tuple(weight.shape) != expected:
                    raise ValueError(f"Actual base tensor shape changed: {key}")
                name = key.removesuffix(".weight") if key.endswith(".weight") else None
                spec = specs.get(name)
                pair = name in pair_names
                scale = base_file.get_tensor("model.diffusion_model." + name + ".weight_scale") if spec else None
                a = lora_file.get_tensor(raw_lora[name + ".lora_A.weight"]) if pair else None
                b = lora_file.get_tensor(raw_lora[name + ".lora_B.weight"]) if pair else None
                value = prepare_weight(key, weight, scale=scale,
                    convrot_groupsize=spec.convrot_groupsize if spec else None,
                    lora_a=a, lora_b=b, strength=.8, fusion_device=fusion_device)
                # Exactly native post-fusion cast policy, including FP32 scalars.
                value = _cast_floating_sd({key: value}, torch.bfloat16)[key]
                parent_name, _, leaf = key.rpartition(".")
                owner = model.get_submodule(parent_name) if parent_name else model
                if leaf in owner._parameters:
                    owner._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
                elif leaf in owner._buffers:
                    owner._buffers[leaf] = value
                else:
                    raise ValueError(f"Unowned native state tensor: {key}")
                counter["state_tensors"] += 1
                counter["inverse_convrot"] += int(spec is not None)
                counter["fused_pairs"] += int(pair)
                del weight, scale, a, b, value
                if counter["state_tensors"] % 25 == 0:
                    progress({**counter, "last_key": key, "elapsed_seconds": time.perf_counter() - started})
        if counter != {"state_tensors": len(shapes), "inverse_convrot": 1344, "fused_pairs": 1660}:
            raise ValueError(f"Incomplete base or LoRA preparation: {counter}")
        if any(p.is_meta or p.device.type != "cpu" for p in (*model.parameters(), *model.buffers())):
            raise ValueError("Uninitialized or non-CPU native model after preparation")
        receipt = {**counter, "seconds": time.perf_counter() - started,
                   "parameter_bytes": sum(p.nbytes for p in model.parameters()),
                   "policy": "CK inverse ConvRot INT8 -> BF16 base; native BF16 LoRA .8; each result retained only on CPU",
                   "fusion_device": str(fusion_device),
                   "original_BF16_checkpoint": False, "INT8_activation_execution": False,
                   "disk_weights_modified": False}
        return model.eval().requires_grad_(False), receipt
    except BaseException:
        if model is not None:
            # The isolated worker still owns the shell; release prepared weights
            # without replacing the original preparation exception.
            try:
                model.to_empty(device="meta")
            except BaseException:
                pass
        raise
