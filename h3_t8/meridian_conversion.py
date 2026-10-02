"""CPU-side Meridian Diffusers→native H3 conversion contracts.

No model load, download, CUDA initialization or registry side effects on import.
This module does not itself write a checkpoint or qualify full-model inference.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TensorRule:
    target: str
    sources: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    operation: str = "copy"
    quantize: bool = False


def build_rules(config):
    """Explicit exhaustive architecture mapping, also supports small CPU fixtures."""
    fields = (
        "hidden_size",
        "num_layers",
        "num_refiner_layers",
        "num_attention_heads",
        "attention_head_dim",
        "ffn_dim",
        "time_embed_dim",
        "freq_dim",
        "time_embed_hidden_dim",
        "text_dim",
        "in_channels",
        "audio_in_channels",
        "rope_freq_dim",
    )
    if any(type(config.get(k)) is not int or config[k] <= 0 for k in fields):
        raise ValueError("Positive integer architecture dimensions required")
    patch = config.get("patch_size")
    if (
        not isinstance(patch, (tuple, list))
        or len(patch) != 3
        or any(type(x) is not int or x <= 0 for x in patch)
    ):
        raise ValueError("Three positive patch dimensions required")
    h, f, t = (config[k] for k in ("hidden_size", "ffn_dim", "time_embed_dim"))
    inner = config["num_attention_heads"] * config["attention_head_dim"]
    video = config["in_channels"] * math.prod(patch)
    rules = []

    def add(dst, src, shape, operation="copy", quantize=False):
        rules.append(TensorRule(dst, (src,), (tuple(shape),), operation, quantize))

    for dst, src, outdim, indim in (
        ("video_patch_proj", "proj_in", h, video),
        ("audio_patch_proj", "audio_proj_in", h, config["audio_in_channels"]),
        ("condition_proj", "context_embedder", h, config["text_dim"]),
        (
            "time_embedder.proj_in",
            "time_embedder.linear_1",
            config["time_embed_hidden_dim"],
            config["freq_dim"],
        ),
        (
            "time_embedder.proj_out",
            "time_embedder.linear_2",
            t,
            config["time_embed_hidden_dim"],
        ),
        ("final_layer.video_out", "proj_out", video, h),
        ("final_layer.audio_out", "audio_proj_out", config["audio_in_channels"], h),
        ("final_layer.adaln_proj.linear", "norm_out.linear", 2 * h, t),
    ):
        add(dst + ".weight", src + ".weight", (outdim, indim))
        add(dst + ".bias", src + ".bias", (outdim,))
    add("final_layer.norm.weight", "norm_out.norm.weight", (h,))
    add("token_refiner.final_norm.weight", "token_refiner.final_norm.weight", (h,))
    for refiner, count in (
        (False, config["num_layers"]),
        (True, config["num_refiner_layers"]),
    ):
        for i in range(count):
            src = (
                f"token_refiner.refiner_blocks.{i}"
                if refiner
                else f"transformer_blocks.{i}"
            )
            dst = f"token_refiner.blocks.{i}" if refiner else f"blocks.{i}"
            rules.append(
                TensorRule(
                    dst + ".attn.qkv_proj.weight",
                    tuple(src + f".attn.to_{part}.weight" for part in "qkv"),
                    ((inner, h),) * 3,
                    "concat_qkv",
                    not refiner,
                )
            )
            for suffix, upstream, shape, op, quant in (
                (
                    "attn.out_proj.weight",
                    "attn.to_out.0.weight",
                    (h, inner),
                    "copy",
                    not refiner,
                ),
                (
                    "attn.q_norm.weight",
                    "attn.norm_q.weight",
                    (config["attention_head_dim"],),
                    "copy",
                    False,
                ),
                (
                    "attn.k_norm.weight",
                    "attn.norm_k.weight",
                    (config["attention_head_dim"],),
                    "copy",
                    False,
                ),
                (
                    "mlp.fc1.weight",
                    "ff.net.0.proj.weight",
                    (2 * f, h),
                    "swap_swiglu",
                    not refiner,
                ),
                ("mlp.fc2.weight", "ff.net.2.weight", (h, f), "copy", not refiner),
                ("norm1.weight", "norm1.weight", (h,), "copy", False),
                ("norm2.weight", "norm2.weight", (h,), "copy", False),
            ):
                add(dst + "." + suffix, src + "." + upstream, shape, op, quant)
            if not refiner:
                add(
                    dst + ".adaln_proj.linear.weight",
                    src + ".adaln_proj.linear.weight",
                    (18 * h, t),
                    quantize=True,
                )
                add(
                    dst + ".adaln_proj.linear.bias",
                    src + ".adaln_proj.linear.bias",
                    (18 * h,),
                )
    return tuple(rules)


def validate_index(config, weight_map):
    rules = build_rules(config)
    expected = {key for rule in rules for key in rule.sources}
    actual = set(weight_map)
    if actual != expected:
        raise ValueError(
            f"Unconsumed/missing teacher keys: extra={sorted(actual - expected)}, missing={sorted(expected - actual)}"
        )
    return rules


def expected_lora_shapes(config):
    """The released DMD targets 6 matrices per main block plus both video projections."""
    rules = build_rules(config)
    return {
        name[:-7]: shape
        for rule in rules
        for name, shape in zip(rule.sources, rule.shapes)
        if name in ("proj_in.weight", "proj_out.weight")
        or (
            name.startswith("transformer_blocks.")
            and name.endswith(".weight")
            and (".attn.to_" in name or ".ff.net." in name)
        )
    }


def validate_adapter(config, metadata, tensor_specs):
    """Reject alternate LoRA mathematics instead of silently applying alpha/r."""
    if (
        metadata.get("peft_type") != "LORA"
        or type(metadata.get("r")) is not int
        or metadata["r"] <= 0
    ):
        raise ValueError("Expected ordinary positive-rank LoRA")
    alpha = metadata.get("lora_alpha")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Finite positive LoRA alpha required")
    for key in (
        "use_dora",
        "use_rslora",
        "use_qalora",
        "fan_in_fan_out",
        "lora_bias",
        "rank_pattern",
        "alpha_pattern",
        "modules_to_save",
        "target_parameters",
        "layer_replication",
        "alora_invocation_tokens",
        "trainable_token_indices",
    ):
        if metadata.get(key):
            raise ValueError(f"Unsupported DMD option: {key}")
    if metadata.get("bias", "none") != "none":
        raise ValueError("DMD bias adaptation is unsupported")
    rank = metadata["r"]
    wanted = {}
    for prefix, (rows, columns) in expected_lora_shapes(config).items():
        wanted[prefix + ".lora_A.weight"] = (rank, columns)
        wanted[prefix + ".lora_B.weight"] = (rows, rank)
    if set(tensor_specs) != set(wanted):
        raise ValueError(
            "DMD targets/pairs do not exactly match the released architecture"
        )
    for name, shape in wanted.items():
        spec = tensor_specs[name]
        if tuple(spec["shape"]) != shape or spec["dtype"] != "F32":
            raise ValueError(f"DMD shape/dtype mismatch: {name}")
    return float(alpha) / rank


def convrot_group(input_columns):
    """Use the native H3 checkpoint's 256/64 policy, then16 for tiny fixtures.

    Do not choose128 merely because2688 is divisible by128: that is not the
    reference H3 conversion policy. Actual runtime dispatch is qualified later.
    """
    if type(input_columns) is not int or input_columns <= 0:
        raise ValueError("Positive integer input width required")
    for group in (256, 64, 16):
        if input_columns % group == 0:
            return group
    raise ValueError(f"No qualified ConvRot group for input width {input_columns}")


def materialize_rule(rule, read_teacher, read_adapter, adapter_targets, scale):
    """Merge in FP32 BEFORE QKV fusion, SwiGLU reorder, rotation or quantization.

    Returns one output matrix and its source storage dtype; caller releases it
    before advancing. No full-model dictionary is materialized.
    """
    import torch

    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Finite positive adapter scale required")
    pieces, dtype = [], None
    for name, shape in zip(rule.sources, rule.shapes):
        source = read_teacher(name)
        if (
            source.device.type != "cpu"
            or tuple(source.shape) != shape
            or source.dtype not in (torch.float32, torch.bfloat16)
        ):
            raise ValueError(f"Teacher CPU shape/dtype mismatch: {name}")
        if not torch.isfinite(source).all():
            raise ValueError(f"Nonfinite teacher: {name}")
        if dtype is not None and dtype != source.dtype:
            raise ValueError("Fused teacher matrices must have matching dtypes")
        dtype = source.dtype
        value = source.float().clone()
        prefix = name.removesuffix(".weight")
        if name.endswith(".weight") and prefix in adapter_targets:
            a = read_adapter(prefix + ".lora_A.weight")
            b = read_adapter(prefix + ".lora_B.weight")
            if (
                a.device.type != "cpu"
                or b.device.type != "cpu"
                or a.dtype != torch.float32
                or b.dtype != torch.float32
            ):
                raise ValueError(f"DMD must merge on CPU in FP32: {prefix}")
            if (
                a.ndim != 2
                or b.ndim != 2
                or a.shape[0] != b.shape[1]
                or (b.shape[0], a.shape[1]) != shape
            ):
                raise ValueError(f"DMD shape mismatch: {prefix}")
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError(f"Nonfinite DMD: {prefix}")
            value.addmm_(b, a, beta=1, alpha=scale)
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite merged teacher: {name}")
        pieces.append(value)
    if rule.operation == "concat_qkv" and len(pieces) == 3:
        output = torch.cat(pieces, dim=0)
    elif (
        rule.operation == "swap_swiglu"
        and len(pieces) == 1
        and pieces[0].shape[0] % 2 == 0
    ):
        first, second = pieces[0].chunk(2, dim=0)
        output = torch.cat((second, first), dim=0)
    elif rule.operation == "copy" and len(pieces) == 1:
        output = pieces[0]
    else:
        raise ValueError(f"Invalid mapping operation: {rule.operation}")
    return output, dtype


def derived_rope(config):
    import torch

    dim, theta = config["rope_freq_dim"], config["rope_theta"]
    if (
        type(dim) is not int
        or dim <= 0
        or type(theta) not in (int, float)
        or not math.isfinite(theta)
        or theta <= 0
    ):
        raise ValueError("Invalid rotary dimensions/base")
    return 1.0 / (
        theta ** (torch.arange(0, 2 * dim, 2, dtype=torch.float32) / (2 * dim))
    )


def encode_native_tensor(rule, value, source_dtype, *, row_chunk=256, cancelled=None):
    """Serialize one merged rule with real CK CPU ConvRot, bounded row scratch.

    Only main-block dense matrices are quantized, matching the existing native
    full H3 policy. Quantization is lossy; the returned error is measured against
    the FP32 merged matrix, not a full-model quality claim.
    """
    import json
    import torch

    if type(row_chunk) is not int or row_chunk <= 0:
        raise ValueError("Positive row chunk required")
    if (
        value.device.type != "cpu"
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
    ):
        raise ValueError("Finite FP32 CPU merged tensor required")
    if source_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Unsupported source storage dtype")
    if cancelled is not None and cancelled():
        raise InterruptedError("Meridian conversion cancelled")
    if not rule.quantize:
        stored = value.to(source_dtype).contiguous()
        if not torch.isfinite(stored).all():
            raise ValueError("Nonfinite high-precision island after storage cast")
        return {rule.target: stored}, dict(quantized=False, dtype=str(source_dtype))
    if value.ndim != 2 or not rule.target.endswith(".weight"):
        raise ValueError("Only matrix weights can use ConvRot")
    from comfy_kitchen.tensor import TensorWiseINT8Layout
    from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight

    group = convrot_group(value.shape[1])
    qdata = torch.empty(value.shape, dtype=torch.int8, device="cpu")
    scales = torch.empty((value.shape[0], 1), dtype=torch.float32, device="cpu")
    squared_error, squared_signal, maximum_error = 0.0, 0.0, 0.0
    for start in range(0, value.shape[0], row_chunk):
        if cancelled is not None and cancelled():
            raise InterruptedError("Meridian conversion cancelled")
        stop = min(start + row_chunk, value.shape[0])
        rows = value[start:stop].contiguous()
        q, params = TensorWiseINT8Layout.quantize(
            rows,
            is_weight=True,
            per_channel=True,
            convrot=True,
            convrot_groupsize=group,
            stochastic_rounding=0,
        )
        if (
            q.dtype != torch.int8
            or params.scale.dtype != torch.float32
            or params.scale.shape != (stop - start, 1)
        ):
            raise ValueError("CK native ConvRot serialization contract changed")
        if not torch.isfinite(params.scale).all() or not (params.scale > 0).all():
            raise ValueError("Invalid ConvRot row scales")
        reconstructed = dequantize_int8_convrot_weight(q, params.scale, group)
        delta = reconstructed.double() - rows.double()
        squared_error += delta.square().sum().item()
        squared_signal += rows.double().square().sum().item()
        maximum_error = max(maximum_error, delta.abs().max().item())
        qdata[start:stop].copy_(q)
        scales[start:stop].copy_(params.scale)
    prefix = rule.target.removesuffix(".weight")
    configuration = dict(
        format="int8_tensorwise", convrot=True, convrot_groupsize=group
    )
    encoded = torch.tensor(
        list(json.dumps(configuration).encode("utf8")), dtype=torch.uint8
    )
    return {
        rule.target: qdata,
        rule.target + "_scale": scales,
        prefix + ".comfy_quant": encoded,
    }, dict(
        quantized=True,
        group=group,
        source_dtype=str(source_dtype),
        shape=list(value.shape),
        rows_per_chunk=row_chunk,
        relative_l2=math.sqrt(squared_error / max(squared_signal, 1e-300)),
        max_absolute_error=maximum_error,
        backend="comfy_kitchen_CPU_ConvRot",
    )
