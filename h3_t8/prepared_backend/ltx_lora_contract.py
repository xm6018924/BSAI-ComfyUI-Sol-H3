"""Header-only LoRA pairing; safe to import in a GPU worker (no env changes)."""


def audit_pairs(header, model_shapes, key_ops, quantized_names):
    mapped = {}
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        target = key_ops.apply_to_key(key)
        if target is None or target in mapped:
            raise ValueError(f"Rejected or colliding LoRA key: {key}")
        mapped[target] = entry
    consumed, rows = set(), []
    for key, entry in mapped.items():
        if not key.endswith(".lora_A.weight"):
            continue
        name = key.removesuffix(".lora_A.weight")
        pair = name + ".lora_B.weight"
        target = name + ".weight"
        if pair not in mapped or target not in model_shapes:
            raise ValueError(f"Missing LoRA pair or actual model target: {name}")
        a, b, weight = tuple(entry["shape"]), tuple(mapped[pair]["shape"]), model_shapes[target]
        if len(a) != 2 or len(b) != 2 or len(weight) != 2 or a[0] != b[1] or (b[0], a[1]) != weight:
            raise ValueError(f"LoRA dimensions do not match {name}: {a}, {b}, {weight}")
        if entry["dtype"] != "BF16" or mapped[pair]["dtype"] != "BF16":
            raise ValueError(f"Distilled pair dtype changed: {name}")
        rows.append({"target": name, "rank": a[0], "weight_shape": list(weight),
                     "base_convrot_int8": name in quantized_names})
        consumed.update((key, pair))
    if not rows or consumed != set(mapped):
        raise ValueError(f"Unconsumed or empty LoRA schema: {sorted(set(mapped) - consumed)[:10]}")
    return rows
