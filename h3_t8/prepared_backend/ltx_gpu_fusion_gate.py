"""Bounded native CPU/GPU LoRA parity and release gate before full loading."""
def run_gate():
    import gc
    import time
    import torch
    from ltx_dequantized_lora import prepare_weight
    from ltx_core.loader.fuse_loras import LoraProduct, aggregate_lora_products, bf16_fuse_rule

    torch.cuda.synchronize()
    cold = torch.cuda.memory_allocated()
    # Establish the native CUDA GEMM handle/workspace baseline, separately from
    # per-weight tensor ownership. First GEMM may allocate persistent workspace.
    with torch.inference_mode():
        warm = torch.ones((64, 256), dtype=torch.bfloat16, device="cuda")
        warm_out = warm @ warm.T
        del warm, warm_out
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    rows = []
    for dtype in (torch.bfloat16, torch.float32):
        gen = torch.Generator().manual_seed(8871)
        base = torch.randn((64, 256), generator=gen).to(dtype)
        a = (torch.randn((16, 256), generator=gen) * .03).bfloat16()
        b = (torch.randn((64, 16), generator=gen) * .03).bfloat16()
        originals = [t.clone() for t in (base, a, b)]
        cpu = prepare_weight("x.weight", base, lora_a=a, lora_b=b)
        start = time.perf_counter()
        actual = prepare_weight("x.weight", base, lora_a=a, lora_b=b, fusion_device="cuda:0")
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        with torch.inference_mode():
            expected = bf16_fuse_rule("x.weight", base.cuda(),
                aggregate_lora_products([LoraProduct(a.cuda(), b.cuda(), .8)], torch.bfloat16), None)["x.weight"].cpu()
        assert torch.equal(actual, expected), "Native CUDA fuse parity failed"
        relative = float((actual.float() - cpu.float()).norm() / cpu.float().norm())
        assert relative < .005 and torch.isfinite(actual).all()
        assert actual.device.type == "cpu" and actual.dtype == dtype
        assert all(torch.equal(t, old) for t, old in zip((base, a, b), originals))
        rows.append({"dtype": str(dtype), "native_CUDA_bitexact": True,
                     "CPU_relative_rmse": relative, "seconds": seconds})
        del expected, actual, cpu
    gc.collect()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert after == before, (before, after)
    return {"status": "native_CUDA_fusion_small_parity_and_release_pass", "cases": rows,
            "cold_allocated": cold, "native_GEMM_warm_baseline": before,
            "allocated_before": before, "allocated_after": after,
            "limits": "Small matrices only; CPU/CUDA tolerance not claimed bitexact across devices"}
