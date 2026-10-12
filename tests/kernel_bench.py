# -*- coding: utf-8 -*-
"""内核级耗时对比：tau 路由 vs Spark(topk_ratio+token_aug) 路由在
(1, N, 56, 128) bf16 张量上的真实 CUDA 内核耗时（RTX 5090 Laptop / SM120）。"""
import time

import torch
import comfy_kitchen as ck


def bench(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


torch.manual_seed(0)
for N in (6167, 12285):
    q = torch.randn(1, N, 56, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, N, 56, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, N, 56, 128, dtype=torch.bfloat16, device="cuda")
    t_tau = bench(lambda: ck.sol_attn(q, k, v, tau=1.3, tail=True, topk_ratio=0.0, token_aug=0))
    t_topk = bench(lambda: ck.sol_attn(q, k, v, tau=1.3, tail=True, topk_ratio=0.1, token_aug=0))
    t_full = bench(lambda: ck.sol_attn(q, k, v, tau=1.3, tail=True, topk_ratio=0.1, token_aug=128))
    t_dense = bench(lambda: ck.sol_attn(q, k, v, tau=0.0, tail=True, topk_ratio=0.0, token_aug=0))
    print(f"N={N}: dense={t_dense*1000:.1f}ms  tau={t_tau*1000:.1f}ms  topk={t_topk*1000:.1f}ms  topk+aug={t_full*1000:.1f}ms")
