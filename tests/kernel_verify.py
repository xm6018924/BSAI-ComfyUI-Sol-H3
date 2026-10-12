# -*- coding: utf-8 -*-
"""内核级验证：comfy_kitchen.sol_attn 在 tau 路由 vs Spark(topk+aug) 路由下
是否产生不同输出。用接近 H3 真实形状的张量 (1, N, 32, 128) bf16。
"""
import torch
import comfy_kitchen as ck

torch.manual_seed(0)
for N in (4096, 8192, 16384, 24192):
    b, h, d = 1, 32, 128
    q = torch.randn(b, N, h, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, N, h, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(b, N, h, d, dtype=torch.bfloat16, device="cuda")

    out_tau = ck.sol_attn(q, k, v, tau=1.3, tail=True, topk_ratio=0.0, token_aug=0)
    out_spark = ck.sol_attn(q, k, v, tau=1.3, tail=True, topk_ratio=0.1, token_aug=128)
    diff = (out_tau - out_spark).abs().max().item()
    same = bool(torch.equal(out_tau, out_spark))
    print(f"N={N}: maxdiff={diff:.6f} bit-identical={same}")

    # 也对比 topk=0.1 vs topk=0.1 (无 aug) 看 aug 是否有效
    out_topk_noaug = ck.sol_attn(q, k, v, tau=1.3, tail=True, topk_ratio=0.1, token_aug=0)
    d2 = (out_spark - out_topk_noaug).abs().max().item()
    print(f"    topk+aug vs topk-only: maxdiff={d2:.6f}")
