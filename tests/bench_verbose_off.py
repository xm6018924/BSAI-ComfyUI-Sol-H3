# -*- coding: utf-8 -*-
"""提交 spark_mode=off + verbose=True 的采样，对比 off/full 的 [sol_attn] 日志。"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, r"G:\BSAI-ComfyUI-intel-XPU-GPU-NPU-aki\ComfyUI\custom_nodes\BSAI-ComfyUI-Sol-H3\tests")
from bench_spark_h3 import BASE, build_prompt, post, get, wait_done  # noqa: E402

p = build_prompt("off", 4242, "bsai_spark_verbose_off")
p["1"]["inputs"]["verbose"] = True
resp = post("/prompt", {"prompt": p, "client_id": "spark-verbose-off"})
print("submit:", resp.get("prompt_id"), resp.get("error", ""))
if "error" in resp and resp.get("error"):
    sys.exit(1)
pid = resp["prompt_id"]
ok, elapsed, hist = wait_done(pid, timeout=3600)
print(f"done ok={ok} wall={elapsed:.2f}s")
print(json.dumps(hist.get("status", {}), ensure_ascii=False))
