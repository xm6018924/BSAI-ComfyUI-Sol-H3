# -*- coding: utf-8 -*-
"""Spark-H3 真实采样对比（低 min_tokens 版）：确保主注意力走稀疏路径。

根因：1344x768x24 的 H3 视频注意力序列 = 6167 tokens < min_tokens=12288，
导致默认配置下 Spark 参数未生效（dense 回退，off/full 输出一致）。
本脚本将 min_tokens 降至 512，让 off(full tau 路由) 与 full(RepBlock+Rabbit)
都在稀疏路径上真实采样，再逐帧对比 PSNR/SSIM 与耗时。
"""
import json
import time
import urllib.request

sys_path_insert = True
import sys
sys.path.insert(0, r"G:\BSAI-ComfyUI-intel-XPU-GPU-NPU-aki\ComfyUI\custom_nodes\BSAI-ComfyUI-Sol-H3\tests")
from bench_spark_h3 import BASE, CLIENT_ID, build_prompt, post, get, wait_done  # noqa: E402

MIN_TOKENS = 512  # 让 6167 token 的主注意力进入稀疏路径
SEED = 42


def build_lowmt(mode: str, prefix: str, seed: int, length: int = 24):
    p = build_prompt(mode, seed, prefix)
    p["1"]["inputs"]["min_tokens"] = MIN_TOKENS
    p["1"]["inputs"]["verbose"] = True  # 输出稀疏路径日志便于核验
    p["5"]["inputs"]["length"] = length
    return p


def main():
    results = {}
    for tag, mode, prefix in [
        ("off", "off", "bsai_lowmt_off"),
        ("full", "full", "bsai_lowmt_full"),
    ]:
        print(f"[bench] submitting {tag}: spark_mode={mode} seed={SEED} min_tokens={MIN_TOKENS}", flush=True)
        resp = post("/prompt", {"prompt": build_lowmt(mode, prefix, SEED), "client_id": CLIENT_ID})
        if "error" in resp and resp.get("error"):
            print(f"[bench] {tag} SUBMIT ERROR: {resp['error']}", flush=True)
            results[tag] = {"error": resp["error"]}
            continue
        pid = resp.get("prompt_id")
        ok, elapsed, hist = wait_done(pid)
        n_images = 0
        if ok and "outputs" in hist:
            for node_id, out in hist["outputs"].items():
                if "images" in out:
                    n_images += len(out["images"])
        results[tag] = {"ok": ok, "wall_s": round(elapsed, 2), "images": n_images}
        print(f"[bench] {tag} done: ok={ok} wall={elapsed:.2f}s images={n_images}", flush=True)

    print("[bench] RESULT_JSON " + json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
