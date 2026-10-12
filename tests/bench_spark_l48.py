# -*- coding: utf-8 -*-
"""48帧长序列对比：off vs full（min_tokens=512，热态）。
验证"序列越长 Spark 加速越明显"的趋势。"""
import json
import sys

sys.path.insert(0, r"G:\BSAI-ComfyUI-intel-XPU-GPU-NPU-aki\ComfyUI\custom_nodes\BSAI-ComfyUI-Sol-H3\tests")
from bench_spark_lowmt import build_lowmt, MIN_TOKENS  # noqa: E402
from bench_spark_h3 import CLIENT_ID, post, wait_done  # noqa: E402

SEED = 42
LENGTH = 48


def run(tag, mode, prefix):
    p = build_lowmt(mode, prefix, SEED, length=LENGTH)
    print(f"[bench] submitting {tag}: spark_mode={mode} seed={SEED} len={LENGTH}", flush=True)
    resp = post("/prompt", {"prompt": p, "client_id": CLIENT_ID})
    if "error" in resp and resp.get("error"):
        print(f"[bench] {tag} SUBMIT ERROR: {resp['error']}", flush=True)
        return {"error": resp["error"]}
    pid = resp["prompt_id"]
    ok, elapsed, hist = wait_done(pid)
    n_images = 0
    if ok and "outputs" in hist:
        for node_id, out in hist["outputs"].items():
            if "images" in out:
                n_images += len(out["images"])
    print(f"[bench] {tag} done: ok={ok} wall={elapsed:.2f}s images={n_images}", flush=True)
    return {"ok": ok, "wall_s": round(elapsed, 2), "images": n_images}


def main():
    results = {
        "off48": run("off48", "off", "bsai_l48_off"),
        "full48": run("full48", "full", "bsai_l48_full"),
    }
    print("[bench] RESULT_JSON " + json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
