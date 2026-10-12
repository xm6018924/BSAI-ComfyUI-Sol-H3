# -*- coding: utf-8 -*-
"""Spark-H3 真实采样对比：spark_mode=off vs full（同种子同参数）。

流程：
1. warmup 一轮（模型/CLIP/内核加载，不计时）
2. off 计时（seed=42）
3. full 计时（seed=42, RepBlock 10% + Rabbit 128）
输出 PNG 帧到 ComfyUI/output/bsai_spark_{off,full}_*，最后打印耗时 JSON。
"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8199"
CLIENT_ID = "spark-bench"

PROMPT = (
    "A cinematic live-action shot, beautiful natural lighting, shallow depth of field, "
    "4K quality, smooth motion. The camera slowly pushes in on a hero standing in a "
    "rain-soaked neon alley, wind rustling their coat, with rich ambient audio: rain, "
    "distant city hum, footsteps."
)
NEG = "blurry, low quality, distorted, extra limbs, bad anatomy, watermark, static, silent, robotic voice"

MODEL = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
LORA = "FastH3-4step-LoRA.safetensors"


def build_prompt(spark_mode: str, seed: int, prefix: str):
    return {
        "1": {
            "class_type": "BSAI_SolH3_Loader",
            "inputs": {
                "model_name": MODEL,
                "precision": "int8",
                "sol_attn": True,
                "tau_start": 0.5,
                "tau_end": 1.0,
                "sink_conditioning": "exact_kv",
                "int8_qk": True,
                "fused_modulation": True,
                "chunk_ff": True,
                "chunk_size": 2,
                "fast_h3_steps": "4步 FastH3 极速",
                "sampler": "euler",
                "cfg": 4.0,
                "shift": 8.0,
                "audio_shift": 3.0,
                "min_tokens": 12288,
                "sol_tau": 1.3,
                "start_percent": 0.2,
                "end_percent": 0.9,
                "morton": False,
                "morton_curve": "2d_frame",
                "centroid_tail": True,
                "routed_cap_percent": 0,
                "reuse_qkv_memory": False,
                "verbose": False,
                "dense_blocks": "",
                "tau_profile": "",
                "sol_tau_ramp": True,
                "dense_first_blocks": 2,
                "spark_mode": spark_mode,
                "spark_keep_percent": 10.0,
                "spark_token_aug": 128,
                "lora_name": LORA,
                "lora_strength": 1.0,
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "type": "minimax",
                "device": "default",
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": PROMPT},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": NEG},
        },
        "5": {
            "class_type": "EmptyHunyuanLatentVideo",
            "inputs": {"width": 1344, "height": 768, "length": 24, "batch_size": 1},
        },
        "6": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
        },
        "7": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "control_after_generate": "fixed",
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["6", 0]},
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": prefix},
        },
        "11": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["8", 0], "vae": ["7", 0]},
        },
    }


def post(url, payload):
    req = urllib.request.Request(
        BASE + url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def get(url, retries=5):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(BASE + url)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - transient server-side resets
            last = exc
            time.sleep(5)
    raise last


def wait_done(prompt_id, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = get("/history/%s" % prompt_id)
        if prompt_id in hist:
            st = hist[prompt_id].get("status", {})
            if st.get("completed") is not None or st.get("status_str") == "success":
                return True, time.time() - t0, hist[prompt_id]
            if st.get("status_str") in ("error", "cancelled"):
                return False, time.time() - t0, hist[prompt_id]
        time.sleep(2)
    return False, time.time() - t0, {}


def main():
    results = {}
    for tag, mode, seed, prefix in [
        ("warmup", "full", 999, "bsai_spark_warmup"),
        ("off", "off", 42, "bsai_spark_off"),
        ("full", "full", 42, "bsai_spark_full"),
    ]:
        print(f"[bench] submitting {tag}: spark_mode={mode} seed={seed}", flush=True)
        resp = post("/prompt", {"prompt": build_prompt(mode, seed, prefix),
                                "client_id": CLIENT_ID})
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
        results[tag] = {
            "ok": ok, "wall_s": round(elapsed, 2), "images": n_images,
            "status": hist.get("status", {}),
        }
        print(f"[bench] {tag} done: ok={ok} wall={elapsed:.2f}s images={n_images}", flush=True)

    print("[bench] RESULT_JSON " + json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
