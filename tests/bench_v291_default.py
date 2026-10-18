# -*- coding: utf-8 -*-
"""v2.9.1 默认参数端到端验证：Loader 默认 min_tokens=1024 / spark_mode=off，
不显式覆盖，验证 tau 稀疏在 H3 视频主注意力真实生效（日志出现 sparse）。"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8199"


def post(url, payload):
    req = urllib.request.Request(
        BASE + url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def get(url, retries=5):
    import time
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(BASE + url)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5)
    raise last


def build_default():
    """Loader 显式传 v2.9.1 默认值（min_tokens=1024 / spark_mode=off），
    验证该默认配置下 tau 稀疏在 H3 视频主注意力真实生效。"""
    return {
        "1": {
            "class_type": "BSAI_SolH3_Loader",
            "inputs": {
                "model_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                "precision": "int8",
                "sol_attn": True,
                "tau_start": 0.5,
                "tau_end": 1.0,
                "sink_conditioning": "exact_kv",
                "int8_qk": True,
                "fused_modulation": True,
                "chunk_ff": True,
                "chunk_size": 2,
                "fast_h3_steps": "4-step FastH3 Turbo",
                "sampler": "euler",
                "cfg": 4.0,
                "shift": 8.0,
                "audio_shift": 3.0,
                "min_tokens": 1024,      # v2.9.1 新默认值
                "sol_tau": 1.3,
                "start_percent": 0.2,
                "end_percent": 0.9,
                "morton": False,
                "morton_curve": "2d_frame",
                "centroid_tail": True,
                "routed_cap_percent": 0,
                "reuse_qkv_memory": False,
                "verbose": True,
                "dense_blocks": "",
                "tau_profile": "",
                "sol_tau_ramp": True,
                "dense_first_blocks": 2,
                "spark_mode": "off",       # v2.9.1 保持默认 off
                "spark_keep_percent": 10.0,
                "spark_token_aug": 128,
                "lora_name": "FastH3-4step-LoRA.safetensors",
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
            "inputs": {"clip": ["2", 0], "text": "a cinematic shot, neon alley, rain"},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": "blurry, low quality"},
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
                "seed": 777,
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
            "inputs": {"images": ["9", 0], "filename_prefix": "bsai_v291_default"},
        },
        "11": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["8", 0], "vae": ["7", 0]},
        },
    }


def main():
    import time
    p = build_default()
    resp = post("/prompt", {"prompt": p, "client_id": "v291-default"})
    if "error" in resp and resp.get("error"):
        print("SUBMIT ERROR:", resp["error"])
        sys.exit(1)
    pid = resp["prompt_id"]
    print("submitted:", pid, flush=True)
    t0 = time.time()
    while time.time() - t0 < 1800:
        hist = get("/history/%s" % pid)
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") in ("success", "error", "cancelled"):
                print("status:", st.get("status_str"), "wall=%.2fs" % (time.time() - t0), flush=True)
                return
        time.sleep(2)
    print("TIMEOUT", flush=True)


if __name__ == "__main__":
    main()
