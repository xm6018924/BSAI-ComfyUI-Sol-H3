# -*- coding: utf-8 -*-
"""
BSAI ComfyUI Sol-H3 自动安装脚本
================================
首次安装/更新时自动：
1. 检查 comfy_kitchen（Sol-Attn内核，已内置则跳过）
2. 下载 FastH3 4步蒸馏 LoRA 到 models/loras/
3. 验证所有依赖
"""
import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
COMFY_ROOT = SCRIPT_DIR.parent.parent  # custom_nodes/.. = ComfyUI/
LORA_DIR = COMFY_ROOT / "models" / "loras"
PYTHON = sys.executable

def log(msg):
    print(f"[BSAI-Sol-H3] {msg}", flush=True)

def pip_install(pkg):
    log(f"pip install {pkg}")
    subprocess.check_call([PYTHON, "-m", "pip", "install", pkg, "--quiet"])

def check_comfy_kitchen():
    """checking whether comfy_kitchen is available"""
    try:
        import comfy_kitchen
        if hasattr(comfy_kitchen, "sol_attn"):
            log("comfy_kitchen.sol_attn ready ✅")
            return True
        log("comfy_kitchen exists but sol_attn is missing; please reinstall comfy_kitchen")
        return False
    except ImportError:
        log("comfy_kitchen not found; Sol-Attn requires the comfy_kitchen kernel")
        return False

def download_fasth3_lora():
    """downloading FastH3 4-step LoRA to models/loras/"""
    target = LORA_DIR / "FastH3-4step-LoRA.safetensors"
    if target.exists():
        log(f"FastH3 LoRA已存在: {target} ✅")
        return

    LORA_DIR.mkdir(parents=True, exist_ok=True)
    log("starting download of FastH3 4-step distilled LoRA (~300MB)...")

    # ModelScope镜像（国内直连）
    urls = [
        "https://modelscope.cn/models/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/resolve/master/dense-datafree/adapter_model.safetensors",
        "https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/resolve/main/dense-datafree/adapter_model.safetensors",
    ]

    import urllib.request
    for url in urls:
        try:
            log(f"尝试: {url}")
            urllib.request.urlretrieve(url, str(target))
            size_mb = target.stat().st_size / 1024 / 1024
            log(f"下载完成: {target} ({size_mb:.1f} MB) ✅")
            return
        except Exception as e:
            log(f"失败: {e}")
            if target.exists():
                target.unlink()
            continue

    log("all download sources failed; please manually download the FastH3 LoRA to models/loras/FastH3-4step-LoRA.safetensors")

def main():
    log("=" * 50)
    log("BSAI ComfyUI Sol-H3 install check")
    log("=" * 50)

    check_comfy_kitchen()
    download_fasth3_lora()

    log("=" * 50)
    log("install check complete! Restart ComfyUI and use the workflows.")
    log("Workflow location: custom_nodes/BSAI-ComfyUI-Sol-H3/workflows/")
    log("=" * 50)

if __name__ == "__main__":
    main()
