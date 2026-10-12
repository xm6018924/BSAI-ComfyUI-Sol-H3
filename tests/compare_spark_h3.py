# -*- coding: utf-8 -*-
"""Spark-H3 off vs full 画质对比：逐帧 PSNR / SSIM。

对比 ComfyUI/output/bsai_spark_off_* 与 bsai_spark_full_*（同种子 42）。
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

OUT = r"G:\BSAI-ComfyUI-intel-XPU-GPU-NPU-aki\ComfyUI\output"
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "bsai_spark"


def load(name):
    img = Image.open(os.path.join(OUT, name)).convert("RGB")
    return np.asarray(img, dtype=np.float32)


def psnr(a, b, peak=255.0):
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(peak * peak / mse)


def ssim(a, b, c1=0.01 ** 2 * 255 ** 2, c2=0.03 ** 2 * 255 ** 2):
    """单尺度 SSIM (8x8 window, 简化为整图均值的近似版)。

    说明：这里用全局统计近似（mean/var/cov），用于同源视频帧的相对对比足够；
    严格逐块 SSIM 由 skimage 提供但本环境未安装。
    """
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = float(np.mean((a - mu_a) * (b - mu_b)))
    return (2 * mu_a * mu_b + c1) * (2 * cov + c2) / \
           ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))


def main():
    offs = sorted(glob.glob(os.path.join(OUT, f"{PREFIX}_off_*.png")))
    fulls = sorted(glob.glob(os.path.join(OUT, f"{PREFIX}_full_*.png")))
    print(f"off frames: {len(offs)}, full frames: {len(fulls)}")

    n = min(len(offs), len(fulls))
    p_list, s_list, maxdiff_list = [], [], []
    for i in range(n):
        a = load(os.path.basename(offs[i]))
        b = load(os.path.basename(fulls[i]))
        p = psnr(a, b)
        s = ssim(a, b)
        md = float(np.abs(a - b).max())
        p_list.append(p)
        s_list.append(s)
        maxdiff_list.append(md)
        if i < 6 or p < 60:
            print(f"frame {i+1:02d}: PSNR={p:7.2f} dB  SSIM={s:.6f}  maxdiff={md:.2f}")

    arr_p = np.array(p_list)
    arr_s = np.array(s_list)
    print("\n--- 汇总 / Summary (off vs full, same seed 42) ---")
    print(f"PSNR   min={arr_p.min():.2f}  mean={arr_p.mean():.2f}  max={arr_p.max():.2f}")
    print(f"SSIM   min={arr_s.min():.6f}  mean={arr_s.mean():.6f}  max={arr_s.max():.6f}")
    print(f"maxdiff min={min(maxdiff_list):.2f}  mean={np.mean(maxdiff_list):.2f}  max={max(maxdiff_list):.2f}")
    identical = all(x == 0 for x in maxdiff_list)
    print(f"identical frames: {identical}")


if __name__ == "__main__":
    main()
