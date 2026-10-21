# -*- coding: utf-8 -*-
"""
BSAI H3 VAE 全量加载统一补丁 (bsai_h3_vae_full)
=================================================
供所有 BSAI H3 相关插件统一引用的公共补丁模块（宿主: BSAI-ComfyUI-Sol-H3）。

问题背景
--------
ComfyUI 0.39 + ``--enable-dynamic-vram`` 时，``comfy/sd.py`` 的 ``VAE.__init__``
在读取 ``disable_offload`` 之前就按初始 ``False`` 创建 patcher（``CoreModelPatcher``
被 main.py 重赋值为 ``ModelPatcherDynamic``，日志 ``prepared for dynamic VRAM
loading. XMB Staged``）；而 MiniMaxH3AudioVAE / VideoVAE 分支未设
``disable_offload=True``，导致 H3 VAE 走 DynamicVRAM 权重量流，每帧解码反复
offload/reload 权重：

- 5 秒音频解码实测 209~324 秒（全量加载实测约 1.6 秒）
- 流式数值不稳定 -> 声音怪异 / 失真

统一修复
--------
全局包装 ``comfy.sd.VAE.__init__ / decode / encode``（幂等、多插件共享一次包装）：

1. 构造时：识别 H3 AudioVAE/VideoVAE -> 置 ``disable_offload=True`` 并把
   **动态 patcher 重建为传统 ModelPatcher**（以 patcher 实际类型为准，
   不被 ``disable_offload`` 已为 True 短路——这是旧补丁失效的关键）；
2. 解码/编码兜底：即使 __init__ 包装被绕过，也按 patcher 实际类型强制重建。

用法（各 BSAI 插件入口，try/except 容错）::

    try:
        import sys, os as _os
        _SOL = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                             "..", "BSAI-ComfyUI-Sol-H3")
        if _SOL not in sys.path:
            sys.path.insert(0, _SOL)
        import bsai_h3_vae_full as _h3v
        _h3v.apply_global_patch("BSAI-xxx")
    except Exception:
        pass

本模块自包含、无副作用（不主动挂载），任意环境下 import 都不会抛异常。
"""

import sys

import torch

_APPLIED = set()       # owner 集合
_GLOBAL_PATCHED = False  # 全局只包装一次（多插件共享，避免层叠包装）


def _h3_audio_cls():
    try:
        from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
        return MiniMaxH3AudioVAE
    except Exception:
        return None


def _h3_video_cls():
    try:
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
        return MiniMaxH3VideoVAE
    except Exception:
        return None


def is_h3_audio_vae(vae):
    """H3 音频 VAE 判定：isinstance 优先，类名兜底（防 import 别名/双模块）。"""
    fsm = getattr(vae, "first_stage_model", None)
    if fsm is None:
        return False
    cls = _h3_audio_cls()
    if cls is not None and isinstance(fsm, cls):
        return True
    return type(fsm).__name__ == "MiniMaxH3AudioVAE"


def is_h3_video_vae(vae):
    """H3 视频 VAE 判定：isinstance 优先，类名兜底。"""
    fsm = getattr(vae, "first_stage_model", None)
    if fsm is None:
        return False
    cls = _h3_video_cls()
    if cls is not None and isinstance(fsm, cls):
        return True
    return type(fsm).__name__ == "MiniMaxH3VideoVAE"


def is_light_vae(vae):
    """LightVAE(Lite) 判定：官方 2026-10-15 #17049 支持 MiniMax-H3 LightVAE，
    编码器 resnet 块被剪枝为 shortcut（encoder_kept_blocks 非空），或仅加载
    编码器/解码器之一。识别到后走全量加载同样有效（Lite 更小，加载更快）。"""
    fsm = getattr(vae, "first_stage_model", None)
    if fsm is None or not is_h3_video_vae(vae):
        return False
    kept = getattr(fsm, "encoder_kept_blocks", None)
    if kept:
        return True
    # 仅加载 decoder / encoder 之一（官方 sd.py 按权重自动推断）
    try:
        has_enc = getattr(fsm, "encoder", None) is not None
        has_dec = getattr(fsm, "decoder", None) is not None
        return (has_enc != has_dec)
    except Exception:
        return False


def _align_video_vae_dtype(vae):
    """int8_convrot VideoVAE 以 fp16 全量加载时，部分层仍带着 fp32 的 weight/bias
    （加载器按权重文件里的源 dtype 装入），PyTorch 2.13 的 conv3d 严格检查会报
    "Input type (struct c10::Half) and bias type (float) should be the same"。
    按 vae_dtype 把残留的 fp32 Parameter 对齐一次（量化权重 dtype 非 fp32，不受影响）。
    AudioVAE 本身以 fp32 加载，天然一致，不处理。
    """
    fsm = getattr(vae, "first_stage_model", None)
    if fsm is None or not is_h3_video_vae(vae):
        return
    if getattr(fsm, "_bsai_dtype_aligned", False):
        return
    dtype = getattr(vae, "vae_dtype", torch.float16)
    if dtype not in (torch.float16, torch.bfloat16):
        fsm._bsai_dtype_aligned = True
        return
    fixed = 0
    for _name, _m in fsm.named_modules():
        for _attr in ("weight", "bias"):
            _p = getattr(_m, _attr, None)
            if isinstance(_p, torch.nn.Parameter) and _p.dtype == torch.float32:
                setattr(_m, _attr, torch.nn.Parameter(_p.to(dtype)))
                fixed += 1
    fsm._bsai_dtype_aligned = True
    if fixed:
        print(f"[BSAI-H3VAE] VideoVAE dtype 对齐: {fixed} 个 fp32 参数已转 {dtype}", flush=True)


def rebuild_patcher_if_dynamic(vae, label="H3 VAE"):
    """若 vae.patcher 仍为 ModelPatcherDynamic，重建为传统 ModelPatcher 全量加载。

    0.39 + --enable-dynamic-vram 时仅置 disable_offload=True 不会改变已创建的
    动态 patcher，必须以 patcher 实际类型为准做幂等重建。保留 fast_disk。
    """
    try:
        import comfy.model_management as _mm
        import comfy.model_patcher as _mp
    except Exception:
        return False
    patcher = getattr(vae, "patcher", None)
    if patcher is None:
        return False
    if type(patcher).__name__ != "ModelPatcherDynamic":
        return False
    try:
        vae.patcher = _mp.ModelPatcher(
            vae.first_stage_model,
            load_device=vae.device,
            offload_device=_mm.vae_offload_device(),
            fast_disk=getattr(patcher, "fast_disk", None),
        )
        print(f"[BSAI-H3VAE] {label}解码修复: H3 VAE 动态 patcher 已重建为传统"
              f"ModelPatcher(全量加载), 不再走 DynamicVRAM 权重量流", flush=True)
        return True
    except Exception:
        return False


def ensure_h3_vae_full(vae):
    """解码/编码前兜底：强制 H3 AudioVAE/VideoVAE 全量加载。

    注意：不能因 disable_offload 已为 True 就提前返回——patcher 可能仍是
    动态的（构造置位发生在 patcher 创建之后），必须以 patcher 实际类型为准。
    """
    audio = is_h3_audio_vae(vae)
    video = is_h3_video_vae(vae)
    if not (audio or video):
        return
    kind = "audio" if audio else "video"
    light = "LightVAE-Lite" if (video and is_light_vae(vae)) else ""
    try:
        if not getattr(vae, "disable_offload", False):
            vae.disable_offload = True
        if rebuild_patcher_if_dynamic(vae, kind):
            tag = f"[BSAI-H3VAE] {kind}解码兜底生效" + (f"({light}): " if light else ": ")
            print(f"{tag}H3 VAE disable_offload=True"
                  f"(dynamic patcher rebuilt to traditional full-load)", flush=True)
        elif light:
            print(f"[BSAI-H3VAE] {kind}({light}) 已全量加载(disable_offload=True)", flush=True)
    except Exception:
        pass


def apply_global_patch(owner="BSAI"):
    """全局包装 comfy.sd.VAE.__init__/decode/encode（幂等，多插件共享一次包装）。

    返回 True 表示补丁已生效（本插件触发或之前已由其他插件触发）。
    """
    global _GLOBAL_PATCHED
    try:
        import comfy.sd as _sd
    except Exception:
        return False
    if _GLOBAL_PATCHED:
        _APPLIED.add(owner)
        return True
    try:
        _orig_init = _sd.VAE.__init__
        _orig_decode = getattr(_sd.VAE, "decode", None)
        _orig_encode = getattr(_sd.VAE, "encode", None)
    except Exception:
        return False

    def _wrapped_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        try:
            if is_h3_audio_vae(self):
                if not getattr(self, "disable_offload", False):
                    self.disable_offload = True
                    print("[BSAI-H3VAE] Audio decode fix active: H3 AudioVAE switched to full load"
                          "(disable_offload=True), eliminates DynamicVRAM jitter", flush=True)
                rebuild_patcher_if_dynamic(self, "audio")
            elif is_h3_video_vae(self):
                if not getattr(self, "disable_offload", False):
                    self.disable_offload = True
                    _lt = "LightVAE-Lite" if is_light_vae(self) else ""
                    print("[BSAI-H3VAE] Video decode fix active: H3 VideoVAE%s switched to full load"
                          "(disable_offload=True), eliminates DynamicVRAM jitter"
                          % (f"({_lt})" if _lt else ""), flush=True)
                rebuild_patcher_if_dynamic(self, "video")
        except Exception:
            pass

    def _wrapped_decode(self, *args, **kwargs):
        ensure_h3_vae_full(self)
        _align_video_vae_dtype(self)
        return _orig_decode(self, *args, **kwargs)

    def _wrapped_encode(self, *args, **kwargs):
        ensure_h3_vae_full(self)
        _align_video_vae_dtype(self)
        return _orig_encode(self, *args, **kwargs)

    try:
        _sd.VAE.__init__ = _wrapped_init
        if _orig_decode is not None:
            _sd.VAE.decode = _wrapped_decode
        if _orig_encode is not None:
            _sd.VAE.encode = _wrapped_encode
    except Exception:
        return False
    _GLOBAL_PATCHED = True
    _APPLIED.add(owner)
    print(f"[BSAI-H3VAE] H3 VAE 全量加载统一补丁已生效（宿主 BSAI-ComfyUI-Sol-H3, "
          f"由 {owner} 触发）: AudioVAE/VideoVAE 将全量加载, 不再走 "
          f"DynamicVRAM 权重量流", flush=True)
    return True
