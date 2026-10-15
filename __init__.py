# === BSAI H3 VAE 全量加载统一补丁（宿主: 本插件 BSAI-ComfyUI-Sol-H3）===
# 修复 0.39 + --enable-dynamic-vram 下 H3 AudioVAE/VideoVAE 走 DynamicVRAM
# 权重量流导致的解码极慢（5s 音频 209~324s -> 全量加载约 1.6s）。
# 统一逻辑见 Sol-H3/bsai_h3_vae_full.py；try/except 容错，宿主缺失则 no-op。
# 2026-10-15 增强: 支持官方 LightVAE(#17049) 识别, Lite 版同样走全量加载。
try:
    import sys as _sys, os as _os
    _SOL = _os.path.dirname(_os.path.abspath(__file__))
    if _SOL not in _sys.path:
        _sys.path.insert(0, _SOL)
    import bsai_h3_vae_full as _h3v
    _h3v.apply_global_patch("BSAI-ComfyUI-Sol-H3")
except Exception:
    pass

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "1.0.0"
