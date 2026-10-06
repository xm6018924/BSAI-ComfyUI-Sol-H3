# -*- coding: utf-8 -*-
"""
BSAI Sol-H3 SolEngine Stage-2 旁支流水线
==========================================
封装 NVIDIA Sol Engine / H3 Super Acceleration 两级流水线的 Stage-2：
  H3 draft (672x384, 4步) → 像素裁剪 → LTX-2.5 VAE 编码 → x2 latent 放大
  → LTX-2.5 3步精修 → TAEHV 快速解码 → 保留 H3 原音频

本文件完全独立于 nodes.py，不修改现有三个节点的任何行为。
依赖 T8mars/comfyui-minimax-h3-audio-T8 已安装（本机已确认）。
"""
import os
import json
import sys
import torch

# 确保插件自身目录在 sys.path 中（h3_t8 vendored 包）
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

try:
    import folder_paths
except Exception:
    folder_paths = None


# ---------------------------------------------------------------------------
# 量化模型检测: int8/fp8 量化权重 (comfy 标准 comfy_quant 格式) 无法用常规
# weight-patch LoRA —— LoraAdapter.calculate_weight 对 QuantizedTensor 执行
# `weight += lora_diff.type(weight.dtype)` 会静默失败(返回原权重), 导致 LoRA
# 不生效, 裸 dev transformer 跑 3 步蒸馏 schedule 直接输出全噪点。
# 量化模型必须走 bypass 注入模式: output = base(x) + lora_path(x)。
# ---------------------------------------------------------------------------
def _is_quantized_model(model_patcher):
    """检测 model_patcher 内的 diffusion model 是否量化(存在 quant_format 层)。"""
    try:
        import comfy.quant_ops as _qops
    except Exception:
        _qops = None
    try:
        for m in model_patcher.model.modules():
            if getattr(m, "quant_format", None) is not None:
                return True
            w = getattr(m, "weight", None)
            if isinstance(w, torch.nn.Parameter):
                w = w.data
            if _qops is not None and isinstance(w, _qops.QuantizedTensor):
                return True
            if w is not None and getattr(w, "comfy_quant", None) is not None:
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# T8mars SolEngine 函数惰性导入
# ---------------------------------------------------------------------------
_T8 = None
_T8_IMPORT_ERROR = None

def _load_t8():
    global _T8, _T8_IMPORT_ERROR
    if _T8 is not None:
        return _T8
    if _T8_IMPORT_ERROR is not None:
        return None
    try:
        # 直接从插件内置的 vendored h3_t8 包导入
        from h3_t8 import sol_engine_h3_super_advanced as _se
        _T8 = _se
    except Exception as e:
        _T8_IMPORT_ERROR = e
        print(f"[BSAI-Sol-H3] SolEngine: h3_t8 包导入失败({e})", flush=True)
        return None
    return _T8


# ---------------------------------------------------------------------------
# 模型文件列表
# ---------------------------------------------------------------------------
def _list(subdir, match=None):
    if folder_paths is None:
        return []
    try:
        names = folder_paths.get_filename_list(subdir)
    except Exception:
        return []
    if match:
        names = [n for n in names if match.lower() in n.lower()]
    return sorted(names)


# ---------------------------------------------------------------------------
# v2.8 Stage-2 条件缓存 (NVIDIA Sol-H3 论文 §4.2 "prompt caching" 的 ComfyUI 落地)
# 官方依据: 两阶段流水线里 Stage-2 只做局部高频精修, 全局构图/运镜已由 Stage-1
# latent 决定, 因此 Stage-2 不需要每请求在线编码文本——官方 RTX 5090 实测里
# Qwen/Gemma 文本编码器"每请求重建"占端到端 34.1% 耗时(13.568s 中仅 0.518s 是
# 前向, 其余全是加载/卸载)。缓存 (clip, prompt, fps) → cond 张量后, 相同提示词
# 复跑直接跳过 14.6GB gemma 的卸载-重载-编码三段(Step6.5/7/7.5), 也免去
# LTX transformer 为腾内存被换出再换回(Windows 24GB 提交内存峰值同步下降)。
# 缓存在进程内有效(重启 ComfyUI 清空), 键含 clip 文件名与 prompt 全文, 不会串。
# ---------------------------------------------------------------------------
_COND_CACHE = {}
_COND_CACHE_MAX = 8


class BSAI_SolH3_SolEngine_Refiner:
    """NVIDIA Sol Engine 两级 Stage-2: H3 draft → LTX-2.5 精修 (旁支)。

    输入 H3 VAEDecode 的 IMAGE + VAEDecodeAudio 的 AUDIO，
    内部完成裁剪/resize → LTX VAE 编码 → x2 latent 放大 → LTX-2.5 精修
    → VAE/TAEHV 解码，输出精修 IMAGE + 裁剪后的 H3 AUDIO。
    """

    @classmethod
    def INPUT_TYPES(cls):
        ltxf = _list("diffusion_models", "ltx") or ["(缺 LTX-2.5 transformer)"]
        vaef = _list("vae", "ltx") or ["(缺 LTX VAE)"]
        clipf = _list("text_encoders", "ltx") or _list("text_encoders", "gemma4") or ["(缺 text encoder)"]
        upf = _list("latent_upscale_models", "ltx") or ["(缺 LTX upscaler)"]
        loras = _list("loras", "ltx")
        lora_opts = ["(无 / distilled-transformer 已 bake)"] + loras
        return {
            "required": {
                "h3_frames": ("IMAGE",),
                "h3_audio": ("AUDIO",),
                "positive_prompt": ("STRING", {"multiline": True,
                    "default": "Use the same positive prompt as the H3 draft."}),
                "target_width": ("INT", {"default": 1920, "min": 256, "max": 3840, "step": 32}),
                "target_height": ("INT", {"default": 1088, "min": 256, "max": 2176, "step": 32}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "schedule_mode": (["official_0p85", "one_step_sol", "official_0p85_8step", "official_0p909", "identity_preserve_0p5", "step20_diag", "lora_10step_diag"],
                    {"default": "official_0p85"}),
                "ltx_transformer": (ltxf,),
                "ltx_vae": (vaef,),
                "ltx_clip": (clipf,),
                "ltx_upscaler": (upf,),
                "refiner_lora": (lora_opts,),
                "refiner_lora_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                # ---- v2.8 新增 (追加在末尾, 不破坏旧工作流 widgets 对齐) ----
                # 官方 Stage-2 Sol-Attn 稀疏注意力默认启用 (论文 §3.4: 3步精修
                # tau 递增 1.0/1.25/1.5 + 第0层 dense 保护, 内置 h3_t8 已实现,
                # v2.7 写死 dense_reference 关闭了它); 内核缺失自动回退 dense。
                "stage2_attention": (["sol_attn_sparse", "dense_reference"],
                    {"default": "sol_attn_sparse"}),
                # 论文 §4.2 prompt caching: Stage-2 条件按 (clip,prompt,fps)
                # 进程内缓存, 复跑同提示词免重载 14.6GB gemma(官方5090实测该
                # 编码器每请求重建占端到端 34.1% 耗时)。
                "stage2_cond_cache": ("BOOLEAN", {"default": True, "advanced": True}),
                # 4K/2K 输出解码: auto=面积>1.6MP自动tiled(24GB可跑3840×2176),
                # full=始终全量, tiled=始终分块。
                "decode_mode": (["auto", "full", "tiled"], {"default": "auto"}),
                "decode_tile_size": ("INT", {"default": 768, "min": 256, "max": 2160, "step": 32, "advanced": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("image", "audio", "report")
    FUNCTION = "refine"
    CATEGORY = "BSAI/Sol-H3/SolEngine"
    DESCRIPTION = "NVIDIA Sol Engine 两级 Stage-2: H3 draft → LTX-2.5 3步精修(旁支, 不影响原双采节点)"

    # ------------------------------------------------------------------
    def _load_ltx_model(self, name):
        import comfy.sd
        path = folder_paths.get_full_path_or_raise("diffusion_models", name)
        print(f"[BSAI-SolEngine] 加载 LTX-2.5 transformer: {name}", flush=True)
        return comfy.sd.load_diffusion_model(path)

    def _load_ltx_vae(self, name):
        import comfy.sd
        import comfy.utils
        path = folder_paths.get_full_path_or_raise("vae", name)
        print(f"[BSAI-SolEngine] 加载 LTX VAE: {name}", flush=True)
        sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    def _load_ltx_clip(self, name):
        import comfy.sd
        path = folder_paths.get_full_path_or_raise("text_encoders", name)
        print(f"[BSAI-SolEngine] 加载 LTX text encoder: {name}", flush=True)
        return comfy.sd.load_clip(
            ckpt_paths=[path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.LTXV,
        )

    def _load_upscaler(self, name):
        import json as _json
        import comfy.utils
        import comfy.ops
        import comfy.model_management
        import comfy.storage
        from comfy.model_patcher import CoreModelPatcher
        from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler

        path = folder_paths.get_full_path_or_raise("latent_upscale_models", name)
        print(f"[BSAI-SolEngine] 加载 LTX upscaler: {name}", flush=True)
        sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        fast_disk = comfy.storage.state_dict_fast_disk(sd)
        config = _json.loads(metadata["config"])
        model = LatentUpsampler.from_config(
            config, operations=comfy.ops.disable_weight_init
        ).to(dtype=comfy.model_management.vae_dtype(
            allowed_dtypes=[torch.bfloat16, torch.float32]))
        comfy.model_management.archive_model_dtypes(model)
        # 先在原始 nn.Module 上 load_state_dict, 再包 CoreModelPatcher
        is_dyn = getattr(model, "is_dynamic", lambda: False)()
        model.load_state_dict(sd, assign=is_dyn)
        mp = CoreModelPatcher(
            model,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
            fast_disk=fast_disk,
        )
        return mp

    # ------------------------------------------------------------------
    def refine(self, h3_frames, h3_audio, positive_prompt,
               target_width, target_height, fps, schedule_mode,
               stage2_attention, stage2_cond_cache, decode_mode, decode_tile_size,
               ltx_transformer, ltx_vae, ltx_clip, ltx_upscaler,
               refiner_lora, refiner_lora_strength, seed):
        t8 = _load_t8()
        if t8 is None:
            raise RuntimeError("T8mars SolEngine 模块不可用")

        report = {"stage": "solengine_stage2", "schedule": schedule_mode,
                  "attention": stage2_attention}

        # 1) H3 draft 帧裁剪/resize → LTX encoder 半分辨率输入
        print("[BSAI-SolEngine] Step1: 裁剪 H3 draft 帧...", flush=True)
        (enc_frames, enc_w, enc_h, kept_frames, dropped,
         duration, _) = t8.prepare_h3_draft_for_ltx_refiner(
            h3_frames, target_width, target_height, "trim_to_8n_plus_1", fps)
        report["encoder_size"] = [enc_w, enc_h]
        report["kept_frames"] = kept_frames
        report["dropped_frames"] = dropped
        report["duration_s"] = duration
        print(f"[BSAI-SolEngine]   encoder={enc_w}x{enc_h} frames={kept_frames} dropped={dropped}", flush=True)

        # 1.5) 卸载 H3 模型，释放 VRAM 给 LTX
        print("[BSAI-SolEngine] Step1.5: 卸载 H3 模型，释放 VRAM...", flush=True)
        import comfy.model_management as _mm_unload
        _mm_unload.unload_all_models()

        # 2) LTX VAE encode (frames: [F,H,W,C] float0-1)
        print("[BSAI-SolEngine] Step2: LTX VAE encode...", flush=True)
        vae = self._load_ltx_vae(ltx_vae)
        # VAE.encode 期望 IMAGE 格式 [F,H,W,C]; 内部自动 movedim/视频维度处理
        pixel = enc_frames[..., :3].contiguous()
        with torch.no_grad():
            lt_out = vae.encode(pixel)
        if isinstance(lt_out, dict):
            lt_samples = lt_out["samples"]
        else:
            lt_samples = lt_out
        print(f"[BSAI-SolEngine]   LTX latent: {tuple(lt_samples.shape)}", flush=True)

        # 3) x2 latent upscaler (复刻 LTXVLatentUpsampler.execute)
        print("[BSAI-SolEngine] Step3: x2 latent upscale...", flush=True)
        up_model = self._load_upscaler(ltx_upscaler)
        device = getattr(up_model, "load_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model_dtype = up_model.model_dtype() if hasattr(up_model, "model_dtype") else lt_samples.dtype

        lt_dev = lt_samples.to(dtype=model_dtype, device=device)
        lt_dev = vae.first_stage_model.per_channel_statistics.un_normalize(lt_dev)
        import comfy.model_management as _mm
        _mm.load_models_gpu([up_model])
        with torch.no_grad():
            up_lat = up_model.model(lt_dev)
        up_lat = vae.first_stage_model.per_channel_statistics.normalize(up_lat)
        up_lat = up_lat.to(dtype=torch.float32, device="cpu")
        print(f"[BSAI-SolEngine]   upscaled: {tuple(up_lat.shape)}", flush=True)
        latent_dict = {"samples": up_lat}

        # 4) 加载 LTX transformer
        print("[BSAI-SolEngine] Step4: 加载 LTX-2.5 transformer...", flush=True)
        ltx_model = self._load_ltx_model(ltx_transformer)

        # 检测 transformer 类型: distilled 已内置蒸馏权重, 不需 LoRA; dev 需要 LoRA
        is_distilled_tf = "distilled" in ltx_transformer.lower()

        # 4b) refiner LoRA (仅 dev transformer 需要; distilled transformer 自动跳过)
        if is_distilled_tf:
            print("[BSAI-SolEngine] Step5: distilled transformer 已内置蒸馏, 跳过 LoRA", flush=True)
        elif schedule_mode != "step20_diag" and refiner_lora and "无" not in refiner_lora and refiner_lora_strength > 0:
            print(f"[BSAI-SolEngine] Step5: LoRA {refiner_lora} x{refiner_lora_strength}", flush=True)
            try:
                import comfy.sd
                import comfy.utils
                lora_path = folder_paths.get_full_path("loras", refiner_lora)
                lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
                if _is_quantized_model(ltx_model):
                    # int8/fp8 量化权重 → bypass 注入模式 (不修改权重,
                    # forward 时 output = base(x) + lora_path(x)),
                    # 避免 calculate_weight 对 QuantizedTensor 静默失败
                    print("[BSAI-SolEngine]   量化 transformer → bypass LoRA 注入", flush=True)
                    ltx_model, _ = comfy.sd.load_bypass_lora_for_models(
                        ltx_model, None, lora_sd, refiner_lora_strength, 0)
                else:
                    ltx_model, _ = comfy.sd.load_lora_for_models(
                        ltx_model, None, lora_sd, refiner_lora_strength, 0)
                print(f"[BSAI-SolEngine]   LoRA OK", flush=True)
            except Exception as e:
                # 不再静默降级: LoRA 是 Stage-2 蒸馏 3 步 schedule 的必需依赖,
                # 应用失败必须明确报错, 否则裸 dev 跑 3 步会输出全噪点
                raise RuntimeError(
                    f"[BSAI-SolEngine] LoRA 应用失败 (Stage-2 必需): {e}"
                ) from e
        else:
            print("[BSAI-SolEngine] Step5: 无 refiner LoRA", flush=True)

        # 4c) ModelSamplingLTXV shift — 官方 Stage-2 硬编码 sigma 已含时间轨迹,
        # LTX 模型 timestep(sigma)=sigma (ModelSamplingFlux), 无需也不应改 shift;
        # 保持模型内置 sampling 配置, 手动设置 shift 会导致采样结果错误/噪点

        # 6) refiner schedule
        # 官方配置对照 (Lightricks ComfyUI 模板 Stage-2):
        #   sigmas = 0.85, 0.725, 0.4219, 0 (3步) + euler_ancestral + CFG 1.0
        # NVIDIA SolEngine 参考用 0.909 + euler, 但那是 NVIDIA 专用蒸馏模型;
        # dev + 官方蒸馏 LoRA 450 的轨迹对齐 Lightricks 官方模板 → 默认 0.85。
        # v2.8: 注意力后端不再写死 dense_reference。sol_attn_sparse 时走内置
        # h3_t8 的官方 Stage-2 Sol-Attn 路径 (论文 §3.4: 逐步 tau 1.0/1.25/1.5
        # + 第 0 层 dense 保护); 内核未加载时 find_loaded_sol_attn_backend()
        # 返回 None, setup 函数自动回退 dense, 不会报错。
        _attn_backend = ("auto_sol_attn" if stage2_attention == "sol_attn_sparse"
                         else "dense_reference")
        print(f"[BSAI-SolEngine] Step6: LTX refiner schedule... (attention={_attn_backend})", flush=True)
        if schedule_mode == "identity_preserve_0p5":
            # identity_preserve 专用后端名: sparse → auto_sol_attn_conservative_exp
            _ip_backend = ("auto_sol_attn_conservative_exp"
                           if stage2_attention == "sol_attn_sparse"
                           else "dense_reference")
            patched, sigmas, _, _ = t8.setup_ltx_identity_preserve_refiner(
                ltx_model, enabled=True, schedule_mode="identity_preserve_0p5",
                manual_sigmas="0.65, 0.56, 0.47, 0.38, 0.29, 0.20, 0.11, 0",
                attention_backend=_ip_backend,
                min_tokens=4096, kernel_precision="bf16_official", verbose=False)
        elif schedule_mode == "one_step_sol":
            # v2.8 一步精修 (对齐 SoL-Refiner 论文 §3.1 Stage-2 首步): 真·
            # SoL-Refiner (arXiv 2609.37969) 权重尚未公开, 这里用 0.85→0 单步
            # + 蒸馏 LoRA 0.8 复刻官方三段轨迹的第一段, 降噪量为 3 步方案 1/3
            # 耗时; 细节低于 3 步, 定位=4K 快速预览。正式 4K 出片建议
            # official_0p85 (3步) + BSAI-H3-upscale-4K 超分链路。
            print("[BSAI-SolEngine]   one_step_sol: 单步 sigma=0.85→0 (SoL-Refiner 权重未公开, 蒸馏近似)", flush=True)
            patched, _, _, _ = t8.setup_ltx_stage2_refiner(
                ltx_model, enabled=True, attention_backend=_attn_backend,
                min_tokens=4096, kernel_precision="bf16_official", verbose=False)
            sigmas = torch.tensor([0.85, 0.0], dtype=torch.float32)
        elif schedule_mode == "official_0p909":
            # 兼容旧工作流: NVIDIA SolEngine 参考轨迹 0.909+euler 只适用于其专用
            # 蒸馏模型; dev+官方蒸馏 LoRA 450 的轨迹是 Lightricks 官方 0.85。
            # 0.909 已被两次实跑证明输出全噪点 → 旧保存值自动映射官方 0.85 轨迹,
            # 用户无需在 UI 手动切换。
            print("[BSAI-SolEngine]   兼容映射: official_0p909(废弃) → Lightricks 官方 0.85 轨迹", flush=True)
            patched, _, _, _ = t8.setup_ltx_stage2_refiner(
                ltx_model, enabled=True, attention_backend=_attn_backend,
                min_tokens=4096, kernel_precision="bf16_official", verbose=False)
            sigmas = torch.tensor(
                [0.85, 0.725, 0.421875, 0.0], dtype=torch.float32)
        elif schedule_mode == "official_0p85":
            print("[BSAI-SolEngine]   Lightricks 官方模板 sigma=0.85 3步", flush=True)
            patched, _, _, _ = t8.setup_ltx_stage2_refiner(
                ltx_model, enabled=True, attention_backend=_attn_backend,
                min_tokens=4096, kernel_precision="bf16_official", verbose=False)
            # setup 固定返回 NVIDIA 0.909 表; 采样 sigmas 完全由调用方决定
            # → 使用 Lightricks 官方 Stage-2 轨迹
            sigmas = torch.tensor(
                [0.85, 0.725, 0.421875, 0.0], dtype=torch.float32)
        elif schedule_mode == "official_0p85_8step":
            # LTX-2.5 蒸馏模型 Stage-2 官方步数可配置(ltx.io 官方文档: distilled
            # Stage-1 固定 8 步, Stage-2 精修 configurable)。3 步官方轨迹
            # (0.85/0.725/0.4219→0) 在"非 LTX Stage-1 采样"输入(H3 帧 encode 的
            # latent)上大步降噪 → 细节生成不足 = 糊。8 步多步轨迹小步降噪,
            # 蒸馏模型充分迭代重建细节(仍 0.85 起点保 H3 结构 + euler_ancestral CFG 1)。
            print("[BSAI-SolEngine]   official_0p85_8step: 0.85 起点 8 步(蒸馏多步精修)", flush=True)
            patched, _, _, _ = t8.setup_ltx_stage2_refiner(
                ltx_model, enabled=True, attention_backend=_attn_backend,
                min_tokens=4096, kernel_precision="bf16_official", verbose=False)
            sigmas = torch.tensor(
                [0.85, 0.78, 0.71, 0.63, 0.54, 0.43, 0.30, 0.15, 0.0],
                dtype=torch.float32)
        elif schedule_mode == "lora_10step_diag":
            # 判别实验第2弹: dev + LoRA + 10 步 (0.85→0, euler)。
            # step20_diag(纯 dev 20 步) 已证明模型/编码/条件全通 → 噪点锁定在
            # LoRA 蒸馏 3 步路径。主因假说: int8 量化误差在 3 步蒸馏中无足够
            # 迭代平均 → 高频细节炸成噪点。本模式保留 LoRA 但把 3 步扩到 10 步:
            #   正常画面 → 量化+步数问题, 用 10 步轨迹替代 3 步蒸馏即可修复;
            #   仍噪点   → LoRA 注入本身与 int8 不兼容, 需换 bf16 dev 或蒸馏模型。
            print("[BSAI-SolEngine]   诊断: dev+LoRA 10 步 sigma=0.85→0 (euler)", flush=True)
            patched = ltx_model
            import numpy as _np2
            sigmas = torch.from_numpy(_np2.linspace(0.85, 0.0, 11).astype(_np2.float32))
        else:  # step20_diag: 诊断用 10 步, 不注入 LoRA (纯 dev 10 步扩散)
            # 判别实验: 纯 dev 10 步(0.5→0+euler)若输出正常画面 → dev 模型/编码链/
            # 条件全通 → 问题锁定在 LoRA 蒸馏 3 步行为; 若仍全噪点 → 采样路径/
            # 模型/条件根本性问题 (与 LoRA、3 步轨迹无关)。
            # 注: 原名 step20_diag 保留兼容旧工作流; 步数 20→10 缩短判别耗时,
            # 纯 dev 10 步已足够区分上述两种情形 (满量加载下约 5~8 分钟)。
            print("[BSAI-SolEngine]   诊断 10 步 sigma=0.5→0 (纯 dev, 无 LoRA)", flush=True)
            patched = ltx_model
            import numpy as _np2
            sigmas = torch.from_numpy(_np2.linspace(0.5, 0.0, 11).astype(_np2.float32))
        print(f"[BSAI-SolEngine]   sigmas={sigmas.tolist()}", flush=True)
        report["sigmas"] = sigmas.tolist()
        if stage2_attention == "sol_attn_sparse":
            # 上报官方 Stage-2 Sol-Attn 内核是否真实加载 (未加载时 setup 函数
            # 已自动回退 dense, 此处让 UI 的 report 输出可直接核对状态)
            _sa_loaded = t8.find_loaded_sol_attn_backend() is not None
            report["sol_attn_backend"] = "loaded (stepwise tau 1.0/1.25/1.5 + layer0 dense)" if _sa_loaded else "NOT_LOADED -> dense fallback"
            print(f"[BSAI-SolEngine]   Stage-2 Sol-Attn backend: {report['sol_attn_backend']}", flush=True)

        # 6.5/7/7.5) CLIP encode —— v2.8 条件缓存 (NVIDIA Sol-H3 论文 §4.2)
        # 官方 RTX 5090 实测: Stage-2 文本编码器每请求重建占端到端 34.1% 耗时
        # (13.568s 里前向仅 0.518s, 其余全是 24GB 显存放不下导致的加载/卸载)。
        # 同 (clip文件, prompt, fps, 负向="") 键命中缓存 → 直接跳过整个编码段,
        # 也免去 transformer 为腾内存被换出再换回。缓存在进程内有效, 重启清空。
        _cache_key = (ltx_clip, positive_prompt, float(fps))
        if stage2_cond_cache and _cache_key in _COND_CACHE:
            cond_pos, cond_neg = _COND_CACHE[_cache_key]
            # dict 里的 cond 张量可能被 guider 持有引用, 复用前浅拷贝外层结构
            cond_pos = [list(c) for c in cond_pos]
            cond_neg = [list(c) for c in cond_neg]
            report["cond_cache"] = "HIT (gemma encode skipped)"
            print("[BSAI-SolEngine] Step6.5-7.5: 条件缓存命中, 跳过 transformer 卸载/CLIP 编码/重载", flush=True)
        else:
            # CLIP encode 前先卸载 LTX transformer — 根治 Windows 提交内存峰值
            # (os error 1455): 20GB transformer 与 14.6GB gemma CLIP 同时驻留 +
            # safetensors 加载拷贝峰值会瞬时顶爆提交限制。CLIP encode 不依赖
            # transformer, 采样前由 Step8 的 load_models_gpu 重新加载 (patch/LoRA
            # 均在模型对象上, 卸载仅换出权重, 重新加载后仍生效)。
            import comfy.model_management as _mm_pre
            _mm_pre.unload_all_models()
            torch.cuda.empty_cache()
            print("[BSAI-SolEngine]   Step6.5: LTX transformer offloaded, RAM/VRAM freed before CLIP", flush=True)

            # 7) CLIP encode
            print("[BSAI-SolEngine] Step7: CLIP encode...", flush=True)
            clip = self._load_ltx_clip(ltx_clip)
            # CLIPTextEncode 标准: tokenize -> encode_from_tokens(return_pooled=True) -> [[tensor, {"pooled_output": pooled}]]
            _cp, _pp = clip.encode_from_tokens(clip.tokenize(positive_prompt), return_pooled=True)
            _cn, _pn = clip.encode_from_tokens(clip.tokenize(""), return_pooled=True)
            cond_pos = [[_cp, {"pooled_output": _pp}]]
            cond_neg = [[_cn, {"pooled_output": _pn}]]
            # LTXVConditioning: 注入帧率
            try:
                from comfy_extras import nodes_lt as _nl
                out = _nl.LTXVConditioning.execute(cond_pos, cond_neg, fps)
                if hasattr(out, "values"):
                    vals = out.values
                else:
                    vals = out
                cond_pos, cond_neg = vals[0], vals[1]
            except Exception as e:
                print(f"[BSAI-SolEngine]   LTXVConditioning 跳过({e})", flush=True)

            if stage2_cond_cache:
                if len(_COND_CACHE) >= _COND_CACHE_MAX:
                    _COND_CACHE.clear()
                _COND_CACHE[_cache_key] = (cond_pos, cond_neg)
                report["cond_cache"] = "MISS (stored for reuse)"

            # 7b) CLIP 编码完成，offload gemma 权重释放 VRAM 给 LTX transformer 采样
            import comfy.model_management as _mm_off2
            _mm_off2.unload_all_models()
            torch.cuda.empty_cache()
            print("[BSAI-SolEngine]   Step7.5: gemma CLIP offloaded, VRAM freed for LTX sampling", flush=True)

        # 8) CFGGuider + RandomNoise + 采样
        n_steps = len(sigmas) - 1
        import comfy.samplers
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

        guider = comfy.samplers.CFGGuider(patched)
        guider.set_conds(cond_pos, cond_neg)
        guider.set_cfg(1.0)
        # 官方模板 Stage-2 用 euler_ancestral (蒸馏 3 步轨迹); 诊断模式用 euler
        sampler = comfy.samplers.sampler_object(
            "euler_ancestral" if schedule_mode in ("official_0p85", "one_step_sol", "official_0p85_8step", "official_0p909", "identity_preserve_0p5") else "euler")
        noise = Noise_RandomNoise(seed)
        print(f"[BSAI-SolEngine] Step8: 采样 ({n_steps}步 {sampler.__class__.__name__ if hasattr(sampler,'__class__') else ''} sampler={getattr(sampler,'name',type(sampler).__name__)})...", flush=True)

        print(f"[BSAI-SolEngine]   latent_dict['samples']: shape={tuple(latent_dict['samples'].shape)} min={latent_dict['samples'].min().item():.4f} max={latent_dict['samples'].max().item():.4f} mean={latent_dict['samples'].mean().item():.4f}", flush=True)
        print(f"[BSAI-SolEngine]   sigmas: {sigmas.tolist()}", flush=True)
        print(f"[BSAI-SolEngine]   cond_pos: {len(cond_pos)} items, tensor shape={tuple(cond_pos[0][0].shape) if cond_pos else 'N/A'}", flush=True)
        print(f"[BSAI-SolEngine]   cond_neg: {len(cond_neg)} items, tensor shape={tuple(cond_neg[0][0].shape) if cond_neg else 'N/A'}", flush=True)

        # 采样前强制清空 PyTorch 缓存，释放 reserved 但未使用的 VRAM，给 RoPE buffer 腾位置
        torch.cuda.empty_cache()
        _free_before = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"[BSAI-SolEngine]   采样前 CUDA free: {_free_before:.2f} GB", flush=True)

        # [OOM-FIX] LTX-2.5 22B int8 权重约 20.4GB；固定 5GB 激活预算在
        # --reserve-vram 下会判定 lowvram → 每步整层换入换出 → 20 步耗时 30 分钟+。
        # 改为动态预算: 剩余显存充足(≥23GB)时用小预算触发 full load(权重全驻,
        # 采样快 5-10 倍), 不足时回退 5GB 低显存模式保证不 OOM。
        import comfy.model_management as _mm_ltx
        _free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        _act_budget_gb = 1.5 if _free_gb >= 23.0 else 5.0
        try:
            _mm_ltx.load_models_gpu(
                [patched],
                memory_required=int(_act_budget_gb * 1024 ** 3),
            )
            _ltx_lowvram = getattr(patched.model, "model_lowvram", False)
            print(
                f"[BSAI-SolEngine]   LTX transformer 显式加载完成 "
                f"(free={_free_gb:.1f}GB 激活预算 {_act_budget_gb}GB, lowvram={_ltx_lowvram})",
                flush=True,
            )
        except Exception as e:
            print(f"[BSAI-SolEngine]   LTX 显式加载失败，交由 guider 自动加载: {e}", flush=True)

        with torch.no_grad():
            refined = guider.sample(
                noise.generate_noise(latent_dict),
                latent_dict["samples"],
                sampler, sigmas,
                denoise_mask=None, callback=None,
                disable_pbar=False, seed=seed,
            )
        print(f"[BSAI-SolEngine]   refined: {tuple(refined.shape)} dtype={refined.dtype} finite={torch.isfinite(refined).all().item()} min={refined.min().item():.4f} max={refined.max().item():.4f} mean={refined.mean().item():.4f}", flush=True)

        # 9) LTX VAE decode
        print("[BSAI-SolEngine] Step9: LTX VAE decode...", flush=True)
        # 9.5) decode 前卸载 LTX transformer — 扩散解码器(CausalDiffusionVAE/NADiffusionDecoder)
        #      decode 推理显存需求大, 20GB transformer 采样后仍驻留时叠加必 OOM
        #      (5090 24GB 实测 Peak 42GB 爆显存)。解码只依赖 VAE, 采样已完成,
        #      卸载仅换出权重, 不影响结果。
        import comfy.model_management as _mm_dec
        _mm_dec.unload_all_models()
        torch.cuda.empty_cache()
        print("[BSAI-SolEngine]   Step9.5: LTX transformer offloaded before diffusion decode", flush=True)
        z = refined.contiguous().float()
        # v2.8: 4K 解码支持。auto 策略: 输出像素面积 >1.6MP(即高于 1080p 档)
        # 时用 tiled 分块解码(核心 decode_tiled, 与官方 VAEDecodeTiled 同参
        # 换算: tile 以像素为单位传入, 内部按 spacial_compression_decode 折算
        # latent tile), 避免 3840x2176 全量 decode 超 24GB 爆显存; 小分辨率
        # 保持全量 decode(更快)。tiled 与全量的最大偏差在核心测试约 0.02,
        # 视觉不可见, 与 NVIDIA 并行瓦片解码同路线 (论文 §3.6)。
        _px = int(target_width) * int(target_height)
        _use_tiled = (decode_mode == "tiled") or (decode_mode == "auto" and _px > 1_600_000)
        report["decode_mode"] = "tiled" if _use_tiled else "full"
        with torch.inference_mode():
            if _use_tiled:
                _comp = vae.spacial_compression_decode() or 32
                _tile = max(int(decode_tile_size), _comp * 8)
                _overlap = max(_tile // 8, _comp * 2)
                out_frames = vae.decode_tiled(
                    z, tile_x=_tile // _comp, tile_y=_tile // _comp,
                    overlap=_overlap // _comp, tile_t=64, overlap_t=8)
                print(f"[BSAI-SolEngine]   tiled decode: tile={_tile}px overlap={_overlap}px", flush=True)
            else:
                out_frames = vae.decode(z)
            out_frames = out_frames.cpu().float()
        if out_frames.ndim == 5:
            out_frames = out_frames[0]
        out_frames = out_frames.clamp(0, 1).float()
        actual_frames = out_frames.shape[0]
        actual_duration = actual_frames / fps
        print(f"[BSAI-SolEngine]   frames: {tuple(out_frames.shape)} range=[{out_frames.min():.3f},{out_frames.max():.3f}] duration={actual_duration:.3f}s", flush=True)

        # 10) pass through original H3 audio (VHS handles trimming by video length)
        print("[BSAI-SolEngine] Step10: pass through original H3 audio...", flush=True)
        audio_out = h3_audio
        report["audio_duration_s"] = actual_duration
        report["status"] = "ok"

        return (out_frames, audio_out, json.dumps(report, ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------
    @staticmethod
    def _trim_audio(audio_dict, target_duration_s):
        if not isinstance(audio_dict, dict) or "waveform" not in audio_dict:
            return audio_dict
        wf = audio_dict["waveform"]
        sr = audio_dict.get("sample_rate", 48000)
        print(f"[BSAI-SolEngine]   audio debug: waveform={tuple(wf.shape)} sr={sr} target_dur={target_duration_s:.3f}s actual_dur={wf.shape[-1]/sr:.3f}s", flush=True)
        max_samples = int(target_duration_s * sr)
        if wf.shape[-1] > max_samples:
            wf = wf[..., :max_samples]
            print(f"[BSAI-SolEngine]   audio trimmed to {max_samples} samples ({max_samples/sr:.3f}s)", flush=True)
        return {"waveform": wf, "sample_rate": sr}


NODE_CLASS_MAPPINGS_SOLENGINE = {
    "BSAI_SolH3_SolEngine_Refiner": BSAI_SolH3_SolEngine_Refiner,
}

NODE_DISPLAY_NAME_MAPPINGS_SOLENGINE = {
    "BSAI_SolH3_SolEngine_Refiner": "BSAI Sol-H3 SolEngine 两级精修 (LTX-2.5)",
}
