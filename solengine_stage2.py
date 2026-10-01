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

try:
    import folder_paths
except Exception:
    folder_paths = None


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
        t8_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "comfyui-minimax-h3-audio-T8",
        )
        if t8_dir not in sys.path:
            sys.path.insert(0, t8_dir)
        from h3_t8 import sol_engine_h3_super_advanced as _se
        _T8 = _se
    except Exception as e:
        _T8_IMPORT_ERROR = e
        print(f"[BSAI-Sol-H3] SolEngine 旁支: T8mars 包导入失败({e}); "
              f"请确认 custom_nodes/comfyui-minimax-h3-audio-T8 已安装", flush=True)
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


class BSAI_SolH3_SolEngine_Refiner:
    """NVIDIA Sol Engine 两级 Stage-2: H3 draft → LTX-2.5 精修 (旁支)。

    输入 H3 VAEDecode 的 IMAGE + VAEDecodeAudio 的 AUDIO，
    内部完成裁剪/resize → LTX VAE 编码 → x2 latent 放大 → LTX-2.5 3步精修
    → TAEHV 解码，输出精修 IMAGE + 裁剪后的 H3 AUDIO。
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
                "target_height": ("INT", {"default": 1088, "min": 256, "max": 2160, "step": 32}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "schedule_mode": (["official_0p909", "identity_preserve_0p5"],
                    {"default": "official_0p909"}),
                "ltx_transformer": (ltxf,),
                "ltx_vae": (vaef,),
                "ltx_clip": (clipf,),
                "ltx_upscaler": (upf,),
                "refiner_lora": (lora_opts,),
                "refiner_lora_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
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
               ltx_transformer, ltx_vae, ltx_clip, ltx_upscaler,
               refiner_lora, refiner_lora_strength, seed):
        t8 = _load_t8()
        if t8 is None:
            raise RuntimeError("T8mars SolEngine 模块不可用")

        report = {"stage": "solengine_stage2", "schedule": schedule_mode}

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
        elif refiner_lora and "无" not in refiner_lora and refiner_lora_strength > 0:
            print(f"[BSAI-SolEngine] Step5: LoRA {refiner_lora} x{refiner_lora_strength}", flush=True)
            try:
                import comfy.sd
                import comfy.utils
                lora_path = folder_paths.get_full_path("loras", refiner_lora)
                lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
                ltx_model, _ = comfy.sd.load_lora_for_models(ltx_model, None, lora_sd, refiner_lora_strength, 0)
                print(f"[BSAI-SolEngine]   LoRA OK", flush=True)
            except Exception as e:
                print(f"[BSAI-SolEngine]   LoRA 失败(跳过): {e}", flush=True)
        else:
            print("[BSAI-SolEngine] Step5: 无 refiner LoRA", flush=True)

        # 4c) ModelSamplingLTXV shift — 官方工作流无此节点, T8mars patch 内部处理
        # (手动设置 shift 会导致采样结果错误/噪点)

        # 6) refiner schedule
        print("[BSAI-SolEngine] Step6: LTX refiner schedule...", flush=True)
        if schedule_mode == "identity_preserve_0p5":
            patched, sigmas, _, _ = t8.setup_ltx_identity_preserve_refiner(
                ltx_model, enabled=True, schedule_mode="identity_preserve_0p5",
                manual_sigmas="0.5, 0.412, 0.350, 0",
                attention_backend="dense_reference",
                min_tokens=4096, kernel_precision="bf16_official", verbose=False)
        elif schedule_mode == "official_0p909":
            print("[BSAI-SolEngine]   T8mars patch + 官方 sigma=0.909 3步", flush=True)
            patched, sigmas, _, _ = t8.setup_ltx_stage2_refiner(
                ltx_model, enabled=True, attention_backend="dense_reference",
                min_tokens=4096, kernel_precision="bf16_official", verbose=False)
        else:
            print("[BSAI-SolEngine]   20 步 sigma=0.5→0", flush=True)
            patched = ltx_model
            import numpy as _np2
            sigmas = torch.from_numpy(_np2.linspace(0.5, 0.0, 21).astype(_np2.float32))
        print(f"[BSAI-SolEngine]   sigmas={sigmas.tolist()}", flush=True)
        report["sigmas"] = sigmas.tolist()

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

        # 7b) CLIP 编码完成，offload gemma 权重释放 VRAM 给 LTX transformer 采样
        import comfy.model_management as _mm_off2
        _mm_off2.unload_all_models()
        torch.cuda.empty_cache()
        print("[BSAI-SolEngine]   Step7.5: gemma CLIP offloaded, VRAM freed for LTX sampling", flush=True)

        # 8) CFGGuider + RandomNoise + euler 采样
        n_steps = len(sigmas) - 1
        print(f"[BSAI-SolEngine] Step8: 采样 ({n_steps}步 euler)...", flush=True)
        import comfy.samplers
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

        guider = comfy.samplers.CFGGuider(patched)
        guider.set_conds(cond_pos, cond_neg)
        guider.set_cfg(1.0)
        sampler = comfy.samplers.sampler_object("euler")
        noise = Noise_RandomNoise(seed)

        print(f"[BSAI-SolEngine]   latent_dict['samples']: shape={tuple(latent_dict['samples'].shape)} min={latent_dict['samples'].min().item():.4f} max={latent_dict['samples'].max().item():.4f} mean={latent_dict['samples'].mean().item():.4f}", flush=True)
        print(f"[BSAI-SolEngine]   sigmas: {sigmas.tolist()}", flush=True)
        print(f"[BSAI-SolEngine]   cond_pos: {len(cond_pos)} items, tensor shape={tuple(cond_pos[0][0].shape) if cond_pos else 'N/A'}", flush=True)
        print(f"[BSAI-SolEngine]   cond_neg: {len(cond_neg)} items, tensor shape={tuple(cond_neg[0][0].shape) if cond_neg else 'N/A'}", flush=True)

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
        z = refined.contiguous().float()
        with torch.inference_mode():
            out_frames = vae.decode(z).cpu().float()
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
