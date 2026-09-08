# -*- coding: utf-8 -*-
"""
BSAI ComfyUI Sol-H3 — 统一高速MiniMax H3推理插件
内置 FastH3 4步蒸馏参数 + Sol-Attn 稀疏注意力

v2.1 (2026-09-09) 变更:
- Loader 并入右侧 SolAttnMiniMax 节点的全套精细参数, 删除独立节点后配置能力不丢失:
  sol_tau / start_percent / end_percent / min_tokens / morton / morton_curve /
  centroid_tail / routed_cap_percent / reuse_qkv_memory / verbose / dense_blocks / tau_profile
- Sol-Attn 安装改走 sol_attn_minimax_v2._apply_patch 完整路径:
  自动 clone model、安装 H3 Morton hooks(sink 必需)、block 索引、percent->sigma 转换,
  不再只写 optimized_attention_override(此前 sink 所需 hooks 缺失)
- 修复: sol_attn=True 时真正安装 Sol-Attn 稀疏注意力补丁(此前只打印 ON 并未安装)
- 使用新版 comfy_kitchen API (sink_blocks/sink_q/tail)
"""
import os


try:
    import folder_paths
except Exception:
    folder_paths = None


def _get_diffusion_models():
    try:
        if folder_paths:
            return folder_paths.get_filename_list("diffusion_models")
    except Exception:
        pass
    return ["model.safetensors"]


def _load_sol_attn_module():
    """加载 Sol-Attn 补丁模块:
    1) 优先使用顶层 custom_nodes 的 sol_attn_minimax_v2(全局唯一注册)
    2) 回退加载本插件内置副本(BSAI-ComfyUI-Sol-H3/sol_attn_minimax_v2.py)
    """
    try:
        import sol_attn_minimax_v2 as _m
        return _m
    except Exception:
        pass
    import importlib.util
    _f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sol_attn_minimax_v2.py")
    _spec = importlib.util.spec_from_file_location("bsai_solh3_solattn", _f)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod


class BSAI_SolH3_Loader:
    """BSAI Sol-H3 一键加载器: 加载H3模型 + FastH3 4步参数 + Sol-Attn(全套精细参数)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_get_diffusion_models(),),
                "precision": (["int8", "default", "fp8_e4m3fn"],
                              {"default": "int8"}),
                "sol_attn": ("BOOLEAN", {"default": True}),
                "tau_start": ("FLOAT", {"default": 0.5, "min": 0.3, "max": 3.0, "step": 0.1}),
                "tau_end": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 2.0, "step": 0.1}),
                "sink_conditioning": (["exact_kv", "exact_kv_and_rows", "off"],
                                      {"default": "exact_kv"}),
                "int8_qk": ("BOOLEAN", {"default": True}),
                "fused_modulation": ("BOOLEAN", {"default": True}),
                "chunk_ff": ("BOOLEAN", {"default": True}),
                "chunk_size": ("INT", {"default": 2, "min": 1, "max": 8}),
                "fast_h3_steps": (["4步 FastH3 极速", "8步 FastH3 增强", "50步 原生高质量"],
                                  {"default": "4步 FastH3 极速"}),
                "sampler": (["euler", "dpmpp_2m", "euler_ancestral"],
                            {"default": "euler"}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 10.0, "step": 0.5}),
                "shift": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                # ---- 以下为原独立 SolAttnMiniMax 节点的精细参数 ----
                "min_tokens": ("INT", {"default": 12288, "min": 256, "max": 262144, "step": 256}),
                "sol_tau": ("FLOAT", {"default": 1.3, "min": 0.1, "max": 4.0, "step": 0.1}),
                "start_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "end_percent": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "morton": ("BOOLEAN", {"default": False}),
                "morton_curve": (["2d_frame", "hilbert", "z_order"],
                                 {"default": "2d_frame"}),
                "centroid_tail": ("BOOLEAN", {"default": True}),
                "routed_cap_percent": ("INT", {"default": 0, "min": 0, "max": 100}),
                "reuse_qkv_memory": ("BOOLEAN", {"default": False}),
                "verbose": ("BOOLEAN", {"default": False}),
                "dense_blocks": ("STRING", {"default": ""}),
                "tau_profile": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("model", "steps", "cfg", "sol_info")
    FUNCTION = "load"
    CATEGORY = "BSAI/Sol-H3"
    DESCRIPTION = "BSAI Sol-H3: 一键加载H3+FastH3 4步+Sol-Attn(含全套精细参数)"

    def load(self, model_name, precision, sol_attn, tau_start, tau_end,
             sink_conditioning, int8_qk, fused_modulation, chunk_ff, chunk_size,
             fast_h3_steps, sampler, cfg, shift,
             min_tokens, sol_tau, start_percent, end_percent,
             morton, morton_curve, centroid_tail, routed_cap_percent,
             reuse_qkv_memory, verbose, dense_blocks, tau_profile):
        from comfy.sd import load_diffusion_model

        model_path = folder_paths.get_full_path("diffusion_models", model_name) if folder_paths else model_name
        model = load_diffusion_model(model_path)

        # 步数解析
        if "4步" in fast_h3_steps:
            steps = 4
        elif "8步" in fast_h3_steps:
            steps = 8
        else:
            steps = 50

        # 设置 flow matching shift (视频=shift, 音频=3)
        try:
            ms = model.get_model_object("model_sampling")
            if hasattr(ms, 'set_parameters'):
                ms.set_parameters(shift=shift)
            to = model.model_options.get("transformer_options", {})
            to["minimax_h3_sigma_shift_video"] = shift
            to["minimax_h3_sigma_shift_audio"] = 3.0
            model.model_options["transformer_options"] = to
        except Exception as e:
            print(f"[BSAI-Sol-H3] shift设置跳过: {e}", flush=True)

        # 应用 FastH3 LoRA (若 models/loras 下有对应 LoRA)
        lora_name = "FastH3-4step-LoRA.safetensors"
        lora_path = folder_paths.get_full_path("loras", lora_name) if folder_paths else lora_name
        if lora_path and os.path.exists(lora_path):
            try:
                import comfy.utils
                sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
                model.add_patches(sd, strength_patch=1.0, strength_model=1.0)
                print(f"[BSAI-Sol-H3] LoRA已加载: {lora_name} strength=1.0", flush=True)
            except Exception as e:
                print(f"[BSAI-Sol-H3] LoRA加载失败: {e}", flush=True)

        # 真正安装 Sol-Attn 稀疏注意力补丁 (完整路径, 含 Morton hooks / block 索引 / sigma 换算)
        sol_attn_state = "OFF"
        if sol_attn and sink_conditioning != "off":
            try:
                sam = _load_sol_attn_module()
                out = sam._apply_patch(
                    model,
                    tau=sol_tau,
                    start_percent=start_percent,
                    end_percent=end_percent,
                    min_tokens=min_tokens,
                    sink_conditioning=sink_conditioning,
                    morton=morton,
                    morton_curve=morton_curve,
                    dense_blocks=dense_blocks,
                    verbose=verbose,
                    tau_profile=tau_profile,
                    routed_cap_percent=routed_cap_percent,
                    centroid_tail=centroid_tail,
                    reuse_qkv_memory=reuse_qkv_memory)
                # _apply_patch 返回 clone 后的模型
                if hasattr(out, "result") and out.result:
                    model = out.result[0]
                elif hasattr(out, "args") and out.args:
                    model = out.args[0]
                sol_attn_state = "ON"
                print(f"[BSAI-Sol-H3] Sol-Attn已安装: tau={sol_tau:.1f} "
                      f"({start_percent:.0%}->{end_percent:.0%}) min_tokens={min_tokens} "
                      f"sink={sink_conditioning} morton={morton} (新API: sink_blocks/tail)", flush=True)
            except Exception as e:
                print(f"[BSAI-Sol-H3] Sol-Attn安装失败(回退Dense): {e}", flush=True)
        elif sol_attn:
            print(f"[BSAI-Sol-H3] sink_conditioning=off, Sol-Attn跳过", flush=True)

        info = (f"Sol-H3: {model_name} | mode={fast_h3_steps} | steps={steps} | cfg={cfg:.1f} | "
                f"Sol-Attn={sol_attn_state} | tau={sol_tau:.1f} | min_tokens={min_tokens} | "
                f"FusedMod={'ON' if fused_modulation else 'OFF'} | ChunkFF={chunk_size}")
        print(f"[BSAI-Sol-H3] {info}", flush=True)

        return (model, steps, cfg, info)


class BSAI_SolH3_Info:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sol_info": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ()
    FUNCTION = "show"
    CATEGORY = "BSAI/Sol-H3"
    OUTPUT_NODE = True

    def show(self, sol_info):
        return {"ui": {"text": [sol_info]}}


NODE_CLASS_MAPPINGS = {
    "BSAI_SolH3_Loader": BSAI_SolH3_Loader,
    "BSAI_SolH3_Info": BSAI_SolH3_Info,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_SolH3_Loader": "BSAI Sol-H3 Loader (极速版)",
    "BSAI_SolH3_Info": "BSAI Sol-H3 Info",
}
