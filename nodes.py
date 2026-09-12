# -*- coding: utf-8 -*-
"""
BSAI ComfyUI Sol-H3 — 统一高速MiniMax H3推理插件
内置 FastH3 4步蒸馏参数 + Sol-Attn 稀疏注意力

v2.4 (2026-09-09) 变更:
- 【双采 Self-Lift 支持】Loader 的 FastH3 LoRA 改为可配置:
  新增 lora_name / lora_strength 参数(默认保持 FastH3-4step-LoRA.safetensors / 1.0),
  双采工作流可将一采设为 8 步 LoRA(权重约0.75), 二采用内置 LoraLoaderModelOnly
  叠加 4 步 LoRA(权重约0.7)。旧工作流加载不受影响(新参数在末尾, 自动补默认值)。
- 【双采 Self-Lift 支持】新增 BSAI_SolH3_LatentUpscaleAlign 节点:
  latent 直接放大(不经过 VAE 编解码, 无画质损失) + 像素32倍数对齐
  (H3 官方分辨率步进, 避免官方节点不取整导致的边缘色条) + 可选 CONST 重加噪
  (音视频 latent 分离, 视频重噪到 sigmas[0], 音频默认锁定), 输出可直接接
  SamplerCustomAdvanced + DisableNoise 完成二采精修。

v2.3 (2026-09-09) 变更:
- 【音频解码修复】修复 Loader 覆盖模型默认 audio_shift 的 bug:
  ms.set_parameters(shift=shift) 会把采样调度里 H3 官方默认 audio_shift=3.0 覆盖为 None,
  导致音频流失去独立缩放、按视频 schedule 采样 -> 音频轨迹错位/声音怪异。
  现改为 set_parameters(shift=shift, audio_shift=audio_shift), 并新增可调 audio_shift 参数。
- 【音频解码修复】新增 AudioVAE 全量加载补丁(对应 ComfyUI 官方 PR #15371):
  ComfyUI 核心 MiniMaxH3AudioVAE 分支未设 disable_offload=True, 577MB 的音频 VAE 走了
  DynamicVRAM 权重量流路径(日志 "prepared for dynamic VRAM loading. 576MB Staged"),
  每帧解码反复 offload/reload 权重 -> 5秒音频解码约需153秒, 且流式数值不稳定导致音频怪异。
  插件在加载时对 comfy.sd.VAE.__init__ 打内存补丁: 检测到 H3 AudioVAE 即置
  disable_offload=True(全量加载, 0.45s 解码, 577MB 常驻显存)。不改核心文件, 装插件即生效。

v2.2 (2026-09-09) 变更:
- Loader 模块加载改为「内置副本优先」, 根治其他电脑根目录旧版 sol_attn_minimax_v2.py
  旧 API (max_blocks) 污染导致的报错。

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

# 可选: 3D latent upscaler 模型支持(Comfyui_Minimax_h3_latent_Upscaler 插件)
_UPSCALER_MOD = None
try:
    import sys as _sys, os as _os
    _up_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                            "Comfyui_Minimax_h3_latent_Upscaler", "nodes")
    if _os.path.isdir(_up_dir) and _up_dir not in _sys.path:
        _sys.path.insert(0, _up_dir)
    import minimax_h3_latent_upscaler_3d as _UPSCALER_MOD
    # 把 h3_latent_upscalers 目录也注册为 latent_upscale_models 的搜索路径
    try:
        import folder_paths as _fp
        _extra_dir = _os.path.join(_fp.models_dir, "h3_latent_upscalers")
        if _os.path.isdir(_extra_dir):
            _fp.add_model_folder_path("latent_upscale_models", _extra_dir)
    except Exception:
        pass
except Exception:
    _UPSCALER_MOD = None


def _get_diffusion_models():
    try:
        if folder_paths:
            return folder_paths.get_filename_list("diffusion_models")
    except Exception:
        pass
    return ["model.safetensors"]


def _get_upscaler_models():
    """获取可用的3D upscaler模型列表; 不可用时返回仅bilinear选项。"""
    opts = ["(bilinear插值, 无需模型)"]
    if _UPSCALER_MOD is not None:
        try:
            names = _UPSCALER_MOD.scan_models()
            names = [n for n in names if not n.startswith("(")]
            opts.extend(sorted(names))
        except Exception:
            pass
    return opts


def _get_loras():
    try:
        if folder_paths:
            return folder_paths.get_filename_list("loras")
    except Exception:
        pass
    return []


def _load_sol_attn_module():
    """加载 Sol-Attn 补丁模块:
    始终使用本插件内置副本(BSAI-ComfyUI-Sol-H3/sol_attn_minimax_v2.py),
    不依赖 custom_nodes 根目录的顶层 sol_attn_minimax_v2.py —— 避免旧版
    顶层文件(旧 API max_blocks)污染新插件, 保证其他电脑安装后开箱即用。
    """
    import importlib.util
    _f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sol_attn_minimax_v2.py")
    _spec = importlib.util.spec_from_file_location("bsai_solh3_solattn", _f)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod


# ============================================================================
# FastVideo FastH3 LoRA → ComfyUI MiniMax-H3 key 转换加载
# ----------------------------------------------------------------------------
# 背景: 官方 FastH3-4step-LoRA.safetensors 是 FastVideo 训练格式
#   (transformer_blocks.N.attn.to_q/to_k/to_v + ff.net.0.proj/ff.net.2 +
#    .diff/.diff_b delta 权重), 而 ComfyUI 0.35 的 MiniMax-H3 模型实现是
#   融合结构 (blocks.N.attn.qkv_proj + mlp.fc1/fc2), 两者 key 完全不对应,
#   直接 add_patches 必然 0 匹配 (ComfyUI 官方 lora.py 无 minimax key 转换)。
# 本转换器把 FastVideo 格式逐类映射到 ComfyUI 结构, 并通过 ComfyUI 官方
#   量化感知 patch 路径 (cast_bias_weight 内 dequantize → calculate_weight)
#   在 int8/convrot 量化模型上也能真正生效。
# ============================================================================

def _load_torch_file_safe(path):
    """load_torch_file 的兼容封装: 部分 FastVideo 导出的 LoRA 文件 header 声明的
    数据长度与实际文件大小不一致, 新版 safetensors 严格校验会报
    'incomplete metadata, file not fully covered', 此时回退手动解析(不校验覆盖)。"""
    try:
        import comfy.utils
        return comfy.utils.load_torch_file(path, safe_load=True)
    except Exception:
        pass
    import struct, json
    import torch
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
        sd = {}
        _dtmap = {'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16,
                  'I8': torch.int8, 'I32': torch.int32, 'I64': torch.int64}
        for k, v in hdr.items():
            if k == '__metadata__':
                continue
            begin, end = v['data_offsets']
            f.seek(8 + n + begin)
            raw = f.read(end - begin)
            dt = _dtmap.get(v.get('dtype'), torch.float32)
            sd[k] = torch.frombuffer(raw, dtype=dt).reshape(v['shape'])
    return sd


def _fv_key_to_comfy(key):
    """FastVideo 模块 key → ComfyUI MiniMax-H3 模块 key (不含 .weight/.bias)。

    注意替换顺序: 先处理含 'audio_proj' 的长名, 再处理 'proj_in/proj_out',
    避免子串误替换。
    """
    k = key
    k = k.replace('transformer_blocks.', 'blocks.')
    k = k.replace('token_refiner.refiner_blocks.', 'token_refiner.blocks.')
    k = k.replace('.attn.to_out.0', '.attn.out_proj')
    k = k.replace('.ff.net.0.proj', '.mlp.fc1')
    k = k.replace('.ff.net.2', '.mlp.fc2')
    # 外围 delta 模块映射 (长名优先)
    k = k.replace('audio_proj_in', 'audio_patch_proj')
    k = k.replace('audio_proj_out', 'final_layer.audio_out')
    k = k.replace('proj_in', 'video_patch_proj')
    k = k.replace('proj_out', 'final_layer.video_out')
    k = k.replace('context_embedder', 'condition_proj')
    k = k.replace('time_embedder.linear_1', 'time_embedder.proj_in')
    k = k.replace('time_embedder.linear_2', 'time_embedder.proj_out')
    k = k.replace('norm_out.norm', 'final_layer.norm')
    k = k.replace('norm_out.linear', 'final_layer.adaln_proj.linear')
    return k


def _fastvideo_lora_to_comfy_patches(sd, rank=64):
    """把 FastVideo 格式 FastH3 LoRA state dict 转换为 ComfyUI patch dict。

    返回 (lora_patches, delta_patches, stats):
      lora_patches:  {模型key: (lora_A, lora_B, alpha)}  —— 标准 LoRA 补丁
      delta_patches: {模型key: ('diff', (diff,))}        —— delta 直写补丁
      stats: {'pairs': n, 'delta': n, 'skipped': [...]}
    """
    import torch
    from comfy.weight_adapter.lora import LoRAAdapter
    lora_patches = {}
    delta_patches = {}
    stats = {'pairs': 0, 'delta': 0, 'skipped': []}

    # ---- 1) 收集标准 lora_A/lora_B 对 ----
    ab = {}
    for k, v in sd.items():
        if k == '__metadata__':
            continue
        if k.endswith('.lora_A.weight'):
            base = k[:-len('.lora_A.weight')]
            bk = base + '.lora_B.weight'
            if bk in sd:
                ab[base] = (v, sd[bk])

    # ---- 2) qkv 融合: to_q/to_k/to_v 三元组 → qkv_proj 块对角 ----
    trio_groups = {}
    for base in list(ab.keys()):
        for role in ('.attn.to_q', '.attn.to_k', '.attn.to_v'):
            if base.endswith(role):
                attn_prefix = base[:base.rfind('.')]
                trio_groups.setdefault(attn_prefix, {})[role.rsplit('.', 1)[-1]] = ab.pop(base)
                break
    for attn_prefix, trio in trio_groups.items():
        if all(r in trio for r in ('to_q', 'to_k', 'to_v')):
            A_q, B_q = trio['to_q']; A_k, B_k = trio['to_k']; A_v, B_v = trio['to_v']
            r = A_q.shape[0]
            hidden = A_q.shape[1]
            inner = B_q.shape[0]
            # 块对角 A: [3r, hidden]; 块对角 B: [3*inner, 3r]
            A = torch.zeros((3 * r, hidden), dtype=A_q.dtype, device=A_q.device)
            A[:r] = A_q; A[r:2*r] = A_k; A[2*r:] = A_v
            B = torch.zeros((3 * inner, 3 * r), dtype=B_q.dtype, device=B_q.device)
            B[:inner, :r] = B_q; B[inner:2*inner, r:2*r] = B_k; B[2*inner:, 2*r:] = B_v
            target = 'diffusion_model.' + _fv_key_to_comfy(attn_prefix) + '.qkv_proj.weight'
            lora_patches[target] = LoRAAdapter(None, (B, A, float(rank), None, None, None))
            stats['pairs'] += 1
        else:
            for role, val in trio.items():
                ab[attn_prefix + '.attn.' + role] = val

    # ---- 3) 1:1 标准 LoRA ----
    for base, (A, B) in ab.items():
        target = 'diffusion_model.' + _fv_key_to_comfy(base) + '.weight'
        lora_patches[target] = LoRAAdapter(None, (B, A, float(rank), None, None, None))
        stats['pairs'] += 1

    # ---- 4) delta (.diff/.diff_b) ----
    for k, v in sd.items():
        if k == '__metadata__':
            continue
        if k.endswith('.diff_b'):
            base = k[:-len('.diff_b')]
            target = 'diffusion_model.' + _fv_key_to_comfy(base) + '.bias'
            delta_patches[target] = ('diff', (v,))
            stats['delta'] += 1
        elif k.endswith('.diff'):
            base = k[:-len('.diff')]
            target = 'diffusion_model.' + _fv_key_to_comfy(base) + '.weight'
            delta_patches[target] = ('diff', (v,))
            stats['delta'] += 1

    return lora_patches, delta_patches, stats


class BSAI_SolH3_Loader:
    """BSAI Sol-H3 一键加载器: 加载H3模型 + FastH3 LoRA + Sol-Attn(全套精细参数)"""

    @classmethod
    def INPUT_TYPES(cls):
        _loras = _get_loras()
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
                "audio_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.5}),
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
                # ---- 双采 Self-Lift: LoRA 可配置(一采 8步LoRA / 二采 4步LoRA) ----
                "lora_name": ((_loras if _loras else ["FastH3-4step-LoRA.safetensors"]),
                              {"default": "FastH3-4step-LoRA.safetensors"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("model", "steps", "cfg", "sol_info")
    FUNCTION = "load"
    CATEGORY = "BSAI/Sol-H3"
    DESCRIPTION = "BSAI Sol-H3: 一键加载H3+FastH3 LoRA+Sol-Attn(含双采双LoRA可配)"

    def load(self, model_name, precision, sol_attn, tau_start, tau_end,
             sink_conditioning, int8_qk, fused_modulation, chunk_ff, chunk_size,
             fast_h3_steps, sampler, cfg, shift, audio_shift,
             min_tokens, sol_tau, start_percent, end_percent,
             morton, morton_curve, centroid_tail, routed_cap_percent,
             reuse_qkv_memory, verbose, dense_blocks, tau_profile,
             lora_name="FastH3-4step-LoRA.safetensors", lora_strength=1.0):
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

        # 设置 flow matching shift (视频=shift, 音频=audio_shift, 默认对齐官方 12.0/3.0 的独立时钟)
        # 注意: set_parameters 必须同时传 audio_shift —— 只传 shift 会把模型配置里官方默认的
        # audio_shift=3.0 覆盖为 None, 音频流失去独立缩放(按视频 schedule 采样), 导致音频怪异。
        try:
            ms = model.get_model_object("model_sampling")
            if hasattr(ms, 'set_parameters'):
                ms.set_parameters(shift=shift, audio_shift=audio_shift)
            to = model.model_options.get("transformer_options", {})
            to["minimax_h3_sigma_shift_video"] = shift
            to["minimax_h3_sigma_shift_audio"] = audio_shift
            model.model_options["transformer_options"] = to
        except Exception as e:
            print(f"[BSAI-Sol-H3] shift设置跳过: {e}", flush=True)

        # 应用 FastH3 LoRA (lora_name/lora_strength 可配, 双采时一采可改用 8 步 LoRA)
        lora_state = "OFF"
        if lora_name and lora_name.strip() and lora_name.strip().lower() != "none":
            lora_path = folder_paths.get_full_path("loras", lora_name) if folder_paths else lora_name
            if lora_path and os.path.exists(lora_path):
                try:
                    sd = _load_torch_file_safe(lora_path)

                    # v2.6: FastVideo FastH3 LoRA → ComfyUI MiniMax-H3 结构转换加载
                    # (to_q/to_k/to_v → qkv_proj 块对角, ff.net → mlp.fc,
                    #  .diff/.diff_b delta → ('diff', ...) 量化感知 patch)
                    lora_patches, delta_patches, lstats = _fastvideo_lora_to_comfy_patches(sd)

                    # 形状过滤: curve 版模型(adaln [*,8]/无 time_embedder)与全宽度
                    # LoRA 部分不兼容, 跳过形状不匹配的 patch, 避免 calculate_weight
                    # 的形状警告与失败; 其余 patch 照常生效。
                    def _shape_filter(patches, exp_shape_fn):
                        keep = {}
                        dropped = 0
                        model_sd = model.model_state_dict()
                        for k, v in patches.items():
                            exp = exp_shape_fn(v)
                            if k in model_sd and tuple(exp) == tuple(model_sd[k].shape):
                                keep[k] = v
                            else:
                                dropped += 1
                        return keep, dropped
                    lora_patches, lora_dropped = _shape_filter(
                        lora_patches, lambda v: (v.weights[0].shape[0], v.weights[1].shape[1]))
                    delta_patches, delta_dropped = _shape_filter(
                        delta_patches, lambda v: tuple(v[1][0].shape))

                    n_lora = 0
                    if lora_patches:
                        n_lora = len(model.add_patches(lora_patches, strength_patch=lora_strength, strength_model=1.0))
                    n_delta = 0
                    if delta_patches:
                        n_delta = len(model.add_patches(delta_patches, strength_patch=lora_strength, strength_model=1.0))

                    if n_lora or n_delta:
                        lora_state = f"{lora_name} x{lora_strength:.2f}"
                        print(f"[BSAI-Sol-H3] LoRA已加载: {lora_name} strength={lora_strength:.2f} "
                              f"(LoRA补丁:{n_lora}/{len(lora_patches)}+跳过{lora_dropped} + delta补丁:{n_delta}/{len(delta_patches)}+跳过{delta_dropped})", flush=True)
                    else:
                        print(f"[BSAI-Sol-H3] LoRA加载失败(0个key匹配模型): {lora_name} "
                              f"(LoRA:{len(lora_patches)}对 delta:{len(delta_patches)}个)", flush=True)
                except Exception as e:
                    print(f"[BSAI-Sol-H3] LoRA加载失败: {e}", flush=True)
            else:
                print(f"[BSAI-Sol-H3] LoRA不存在, 跳过: {lora_name}", flush=True)

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
                f"shift={shift:.1f}/audio={audio_shift:.1f} | LoRA={lora_state} | "
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


# ---------------------------------------------------------------------------
# 双采 Self-Lift: Latent 放大 + 像素32倍数对齐 + 可选 CONST 重加噪
# ---------------------------------------------------------------------------
def _snap_to_multiple(value: int, multiple: int) -> int:
    """向上取整到 multiple 的倍数(至少为 multiple, 保证合法最小尺寸)。"""
    multiple = max(1, int(multiple))
    return max(multiple, ((int(value) + multiple - 1) // multiple) * multiple)


def _is_nested(samples) -> bool:
    try:
        import comfy.nested_tensor as _nt
        return isinstance(samples, _nt.NestedTensor)
    except Exception:
        return bool(getattr(samples, "is_nested", False))


def _extract_members(samples):
    """NestedTensor -> (members列表, was_nested); 普通Tensor -> ([tensor], False)。"""
    if _is_nested(samples):
        members = list(samples.unbind())
        return members, True
    if hasattr(samples, "unbind") and hasattr(samples, "tensors"):
        members = list(samples.unbind())
        if members and all(isinstance(m, type(samples[0])) for m in members):
            return members, True
    return [samples], False


def _wrap_members(members, was_nested):
    if was_nested:
        import comfy.nested_tensor as _nt
        return _nt.NestedTensor(members)
    return members[0]


def _upscale_video_latent(video, scale, align_to_px, method, upscaler_model=""):
    """视频 latent [B,C,T,H,W] 空间放大 + 像素32倍数对齐(不经过 VAE, 无编解码损失)。

    H3 官方支持的分辨率步进为 32px(1344x768 等), latent 是像素的 1/8。
    放大后先把像素目标取整到 32 的倍数, 再换算回 latent 尺寸, 避免官方 latent
    放大节点不取整导致的渲染分辨率偏移与边缘色条。

    upscaler_model: 如果选了3D upscaler模型文件名, 用神经网络语义放大;
                    否则用 bilinear 等插值方法。
    """
    import torch
    import comfy.utils

    if method == "nearest":
        method = "nearest-exact"
    if video.ndim < 4:
        raise ValueError(f"视频latent至少需要4维 [B,C,H,W], 实际 {tuple(video.shape)}")

    h_lat, w_lat = video.shape[-2], video.shape[-1]
    latent_mult = max(1, align_to_px // 8)  # 像素步进32 -> latent步进4
    new_h = _snap_to_multiple(round(h_lat * scale), latent_mult)
    new_w = _snap_to_multiple(round(w_lat * scale), latent_mult)
    if new_h == h_lat and new_w == w_lat:
        return video

    orig_dtype = video.dtype

    # 3D upscaler 模型语义放大(比插值细节更丰富)
    use_model = (upscaler_model and _UPSCALER_MOD is not None
                 and not upscaler_model.startswith("("))
    if use_model:
        try:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_dtype = torch.bfloat16
            s = video.to(device=dev, dtype=model_dtype, copy=True)
            if s.ndim == 4:
                s = s.unsqueeze(2)
            T = s.shape[2]
            model = _UPSCALER_MOD.load_model(upscaler_model, dev, "bf16")
            norm_mean, norm_std = _UPSCALER_MOD._make_norm_tensors(dev, model_dtype)
            with torch.inference_mode():
                s.sub_(norm_mean).div_(norm_std)
                out = model(s, scale=float(scale), target_size=(T, new_h, new_w))
                del s
                out.mul_(norm_std).add_(norm_mean)
            out = out.to(device="cpu", dtype=orig_dtype)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[BSAI-Sol-H3] 3D upscaler: in={tuple(video.shape)} -> out={tuple(out.shape)} "
                  f"model={upscaler_model}")
            return out
        except Exception as e:
            print(f"[BSAI-Sol-H3] 3D upscaler 失败({e}), fallback bilinear")

    # fp16/bf16 插值会量化新网格, 二采出现斑点; 统一 fp32 放大后转回原精度
    samples = video.float() if video.dtype in (torch.float16, torch.bfloat16) else video
    out = comfy.utils.common_upscale(samples, new_w, new_h, method, "disabled")
    return out.to(dtype=orig_dtype)


class BSAI_SolH3_LatentUpscaleAlign:
    """双采 Self-Lift 核心节点: latent 直接放大(不经过VAE) + 像素32倍数对齐 + CONST 重加噪。

    - 放大: 视频 latent 空间插值放大(无 VAE 编解码画质损失), 音频 latent 保持不变;
    - 对齐: 目标分辨率取整到 32px 倍数(H3 官方步进), 避免官方节点取整导致的边缘色条;
    - 重加噪: 可选把放大后的视频 latent 重噪到 sigmas[0](CONST re-noise), 音频默认锁定
      (audio_denoise<0.5 时音频零噪声+noise_mask=0), 输出可直接接
      SamplerCustomAdvanced + DisableNoise 完成二采精修(非 ancestral 采样器)。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1,
                                    "tooltip": "latent 放大倍率, 双采推荐 1.5-2.0(目标分辨率=一采x倍率)"}),
                "align_to": ("INT", {"default": 32, "min": 8, "max": 256, "step": 8,
                                     "tooltip": "像素分辨率对齐步长(H3 官方为 32px), 避免边缘色条"}),
                "method": (["nearest-exact", "area", "bilinear", "bicubic", "bislerp"],
                           {"default": "bilinear"}),
                "upscaler_model": (_get_upscaler_models(),
                                   {"default": "(bilinear插值, 无需模型)",
                                    "tooltip": "选择3D upscaler模型做神经网络语义放大(比bilinear插值细节更丰富); 选bilinear则用插值"}),
                "add_noise": ("BOOLEAN", {"default": True,
                                          "tooltip": "True=CONST 重加噪到 sigmas[0] 供二采; False=仅放大对齐"}),
                "audio_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                            "tooltip": "0=二采锁定一采音频(默认); >=0.5=音频一起重噪重绘"}),
            },
            "optional": {
                "model": ("MODEL",),
                "noise": ("NOISE",),
                "sigmas": ("SIGMAS",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "upscale_align"
    CATEGORY = "BSAI/Sol-H3"
    DESCRIPTION = "双采 Self-Lift: latent 放大(不经过VAE) + 像素32倍数对齐 + CONST 重加噪, 输出接 SamplerCustomAdvanced+DisableNoise 二采"

    def upscale_align(self, samples, scale, align_to, method, upscaler_model,
                      add_noise, audio_denoise,
                      model=None, noise=None, sigmas=None):
        out = dict(samples) if isinstance(samples, dict) else {"samples": samples}
        latent_image = out["samples"]

        # 1) 解包 NestedTensor (H3: [0]=video [B,24,T,H,W], [1]=audio [B,32,2,Ta], ...)
        members, was_nested = _extract_members(latent_image)
        if not members:
            raise ValueError("latent 为空")
        video = members[0]
        if video.ndim not in (4, 5):
            raise ValueError(f"视频latent维度异常: {tuple(video.shape)}")
        up_video = _upscale_video_latent(video, float(scale), int(align_to), method, upscaler_model)
        up_members = [up_video] + list(members[1:])
        out["samples"] = _wrap_members(up_members, was_nested)

        # 2) CONST 重加噪 (放大后视频重噪到 sigmas[0], 音频默认锁定)
        if add_noise and model is not None and noise is not None and sigmas is not None:
            out = self._add_noise(model, noise, sigmas, out, float(audio_denoise))

        return (out,)

    def _add_noise(self, model, noise, sigmas, latent, audio_denoise):
        """NestedTensor 感知的 CONST AddNoise(对齐官方 AddNoise 语义):
        process_latent_in/out 作用于整体 NestedTensor(音频走独立 audio_scale),
        在 sigmas[0] 处混合, 再 inverse_noise_scaling, 使 SamplerCustomAdvanced+DisableNoise
        以 σ·ε+(1-σ)·x 而非 (1-σ)²·x 重构。音频锁定: 音频成员噪声置零 + noise_mask 音频=0。
        """
        import torch
        if len(sigmas) == 0 or "samples" not in latent:
            return latent

        out = dict(latent)
        latent_image = latent["samples"]
        members, was_nested = _extract_members(latent_image)

        # 生成噪声
        noisy = noise.generate_noise(latent)
        n_members, n_was_nested = _extract_members(noisy)
        if len(n_members) != len(members):
            raise ValueError(f"噪声成员数 {len(n_members)} 与 latent 成员数 {len(members)} 不一致")

        lock_audio = len(members) >= 2 and float(audio_denoise) < 0.5
        if lock_audio:
            # 音频成员噪声置零(保持一采音频)
            n_members = [torch.zeros_like(t) if i == 1 else t for i, t in enumerate(n_members)]

        # noise_mask: 音频=0(保持), 视频=1(重噪); H3 joint DiT 仍需 mask 显式锁定
        if lock_audio and was_nested:
            mask_members = []
            for i, m in enumerate(members):
                if i == 0:
                    mask_members.append(torch.ones(
                        (m.shape[0], 1, m.shape[2], m.shape[3], m.shape[4]),
                        device=m.device, dtype=torch.float32))
                elif i == 1:
                    mask_members.append(torch.zeros(
                        (m.shape[0], 1, m.shape[2], m.shape[3]),
                        device=m.device, dtype=torch.float32))
                else:
                    mask_members.append(torch.ones_like(m[:1, :1]))
            out["noise_mask"] = _wrap_members(mask_members, was_nested=True)

        model_sampling = model.get_model_object("model_sampling")
        process_latent_out = model.get_model_object("process_latent_out")
        process_latent_in = model.get_model_object("process_latent_in")
        sigma_start = sigmas[0]

        latent_image = process_latent_in(latent_image)
        lat_members, _ = _extract_members(latent_image)
        mixed = []
        for lat, noi in zip(lat_members, n_members):
            m = model_sampling.noise_scaling(sigma_start, noi, lat)
            if hasattr(model_sampling, "inverse_noise_scaling"):
                m = model_sampling.inverse_noise_scaling(sigma_start, m)
            mixed.append(m)
        mixed_nt = _wrap_members(mixed, was_nested=was_nested or n_was_nested)
        mixed_nt = process_latent_out(mixed_nt)
        m_members, _ = _extract_members(mixed_nt)
        out["samples"] = _wrap_members(
            [torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0) for t in m_members],
            was_nested=was_nested)
        # debug: print final member shapes
        _final, _ = _extract_members(out["samples"])
        print(f"[BSAI-Sol-H3] output members: {[tuple(m.shape) for m in _final]}")
        if "noise_mask" in out:
            _nm, _ = _extract_members(out["noise_mask"])
            print(f"[BSAI-Sol-H3] noise_mask members: {[tuple(m.shape) for m in _nm]}")
        return out


NODE_CLASS_MAPPINGS = {
    "BSAI_SolH3_Loader": BSAI_SolH3_Loader,
    "BSAI_SolH3_Info": BSAI_SolH3_Info,
    "BSAI_SolH3_LatentUpscaleAlign": BSAI_SolH3_LatentUpscaleAlign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_SolH3_Loader": "BSAI Sol-H3 Loader (极速版)",
    "BSAI_SolH3_Info": "BSAI Sol-H3 Info",
    "BSAI_SolH3_LatentUpscaleAlign": "BSAI Sol-H3 Latent 放大对齐 (双采)",
}


def _patch_audio_vae_offload():
    """内存级补丁: 修复 ComfyUI 核心 H3 AudioVAE 的 DynamicVRAM 权重量流问题。

    官方 PR #15371 语义: MiniMax H3 音频 VAE(约577MB) 在 comfy/sd.py 的
    MiniMaxH3AudioVAE 分支未设 disable_offload=True, 继承了全局默认 False,
    导致音频 VAE 走 DynamicVRAM streaming 路径(日志: "prepared for dynamic
    VRAM loading. 576MB Staged"), 每帧解码反复 offload/reload 权重:
    - 5 秒音频解码耗时约 153 秒(全量加载仅 0.45 秒)
    - 流式数值不稳定, 解码状态异常 -> 声音怪异/失真
    本补丁不改动 ComfyUI 核心文件: 插件加载时包装 VAE.__init__, 检测到
    MiniMaxH3AudioVAE 即置 disable_offload=True, 全量加载、解码快且稳定。
    其他电脑安装本插件即自动生效, 开箱即用。
    """
    try:
        import comfy.sd as _sd
        from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
    except Exception:
        return
    if getattr(_patch_audio_vae_offload, "_applied", False):
        return
    _patch_audio_vae_offload._applied = True

    _orig_init = _sd.VAE.__init__

    def _wrapped_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        try:
            if isinstance(getattr(self, "first_stage_model", None), MiniMaxH3AudioVAE) \
                    and not getattr(self, "disable_offload", False):
                self.disable_offload = True
                print("[BSAI-Sol-H3] 音频解码修复: H3 AudioVAE 已切换全量加载"
                      "(disable_offload=True), 消除 DynamicVRAM 抖动", flush=True)
        except Exception:
            pass

    _sd.VAE.__init__ = _wrapped_init
    print("[BSAI-Sol-H3] 音频解码修复已就绪: H3 AudioVAE 将全量加载(PR #15371 语义), "
          "不再走 DynamicVRAM 权重量流", flush=True)


_patch_audio_vae_offload()
