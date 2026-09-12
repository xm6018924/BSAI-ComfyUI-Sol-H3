# BSAI ComfyUI Sol-H3 ｜ BSAI ComfyUI Sol-H3

**MiniMax H3 一键高速推理插件** — Sol-Attn 稀疏注意力 + FastH3 4 步蒸馏 + Fused Modulation + Chunk FeedForward + 双采 Self-Lift
**One-click high-speed inference for MiniMax H3** — Sol-Attn sparse attention + FastH3 4-step distillation + Fused Modulation + Chunk FeedForward + Dual-Sample Self-Lift

> 中文说明见下方；English documentation follows each section.
> 无需修改 ComfyUI 核心文件，clone 即用，开箱即得 4 步极速 + Sol-Attn 稀疏注意力。
> No ComfyUI core file is modified — clone and run. 4-step ultra-fast + Sol-Attn out of the box.

---

## 插件介绍 / Introduction

**BSAI Sol-H3** 是一个为 MiniMax H3（文生视频+音频联合模型）打造的"一键高速推理"加载器插件。它把四套官方/社区优化整合到一个节点里，让用户**不用自己拼装**就能直接获得 4 步蒸馏 + 稀疏注意力的极速体验：

**BSAI Sol-H3** is a one-click high-speed inference loader for the MiniMax H3 (text-to-video + audio joint model). It bundles four proven optimizations into a single node so you get 4-step distillation + sparse attention speed **without assembling anything yourself**:

| 组件 / Component | 来源 / Source | 作用 / Role |
|---|---|---|
| **Sol-Attn** | [Saganaki22/ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn) | Triton 稀疏注意力，SM86-SM121（RTX 30/40/50 系）/<br>Triton sparse attention for SM86–SM121 (RTX 30/40/50) |
| **FastH3 V1** | [Hao AI Lab (UCSD)](https://haoailab.com/blogs/fasth3-preview/) | 4 步蒸馏 adapter，约 11x 加速 /<br>4-step distilled adapter, ~11x speedup |
| **Fused Modulation** | Sol-Attn 内置 / built-in | Triton 融合 AdaLN+门控残差，bit-exact |
| **Chunk FeedForward** | Sol-Attn 内置 / built-in | MLP 显存峰值降低最多 1.9GB /<br>MLP VRAM peak reduced by up to 1.9GB |
| **双采 Self-Lift** | 本插件 / this plugin | latent 直接放大+32px 对齐+CONST 重加噪，二采精修 /<br>latent upscale + 32px align + CONST re-noise for second-pass refinement |

### 亮点 / Highlights
- **一个节点替代多个优化节点**：模型加载、LoRA 挂载、Sol-Attn 稀疏注意力安装（含 Morton hooks / block 索引 / sigma 换算）全部在 `BSAI Sol-H3 Loader` 内完成。
- **双采 Self-Lift 开箱即用**：`LatentUpscaleAlign` 不经过 VAE 编解码直接放大 latent（无画质损失），按 H3 官方 32px 步进对齐（避免边缘色条），可选 CONST 重加噪，音频默认锁定，输出可直接接 `SamplerCustomAdvanced + DisableNoise` 精修。
- **音频问题根治**：内置音频修复（`audio_shift` 独立缩放 + H3 AudioVAE 全量加载补丁，对应官方 PR #15371），5 秒音频解码从约 153 秒降到约 0.45 秒，声音不再怪异。
- **delta 权重（.diff/.diff_b）智能加载**：v2.5.4 起自动识别 delta 格式 LoRA，自动匹配模型参数 key 并反量化应用，兼容标准 LoRA 与直连 delta 两种格式。
- **3D Latent Upscaler 支持**：可选神经网络语义放大（比 bilinear 插值保留更多细节），模型文件自动搜索 `models/latent_upscale_models/` 与 `models/h3_latent_upscalers/`。

---

## 性能 / Performance

| 配置 / Configuration | 5 秒 720p 视频 / 5s 720p video | 对比 / vs baseline |
|---|---|---|
| 原生 H3（50 步）/ Native H3 (50 steps) | ~30 分钟 / ~30 min | 基准 / baseline |
| Sol-Attn（50 步）/ Sol-Attn (50 steps) | ~6.5 分钟 / ~6.5 min | ~4.5x 加速（RTX 5090）/ ~4.5x on RTX 5090 |
| **FastH3 + Sol（4 步）/ 4-step** | **~2-3 分钟 / ~2-3 min** | **~11x 加速 / ~11x** |
| NVIDIA 8xB300（4 步）/ server grade | 1.653 秒 / 1.653 s | 服务器级 / datacenter |

> 实际速度取决于 GPU、分辨率与帧数。/ Actual speed depends on GPU, resolution and frame count.

---

## 节点 / Nodes

本插件提供 3 个节点，全部位于 `BSAI/Sol-H3` 分类下 / This plugin provides 3 nodes, all under the `BSAI/Sol-H3` category:

| 节点 / Node | 作用 / Role |
|---|---|
| **BSAI Sol-H3 Loader (极速版)** | 一键加载 H3 模型 + FastH3 LoRA + Sol-Attn（全套精细参数），输出 `model / steps / cfg / sol_info`。/ One-click model + LoRA + Sol-Attn loader; outputs `model / steps / cfg / sol_info`. |
| **BSAI Sol-H3 Latent 放大对齐 (双采)** | 双采核心节点：latent 直接放大（不经过 VAE）+ 像素 32 倍数对齐 + 可选 CONST 重加噪（音频默认锁定）。/ Core dual-sample node: latent upscale (no VAE round-trip) + 32px pixel alignment + optional CONST re-noise (audio locked by default). |
| **BSAI Sol-H3 Info** | 显示 Loader 返回的优化状态信息（UI 文本）。/ Displays the optimization status string from the Loader. |

---

## 安装 / Installation

### 方式一：命令行 / Option 1: Command line
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/BSAI-ComfyUI-Sol-H3.git
cd BSAI-ComfyUI-Sol-H3
python install.py          # 自动检查 comfy_kitchen 并下载 FastH3 4步 LoRA（~300MB）
python -m pip install -r requirements.txt   # 如有依赖变更
```

### 方式二：ComfyUI Manager / Option 2: ComfyUI Manager
在 Manager 中搜索 `BSAI-ComfyUI-Sol-H3` 安装，重启 ComfyUI。/ Search `BSAI-ComfyUI-Sol-H3` in Manager, install, restart ComfyUI.

### 首次安装自动做的事 / What `install.py` does
1. 检查 `comfy_kitchen`（Sol-Attn 内核，随 ComfyUI 提供）。
2. 自动下载 `FastH3-4step-LoRA.safetensors`（~300MB）到 `models/loras/`（ModelScope 国内直连优先，HF 兜底）。
3. 启动 ComfyUI 后，日志应出现 `Sol-Attn已安装: ... (新API: sink_blocks/tail)` 且无 `kernel failed`。

---

## 使用方法 / Usage

### 基本连接（文生视频+音频）/ Basic wiring (T2V + audio)

```
BSAI Sol-H3 Loader ──model──> KSampler ──LATENT──> VAEDecode ──IMAGE──> VHS_VideoCombine
CLIPLoader ──> CLIPTextEncode(正向) ─┘              └──> VAEDecodeAudio ──AUDIO──┘
             └─> CLIPTextEncode(负向)
EmptyHunyuanLatentVideo ──> KSampler(latent_image)
VAELoader(video VAE) ──> VAEDecode
VAELoader(audio VAE) ──> VAEDecodeAudio
```

也可将 Loader 输出的 `steps`（INT）接 `EmptyHunyuanLatentVideo` 的帧数无关（帧数由 latent 决定）；`steps`/`cfg` 可直接接入 KSampler 的 `steps`/`cfg` 输入端。

> ⚠️ **重要：勿叠加其他 attention 优化节点 / IMPORTANT: do NOT stack extra attention-optimization nodes**
>
> `sol_attn=True` 时 Loader 已内置 Sol-Attn 稀疏注意力补丁。**同一条模型链上不要再挂**：
> - KJNodes `MiniMaxH3MemoryEfficientSageAttentionPatch`（int8/fp8 Sage，会在 Sol-Attn 内层重复量化，显存翻倍触发 OOM）
> - 旧版单文件节点 `SolAttnMiniMax`（`custom_nodes/sol_attn_minimax_v2.py` 旧版残留，请删除该文件，插件 v2.2+ 始终使用内置副本）
> - T8 `MiniMaxLowVRAMAttention`（另一套 attention override）
>
> 多套叠加会形成 `Sol-Attn → kitchen → Sage int8/fp8` 链式 patch，每层都分配中间张量，RTX 5090 Laptop 24GB 上主采样即 OOM。
> When `sol_attn=True`, the Loader already installs the Sol-Attn sparse-attention patch. Do **not** add any of these on the same model path: KJNodes `MiniMaxH3MemoryEfficientSageAttentionPatch` (re-quantizes inside Sol-Attn, doubles VRAM → OOM), the legacy single-file `SolAttnMiniMax` node (delete the stale `custom_nodes/sol_attn_minimax_v2.py`; v2.2+ always loads the built-in copy), or T8 `MiniMaxLowVRAMAttention`. Stacking them chains `Sol-Attn → kitchen → Sage int8/fp8`, each layer allocating intermediate tensors, and OOMs the main pass even on a 24GB RTX 5090 Laptop.
>
> 保留 `ModelAttentionBackend`（comfy kitchen attention）**不影响**——它是底层 kernel 后端，Sol-Attn 正常工作依赖它。Keep `ModelAttentionBackend` (comfy kitchen attention) — it is the underlying kernel backend Sol-Attn relies on.
>
> ❗ **更正 / Correction：`ModelAttentionBackend` 必须删除/断开，不能保留。** 该节点执行 `set_model_optimized_attention`，写入的正是 `transformer_options["optimized_attention_override"]`——与 Loader 内置 Sol-Attn **同一个槽位**；节点图按依赖顺序执行，Loader 先装 Sol-Attn，`ModelAttentionBackend` 后执行会**直接覆盖**，Sol-Attn 稀疏注意力被禁用。int8 权重的 int8_linear 走 comfy_kitchen 的张量层（ops），不需要该节点。**请删掉工作流中的 `ModelAttentionBackend`（302/264）节点，将 Loader → ChunkFF → SigmaShift 直接连到采样器**。

### Loader 参数说明 / Loader Parameters

| # | 参数 / Parameter | 默认 / Default | 说明 / Description |
|---|---|---|---|
| 1 | model_name | - | H3 模型文件（`diffusion_models` 目录）<br>H3 model file in `diffusion_models` |
| 2 | precision | int8 | 模型加载精度：int8 / default / fp8_e4m3fn |
| 3 | sol_attn | ON | 是否安装 Sol-Attn 稀疏注意力补丁<br>Install Sol-Attn sparse attention patch |
| 4 | tau_start | 0.5 | 渐变起始 tau（信息/兼容旧工作流）<br>Gradient start tau (informational) |
| 5 | tau_end | 1.0 | 渐变结束 tau（信息/兼容旧工作流）<br>Gradient end tau (informational) |
| 6 | sink_conditioning | exact_kv | 锚点策略：exact_kv / exact_kv_and_rows / off<br>Sink strategy |
| 7 | int8_qk | ON | INT8 QK 量化 / INT8 QK quantization |
| 8 | fused_modulation | ON | 融合调制优化 / Fused modulation |
| 9 | chunk_ff | ON | FFN 分块显存优化 / Chunked FFN memory saving |
| 10 | chunk_size | 2 | FFN 分块大小 1-8 / FFN chunk size |
| 11 | fast_h3_steps | 4步 FastH3 极速 | 4步 FastH3 极速 / 8步 FastH3 增强 / 50步 原生高质量<br>4-step Ultra / 8-step Enhanced / 50-step Native |
| 12 | sampler | euler | 采样器：euler / dpmpp_2m / euler_ancestral |
| 13 | cfg | 4.0 | 提示词引导强度 / CFG guidance |
| 14 | shift | 8.0 | 视频流 flow matching 时间偏移（官方默认 12.0）<br>Video flow-matching shift (official default 12.0) |
| 15 | **audio_shift** | 3.0 | 音频流独立时间偏移（官方默认 3.0；修音频怪异核心参数）<br>Audio shift — fixes weird audio |
| 16 | **min_tokens** | 12288 | 序列长度阈值：低于此长度用 Dense（省显存/保质量）<br>Below this token count, attention stays dense |
| 17 | **sol_tau** | 1.3 | Sol-Attn 稀疏强度（越大越稀疏、越省显存）<br>Sol-Attn sparsity strength |
| 18 | **start_percent** | 0.2 | 稀疏生效起始采样比例（前 20% 步数 Dense 热身）<br>Sparse starts after 20% of steps (dense warmup) |
| 19 | **end_percent** | 0.9 | 稀疏生效结束采样比例（最后 10% 步数 Dense 收尾）<br>Sparse ends at 90% (dense tail) |
| 20 | **morton** | OFF | Morton 空间排序（提升稀疏命中率）<br>Morton space-filling ordering |
| 21 | **morton_curve** | 2d_frame | 排序曲线：2d_frame / hilbert / z_order |
| 22 | **centroid_tail** | ON | 质心尾块保留（提升长序列质量）<br>Keep centroid tail blocks |
| 23 | **routed_cap_percent** | 0 | 路由容量上限百分比（0=不限制）<br>Routing capacity cap % (0 = unlimited) |
| 24 | **reuse_qkv_memory** | OFF | 复用 QKV 中间缓冲（仅限已知安全模型）<br>Reuse QKV intermediate buffers |
| 25 | **verbose** | OFF | Sol-Attn 详细日志 / Verbose logging |
| 26 | **dense_blocks** | 空 | 强制全 Dense 的块索引，如 `0,1,2` / Force-dense block list |
| 27 | **tau_profile** | 空 | 逐块 tau 覆盖，如 `0-4=2.0; 20-24=0.8`（分号/换行分隔）<br>Per-block tau overrides |
| 28 | **lora_name** | FastH3-4step-LoRA.safetensors | 主 LoRA（双采一采可改 8 步 LoRA 文件名，权重约 0.75）<br>Main LoRA (use an 8-step LoRA at ~0.75 for first pass) |
| 29 | **lora_strength** | 1.0 | 主 LoRA 强度（0=禁用 LoRA）<br>LoRA strength (0 = disable) |

> **delta 权重说明 / Delta weights**: v2.5.4 起 `lora_name` 同时支持标准 LoRA（`lora_A`/`lora_B`）与直连 delta 权重（`.diff`/`.diff_b` 后缀，按模型 key 自动匹配反量化应用）。/ Since v2.5.4 both standard LoRA and raw delta weights (`.diff`/`.diff_b`) are auto-detected and applied.

### LatentUpscaleAlign 参数 / LatentUpscaleAlign Parameters

| 参数 / Parameter | 默认 / Default | 说明 / Description |
|---|---|---|
| samples | LATENT | 一采输出的 latent（音视频 NestedTensor）|
| scale | 2.0 | latent 放大倍率，双采推荐 1.5-2.0 |
| align_to | 32 | 像素分辨率对齐步长（H3 官方 32px，避免边缘色条）|
| method | bilinear | 放大方法：**必须 bilinear**（nearest-exact 会导致 latent 块化→二采鬼影）|
| upscaler_model | (bilinear插值) | 可选 3D latent upscaler 模型做神经网络语义放大 |
| add_noise | True | True=CONST 重加噪到 sigmas[0] 供二采；False=仅放大对齐 |
| audio_denoise | 0.0 | 0=二采锁定一采音频（默认）；≥0.5=音频一起重噪重绘 |

---

## 示例工作流 / Example Workflows

仓库 `workflows/` 目录提供 2 个可直接运行的示例（**纯 ComfyUI 核心节点 + 本插件，不依赖任何第三方插件**）。/ Two ready-to-run examples in `workflows/` (pure ComfyUI core nodes + this plugin, no third-party dependency):

### 工作流 1：`SolH3_Fast_4step.json` — 基础 4 步极速（文生视频+音频）/ Basic 4-step Ultra (T2V + Audio)

**节点链 / Node chain**:
```
BSAI Sol-H3 Loader ──model──> KSampler ──LATENT──> VAEDecode ──IMAGE──> VHS_VideoCombine
CLIPLoader ──> 2×CLIPTextEncode (正/负)                └──> VAEDecodeAudio ──AUDIO──┘
EmptyHunyuanLatentVideo (1344×768×24) ──> KSampler
VAELoader×2 (video VAE + audio VAE)
```

**使用步骤 / How to use**:
1. 把 `workflows/SolH3_Fast_4step.json` 拖入 ComfyUI 画布。
2. **Loader 节点**：确认 `model_name` 指向你本机的 H3 模型文件；`fast_h3_steps` 保持「4步 FastH3 极速」；精度 `int8`（显存紧张）或 `default`。
3. **CLIPLoader**：选择你的 H3 CLIP 文本模型（如 `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`）。
4. **EmptyHunyuanLatentVideo**：`width/height/length` 默认 1344×768×24（5 秒 @24fps），可改 672×384 快速测试。
5. **两个 VAELoader**：分别选视频 VAE（`minimax_h3_video_vae_fp16.safetensors`）与音频 VAE（`minimax_h3_audio_vae_fp32.safetensors`）。
6. **CLIPTextEncode**：正向框写你的提示词（含画面+音效描述，如示例的雨夜霓虹小巷）；负向框保持默认负面词。
7. 点「运行」→ 视频+音频输出到 `ComfyUI/output/`。

**参数建议 / Tips**: 先用 672×384×16 测试；确认无问题再上 1344×768×24。`shift=8.0`（视频）、`audio_shift=3.0`（音频）为推荐默认值。

### 工作流 2：`SolH3_Self-Lift_DualSample.json` — 双采 Self-Lift 精修 / Dual-Sample Self-Lift Refinement

**流程 / Pipeline**:
```
一采 First pass:
  Loader(8步LoRA ~0.75) ──> KSampler(8步 euler/simple, 672×384×24)
        └──> LatentUpscaleAlign(scale=2.0 → 1344×768, bilinear, CONST重噪, 音频锁定)
二采 Second pass:
  LoraLoaderModelOnly(FastH3-4step-LoRA 0.7) ──> BasicGuider
  BasicScheduler(beta, 4步, denoise 0.55) ──> SamplerCustomAdvanced + DisableNoise
        └──> VAEDecode + VAEDecodeAudio ──> VHS_VideoCombine
```

**为什么要双采 / Why dual-sample**: 一采低分辨率快速出草稿（省显存/快），LatentUpscaleAlign 把 latent 直接放大到 2x 并 32px 对齐（无 VAE 编解码损失），二采用 4 步 LoRA + Beta 调度在原 latent 上精修细节，得到高清成片——比一次直接生成高分辨率更快、更省显存，且音频全程锁定一采结果不受影响。

**使用步骤 / How to use**:
1. 拖入 `SolH3_Self-Lift_DualSample.json`。
2. **Loader**：`lora_name` 指向你的 8 步 LoRA（一采保底），`lora_strength≈0.75`；`fast_h3_steps` 选「8步 FastH3 增强」（一采 8 步；>3 步画面更稳）。
3. **LoraLoaderModelOnly**：选 `FastH3-4step-LoRA.safetensors`，强度 `0.7`（二采收敛快、边缘清晰）。
4. **LatentUpscaleAlign**：`scale=2.0`（672×384→1344×768）、`method=bilinear`（**不要改**）、`add_noise=True`、`audio_denoise=0.0`（锁定一采音频）；如装有 3D upscaler 模型可选 `upscaler_model`。
5. **BasicScheduler**：`scheduler=beta`、`steps=4`、`denoise=0.55`（二采精修强度）。
6. 提示词与 CLIP/VAE 设置同工作流 1。运行即可。

**经验参数（社区验证 + v2.4.2 实测） / Proven settings**:
- 一采步数 ≤3 步快速预览更省抽卡成本（>3 步易画面异常）。
- 放大倍数 1.3–2（显存峰值=放大后分辨率那次，不能超单采预算）。
- 放大方法必须 bilinear（nearest-exact 会导致 latent 块化→二采鬼影/重影）。
- 二采 denoise 0.5–0.6（0.35 太低修不干净块化会留鬼影）。
- 显存紧张时 `precision=int8` + `chunk_ff=ON` 可显著降低峰值。

---

## 常见问题 / FAQ & Troubleshooting

**Q1: 报错 `sol_attn() got an unexpected keyword argument 'max_blocks'`？**
> 旧版根目录顶层文件污染。检查 `ComfyUI/custom_nodes/sol_attn_minimax_v2.py` 是否存在——若非本插件内置副本，请删除或覆盖为最新版（v2.2 起插件始终加载内置副本，不受影响）。/ Remove any stale top-level `sol_attn_minimax_v2.py` in `custom_nodes/`.

**Q2: 报错 `got an unexpected keyword argument 'sink_blocks'`？**
> `comfy_kitchen` 过旧，请升级 ComfyUI / comfy_kitchen 到支持 `sink_blocks/sink_q/tail` 新 API 的版本。/ Upgrade comfy_kitchen.

**Q3: 音频怪异或解码极慢？**
> 检查 Loader 的 `audio_shift` 是否为默认 3.0；启动日志应有 `[BSAI-Sol-H3] 音频解码修复已就绪`（AudioVAE 全量加载补丁已生效），生成时无 `prepared for dynamic VRAM loading. 576MB Staged` 字样。/ Ensure `audio_shift=3.0`; the AudioVAE full-load patch is applied automatically.

**Q4: 双采出鬼影/重影？**
> 放大方法改回 `bilinear`；二采 `denoise` 提到 0.5-0.6；一采步数 ≤8 且 LoRA 强度适当。/ Use bilinear upscale; raise second-pass denoise to 0.5-0.6.

**Q5: 没有 `workflows/` 目录？**
> 从 GitHub 最新版获取（v2.5+ 已内置两个示例工作流）。/ Get the latest from GitHub — v2.5+ ships both example workflows.

**Q6: Sol-Attn 未生效（速度慢、显存高）？**
> 确认 Loader `sol_attn=ON` 且 `sink_conditioning≠off`；日志应出现 `Sol-Attn已安装`，且无 `kernel failed`。/ Verify `sol_attn=ON`, `sink_conditioning≠off`, and no `kernel failed` in logs.

**Q7: 报错 `ERROR lora diffusion_model.blocks.N.adaln_proj.linear.weight shape '[...]' is invalid for input of size ...`（blocks.0~N 刷屏）？**
> 这是 **LoRA 与模型架构不匹配**。H3 的 `adaln_proj` 有**两种结构**：标准模型（`10Eros_Max_h3_TURBO-hybrid_beta4*`、`minimax_h3_fl2va/ref2va_int8_convrot`、bf16 系列）该层输入是 **2688 维**（权重 `[96768, 2688]`）；curve 模型（`minimax_h3_fastvideo_*4step*`、`*_pruned_*`、`hybrid_b25-49`、`Dasiwa*`）是 **8 维**（权重 `[96768, 8]`）。LoRA 也分两派：`H3电影质感V0.4 .safetensors`（rank16，A=[16, 2688]）为**标准架构**训练；`minimax_h3_lms_v1.0_r64-细节纹理增强lora.safetensors`（部分版本文件名无连字符）的 A=[64, 8]，为 **curve 架构**训练。**挂反了，官方 `Lora Loader Stack (rgthree)` 路径就会 reshape 失败逐 block 刷屏**：
> - 标准模型 + `lms`（8 维 LoRA）→ `shape '[96768, 2688]' is invalid for input of size 774144`（774144 = 96768×8，LoRA 元素数）
> - **curve 模型（fastvideo 4步极速等）+ `质感V0.4`（2688 维 LoRA）→ `shape '[96768, 8]' is invalid for input of size 260112384`（260112384 = 96768×2688）**
> 与电脑/磁盘无关：**任何机器**这样组合都会报错（"C 盘正常"只是因为该组合从未出现在 C 盘工作流里）。
> 解决（按需二选一）：① 把不匹配的 LoRA strength 改为 **0** 或从栈中移除；② 换模型使架构匹配——`质感V0.4` 配**标准**模型（10Eros_TURBO/fl2va/ref2va），`lms` 配 **curve** 模型（fastvideo 4步极速等）。`FastH3-4step-LoRA` 由 Sol-H3 Loader **内置转换加载**（日志出现 `[BSAI-Sol-H3] LoRA已加载: FastH3-4step-LoRA... 258/258` 即为生效），**不要**再挂官方栈。
> EN: This is a **LoRA ↔ model architecture mismatch** on `adaln_proj`. Standard H3 models (e.g. `10Eros_Max_h3_TURBO-hybrid_beta4*`, `minimax_h3_fl2va/ref2va_int8_convrot`, bf16) use a **2688-dim** input (weight `[96768, 2688]`); curve models (`minimax_h3_fastvideo_*4step*`, `*_pruned_*`, `hybrid_b25-49`, `Dasiwa*`) use an **8-dim** input (weight `[96768, 8]`). LoRAs match one side: `H3电影质感V0.4 .safetensors` (rank16, A=[16, 2688]) was trained for **standard** models; `minimax_h3_lms_v1.0_r64-细节纹理增强lora.safetensors` (A=[64, 8]) for **curve** models. Putting the wrong pair into the official `Lora Loader Stack (rgthree)` makes reshape fail and spam per-block errors:
> - Standard model + `lms` (8-dim LoRA) → `shape '[96768, 2688]' ... size 774144` (96768×8)
> - **Curve model (fastvideo 4-step etc.) + `质感V0.4` (2688-dim LoRA) → `shape '[96768, 8]' ... size 260112384` (96768×2688)**
> This is machine-independent: any PC hits it with that combo ("C-drive works" only because the combo never appears in those workflows). Fix: set the mismatched LoRA's strength to **0** (or remove it), or swap the model so architectures match — `质感V0.4` ↔ standard (10Eros_TURBO/fl2va/ref2va), `lms` ↔ curve (fastvideo 4-step). Keep `FastH3-4step-LoRA` out of the official stack; the Sol-H3 Loader converts and loads it natively (`[BSAI-Sol-H3] LoRA已加载: FastH3-4step-LoRA... 258/258`).

**Q8: 加了 `ModelAttentionBackend`（comfy kitchen attention）后 Sol-Attn 不生效 / 显存没降？**
> 删掉这个节点（或断开它的 model 链）。它执行 `set_model_optimized_attention`，写入的 `transformer_options["optimized_attention_override"]` 与 Loader 内置 Sol-Attn 是**同一个槽位**；按节点执行顺序它在 Loader 之后运行，会直接**覆盖** Sol-Attn，稀疏注意力失效。int8 权重的计算走 comfy_kitchen 张量层（ops），不依赖该节点。/ Remove it. It writes the same `optimized_attention_override` slot as the Loader's built-in Sol-Attn and overwrites it when executed later, disabling sparse attention. INT8 math runs in comfy_kitchen's tensor layer and does not need this node.

---

## 硬件要求 / Requirements

- NVIDIA GPU: SM86 (RTX 30 系) / SM89 (RTX 40 系) / SM120 (RTX 50 系) / SM121 (DGX Spark)
- PyTorch + CUDA，bfloat16 支持 / bfloat16 support
- Triton 3.6.0+
- ComfyUI 0.30.0+

## 版本历史 / Changelog

- **v2.6.1 (2026-09-12)**: docs(FAQ): Q7 扩展为双向 LoRA 兼容矩阵——新增反向案例（curve 模型如 fastvideo 4步极速 × 标准架构 LoRA `质感V0.4` → `shape '[96768, 8]' ... size 260112384`），修正原"质感V0.4 可任意挂官方栈"的误导；兼容性结论与电脑/磁盘无关。
- **v2.6 (2026-09-12)**: **FastVideo LoRA 真正生效 / FastVideo LoRA fully works** — 内置 FastVideo→ComfyUI 结构转换（`transformer_blocks.attn.to_q/to_k/to_v`→`blocks.attn.qkv_proj` 块对角融合、`ff.net.0.proj/2`→`mlp.fc1/fc2`、`.diff/.diff_b`→量化感知 delta patch），并加形状过滤自动跳过 curve 版模型（adaln 8 维 / 无 time_embedder）不兼容的 patch；同时修复 FastVideo 导出 LoRA 文件 header 数据长度与实际文件不一致导致的 safetensors 0.8.0 严格校验失败（自动回退手动解析）。标准模型（adaln 全宽 2688）上 FastH3-4step-LoRA **258+85 全部生效**；curve 版模型（如 10Eros TURBO-hybrid）主体 208+80 生效、不兼容部分自动跳过。
- **v2.5.4 (2026-09-11)**: Loader 支持 delta 权重（`.diff`/`.diff_b`）智能加载（自动 key 匹配 + 反量化应用）；示例工作流恢复并更新至 29 参数结构。
- **v2.5.3 (2026-09-11)**: 分离标准 LoRA 与 delta 格式；shape/cov 调试打印。
- **v2.5.2**: 示例工作流默认选 3D upscaler 模型（fp16）。
- **v2.5.1**: 自动注册 `h3_latent_upscalers` 目录为模型搜索路径。
- **v2.5**: `LatentUpscaleAlign` 支持 3D latent upscaler 模型语义放大。
- **v2.4**: 双采 Self-Lift：Loader 双 LoRA 可配（lora_name/lora_strength）+ 新增 `LatentUpscaleAlign` 节点。
- **v2.3**: 音频修复：`audio_shift` 独立缩放 + AudioVAE 全量加载补丁（PR #15371）。
- **v2.2**: Loader 改为内置副本优先，根治旧顶层文件污染。
- **v2.1**: Loader 并入 Sol-Attn 全套精细参数，`_apply_patch` 完整路径安装。

## 许可证 / License

MIT（本插件 / this plugin）｜Sol-Attn 内核：Apache 2.0 (NVIDIA)｜H3 权重：MiniMax Community License
