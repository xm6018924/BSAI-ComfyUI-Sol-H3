# BSAI ComfyUI Sol-H3

**MiniMax H3 一键高速推理插件** — Sol-Attn 稀疏注意力 + FastH3 4步蒸馏 + Fused Modulation + Chunk FeedForward

## 技术栈

| 组件 | 来源 | 作用 |
|---|---|---|
| **Sol-Attn** | [Saganaki22/ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn) | Triton稀疏注意力，SM86-SM121（RTX 30/40/50系） |
| **FastH3 V1** | [Hao AI Lab (UCSD)](https://haoailab.com/blogs/fasth3-preview/) | 4步蒸馏adapter，~11x加速 |
| **Fused Modulation** | Sol-Attn内置 | Triton融合AdaLN+门控残差，bit-exact |
| **Chunk FeedForward** | Sol-Attn内置 | MLP显存峰值降低最多1.9GB |

## 性能

| 配置 | 5秒720p视频 | 对比 |
|---|---|---|
| 原生H3 (50步) | ~30分钟 | 基准 |
| Sol-Attn (50步) | ~6.5分钟 | ~4.5x加速 (RTX 5090) |
| **FastH3+Sol (4步)** | **~2-3分钟** | **~11x加速** |
| NVIDIA 8xB300 (4步) | 1.653秒 | 服务器级 |

## 节点

- **BSAI Sol-H3 Loader** — 一键加载模型+应用所有优化（27 参数：模型/精度/Sol-Attn/tau/sink条件/INT8-QK/融合调制/分块FFN/步数/采样器/CFG/Shift/AudioShift）
- **BSAI Sol-H3 Info** — 显示优化状态

## v2.3 更新（2026-09-09）— 音频解码修复

- **修复音频怪异/失真（关键）**：Loader 之前调用 `set_parameters(shift=shift)` 时把采样调度里 H3 官方默认的 `audio_shift=3.0` 覆盖为 `None`，音频流失去独立缩放、被按视频 schedule 采样，导致音频轨迹错位、声音怪异。现在改为 `set_parameters(shift=shift, audio_shift=audio_shift)`，并新增可调 `audio_shift` 参数（默认 3.0）。
- **修复音频解码慢/异常（关键）**：ComfyUI 核心的 MiniMax H3 音频 VAE（约 577MB）未设 `disable_offload=True`，走了 DynamicVRAM 权重量流路径（日志 `prepared for dynamic VRAM loading. 576MB Staged`），5 秒音频解码耗时约 153 秒且流式数值不稳定。插件现内置内存级补丁（对应官方 PR #15371）：加载时自动检测 H3 AudioVAE 并切换全量加载（解码约 0.45 秒，577MB 常驻显存）。**不改核心文件，装插件即生效，其他电脑开箱即用**。
- Loader 新增第 14 号参数 `audio_shift`（默认 3.0）：音频流 flow matching 时间偏移（官方默认 3.0）。
- **示例工作流已更新**：`workflows/SolH3_Fast_4step.json` 重制为 v2.3 结构——27 参数 Loader（含 `audio_shift`）、1344×768×24 帧、移除已并入 Loader 的旧 `SolAttnMiniMax` 独立节点，文生视频+音频一键跑通。

## v2.2 更新（2026-09-09）

- **Loader 模块加载改为「内置副本优先」**：不再 `import` custom_nodes 根目录的顶层 `sol_attn_minimax_v2.py`，始终加载本插件自带的 `sol_attn_minimax_v2.py`。彻底杜绝「其他电脑从旧打包拷贝后，根目录残留旧版顶层文件（旧 API `max_blocks`）污染新版插件」导致的 `sol_attn() got an unexpected keyword argument 'max_blocks'` 报错。插件现在完全自包含，clone 即用。
- **升级必读**：如果你或分发对象在 `ComfyUI/custom_nodes/` 根目录仍留有旧版 `sol_attn_minimax_v2.py`（独立 SolAttnMiniMax 节点），请将其**删除**（或替换为本插件内置的同名文件）。v2.1 起该节点已并入 Loader，不再需要顶层文件；保留旧文件只会带来 API 冲突。

## v2.1 更新（2026-09-09）

- **Loader 并入原独立 SolAttnMiniMax 节点的全套精细参数**：删除右侧独立节点后配置能力不丢失（见下表参数 15-26）。
- **Sol-Attn 安装改走 `_apply_patch` 完整路径**：自动 clone model、安装 H3 Morton hooks（sink 锚点必需）、block 索引（dense_blocks/tau_profile 用）、采样百分比→sigma 换算。
- **修复 Sol-Attn 实际未安装问题**：旧版 `sol_attn=True` 仅打印 ON，未真正安装稀疏注意力补丁，导致注意力退化全 Dense、显存 OOM。
- **适配新版 comfy_kitchen API**：`sink_blocks`/`sink_q`/`tail`（旧名 `max_blocks`/`centroid_tail` 已移除）。

### Loader 参数说明

| # | 参数 | 默认 | 说明 |
|---|---|---|---|
| 1 | model_name | - | H3 模型文件（diffusion_models） |
| 2 | precision | int8 | 模型加载精度：int8 / default / fp8_e4m3fn |
| 3 | sol_attn | ON | 是否安装 Sol-Attn 稀疏注意力补丁 |
| 4 | tau_start | 0.5 | 渐变起始 tau（信息/兼容旧工作流） |
| 5 | tau_end | 1.0 | 渐变结束 tau（信息/兼容旧工作流） |
| 6 | sink_conditioning | exact_kv | 锚点策略：exact_kv / exact_kv_and_rows / off |
| 7 | int8_qk | ON | INT8 QK 量化（打印记录） |
| 8 | fused_modulation | ON | 融合调制优化（打印记录） |
| 9 | chunk_ff / chunk_size | ON / 2 | FFN 分块显存优化（打印记录） |
| 10 | fast_h3_steps | 4步 | 4步 FastH3 极速 / 8步增强 / 50步原生 |
| 11 | sampler | euler | 采样器 |
| 12 | cfg | 4.0 | 提示词引导强度 |
| 13 | shift | 8.0 | Flow matching 时间偏移（视频时钟；官方默认 12.0） |
| 14 | **audio_shift** | 3.0 | 音频流独立时间偏移（官方默认 3.0；修音频怪异核心参数） |
| 15 | **min_tokens** | 12288 | 序列长度阈值：低于此长度的注意力用 Dense（省显存/保质量） |
| 16 | **sol_tau** | 1.3 | Sol-Attn 稀疏强度（越大越稀疏、越省显存） |
| 17 | **start_percent** | 0.2 | 稀疏生效起始采样比例（前 20% 步数 Dense 热身） |
| 18 | **end_percent** | 0.9 | 稀疏生效结束采样比例（最后 10% 步数 Dense 收尾） |
| 19 | **morton** | OFF | Morton 空间排序（提升稀疏命中率；需 H3） |
| 20 | **morton_curve** | 2d_frame | 排序曲线：2d_frame / hilbert / z_order |
| 21 | **centroid_tail** | ON | 质心尾块保留（提升长序列质量） |
| 22 | **routed_cap_percent** | 0 | 路由容量上限百分比（0=不限制） |
| 23 | **reuse_qkv_memory** | OFF | 复用 QKV 中间缓冲（仅限已知安全的模型） |
| 24 | **verbose** | OFF | Sol-Attn 详细日志 |
| 25 | **dense_blocks** | 空 | 强制全 Dense 的块索引列表，如 `0,1,2` |
| 26 | **tau_profile** | 空 | 逐块 tau 覆盖，如 `0-4=2.0; 20-24=0.8`（分号/换行分隔） |

> 注意：升级后旧工作流若出现参数错位，请在 Loader 面板重新设置一次参数（面板顺序已与 v2.3 对齐）。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/BSAI-ComfyUI-Sol-H3.git
cd BSAI-ComfyUI-Sol-H3
python install.py
```

### 其他电脑安装 / 升级排错

1. **必须使用最新版**：若旧电脑曾用旧打包（v2.0 之前），请先删除旧目录再重新 clone，或 `git pull` 到最新（当前 v2.3）。
2. **删除根目录旧顶层文件**：检查 `ComfyUI/custom_nodes/sol_attn_minimax_v2.py` 是否存在。若存在且不是本插件内置副本（对比文件大小约 36KB），**请删除或覆盖为最新版**——否则旧文件（旧 API `max_blocks`）会在运行时触发：
   `TypeError: sol_attn() got an unexpected keyword argument 'max_blocks'` 并退化为全 Dense 注意力（速度变慢、显存暴涨）。
3. **依赖项**：`comfy_kitchen`（随 ComfyUI 升级）必须为支持 `sink_blocks/sink_q/tail` 新 API 的版本；若报 `got an unexpected keyword argument 'sink_blocks'`，说明 `comfy_kitchen` 过旧，请升级 ComfyUI/comfy_kitchen。
4. **音频问题**：v2.3 已内置两项音频修复（audio_shift 独立缩放 + AudioVAE 全量加载补丁）。启动日志应出现 `[BSAI-Sol-H3] 音频解码修复已就绪`，且生成时无 `prepared for dynamic VRAM loading. 576MB Staged` 字样。若音频仍怪异，请检查工作流里 Loader 的 `audio_shift` 参数是否为默认 3.0。
5. 安装后启动 ComfyUI，日志应出现：`Sol-Attn已安装: ... (新API: sink_blocks/tail)` 且无 `kernel failed`。

## 硬件要求

- NVIDIA GPU: SM86 (RTX 30系) / SM89 (RTX 40系) / SM120 (RTX 50系) / SM121 (DGX Spark)
- PyTorch + CUDA, bfloat16支持
- Triton 3.6.0+
- ComfyUI 0.30.0+

## 许可证

MIT (本插件) | Sol-Attn内核: Apache 2.0 (NVIDIA) | H3权重: MiniMax Community License
