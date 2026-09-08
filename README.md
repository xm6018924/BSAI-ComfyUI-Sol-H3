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

- **BSAI Sol-H3 Loader** — 一键加载模型+应用所有优化（14 参数：模型/精度/Sol-Attn/tau/sink条件/INT8-QK/融合调制/分块FFN/步数/采样器/CFG/Shift）
- **BSAI Sol-H3 Info** — 显示优化状态

## v2.1 更新（2026-09-09）

- **Loader 并入原独立 SolAttnMiniMax 节点的全套精细参数**：删除右侧独立节点后配置能力不丢失（见下表参数 15-26）。
- **Sol-Attn 安装改走 `_apply_patch` 完整路径**：自动 clone model、安装 H3 Morton hooks（sink 锚点必需）、block 索引（dense_blocks/tau_profile 用）、采样百分比→sigma 换算。
- **修复 Sol-Attn 实际未安装问题**：旧版 `sol_attn=True` 仅打印 ON，未真正安装稀疏注意力补丁，导致注意力退化全 Dense、显存 OOM。
- **适配新版 comfy_kitchen API**：`sink_blocks`/`sink_q`/`tail`（旧名 `max_blocks`/`centroid_tail` 已移除）。
- 插件内置 `sol_attn_minimax_v2.py`（与顶层节点同源），Loader 优先复用顶层模块、回退使用内置副本。

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
| 13 | shift | 8.0 | Flow matching 时间偏移（视频时钟；音频固定 3.0） |
| 14 | **min_tokens** | 12288 | 序列长度阈值：低于此长度的注意力用 Dense（省显存/保质量） |
| 15 | **sol_tau** | 1.3 | Sol-Attn 稀疏强度（越大越稀疏、越省显存） |
| 16 | **start_percent** | 0.2 | 稀疏生效起始采样比例（前 20% 步数 Dense 热身） |
| 17 | **end_percent** | 0.9 | 稀疏生效结束采样比例（最后 10% 步数 Dense 收尾） |
| 18 | **morton** | OFF | Morton 空间排序（提升稀疏命中率；需 H3） |
| 19 | **morton_curve** | 2d_frame | 排序曲线：2d_frame / hilbert / z_order |
| 20 | **centroid_tail** | ON | 质心尾块保留（提升长序列质量） |
| 21 | **routed_cap_percent** | 0 | 路由容量上限百分比（0=不限制） |
| 22 | **reuse_qkv_memory** | OFF | 复用 QKV 中间缓冲（仅限已知安全的模型） |
| 23 | **verbose** | OFF | Sol-Attn 详细日志 |
| 24 | **dense_blocks** | 空 | 强制全 Dense 的块索引列表，如 `0,1,2` |
| 25 | **tau_profile** | 空 | 逐块 tau 覆盖，如 `0-4=2.0; 20-24=0.8`（分号/换行分隔） |

> 注意：升级后旧工作流若出现参数错位，请在 Loader 面板重新设置一次参数（面板顺序已与 v2.1 对齐）。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/BSAI-ComfyUI-Sol-H3.git
cd BSAI-ComfyUI-Sol-H3
python install.py
```

## 硬件要求

- NVIDIA GPU: SM86 (RTX 30系) / SM89 (RTX 40系) / SM120 (RTX 50系) / SM121 (DGX Spark)
- PyTorch + CUDA, bfloat16支持
- Triton 3.6.0+
- ComfyUI 0.30.0+

## 许可证

MIT (本插件) | Sol-Attn内核: Apache 2.0 (NVIDIA) | H3权重: MiniMax Community License
