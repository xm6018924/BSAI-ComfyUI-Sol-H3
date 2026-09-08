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

## v2 更新（2026-09-09）

- **修复 Sol-Attn 实际未安装问题**：旧版 `sol_attn=True` 仅打印 ON，未真正安装稀疏注意力补丁，导致注意力退化全 Dense、显存 OOM。v2 在 Loader 内真正调用 `sol_attn_minimax_v2.make_override` 安装补丁。
- **适配新版 comfy_kitchen API**：Sol-Attn 内核参数已从旧名 `max_blocks`/`centroid_tail` 迁移到 `sink_blocks`/`sink_q`/`tail`；修复因参数名不匹配导致的 `sol_attn() got an unexpected keyword argument 'max_blocks'` 报错与内核持续失败回退。
- 插件内置 `sol_attn_minimax_v2.py`（与顶层节点同源），Loader 安装补丁时优先复用顶层模块、回退使用内置副本，避免重复注册。

### Loader 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| model_name | - | H3 模型文件（diffusion_models） |
| precision | int8 | 模型加载精度：int8 / default / fp8_e4m3fn |
| sol_attn | ON | 是否安装 Sol-Attn 稀疏注意力补丁 |
| tau_start / tau_end | 0.5 / 1.0 | Sol-Attn 稀疏强度区间（越大越稀疏） |
| sink_conditioning | exact_kv_and_rows | 锚点策略：exact_kv / exact_kv_and_rows / off |
| int8_qk | ON | INT8 QK 量化（打印记录） |
| fused_modulation | ON | 融合调制优化（打印记录） |
| chunk_ff / chunk_size | ON / 2 | FFN 分块显存优化（打印记录） |
| fast_h3_steps | 4步 | 4步 FastH3 极速 / 8步增强 / 50步原生 |
| sampler | euler | 采样器 |
| cfg | 4.0 | 提示词引导强度 |
| shift | 8.0 | Flow matching 时间偏移（视频时钟；音频固定 3.0） |

> 注意：升级 v2 后旧工作流若出现参数错位，请在 Loader 面板重新设置一次参数（面板顺序已与 v2 对齐）。

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
