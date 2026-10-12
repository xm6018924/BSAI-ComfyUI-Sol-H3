# Spark-H3 真实采样对比结论（BSAI-Sol-H3 v2.9）

> 对比环境：CUDA ComfyUI 0.39.0（python_embeded_cuda，torch 2.13.0+cu130），
> RTX 5090 Laptop 24GB（SM120），minimax_h3_ref2va_pruned_int8_convrot + FastH3-4step-LoRA，
> 1344×768，seed=42，euler/simple 4 步。2026-10-08 实测。

## 一、三层验证结论

### 1. 内核层（comfy_kitchen.sol_attn，bf16，(1,N,56,128)）

| 序列长度 N | dense | tau=1.3 路由（off） | topk=0.1（Spark RepBlock） | topk+aug=128（Spark full） |
|---|---:|---:|---:|---:|
| 6167（24帧主注意力） | 5.0 ms | 2.1 ms | 2.9 ms | 4.0 ms |
| 12285（48帧主注意力） | 24.6 ms | 6.4 ms | 8.0 ms | 11.5 ms |

- 稀疏均优于 dense：tau 比 dense 快 **2.4×~3.8×**；Spark full 比 dense 快 **1.3×~2.1×**。
- **Spark full 比 tau 路由慢 1.8~1.9×**（topk 分组路由 + 尾部 token_aug 精确修正的开销）。
- Spark 参数真实有效：同一输入下 topk+aug 与 tau 输出不同（maxdiff 0.06~0.21）。

### 2. 采样层（真实视频采样端到端 wall 时间）

| 场景 | off（tau 路由） | full（RepBlock 10% + Rabbit 128） | 结论 |
|---|---:|---:|---|
| 默认 min_tokens=12288，24帧 | 40.71 s | 33.11 s | **两者逐像素一致**（maxdiff=0.00）——Spark 未生效 |
| min_tokens=512，24帧（Spark 生效） | 32.66 s | 30.37 s | 输出有差异（PSNR 26.32dB/SSIM 0.96），full 略快 ~7% |
| min_tokens=512，48帧（Spark 生效） | 38.19 s | 68.80 s | 输出有差异（PSNR 34.05dB/SSIM 0.94），**full 反而慢 80%** |

### 3. 关键归因

- **默认 min_tokens=12288 门槛过高**：1344×768×24 的 H3 视频注意力序列只有 **6167 tokens**（空间 1008×6 帧 + 文本），低于门槛 → 内核全部 dense 回退 → Spark 参数虽已透传但**完全未生效**（verbose 日志实证：`dense (1, 6167, 56, 128): seq 6167 < 12288`）。此场景下 off/full 输出逐像素一致，恰恰证明"参数没进入计算"。
- 官方视频宣称的"10 秒 768p 提速 1.73×"的基线是 **dense/Veda 密集注意力**；本插件 off 模式（tau 稀疏）已比 dense 快 2.4~3.8×，Spark 相对 dense 仍快，但**相对 tau 稀疏是倒退**。

## 二、对插件升级的定夺建议

1. **不建议将 spark_mode=full 设为默认**：在本机 H3 视频场景（~6k~12k tokens）下，Spark 相对现有 tau 稀疏路由无加速优势，48 帧反而明显变慢；画质与 tau 相当（同种子差异 PSNR 26~34dB，属同源微差，肉眼难辨优劣）。
2. **保留 v2.9 的 Spark 参数透传作为可选能力**：官方 Spark 的价值（10% 密度契约 + 尾部精确修正）在**数万 token 的超长序列**（官方 10s 768p 场景）才可能体现；后续若跑更长序列可对照实验。
3. **真正值得改的是默认 min_tokens**：12288 对 H3 视频（6k~12k tokens）偏高，导致用户开了稀疏却实际跑 dense。建议默认下调（如 1024），让 tau 稀疏真实生效，用户即可获得 2.4~3.8× 注意力加速（本插件现有能力），而不是依赖 Spark。

## 三、定夺结果（2026-10-08 用户确认「按你的建议来」）

- ✅ **min_tokens 默认值 12288 → 1024**（落地为 v2.9.1，nodes.py/README/插件总清单已同步）
- ✅ **spark_mode 保持默认 off**，v2.9 的 Spark 参数透传保留为可选能力
- 验证与交付：py_compile + 单元测试通过后交付本地；GitHub 推送待用户验收后执行

## 四、交付物

- 输出帧：`ComfyUI/output/bsai_spark_{off,full,warmup}_*.png`（默认门槛，off/full 一致）
  `ComfyUI/output/bsai_lowmt_{off,full}_*.png`（24帧 Spark 生效）与 `bsai_l48_{off,full}_*.png`（48帧 Spark 生效）
- 脚本：`tests/bench_spark_h3.py`（默认对比）、`tests/bench_spark_lowmt.py`（低门槛对比）、`tests/bench_spark_l48.py`（48帧）、`tests/compare_spark_h3.py`（PSNR/SSIM）、`tests/kernel_verify.py`（内核差异）、`tests/kernel_bench.py`（内核耗时）
- 原始计时：项目根 `_bench_log.txt`（RESULT_JSON）
