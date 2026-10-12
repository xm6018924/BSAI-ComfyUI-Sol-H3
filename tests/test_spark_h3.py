# -*- coding: utf-8 -*-
"""Spark-H3 契约单元测试 (v2.9)

覆盖:
1. comfy_kitchen 生产级 sol_attn 内核接受 topk_ratio/token_aug 参数
   (SparkRepBlock / SparkRabbit 的内核侧契约, 本机 sm_120 RTX 5090)
2. sol_attn_minimax_v2.make_override 构建的 override 正确透传两个 Spark 参数
3. 参数合法性: token_aug 必须是 64 的倍数(0..256), topk_ratio 0..1
4. 无 CUDA / 无内核时行为: _ineligible 判定回退 dense (不抛错)

运行: python_embeded\\python.exe tests\\test_spark_h3.py
"""
import os
import sys
import unittest

_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PLUGIN_DIR)
# comfy_api 是 ComfyUI 根目录下的命名空间包; 独立运行测试时向上查找补上。
_cursor = os.path.dirname(_PLUGIN_DIR)
for _ in range(6):
    if os.path.isdir(os.path.join(_cursor, "comfy_api")):
        sys.path.insert(0, _cursor)
        break
    _parent = os.path.dirname(_cursor)
    if _parent == _cursor:
        break
    _cursor = _parent

import torch  # noqa: E402


def _cuda_ready():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


class TestKernelContract(unittest.TestCase):
    """内核侧契约: 生产级 sol_attn 支持 Spark 双参数且数值有限。"""

    def setUp(self):
        if not _cuda_ready():
            self.skipTest("无 CUDA sm_80+ 设备")

    def test_topk_ratio_and_token_aug_accepted(self):
        import comfy_kitchen as ck
        b, t, h, d = 1, 256, 4, 128
        q = torch.randn(b, t, h, d, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(b, t, h, d, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(b, t, h, d, dtype=torch.bfloat16, device="cuda")
        out = ck.sol_attn(q, k, v, tau=1.0, topk_ratio=0.1, token_aug=128)
        self.assertEqual(tuple(out.shape), (b, t, h, d))
        self.assertTrue(torch.isfinite(out).all())

    def test_token_aug_64_multiple_validation(self):
        import comfy_kitchen as ck
        b, t, h, d = 1, 256, 4, 128
        q = torch.randn(b, t, h, d, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(b, t, h, d, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(b, t, h, d, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises((ValueError, RuntimeError)):
            ck.sol_attn(q, k, v, token_aug=100)  # 非 64 倍数

    def test_topk_ratio_out_of_range(self):
        import comfy_kitchen as ck
        b, t, h, d = 1, 256, 4, 128
        q = torch.randn(b, t, h, d, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(b, t, h, d, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(b, t, h, d, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises((ValueError, RuntimeError)):
            ck.sol_attn(q, k, v, topk_ratio=1.5)


class TestOverrideContract(unittest.TestCase):
    """override 侧契约: make_override 透传 Spark 参数到 _run。"""

    def _build_override(self, **extra):
        import sol_attn_minimax_v2 as sam
        return sam.make_override(
            tau=1.0, min_tokens=256,
            sigma_start=None, sigma_end=None,
            sink_conditioning="off",
            **extra)

    def test_override_builds_with_spark_params(self):
        ov = self._build_override(topk_ratio=0.1, token_aug=128)
        self.assertTrue(callable(ov))

    def test_override_defaults_off(self):
        ov = self._build_override()
        self.assertTrue(callable(ov))

    def test_override_masked_falls_back_dense(self):
        ov = self._build_override(topk_ratio=0.1, token_aug=128)
        b, t, h, d = 1, 512, 4, 128
        q = torch.randn(b, t, h, d, dtype=torch.bfloat16, device="cuda") \
            if _cuda_ready() else torch.randn(b, t, h, d, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        mask = torch.zeros(b, 1, t, t, dtype=torch.bool)
        out = ov(lambda *a, **kw: torch.ones_like(q), q, k, v, h,
                 mask=mask, skip_reshape=True, skip_output_reshape=True)
        self.assertTrue(torch.all(out == 1))  # dense 回退路径

    def test_apply_patch_signature_accepts_spark(self):
        import inspect
        import sol_attn_minimax_v2 as sam
        sig = inspect.signature(sam._apply_patch)
        self.assertIn("topk_ratio", sig.parameters)
        self.assertIn("token_aug", sig.parameters)
        self.assertEqual(sig.parameters["topk_ratio"].default, 0.0)
        self.assertEqual(sig.parameters["token_aug"].default, 0)


class TestSparkParamMapping(unittest.TestCase):
    """Loader 侧参数映射: spark_mode → 内核参数(与 nodes.py 同逻辑)。"""

    @staticmethod
    def _map(mode, keep=10.0, aug=128):
        topk, augv = 0.0, 0
        if mode == "rep_block":
            topk = max(0.01, min(1.0, keep / 100.0))
        elif mode == "rabbit":
            augv = max(0, min(256, aug // 64 * 64))
        elif mode == "full":
            topk = max(0.01, min(1.0, keep / 100.0))
            augv = max(0, min(256, aug // 64 * 64))
        return topk, augv

    def test_off(self):
        self.assertEqual(self._map("off"), (0.0, 0))

    def test_rep_block_10pct(self):
        topk, aug = self._map("rep_block")
        self.assertAlmostEqual(topk, 0.1)
        self.assertEqual(aug, 0)

    def test_rabbit_128(self):
        topk, aug = self._map("rabbit")
        self.assertEqual(topk, 0.0)
        self.assertEqual(aug, 128)

    def test_full(self):
        topk, aug = self._map("full")
        self.assertAlmostEqual(topk, 0.1)
        self.assertEqual(aug, 128)

    def test_token_aug_clamped_to_64(self):
        _, aug = self._map("rabbit", aug=100)
        self.assertEqual(aug, 64)  # 100//64*64

    def test_keep_percent_clamped(self):
        topk, _ = self._map("rep_block", keep=0.5)
        self.assertAlmostEqual(topk, 0.01)  # 下限 1%


if __name__ == "__main__":
    unittest.main(verbosity=2)
