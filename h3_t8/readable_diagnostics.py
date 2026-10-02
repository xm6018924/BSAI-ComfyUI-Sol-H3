"""Read-only, evidence-qualified provenance and delivery explanations."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _digest(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


# This is an import-time *disk observation*, not proof of another module's bytecode.
IMPORT_OBSERVATIONS = {str(Path(module.__file__).resolve()): _digest(module.__file__)
                       for module in list(sys.modules.values())
                       if getattr(module, "__file__", None)
                       and str(ROOT / "h3_t8") in str(Path(module.__file__).resolve())}


def diagnose_node_source(node_id, node_classes):
    def matches(cls):
        if cls.__name__ == node_id or getattr(cls, 'NODE_ID', None) == node_id:
            return True
        schema = getattr(cls, 'define_schema', None)
        return callable(schema) and schema().node_id == node_id
    selected = [c for c in node_classes if matches(c)]
    rows = []
    warnings = []
    for cls in selected:
        module = sys.modules.get(cls.__module__)
        path = getattr(module, "__file__", None) or inspect.getfile(cls)
        path = str(Path(path).resolve())
        before, now = IMPORT_OBSERVATIONS.get(path), _digest(path)
        changed = before is not None and now != before
        if changed:
            warnings.append("磁盘文件在诊断模块导入后变化；当前运行实例可能仍是旧实现，请自行重启后核验。")
        rows.append(dict(node_id=node_id, class_name=cls.__name__, loaded_module=cls.__module__,
                         loaded_module_path=path, disk_sha256=now, import_observed_disk_sha256=before,
                         disk_changed_since_observation=changed,
                         memory_bytecode_matches_disk="not_proven_by_file_hash"))
    if not rows:
        warnings.append("当前实例没有找到该节点；磁盘存在不表示实例已注册／加载。")
    copies = []
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        candidate = getattr(module, node_id, None) if path else None
        if inspect.isclass(candidate):
            provider = sys.modules.get(candidate.__module__)
            provider_path = getattr(provider, '__file__', None)
            if provider_path:
                copies.append(dict(module=candidate.__module__, path=str(Path(provider_path).resolve()),
                                   imported_by=name))
    if len({item["path"] for item in copies}) > 1:
        warnings.append("检测到同名节点由多个已加载路径提供；不自动删除或停用。")
    try:
        import tomllib
        version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf8"))["project"]["version"]
    except (OSError, ValueError, KeyError, ImportError):
        version = None
    report = dict(schema="t8.node.source_diagnostic.v1", status="loaded" if rows else "not_found",
                  formal_project=str(ROOT), disk_package_version=version,
                  running_package_version="unknown_unless_runtime_declares_it",
                  nodes=rows, loaded_copies=copies, warnings=warnings,
                  mutation=False, restarted_service=False, blocks_sampling=False)
    summary = "\n".join([f"节点：{node_id}", f"磁盘版本：{version or '未知'}（不等于实例已加载版本）",
                         *[f"实际模块：{row['loaded_module_path']}" for row in rows], *warnings])
    return summary, json.dumps(report, ensure_ascii=False, indent=2)


DELIVERY = {
    "return_exact_first_pass_audio_tensor": "交付：保留一采音频 tensor；二采可能参与联合计算，但不交付其音频预测。",
    "publish_refined_joint_audio": "交付：二采联合精修音频。",
    "exact_input_tensor_passthrough": "当前步骤：输入音频 tensor 原样通过；不能据此判断最终交付使用哪条音轨。",
    "completed_second_pass_output": "上下文：使用已完成二采音频；这不是粗采未完成音频锁定。",
}


def explain_audio_report(report_json):
    if not report_json.strip():
        return "尚无报告，音频来源和缓存状态未知。", json.dumps({"status": "unknown", "evidence": []}, ensure_ascii=False)
    input_format = 'json'
    try:
        value = json.loads(report_json)
    except (ValueError, TypeError) as error:
        # Existing Audio Conditioning intentionally emits a text report. Only
        # recognize exact machine fields, never interpret arbitrary user prose.
        fields = dict(line.split('=', 1) for line in report_json.splitlines()
                      if line.startswith(('task=', 'audio_mode=', 'source_audio_tag=', 'frames=', 'canvas=')))
        if 'audio_mode' not in fields:
            raise ValueError("连接已有 report_json／Conditioning report；这个框不是提示词或音频设置。") from error
        value, input_format = fields, 'conditioning_text'
    if not isinstance(value, dict):
        raise ValueError("report_json must contain an object")
    evidence, lines = [], []

    def visit(obj, path="$"):
        if not isinstance(obj, dict):
            return
        for key, val in obj.items():
            location = f"{path}.{key}"
            if key in {"final_audio_policy", "delivery_audio_source", "audio_policy", "audio_source"} and isinstance(val, str):
                text = DELIVERY.get(val)
                if text is None:
                    text = f"{key}={val}；未经识别的来源声明，不自动解释成音色克隆或保原音。"
                evidence.append(dict(path=location, value=val, explanation=text))
                lines.append(text)
            elif key == "audio_mode" and isinstance(val, str):
                text = {"lock_source": "条件：锁定源录音驱动画面，不是生成新台词的音色克隆。",
                        "native": "条件：原生生成音频；参考录音（若连接）不是直接交付音轨。",
                        "reference_only": "条件：源声音仅作参考；不直接保留该音轨。",
                        "remix_source": "条件：源音频参与重混；不保证原音不变。"}.get(val, f"条件音频模式：{val}（未识别）")
                evidence.append(dict(path=location, value=val, explanation=text))
                lines.append(text)
            elif key in {"low_reused", "high_reused", "cache_hit", "reused"} and isinstance(val, bool):
                text = f"{location}：{'复用已校验结果' if val else '未命中该项缓存'}。"
                evidence.append(dict(path=location, value=val, explanation=text))
                lines.append(text)
            elif key == 'effective_source' and path.endswith('.audio_policy') and isinstance(val, str):
                text = {'legacy_policy': '二采输入：原生联合续采；生成区粗采音频继续演化，已知上下文音频按模板锁定。',
                        'first_pass': '二采输入：来自一采音频；是否锁定须同时看effective_strength，不能仅凭来源称完成。',
                        'highres_template': '二采输入：高分辨率条件模板。'}.get(val, f'二采输入策略：{val}（未识别）。')
                evidence.append(dict(path=location, value=val, explanation=text))
                lines.append(text)
            elif key in {'migration', 'context_handoff', 'first_pass_complete_trajectory', 'effective_strength'} and path.endswith('.audio_policy'):
                text = f'{location}={val}（原报告声明；不是最终交付音轨证明）。'
                evidence.append(dict(path=location, value=val, explanation=text))
                lines.append(text)
            elif isinstance(val, dict):
                visit(val, location)
            elif isinstance(val, list):
                for index, item in enumerate(val):
                    visit(item, f"{location}[{index}]")
    visit(value)
    if not evidence:
        lines.append("报告未提供可识别的音频／缓存证据，保持未知；不重新采样或改缓存。")
    lines.append("说明节点只读：不改变音频、MODEL或缓存身份；未知字段不等于功能失败。")
    report = dict(schema="t8.audio.readable_report.v1", status="evidence_found" if evidence else "unknown",
                  evidence=evidence, numerical_mutation=False, sampling_started=False,
                  input_format=input_format,
                  input_report_sha256=hashlib.sha256(report_json.encode("utf8")).hexdigest())
    return "\n".join(dict.fromkeys(lines)), json.dumps(report, ensure_ascii=False, indent=2)
