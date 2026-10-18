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
            warnings.append("Disk files changed after the diagnostics module was imported; the running instance may still be the old implementation. Please restart and verify.")
        rows.append(dict(node_id=node_id, class_name=cls.__name__, loaded_module=cls.__module__,
                         loaded_module_path=path, disk_sha256=now, import_observed_disk_sha256=before,
                         disk_changed_since_observation=changed,
                         memory_bytecode_matches_disk="not_proven_by_file_hash"))
    if not rows:
        warnings.append("This node was not found in the current instance; presence on disk does not mean it is registered/loaded.")
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
        warnings.append("The same node is provided by multiple loaded paths; none are auto-removed or disabled.")
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
    summary = "\n".join([f"节点：{node_id}", f"磁盘版本：{version or 'unknown'}（不等于实例已加载版本）",
                         *[f"实际模块：{row['loaded_module_path']}" for row in rows], *warnings])
    return summary, json.dumps(report, ensure_ascii=False, indent=2)


DELIVERY = {
    "return_exact_first_pass_audio_tensor": "Delivery: keeps the first-pass audio tensor; the second pass may participate in joint computation but its audio prediction is not delivered.",
    "publish_refined_joint_audio": "Delivery: second-pass joint refined audio.",
    "exact_input_tensor_passthrough": "Current step: input audio tensor passes through unchanged; this does not indicate which track is finally delivered.",
    "completed_second_pass_output": "Context: uses completed second-pass audio; this is not an unfinished first-pass audio lock.",
}


def explain_audio_report(report_json):
    if not report_json.strip():
        return "No report yet; audio source and cache state are unknown.", json.dumps({"status": "unknown", "evidence": []}, ensure_ascii=False)
    input_format = 'json'
    try:
        value = json.loads(report_json)
    except (ValueError, TypeError) as error:
        # Existing Audio Conditioning intentionally emits a text report. Only
        # recognize exact machine fields, never interpret arbitrary user prose.
        fields = dict(line.split('=', 1) for line in report_json.splitlines()
                      if line.startswith(('task=', 'audio_mode=', 'source_audio_tag=', 'frames=', 'canvas=')))
        if 'audio_mode' not in fields:
            raise ValueError("Connect an existing report_json / Conditioning report; this box is not a prompt or audio setting.") from error
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
                text = {"lock_source": "Condition: lock source recording to drive the image; this is not voice cloning for new dialogue.",
                        "native": "Condition: natively generate audio; the reference recording (if connected) is not the delivered track.",
                        "reference_only": "Condition: source sound is reference only; that track is not directly retained.",
                        "remix_source": "Condition: source audio participates in remix; the original audio is not guaranteed unchanged."}.get(val, f"Condition audio mode: {val} (unrecognized)")
                evidence.append(dict(path=location, value=val, explanation=text))
                lines.append(text)
            elif key in {"low_reused", "high_reused", "cache_hit", "reused"} and isinstance(val, bool):
                text = f"{location}：{'reusing validated result' if val else 'no cache hit for this item'}。"
                evidence.append(dict(path=location, value=val, explanation=text))
                lines.append(text)
            elif key == 'effective_source' and path.endswith('.audio_policy') and isinstance(val, str):
                text = {'legacy_policy': 'Second-pass input: native joint continuation; first-pass audio in the generation region continues to evolve, known context audio is template-locked.',
                        'first_pass': 'Second-pass input: from first-pass audio; whether it is locked also depends on effective_strength; do not claim completion by source alone.',
                        'highres_template': 'Second-pass input: high-resolution conditioning template.'}.get(val, f'Second-pass input policy: {val} (unrecognized).')
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
        lines.append("The report provides no recognizable audio/cache evidence; state remains unknown; no resampling or cache change.")
    lines.append("The notes node is read-only: it does not alter audio, MODEL, or cache identity; unknown fields do not mean a feature failed.")
    report = dict(schema="t8.audio.readable_report.v1", status="evidence_found" if evidence else "unknown",
                  evidence=evidence, numerical_mutation=False, sampling_started=False,
                  input_format=input_format,
                  input_report_sha256=hashlib.sha256(report_json.encode("utf8")).hexdigest())
    return "\n".join(dict.fromkeys(lines)), json.dumps(report, ensure_ascii=False, indent=2)
