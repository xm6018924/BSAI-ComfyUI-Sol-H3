"""Prepared Tao/LTX generation nodes; no model/runtime discovery at import."""
from copy import deepcopy
import json
from pathlib import Path

import folder_paths
from comfy_api.latest import io, InputImpl, ui

PreparedBundle = io.Custom('T8_PREPARED_GENERATION_BUNDLE')
CATEGORY = 'T8/MiniMax H3/Experimental Prepared Generation'


class MiniMaxH3PreparedGenerationBundleEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3PreparedGenerationBundleEXPT8',
            display_name='Tao / LTX · 读取已准备生成输入 (T8 EXP)', category=CATEGORY,
            is_experimental=True,
            description='读取本地准备清单，不下载、不编码提示词、不运行模型。需要匹配的教师/文本缓存或LTX AV输入；完整文件身份在生成节点执行时再次检查。',
            inputs=[io.String.Input('prepared_bundle_path', default='',
                tooltip='由准备工具生成的本地JSON绝对路径，不是工作流JSON。不同提示词需要重新准备匹配的文本/音频条件。')],
            outputs=[PreparedBundle.Output('prepared_bundle'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, prepared_bundle_path):
        from .prepared_generation_contract import read_bundle
        bundle = read_bundle(prepared_bundle_path)
        report = {'status': 'structure_checked_only', 'kind': bundle['kind'],
            'asset_count': len(bundle['assets']), 'directory_count': len(bundle['trees']),
            'models_loaded': False, 'content_hashes_verified': False,
            'next': 'Connect prepared generation; full identity verification happens on execution.'}
        return io.NodeOutput(bundle, json.dumps(report, ensure_ascii=False, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')


class MiniMaxH3PreparedVideoEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3PreparedVideoEXPT8',
            display_name='Tao / LTX · 串行生成与解码 (T8 EXP)', category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description='完整身份检查→生成latent→生成进程退出→独立VAE解码。Windows单GPU实验入口。已有严格匹配缓存只复用；报告区分新生成/恢复/旧回执迁移。尚未代表所有输入/模型质量通过。',
            inputs=[PreparedBundle.Input('prepared_bundle'),
                io.Int.Input('noise_seed', default=8301, min=0, max=0xffffffffffffffff,
                    tooltip='视频噪声种子；Tao多请求依次加1（uint64回绕）。教师音频种子来自准备清单，不随这里改变。'),
                io.String.Input('chain_id', default='prepared_trial_01',
                    tooltip='只用字母数字下划线连字符。改输入/种子/代码或关闭恢复时请换新ID；不会覆盖旧结果。'),
                io.Boolean.Input('resume_existing', default=True,
                    tooltip='仅复用身份和哈希完全匹配的阶段。只生成成功而解码失败时只补解码。'),
                io.String.Input('serial_lease_path', default='', advanced=True,
                    tooltip='本地串行GPU锁文件的绝对路径，必须与同时使用的其他研究入口一致。不要通过换锁文件绕过占用。')],
            outputs=[io.Video.Output('video'), io.String.Output('saved_path'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, prepared_bundle, noise_seed, chain_id, resume_existing, serial_lease_path):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from .prepared_identity import absolute_path
        from .prepared_generation_runtime import run_prepared
        lease = Path(absolute_path(serial_lease_path))
        path, report = run_prepared(deepcopy(prepared_bundle),
            output_directory=folder_paths.get_output_directory(), chain_id=chain_id,
            noise_seed=noise_seed, resume_existing=resume_existing, lease_path=lease,
            interrupt=throw_exception_if_processing_interrupted)
        relative = Path(path).resolve().relative_to(Path(folder_paths.get_output_directory()).resolve())
        return io.NodeOutput(InputImpl.VideoFromFile(str(path)), str(path),
            json.dumps(report, ensure_ascii=False, indent=2),
            ui=ui.PreviewVideo([ui.SavedResult(relative.name, str(relative.parent), io.FolderType.output)]))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')


PREPARED_GENERATION_NODE_CLASSES = [MiniMaxH3PreparedGenerationBundleEXPT8, MiniMaxH3PreparedVideoEXPT8]
