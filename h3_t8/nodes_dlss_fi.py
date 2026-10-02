from __future__ import annotations

import json
from pathlib import Path

from comfy_api.latest import io, ui


class MiniMaxH3DLSSFrameInterpolationEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MiniMaxH3DLSSFrameInterpolationEXPT8',
            display_name='MiniMax H3 DLSS 视频插帧 2x (T8 EXP)',
            category='T8/MiniMax H3/Post FX/DLSS-FI',
            description='独立成片插帧：24→48 / 30→60fps，时长和原音轨不变。不是超分，也不加速 H3 采样。'
                        '仅 Windows 单 RTX 显卡、8bit SDR 恒定帧率文件；需要自行提供独立 DLSSG 运行文件。',
            is_experimental=True, is_output_node=True,
            inputs=[
                io.Video.Input('source_video'),
                io.String.Input('runtime_directory', default='', tooltip='留空使用 ComfyUI/models/DLSS-FI/310.7.0；'
                                '也可填已安装运行目录。需要 dlssg-worker.exe 和 nvngx_dlssg.dll，不是 DLSS-NR 目录。'),
                io.String.Input('cut_frames', default='', advanced=True,
                                tooltip='硬切后第一帧的索引，从 0 开始，例如 24,48。留空表示无硬切；不会自动识别镜头。'),
                io.Int.Input('timeout_seconds', default=600, min=30, max=3600, advanced=True,
                             tooltip='完整文件处理的最长秒数；超时停止本节点自己的子进程，不重试。'),
            ],
            outputs=[io.Video.Output('interpolated_video'), io.Video.Output('source_video'),
                     io.String.Output('saved_path'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, source_video, runtime_directory='', cut_frames='', timeout_seconds=600):
        import folder_paths
        from comfy_api.latest import InputImpl
        import comfy.model_management
        from .dlss_fi_backend.entry import file_video_path, process_file

        check = comfy.model_management.throw_exception_if_processing_interrupted
        source = file_video_path(source_video, InputImpl.VideoFromFile)
        runtime = (Path(runtime_directory.strip()) if runtime_directory.strip() else
                   Path(folder_paths.models_dir)/'DLSS-FI/310.7.0')
        try:
            saved, report = process_file(source, runtime, folder_paths.get_output_directory(),
                                         cuts=cut_frames, timeout=timeout_seconds, check=check)
        except Exception:
            check()  # Preserve ComfyUI's cancellation semantics after owned-tree cleanup.
            raise
        return io.NodeOutput(InputImpl.VideoFromFile(str(saved)), source_video, str(saved),
            json.dumps(report, ensure_ascii=False, indent=2),
            ui=ui.PreviewVideo([ui.SavedResult(saved.name, '', io.FolderType.output)]))

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float('nan')
