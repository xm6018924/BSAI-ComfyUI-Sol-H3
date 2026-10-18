from __future__ import annotations

import json
from pathlib import Path

from comfy_api.latest import io, ui


class MiniMaxH3DLSSFrameInterpolationEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MiniMaxH3DLSSFrameInterpolationEXPT8',
            display_name='MiniMax H3 DLSS Video Frame Interpolation 2x (T8 EXP)',
            category='T8/MiniMax H3/Post FX/DLSS-FI',
            description='Standalone file frame interpolation: 24->48 / 30->60 fps; duration and original audio track are unchanged. This is not super-resolution and does not speed up H3 sampling.'
                        'Windows-only, single RTX GPU, 8-bit SDR constant-frame-rate files. You must provide the standalone DLSSG runtime yourself.',
            is_experimental=True, is_output_node=True,
            inputs=[
                io.Video.Input('source_video'),
                io.String.Input('runtime_directory', default='', tooltip='models/DLSS-FI/310.7.0；'
                                'or set an installed runtime directory. Requires dlssg-worker.exe and nvngx_dlssg.dll; this is not the DLSS-NR directory.'),
                io.String.Input('cut_frames', default='', advanced=True,
                                tooltip='Index of the first frame after a hard cut, 0-based, e.g. 24,48. Leave empty for no hard cut; shot detection is not automatic.'),
                io.Int.Input('timeout_seconds', default=600, min=30, max=3600, advanced=True,
                             tooltip='Maximum seconds for full-file processing; on timeout the node kills its own subprocess and does not retry.'),
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
