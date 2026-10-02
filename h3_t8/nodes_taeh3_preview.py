"""Opt-in MODEL observer; existing samplers/workflows require no new inputs."""
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .taeh3_sampling_preview import PreviewSettings, attach_preview


class MiniMaxH3TAEH3SamplingPreviewEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = [name for name in folder_paths.get_filename_list('vae_approx') if name.startswith('taeh3')]
        return io.Schema(node_id=cls.__name__, display_name='H3 一采动态预览＋安全取消（TAEH3 EXP/T8）',
            category='T8/MiniMax H3/Experimental', is_experimental=True,
            description='MODEL串联到原采样器。预览实际x0的连续开头片段，仅近似观察运动，不是最终画质／声音。'
                '不会增加扩散步；有少量解码开销。LOW/HIGH来自实际阶段；普通采样标为未识别。'
                '取消仅作用于本面板绑定的请求，不清队列、不结束服务。缺少预览模型时原采样继续。'
                '模型下载：https://huggingface.co/t8star/Taeh3-Comfy；保留models/vae_approx目录。',
            inputs=[io.Model.Input('model'), io.Boolean.Input('enabled', default=True),
                io.Combo.Input('checkpoint', options=files or ['taeh3.safetensors'],
                    tooltip='下载：https://huggingface.co/t8star/Taeh3-Comfy；放models/vae_approx。'
                        'taeh3.safetensors为时序版；2D另存taeh3_2d_kijai.safetensors，不覆盖时序版。'),
                io.Combo.Input('phase', options=['low', 'high', 'all']),
                io.Int.Input('update_every_steps', default=2, min=1, max=1000),
                io.Int.Input('min_interval_ms', default=500, min=0, max=60000),
                io.Int.Input('max_resolution', default=256, min=64, max=512),
                io.Combo.Input('latent_prefix', options=['7', '2', '12']),
                io.Int.Input('frames', default=12, min=1, max=24),
                io.Int.Input('fps', default=12, min=1, max=24),
                io.Int.Input('jpeg_quality', default=75, min=30, max=95)],
            hidden=[io.Hidden.unique_id], outputs=[io.Model.Output('model')])

    @classmethod
    def execute(cls, model, enabled, checkpoint, latent_prefix, **kwargs):
        if not enabled:
            return io.NodeOutput(model)
        path = folder_paths.get_full_path('vae_approx', checkpoint)
        if path is None:
            path = str(Path(folder_paths.get_folder_paths('vae_approx')[0]) / checkpoint)
        settings = PreviewSettings(model_path=path, latent_prefix=int(latent_prefix), **kwargs)
        return io.NodeOutput(attach_preview(model, settings, cls.hidden.unique_id))
