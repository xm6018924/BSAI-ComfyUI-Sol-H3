"""Opt-in MODEL observer; existing samplers/workflows require no new inputs."""
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .taeh3_sampling_preview import PreviewSettings, attach_preview


class MiniMaxH3TAEH3SamplingPreviewEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = [name for name in folder_paths.get_filename_list('vae_approx') if name.startswith('taeh3')]
        return io.Schema(node_id=cls.__name__, display_name='H3 First-Pass Live Preview + Safe Cancel (TAEH3 EXP/T8)',
            category='T8/MiniMax H3/Experimental', is_experimental=True,
            description='Chain the MODEL to the original sampler. Previews a continuous head segment of the actual x0; for approximate motion observation only, not final quality/audio.'
                'Does not add diffusion steps; small decode overhead. LOW/HIGH comes from the actual stage; plain sampling is marked unrecognized.'
                'Cancel affects only the request bound to this panel; it does not clear the queue or stop the server. If the preview model is missing, the original sampling continues.'
                'Model download: https://huggingface.co/t8star/Taeh3-Comfy; keep the models/vae_approx directory.',
            inputs=[io.Model.Input('model'), io.Boolean.Input('enabled', default=True),
                io.Combo.Input('checkpoint', options=files or ['taeh3.safetensors'],
                    tooltip='Download: https://huggingface.co/t8star/Taeh3-Comfy; place in models/vae_approx.'
                        'taeh3.safetensors is the temporal version; save the 2D version as taeh3_2d_kijai.safetensors and do not overwrite the temporal one.'),
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
