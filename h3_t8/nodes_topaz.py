"""Optional official Topaz postprocessing. No discovery/installation at import."""
import json
from pathlib import Path
import tempfile
import uuid

import folder_paths
from comfy_api.latest import io, InputImpl


CATEGORY = 'T8/MiniMax H3/Post FX/Topaz'
TopazRuntime = io.Custom('T8_OFFICIAL_TOPAZ_RUNTIME')
TOPAZ_REGULAR_MODEL_IDS = [
    'iris-3', 'prob-4', 'rhea-1', 'nyx-3', 'ahq-12', 'amq-13', 'alq-13',
    'thf-4', 'thd-3', 'ghq-5', 'gcg-5', 'ganim-1', 'nxf-1', 'nxl-1',
]
TOPAZ_MODEL_GUIDE = (
    'iris-3 Iris: real faces, low-res portraits; prob-4 Proteus: general-purpose with the most parameters;'
    'rhea-1 Rhea: medium-quality detail recovery; nyx-3/nxf-1/nxl-1 Nyx: strong denoise;'
    'ahq-12/amq-13/alq-13 Artemis: for high/medium/low-quality footage respectively;'
    'thf-4 Theia Fidelity: fidelity-leaning; thd-3 Theia Detail: detail-leaning;'
    'ghq-5 Gaia HQ: high-quality live-action/CG; gcg-5 Gaia CG: CG, aliasing, and moiré;'
    'ganim-1 Gaia Animation: animation. In the current official definition, Gaia Animation is usually 2x only, Nyx Fast/XL 1x only,'
    'Nyx is usually 1x/2x; other common models usually have 1x/2x/4x. Per-install versions follow the environment report,'
    'A missing scale factor is not a failed download.')
TOPAZ_FI_MODEL_IDS = ['apo-8', 'apf-2', 'chr-2', 'chf-3', 'aion-1']
TOPAZ_FI_GUIDE = ('apo-8 Apollo: high-quality general, suited to nonlinear motion; apf-2 Apollo Fast: speed-first;'
    'chr-2 Chronos: general high-quality variable frame rate; chf-3 Chronos Fast: speed-first;'
    'aion-1 Aion: new general interpolation. Download the corresponding model in the official Topaz app first.')


def _parameter_overrides(mode, auto_estimate_frames, preblur, noise, details, halo,
                         sharpen, compression, add_noise, grain, grain_size,
                         keep_color, input_blend, parameters_json,
                         supported_model_parameters=None):
    try:
        custom = json.loads(parameters_json)
    except json.JSONDecodeError as error:
        raise ValueError('Advanced parameters JSON is invalid; for normal use keep it as {}') from error
    if not isinstance(custom, dict):
        raise ValueError('Advanced parameters JSON must be an object, e.g. {}')
    if mode == 'model_defaults':
        values = {}
    elif mode == 'auto_estimate':
        values = {'estimate': auto_estimate_frames}
    elif mode == 'manual':
        requested = {'preblur': preblur, 'noise': noise, 'details': details,
            'halo': halo, 'blur': sharpen, 'compression': compression}
        supported = (set(requested) if supported_model_parameters is None
            else {str(name).lower() for name in supported_model_parameters})
        values = {'estimate': 0, **{name: value for name, value in requested.items()
            if name in supported}, 'prenoise': add_noise, 'grain': grain,
            'gsize': grain_size, 'kcolor': int(keep_color), 'blend': input_blend}
    else:
        raise ValueError('unknown Topaz Parameter mode')
    values.update(custom)
    return values


class MiniMaxH3TopazEnvironmentEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TopazEnvironmentEXPT8',
            display_name='Official Topaz · Environment Check (T8 EXP)', category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description='Checks the official installation, signature, and filters; lists model IDs, per-scale candidate weights, and available parameters in the report. Files present does not mean they run. Does not install, download, or run enhancement; Starlight requires a separate official Neuroserver and valid license.',
            inputs=[io.String.Input('install_directory', default='', tooltip='Official Topaz Video program directory, not a model directory.'),
                io.String.Input('model_definitions_directory', default='', tooltip='The models JSON definitions directory for the official install.'),
                io.String.Input('model_data_directory', default='', tooltip='The model data directory configured by official Topaz; it may differ from the definitions directory.')],
            outputs=[TopazRuntime.Output('topaz_runtime'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, install_directory, model_definitions_directory, model_data_directory):
        from .topaz_contract import OfficialTopaz
        from .topaz_runtime import audit_installation
        if not all(isinstance(p, str) and p.strip() for p in
                (install_directory, model_definitions_directory, model_data_directory)):
            raise ValueError('Please fill in the official Topaz program, model definitions, and model data directories. Not installing Topaz does not affect other T8 nodes.')
        runtime = OfficialTopaz(install_directory, model_definitions_directory, model_data_directory)
        report = audit_installation(runtime)
        handle = {'schema': 't8_official_topaz_paths_v1', 'install': str(runtime.install),
            'definitions': str(runtime.definitions), 'data': str(runtime.data)}
        return io.NodeOutput(handle, json.dumps(report, ensure_ascii=False, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')


class MiniMaxH3TopazVideoEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TopazVideoEXPT8',
            display_name='Official Topaz · Video Upscale (T8 EXP)', category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description='Official tvai_up upscale; by default accepts common 8',
            inputs=[TopazRuntime.Input('topaz_runtime'), io.Video.Input('source_video'),
                io.Combo.Input('model_id', options=TOPAZ_REGULAR_MODEL_IDS, default='iris-3',
                    display_name='Upscale model (see tooltip for use)', tooltip=TOPAZ_MODEL_GUIDE),
                io.Combo.Input('scale', options=['1x', '2x', '4x'], default='2x', tooltip='Does not change frame rate; prepare the corresponding scale weights in official Topaz.'),
                io.Float.Input('vram_fraction', default=.8, min=.1, max=1., step=.05, advanced=True,
                    tooltip='Fraction of VRAM Topaz may use; this is not a startup threshold. Default 0.80; try 0.60 under VRAM pressure.'),
                io.String.Input('parameters_json', default='{}', multiline=True, advanced=True,
                    display_name='Advanced parameters JSON (usually keep {})',
                    tooltip='For precise override of standalone controls or future parameters only; normal users should leave it empty.'),
                io.Combo.Input('size_mode', options=['scale', 'target_dimensions'], default='scale',
                    optional=True, advanced=True, tooltip='scale uses the ratio above. target_dimensions uses the width/height below and ignores ratio; keep the original aspect, 1x to 4x, no crop or stretch.'),
                io.Int.Input('target_width', default=0, min=0, max=8192, step=2, optional=True, advanced=True,
                    tooltip='Active only in custom-size mode; enter an even number between 32 and 8192.'),
                io.Int.Input('target_height', default=0, min=0, max=8192, step=2, optional=True, advanced=True,
                    tooltip='Active only in custom-size mode; width and height must both be entered and preserve the source video aspect.'),
                io.String.Input('output_directory', default='', optional=True, advanced=True,
                    tooltip='Optional master output directory. Leave empty to write to ComfyUI/output/MiniMaxH3-Topaz; under disk pressure, enter another local directory.'),
                io.String.Input('custom_model_id', default='', optional=True, advanced=True,
                    tooltip='For new official model IDs not yet in the list only; if entered, overrides model_id.'),
                io.Combo.Input('output_profile', options=['delivery_h264', 'delivery_hevc_main10', 'lossless_master'],
                    default='delivery_h264', optional=True, advanced=True,
                    display_name='Advanced output policy (normal users keep default)',
                    tooltip='Default delivery_h264 accepts common 8'),
                io.Combo.Input('parameter_mode', options=['model_defaults', 'auto_estimate', 'manual'],
                    default='model_defaults', optional=True, display_name='Parameter mode',
                    tooltip='model_defaults uses model defaults; auto_estimate lets Topaz sample frames to estimate; manual uses the standalone parameters below.'),
                io.Int.Input('auto_estimate_frames', default=8, min=1, max=100, optional=True,
                    advanced=True, display_name='Auto-analyze frames', tooltip='Active only in auto_estimate; default 8 is stable, complex long films can try 12-20; more means slower startup.'),
                io.Float.Input('preblur', default=0., min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='De-alias (-) / De-blur (+)',
                    tooltip='Negative leans de-alias/moiré, positive leans de-blur; e.g. -0.20 or +0.20. Active only in manual.'),
                io.Float.Input('noise', default=.5, min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='Denoise', tooltip='e.g. 0.20 light, 0.45 medium; too high waxes skin. Active only in manual.'),
                io.Float.Input('details', default=.5, min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='Recover detail', tooltip='e.g. 0.25 natural, 0.50 stronger; too high may invent fake texture. Active only in manual.'),
                io.Float.Input('halo', default=.5, min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='De-halo', tooltip='e.g. 0.20 light, 0.50 for obvious edge halos; too high loses edges. Active only in manual.'),
                io.Float.Input('sharpen', default=.5, min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='Sharpen', tooltip='Internally maps to the Topaz blur parameter; e.g. 0.15 light, 0.35 sharper. Active only in manual.'),
                io.Float.Input('compression', default=.5, min=-1., max=1., step=.05, optional=True,
                    advanced=True, display_name='De-block', tooltip='e.g. 0.25 mild web compression, 0.55 obvious blocking. Active only in manual.'),
                io.Float.Input('add_noise', default=0., min=0., max=.1, step=.005, optional=True,
                    advanced=True, display_name='Input pre-noise', tooltip='Default 0; e.g. 0.01 helps retain texture on over-smoothed footage. Active only in manual.'),
                io.Float.Input('grain', default=0., min=0., max=1., step=.05, optional=True,
                    advanced=True, display_name='Output grain', tooltip='Default 0; e.g. 0.10 light grain, 0.20 obvious film look. Active only in manual.'),
                io.Float.Input('grain_size', default=0., min=0., max=5., step=.1, optional=True,
                    advanced=True, display_name='Grain size', tooltip='Only meaningful when output grain > 0; 1.0 fine, 2.0 coarser.'),
                io.Boolean.Input('keep_color', default=True, optional=True, advanced=True,
                    display_name='Model color correction', tooltip='Default true; turning off is recommended only for color comparison tests.'),
                io.Float.Input('input_blend', default=0., min=0., max=1., step=.05, optional=True,
                    advanced=True, display_name='Blend with original', tooltip='0 = fully enhanced, 1 = fully original; 0.10-0.20 can reduce over-sharpening or waxiness.'),
                io.Int.Input('engine_instances', default=0, min=0, max=3, optional=True,
                    advanced=True, display_name='Extra inference instances',
                    tooltip='0 is safest. On this machine 1 instance took 60 frames from 11.5s to 10.5s; small gain but more VRAM; not recommended above 1 on 16GB cards.'),
                io.Int.Input('gpu_device_index', default=0, min=0, max=15, optional=True,
                    advanced=True, display_name='Topaz GPU index',
                    tooltip='Keep 0 for a single card; on multi-GPU, choose the GPU index used by official Topaz.')],
            outputs=[io.Video.Output('enhanced_video'), io.Video.Output('source_video'),
                io.String.Output('saved_path'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, topaz_runtime, source_video, model_id, scale, vram_fraction=.8, parameters_json='{}',
                size_mode='scale', target_width=0, target_height=0, output_directory='', custom_model_id='',
                output_profile='delivery_h264', parameter_mode='model_defaults', auto_estimate_frames=8,
                preblur=0., noise=.5, details=.5, halo=.5, sharpen=.5, compression=.5,
                add_noise=0., grain=0., grain_size=0., keep_color=True, input_blend=0., engine_instances=0,
                gpu_device_index=0):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from comfy.utils import ProgressBar
        from .dlss_fi_backend.entry import file_video_path
        from .topaz_contract import OfficialTopaz
        from .topaz_runtime import run_regular
        if not isinstance(topaz_runtime, dict) or topaz_runtime.get('schema') != 't8_official_topaz_paths_v1':
            raise ValueError('Connect the official Topaz environment node')
        if scale not in ('1x', '2x', '4x') or not isinstance(parameters_json, str) or len(parameters_json) > 8192:
            raise ValueError('Invalid Topaz scale or parameter JSON')
        if not isinstance(custom_model_id, str):
            raise ValueError('Custom Topaz model ID must be a string')
        model_id = custom_model_id.strip() or model_id
        runtime = OfficialTopaz(topaz_runtime['install'], topaz_runtime['definitions'], topaz_runtime['data'])
        _, definition = runtime.model(model_id)
        declared = {str(item.get('name', '')).lower() for item in definition.get('parameters', [])}
        parameters = _parameter_overrides(parameter_mode, auto_estimate_frames, preblur,
            noise, details, halo, sharpen, compression, add_noise, grain, grain_size,
            keep_color, input_blend, parameters_json, supported_model_parameters=declared)
        manual_controls = {'preblur', 'noise', 'details', 'halo', 'blur', 'compression'}
        parameter_audit = {'mode': parameter_mode,
            'ignored_unsupported_model_controls': sorted(manual_controls - declared)
                if parameter_mode == 'manual' else []}
        try:
            source = file_video_path(source_video, InputImpl.VideoFromFile)
        except ValueError as error:
            raise ValueError('Topaz requires an untrimmed, uncropped local file VIDEO. Save edited/frame-batch VIDEO first.') from error
        if not isinstance(output_directory, str):
            raise ValueError('Topaz output directory must be a local path string')
        if output_directory.strip():
            root = Path(output_directory.strip())
            if not root.is_absolute() or root.parent == root:
                raise ValueError('Custom Topaz output directory must be an absolute non-root path')
            root.mkdir(parents=True, exist_ok=True)
            root = root.resolve(strict=True)
        else:
            output = Path(folder_paths.get_output_directory()).resolve(strict=True)
            root = output / 'MiniMaxH3-Topaz'
            root.mkdir(exist_ok=True)
            root = root.resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            raise ValueError('Topaz output directory must be a real local directory')
        progress_bar = ProgressBar(100)
        path, report = run_regular(runtime, source, root / ('task-' + uuid.uuid4().hex),
            model_id=model_id, width=target_width if size_mode == 'target_dimensions' else None,
            height=target_height if size_mode == 'target_dimensions' else None,
            scale=int(scale[0]), size_mode=size_mode,
            output_profile=output_profile,
            instances=engine_instances,
            parameter_audit=parameter_audit,
            progress=lambda value: progress_bar.update_absolute(value, 100),
            lease_path=Path(tempfile.gettempdir()) / 'T8-Topaz-serial.lock',
            device=gpu_device_index, vram=vram_fraction, parameters=parameters,
            interrupt=throw_exception_if_processing_interrupted)
        return io.NodeOutput(InputImpl.VideoFromFile(str(path)), source_video, str(path),
            json.dumps(report, ensure_ascii=False, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')


class MiniMaxH3TopazFrameInterpolationEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TopazFrameInterpolationEXPT8',
            display_name='Official Topaz · Video Frame Interpolation (T8 EXP)', category=CATEGORY,
            is_experimental=True, is_output_node=True,
            description='Standalone official tvai_fi interpolation: 2x/4x raises frame rate while keeping duration and size; by default accepts common SDR bit depths and saves high-quality 8-bit H.264 MP4, converting non-compatible audio to AAC. Not slow motion, not mixed with upscale models, and the Topaz app need not stay open.',
            inputs=[TopazRuntime.Input('topaz_runtime'), io.Video.Input('source_video'),
                io.Combo.Input('model_id', options=TOPAZ_FI_MODEL_IDS, default='apo-8',
                    display_name='Interpolation model (see tooltip for use)', tooltip=TOPAZ_FI_GUIDE),
                io.Combo.Input('multiplier', options=['2x', '4x'], default='2x',
                    display_name='Frame rate multiplier', tooltip='2x: 24->48, 30->60; 4x: 24->96. Duration is unchanged.'),
                io.Float.Input('duplicate_threshold', default=.01, min=-.01, max=.2, step=.005,
                    advanced=True, display_name='Duplicate-frame detection threshold',
                    tooltip='0 or negative disables duplicate-frame detection; too high may wrongly remove normal static frames. Usually keep 0.01.'),
                io.Float.Input('vram_fraction', default=.8, min=.1, max=1., step=.05, advanced=True),
                io.String.Input('output_directory', default='', optional=True, advanced=True,
                    tooltip='Leave empty to write to ComfyUI/output/MiniMaxH3-Topaz; another local drive is also allowed.'),
                io.String.Input('custom_model_id', default='', optional=True, advanced=True,
                    tooltip='For official interpolation model IDs not yet in the list only; if entered, overrides model_id.'),
                io.Int.Input('engine_instances', default=0, min=0, max=3, optional=True,
                    advanced=True, display_name='Extra inference instances', tooltip='0 is safest; more uses more VRAM.'),
                io.Combo.Input('output_profile', options=['delivery_h264', 'delivery_hevc_main10'],
                    default='delivery_h264', optional=True, advanced=True,
                    display_name='Advanced output policy (normal users keep default)',
                    tooltip='Default delivery_h264 accepts common 8/10/16-bit SDR and uniformly outputs 8-bit H.264, no bit-depth selection needed. Main10 is for advanced users who explicitly need retained 10-bit. HDR must still be converted to SDR first.'),
                io.Int.Input('gpu_device_index', default=0, min=0, max=15, optional=True,
                    advanced=True, display_name='Topaz GPU index', tooltip='Keep 0 for a single card; on multi-GPU choose the GPU Topaz uses.')],
            outputs=[io.Video.Output('interpolated_video'), io.Video.Output('source_video'),
                io.String.Output('saved_path'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, topaz_runtime, source_video, model_id, multiplier,
                duplicate_threshold=.01, vram_fraction=.8, output_directory='', custom_model_id='',
                engine_instances=0, output_profile='delivery_h264', gpu_device_index=0):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from comfy.utils import ProgressBar
        from .dlss_fi_backend.entry import file_video_path
        from .topaz_contract import OfficialTopaz
        from .topaz_runtime import run_interpolation
        if not isinstance(topaz_runtime, dict) or topaz_runtime.get('schema') != 't8_official_topaz_paths_v1':
            raise ValueError('Please connect the Official Topaz Environment Check node')
        model_id = custom_model_id.strip() or model_id
        runtime = OfficialTopaz(topaz_runtime['install'], topaz_runtime['definitions'], topaz_runtime['data'])
        try:
            source = file_video_path(source_video, InputImpl.VideoFromFile)
        except ValueError as error:
            raise ValueError('Topaz interpolation requires an uncropped local-file VIDEO; save frame batches as a video first.') from error
        if output_directory.strip():
            root = Path(output_directory.strip())
            if not root.is_absolute() or root.parent == root:
                raise ValueError('The custom Topaz output directory must be an absolute non-drive-root path')
            root.mkdir(parents=True, exist_ok=True)
            root = root.resolve(strict=True)
        else:
            root = Path(folder_paths.get_output_directory()).resolve(strict=True) / 'MiniMaxH3-Topaz'
            root.mkdir(exist_ok=True)
            root = root.resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            raise ValueError('The Topaz output directory must be a real local directory')
        progress_bar = ProgressBar(100)
        path, report = run_interpolation(runtime, source, root / ('task-' + uuid.uuid4().hex),
            model_id=model_id, multiplier=int(multiplier[0]),
            lease_path=Path(tempfile.gettempdir()) / 'T8-Topaz-serial.lock',
            device=gpu_device_index, vram=vram_fraction, instances=engine_instances,
            duplicate_threshold=duplicate_threshold, output_profile=output_profile,
            progress=lambda value: progress_bar.update_absolute(value, 100),
            interrupt=throw_exception_if_processing_interrupted)
        return io.NodeOutput(InputImpl.VideoFromFile(str(path)), source_video, str(path),
            json.dumps(report, ensure_ascii=False, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')


TOPAZ_NODE_CLASSES = [MiniMaxH3TopazEnvironmentEXPT8, MiniMaxH3TopazVideoEXPT8,
                      MiniMaxH3TopazFrameInterpolationEXPT8]
