"""Optional TRT VAE nodes; registration is separate from research qualification."""
import json
from comfy_api.latest import io

from .trt_vae_loader import runtime_directory,bundle_options,resolve_bundle,inspect_runtime,load_vae

CATEGORY = "T8/MiniMax H3/Acceleration/TRT VAE"


def loader_inputs(full):
    import folder_paths
    native = folder_paths.get_filename_list('vae')
    decoder = bundle_options(folder_paths.models_dir,'decoder')
    inputs = [io.Combo.Input('native_video_vae',options=native or ['minimax_h3_video_vae_fp16.safetensors'],
                            default='minimax_h3_video_vae_fp16.safetensors' if 'minimax_h3_video_vae_fp16.safetensors' in native else None,
                            tooltip='Original H3 video VAE; do not select an audio VAE. This weight is still needed even when encoding stays native.'),
              io.Combo.Input('decoder_engine',options=decoder,tooltip='Locally compiled decode engine; single-frame or small sizes require the flex engine.'),
              io.String.Input('runtime_directory',default='',tooltip='Absolute path to the standalone TensorRT site-packages; leave empty to use models/vae/h3_trt/runtime/site-packages.'),
              io.Int.Input('cpu_output_budget_mib',default=1024,min=64,max=8192,advanced=True,
                           tooltip='Post-decode CPU pixel buffer cap; this is not a VRAM limit.')]
    if full:
        encoders = bundle_options(folder_paths.models_dir,'encoder')
        inputs += [io.Combo.Input('image_encoder_engine',options=encoders,tooltip='Select the T1 single-frame encode engine; does not pad a single image to 17 frames.'),
                   io.Combo.Input('video_encoder_engine',options=encoders,tooltip='Select the T17 video encode engine; chunks internally.')]
    return inputs


class MiniMaxH3TRTVAECheckEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TRTVAECheckEXPT8',display_name='MiniMax H3 TRT VAE Install Check (T8 EXP)',
            category=CATEGORY,is_experimental=True,is_output_node=True,
            description='Only checks that standalone runtime files are present; does not compile, load the GPU, or install dependencies. Present files do not mean inference is validated.',
            inputs=[io.String.Input('runtime_directory',default='',tooltip='vae/h3_trt/runtime/site-packages。')],
            outputs=[io.Boolean.Output('files_present'),io.String.Output('status'),io.String.Output('report_json')])

    @classmethod
    def execute(cls,runtime_directory=''):
        import folder_paths
        from .trt_vae_loader import runtime_directory as resolve_runtime
        report = inspect_runtime(resolve_runtime(runtime_directory,folder_paths.models_dir))
        return io.NodeOutput(not report['missing'],report['message'],json.dumps(report,ensure_ascii=False,indent=2))

    @classmethod
    def fingerprint_inputs(cls,**kwargs):
        return float('nan')


class MiniMaxH3TRTVAEDecoderEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TRTVAEDecoderEXPT8',display_name='MiniMax H3 TRT VAE Decode Acceleration (T8 EXP)',
            category=CATEGORY,is_experimental=True,
            description='Recommended first mode: replaces only video decode; image/reference-video encoding stays native. Connect where the original video_vae was; the audio VAE is unchanged.',
            inputs=loader_inputs(False),outputs=[io.Vae.Output('video_vae'),io.String.Output('report_json')])

    @classmethod
    def execute(cls,native_video_vae,decoder_engine,runtime_directory='',cpu_output_budget_mib=1024):
        return execute_loader(native_video_vae,decoder_engine,runtime_directory,cpu_output_budget_mib)


class MiniMaxH3TRTVAEFullEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TRTVAEFullEXPT8',display_name='MiniMax H3 TRT VAE Encode + Decode (T8 EXP)',
            category=CATEGORY,is_experimental=True,
            description='Optional full mode: reference image/video encoding also uses TRT; encode differences may affect later sampling. Single images show no speedup; decoding-only acceleration is recommended by default.',
            inputs=loader_inputs(True),outputs=[io.Vae.Output('video_vae'),io.String.Output('report_json')])

    @classmethod
    def execute(cls,native_video_vae,decoder_engine,image_encoder_engine,video_encoder_engine,
                runtime_directory='',cpu_output_budget_mib=1024):
        return execute_loader(native_video_vae,decoder_engine,runtime_directory,cpu_output_budget_mib,
                              (image_encoder_engine,video_encoder_engine))


def execute_loader(native_name,decoder_name,site,budget,encoders=None):
    import folder_paths
    import comfy.model_management as mm
    native = folder_paths.get_full_path_or_raise('vae',native_name)
    decoder,_ = resolve_bundle(decoder_name,folder_paths.models_dir,'decoder')
    paths = [resolve_bundle(n,folder_paths.models_dir,'encoder')[0] for n in encoders] if encoders is not None else None
    vae,report = load_vae(native_path=native,decoder=decoder,runtime_site=runtime_directory(site,folder_paths.models_dir),
                         encoder_paths=paths,output_budget_mib=budget,check=mm.throw_exception_if_processing_interrupted)
    return io.NodeOutput(vae,json.dumps(report,ensure_ascii=False,indent=2))


class MiniMaxH3TRTVAECompileEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TRTVAECompileEXPT8',display_name='MiniMax H3 TRT VAE Local Compile (T8 EXP)',
            category=CATEGORY,is_experimental=True,is_output_node=True,
            description='Run only when preparing an engine. Serial compilation in a standalone subprocess; does not alter existing engines, download, or install dependencies; cancel only ends this subprocess.',
            inputs=[io.Combo.Input('kind',options=['decoder-flex','decoder','encoder-t1','encoder','decoder-w4a16'],
                                   default='decoder-flex',tooltip='flex decode is recommended; W4A16 passed short-film review overall, but quantization differences are larger; no VRAM savings or speedup is promised.'),
                    io.String.Input('runtime_directory',default='',tooltip='vae/h3_trt/runtime/site-packages。'),
                    io.Int.Input('timeout_seconds',default=1800,min=1,max=3600,advanced=True,
                                 tooltip='Per-compile time limit. Compilation requires at least 12000 MiB free VRAM and 24 GiB free RAM.')],
            outputs=[io.String.Output('engine_directory'),io.String.Output('report_json')])

    @classmethod
    def fingerprint_inputs(cls,**kwargs):
        return float('nan')

    @classmethod
    def execute(cls,kind='decoder-flex',runtime_directory='',timeout_seconds=1800):
        import os
        from pathlib import Path
        import uuid
        import folder_paths
        import comfy.model_management as mm
        from .trt_vae_compile import main as compile_engine
        from .trt_vae_loader import runtime_directory as resolve_runtime
        names = {'decoder-flex':'minimax_h3_vae_decoder_flex.onnx','decoder':'minimax_h3_vae_decoder.onnx',
                 'encoder-t1':'minimax_h3_vae_encoder_t1.onnx','encoder':'minimax_h3_vae_encoder.onnx',
                 'decoder-w4a16':'minimax_h3_vae_decoder_w4a16_awq.onnx'}
        if os.name != 'nt':
            raise ValueError('The TRT VAE compiler is validated on Windows only; it does not affect other native H3 nodes.')
        if kind not in names or type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3600:
            raise ValueError('Invalid compile type or timeout')
        site = resolve_runtime(runtime_directory,folder_paths.models_dir)
        report = inspect_runtime(site)
        if report['missing']:
            raise ValueError(report['message'])
        root = Path(folder_paths.models_dir)/'vae/h3_trt'
        model = root/names[kind]
        if not model.is_file():
            raise ValueError(f'缺少编译源文件：{model}。请先按TRT VAENotes准备模型；节点不自动下载。')
        log = Path(folder_paths.get_output_directory())/'MiniMaxH3/TRT-VAE/builds'/uuid.uuid4().hex
        result = compile_engine(['--runtime-site',str(site),'--onnx',str(model),
            '--run-dir',str(log.resolve()),'--kind',kind,'--timeout',str(timeout_seconds),'--execute'],
            check_interrupt=mm.throw_exception_if_processing_interrupted)
        return io.NodeOutput(result['bundle'],json.dumps(result,ensure_ascii=False,indent=2))


TRT_VAE_NODE_CLASSES = [MiniMaxH3TRTVAECheckEXPT8,MiniMaxH3TRTVAEDecoderEXPT8,
                        MiniMaxH3TRTVAEFullEXPT8,MiniMaxH3TRTVAECompileEXPT8]
