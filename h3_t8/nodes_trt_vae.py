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
                            tooltip='原版H3视频VAE；不选择音频VAE。编码保持原生时仍需这份权重。'),
              io.Combo.Input('decoder_engine',options=decoder,tooltip='本机编译的解码引擎；单帧或小尺寸需要flex引擎。'),
              io.String.Input('runtime_directory',default='',tooltip='TensorRT独立site-packages绝对路径；留空使用models/vae/h3_trt/runtime/site-packages。'),
              io.Int.Input('cpu_output_budget_mib',default=1024,min=64,max=8192,advanced=True,
                           tooltip='解码后的CPU像素缓冲上限，不是显存上限。')]
    if full:
        encoders = bundle_options(folder_paths.models_dir,'encoder')
        inputs += [io.Combo.Input('image_encoder_engine',options=encoders,tooltip='选择T1单帧编码引擎，不把单图补成17帧。'),
                   io.Combo.Input('video_encoder_engine',options=encoders,tooltip='选择T17视频编码引擎；内部自动分块。')]
    return inputs


class MiniMaxH3TRTVAECheckEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TRTVAECheckEXPT8',display_name='MiniMax H3 TRT VAE 安装检查 (T8 EXP)',
            category=CATEGORY,is_experimental=True,is_output_node=True,
            description='只检查独立运行文件是否齐全，不编译、不加载GPU，也不安装依赖。文件齐全不代表实际推理已验收。',
            inputs=[io.String.Input('runtime_directory',default='',tooltip='留空使用models/vae/h3_trt/runtime/site-packages。')],
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
        return io.Schema(node_id='MiniMaxH3TRTVAEDecoderEXPT8',display_name='MiniMax H3 TRT VAE 解码加速 (T8 EXP)',
            category=CATEGORY,is_experimental=True,
            description='推荐先用此模式：只替换视频解码，图片/参考视频编码保持原生。连接到原来video_vae的位置，音频VAE不变。',
            inputs=loader_inputs(False),outputs=[io.Vae.Output('video_vae'),io.String.Output('report_json')])

    @classmethod
    def execute(cls,native_video_vae,decoder_engine,runtime_directory='',cpu_output_budget_mib=1024):
        return execute_loader(native_video_vae,decoder_engine,runtime_directory,cpu_output_budget_mib)


class MiniMaxH3TRTVAEFullEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TRTVAEFullEXPT8',display_name='MiniMax H3 TRT VAE 编码＋解码 (T8 EXP)',
            category=CATEGORY,is_experimental=True,
            description='可选完整模式：参考图/视频编码也用TRT，编码差异可能影响后续采样。单图未体现提速，默认建议只加速解码。',
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
        return io.Schema(node_id='MiniMaxH3TRTVAECompileEXPT8',display_name='MiniMax H3 TRT VAE 本机编译 (T8 EXP)',
            category=CATEGORY,is_experimental=True,is_output_node=True,
            description='只在准备引擎时运行。独立子进程串行编译，不改已有引擎，不下载或安装依赖；取消只结束本次子进程。',
            inputs=[io.Combo.Input('kind',options=['decoder-flex','decoder','encoder-t1','encoder','decoder-w4a16'],
                                   default='decoder-flex',tooltip='推荐flex解码；W4A16已审短片整体可接受，但量化差异更大，不承诺省显存或更快。'),
                    io.String.Input('runtime_directory',default='',tooltip='留空使用models/vae/h3_trt/runtime/site-packages。'),
                    io.Int.Input('timeout_seconds',default=1800,min=1,max=3600,advanced=True,
                                 tooltip='单次编译时间上限。编译需要至少12000MiB空闲显存和24GiB空闲内存。')],
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
            raise ValueError('当前TRT VAE编译器仅验证Windows；不影响其他原生H3节点。')
        if kind not in names or type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3600:
            raise ValueError('编译类型或超时时间无效')
        site = resolve_runtime(runtime_directory,folder_paths.models_dir)
        report = inspect_runtime(site)
        if report['missing']:
            raise ValueError(report['message'])
        root = Path(folder_paths.models_dir)/'vae/h3_trt'
        model = root/names[kind]
        if not model.is_file():
            raise ValueError(f'缺少编译源文件：{model}。请先按TRT VAE说明准备模型；节点不自动下载。')
        log = Path(folder_paths.get_output_directory())/'MiniMaxH3/TRT-VAE/builds'/uuid.uuid4().hex
        result = compile_engine(['--runtime-site',str(site),'--onnx',str(model),
            '--run-dir',str(log.resolve()),'--kind',kind,'--timeout',str(timeout_seconds),'--execute'],
            check_interrupt=mm.throw_exception_if_processing_interrupted)
        return io.NodeOutput(result['bundle'],json.dumps(result,ensure_ascii=False,indent=2))


TRT_VAE_NODE_CLASSES = [MiniMaxH3TRTVAECheckEXPT8,MiniMaxH3TRTVAEDecoderEXPT8,
                        MiniMaxH3TRTVAEFullEXPT8,MiniMaxH3TRTVAECompileEXPT8]
