"""Public loader plumbing; optional TRT is never imported by module/schema load."""
from pathlib import Path

from .trt_vae_build import TRT_VERSION, digest_file
from .trt_vae_backend import inspect_bundle, ScopedBackend
from .trt_vae_interface import H3VAEInterface

NATIVE_VAE_SHA = "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"


def runtime_directory(value, models_directory):
    value = str(value).strip()
    path = Path(value) if value else Path(models_directory)/"vae/h3_trt/runtime/site-packages"
    if not path.is_absolute():
        raise ValueError("TensorRT运行目录请填写绝对路径，或留空使用模型目录中的默认位置")
    return path.resolve()


def bundle_options(models_directory, kind):
    """Metadata-only inventory; no GPU, runtime import, hashing GB files or deletion."""
    if kind not in ("encoder","decoder"):
        raise ValueError("Unknown engine kind")
    root = Path(models_directory)/"vae/h3_trt/engines"
    values = []
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if path.name.startswith('.') or path.is_symlink() or not path.is_dir():
                continue
            try:
                inspect_bundle(path,kind)
            except (OSError,ValueError,KeyError,TypeError):
                continue  # Corrupt/incomplete bundles are not selectable; explicit load still errors.
            values.append(path.name)
    return values or ["未找到引擎：请先编译"]


def resolve_bundle(value, models_directory, kind):
    value = str(value).strip()
    if not value or value == "未找到引擎：请先编译":
        raise ValueError("尚未选择TensorRT引擎。先完成本机编译，再刷新节点选项。")
    path = Path(value)
    if not path.is_absolute():
        if path.name != value or value in ('.','..'):
            raise ValueError("请选择引擎目录名，或填写明确的绝对路径")
        path = Path(models_directory)/"vae/h3_trt/engines"/value
    return inspect_bundle(path,kind)


def inspect_runtime(site):
    """Static installation presence only, explicitly not GPU/inference readiness."""
    site = Path(site).resolve()
    required = [site/"tensorrt/__init__.py",site/"tensorrt_bindings/__init__.py",
                site/"tensorrt_libs/nvinfer_10.dll",site/"tensorrt_libs/nvonnxparser_10.dll"]
    missing = [str(p.relative_to(site)) for p in required if not p.is_file()]
    versions = []
    for name in ("tensorrt_cu13","tensorrt_cu13_bindings","tensorrt_cu13_libs"):
        metadata = site/f"{name}-{TRT_VERSION}.dist-info/METADATA"
        if not metadata.is_file():
            missing.append(str(metadata.relative_to(site)))
        else:
            lines = metadata.read_text(encoding="utf8").splitlines()
            declared = [line.split(':',1)[1].strip() for line in lines if line.startswith('Version:')]
            if declared != [TRT_VERSION]:
                missing.append(str(metadata.relative_to(site))+":版本不符")
            versions.extend(declared)
    return {"status":"static_files_present_not_execution_qualified" if not missing else "missing_runtime_files",
            "runtime_directory":str(site),"runtime_version":TRT_VERSION,"missing":missing,
            "gpu_initialized":False,"installed_or_downloaded":False,
            "message":"文件齐全；首次执行还会校验GPU、驱动、DLL、引擎和可用显存" if not missing else
                      "运行文件不完整。请按TRT VAE说明安装到独立目录，不要替换ComfyUI的torch或CUDA。"}


def load_vae(*, native_path, decoder, runtime_site, encoder_paths=None, output_budget_mib=1024,
             check=lambda:None, native_factory=None, backend_factory=ScopedBackend):
    if type(output_budget_mib) is not int or not 64 <= output_budget_mib <= 8192:
        raise ValueError("CPU输出缓冲上限必须为64..8192MiB；这不是显存上限")
    site = Path(runtime_site).resolve()
    report = inspect_runtime(site)
    if report['missing']:
        raise ValueError(report['message']+' 缺少：'+', '.join(report['missing']))
    root, manifest = inspect_bundle(decoder,'decoder')
    if Path(manifest['source_request']['runtime_site']).resolve() != site:
        raise ValueError("引擎绑定的TensorRT运行目录不同，请选择编译时使用的目录或明确重新编译")
    bundles = {'decoder':str(root)}
    if encoder_paths is not None:
        from .trt_vae_backend import select_bundle
        if len(encoder_paths) != 2:
            raise ValueError("完整VAE需分别选择T1单帧和T17视频编码引擎")
        for shape in ((1,3,1,256,256),(1,3,17,256,256)):
            _, item = select_bundle(encoder_paths,'encoder',[shape])
            if Path(item['source_request']['runtime_site']).resolve() != site:
                raise ValueError("编码和解码引擎必须使用同一TensorRT运行目录")
        bundles['encoder'] = list(map(str,encoder_paths))
    check()
    native_path = Path(native_path).resolve(strict=True)
    if digest_file(native_path) != NATIVE_VAE_SHA:
        raise ValueError("此引擎对应原版minimax_h3_video_vae_fp16.safetensors；不能静默替换其他VAE权重")
    if native_factory is None:
        import torch
        import comfy.sd
        import comfy.utils
        import comfy.model_management as mm
        def native_factory(path):
            state,metadata = comfy.utils.load_torch_file(str(path),return_metadata=True)
            vae = comfy.sd.VAE(sd=state,metadata=metadata,device=mm.get_torch_device(),dtype=torch.float16)
            # This node owns this new VAE. Match the eventual native FP16 load
            # now, so encode-before-decode cannot change normalization buffers.
            vae.first_stage_model.to(dtype=torch.float16)
            return vae
    native = native_factory(native_path)
    native.throw_exception_if_invalid()
    check()
    backend = backend_factory(bundles,str(site),check=check)
    result = H3VAEInterface(native,backend,encode_backend='trt' if encoder_paths is not None else 'native',
                           max_output_bytes=output_budget_mib*1024**2,check=check)
    return result,{"status":"vae_interface_prepared_not_executed","encoder":"trt" if encoder_paths is not None else "native",
                   "decoder_engine":str(root),"runtime_directory":str(site),"cpu_output_budget_mib":output_budget_mib,
                   "message":"仅包装本机已编译引擎；生成时按需加载和释放。不自动编译、不安装依赖、不改音频VAE。"}
