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
        raise ValueError("Please enter an absolute path for the TensorRT runtime directory, or leave empty to use the default location in the model directory")
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
    return values or ["Engine not found: please compile first"]


def resolve_bundle(value, models_directory, kind):
    value = str(value).strip()
    if not value or value == "Engine not found: please compile first":
        raise ValueError("No TensorRT engine selected yet. Complete local compilation first, then refresh the node options.")
    path = Path(value)
    if not path.is_absolute():
        if path.name != value or value in ('.','..'):
            raise ValueError("Please choose an engine directory name or enter an explicit absolute path")
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
                missing.append(str(metadata.relative_to(site))+": version mismatch")
            versions.extend(declared)
    return {"status":"static_files_present_not_execution_qualified" if not missing else "missing_runtime_files",
            "runtime_directory":str(site),"runtime_version":TRT_VERSION,"missing":missing,
            "gpu_initialized":False,"installed_or_downloaded":False,
            "message":"Files are present; the first run also verifies GPU, driver, DLLs, engine, and available VRAM" if not missing else
                      "Runtime files are incomplete. Install to a standalone directory per the TRT VAE guide; do not replace ComfyUI's torch or CUDA."}


def load_vae(*, native_path, decoder, runtime_site, encoder_paths=None, output_budget_mib=1024,
             check=lambda:None, native_factory=None, backend_factory=ScopedBackend):
    if type(output_budget_mib) is not int or not 64 <= output_budget_mib <= 8192:
        raise ValueError("CPU output buffer cap must be 64..8192 MiB; this is not a VRAM limit")
    site = Path(runtime_site).resolve()
    report = inspect_runtime(site)
    if report['missing']:
        raise ValueError(report['message']+' missing: '+', '.join(report['missing']))
    root, manifest = inspect_bundle(decoder,'decoder')
    if Path(manifest['source_request']['runtime_site']).resolve() != site:
        raise ValueError("The engine is bound to a different TensorRT runtime directory; choose the compile-time directory or explicitly recompile")
    bundles = {'decoder':str(root)}
    if encoder_paths is not None:
        from .trt_vae_backend import select_bundle
        if len(encoder_paths) != 2:
            raise ValueError("The full VAE requires selecting both the T1 single-frame and T17 video encode engines")
        for shape in ((1,3,1,256,256),(1,3,17,256,256)):
            _, item = select_bundle(encoder_paths,'encoder',[shape])
            if Path(item['source_request']['runtime_site']).resolve() != site:
                raise ValueError("Encode and decode engines must use the same TensorRT runtime directory")
        bundles['encoder'] = list(map(str,encoder_paths))
    check()
    native_path = Path(native_path).resolve(strict=True)
    if digest_file(native_path) != NATIVE_VAE_SHA:
        raise ValueError("This engine corresponds to the original minimax_h3_video_vae_fp16.safetensors; do not silently swap in other VAE weights")
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
                   "message":"Wraps only the locally compiled engine; loads/releases on demand at generation. Does not auto-compile, install dependencies, or alter the audio VAE."}
