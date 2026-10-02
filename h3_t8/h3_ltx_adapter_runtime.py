"""Owned, pinned external learned adapter; no persistent GPU cache or downloads."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
from types import ModuleType
import uuid


SOURCE_REVISION = "144085566a866f9784f3798d4c8d1603f3adbccf"
MODEL_REVISION = "1792c42689a0f22de880eaf57a187c6a373a636d"
MODEL_SHA = "170199a390c40ac97f5895bc9c8cc29817e74fb9193c858a85d8c0f1f30724ac"
CONFIG_SHA = "49e13820453c4e75b8c17c8248b46cfa6ab05c2b91fd52895a07ffd5d4dc3dd3"
# UTF8 LF-normalized hashes tolerate Git checkout line endings, not code changes.
SOURCE_HASHES = {
    "__init__.py": "c200de6b2632edb6f5f87908af707d4599411ae8c51ca2440d46c9dce167ee19",
    "constants.py": "fc58ee14f66d845213029e154c47633003f4eafc6a6408d9638acecb069f680a",
    "geometry.py": "6ed4ff18ed60faa4038c7e31d436640f33854471ab0362436ac64cf68f65f419",
    "model.py": "f7a894a571197af2057de604916daadba3fe4b488474b5865cf1a4745a4c48ba",
    "adapter.py": "a0b6c08ca3ddc5ba79c6348a78f09c49fa07a9b556ae18677347275f65ab0a5f",
}


def file_hash(path, check_cancel=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            check_cancel()
            digest.update(block)
    return digest.hexdigest()


def inspect_assets(source_directory, model_directory, check_cancel=lambda: None):
    if not str(source_directory).strip() or not str(model_directory).strip():
        raise ValueError("Set the pinned h3_ltx_adapter source directory and model directory")
    source, model = Path(source_directory).resolve(), Path(model_directory).resolve()
    texts = {}
    for name, expected in SOURCE_HASHES.items():
        check_cancel()
        path = source / name
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise ValueError(f"Missing or oversized adapter source: {name}")
        text = path.read_text(encoding="utf8").replace("\r\n", "\n")
        if hashlib.sha256(text.encode("utf8")).hexdigest() != expected:
            raise ValueError(f"Unqualified adapter source: {name}; expected Sana {SOURCE_REVISION}")
        texts[name] = text
    for name, expected in (("config.json", CONFIG_SHA), ("model.safetensors", MODEL_SHA)):
        if file_hash(model / name, check_cancel) != expected:
            raise ValueError(f"Pinned H3-to-LTX asset hash mismatch: {name}")
    return source, model, texts


@contextmanager
def owned_adapter(source_directory, model_directory, *, device="cpu", precision="float32",
                  check_cancel=lambda: None):
    """Import only authenticated source bytes; unload only this call's model/modules."""
    if device not in ("cpu", "cuda") or precision not in ("float32", "bfloat16"):
        raise ValueError("Explicit cpu/cuda and float32/bfloat16 adapter execution required")
    source, model, texts = inspect_assets(source_directory, model_directory, check_cancel)
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA adapter selected but unavailable; no silent CPU fallback")
    name = "_t8_h3_ltx_owned_" + uuid.uuid4().hex
    loaded, hooks, adapter = [], [], None
    try:
        package = ModuleType(name)
        package.__path__ = []  # Never import unverified sibling files from disk.
        package.__package__ = name
        sys.modules[name] = package
        loaded.append(name)
        for filename in ("constants.py", "geometry.py", "model.py", "adapter.py", "__init__.py"):
            check_cancel()
            key = name if filename == "__init__.py" else name + "." + filename[:-3]
            module = package if key == name else ModuleType(key)
            module.__package__ = name
            module.__file__ = str(source / filename)
            if key != name:
                sys.modules[key] = module
                loaded.append(key)
            exec(compile(texts[filename], module.__file__, "exec"), module.__dict__)
        # Upstream initializes then loads weights. Preserve the host RNG and never
        # initialize CUDA merely to enumerate RNG devices during CPU loading.
        with torch.random.fork_rng(devices=[]):
            adapter = package.H3ToLTXAdapter.from_pretrained(model, device="cpu", dtype=getattr(torch, precision))
        check_cancel()
        adapter.model.to(device=device)
        adapter.device = torch.device(device)
        for block in adapter.model.modules():
            hooks.append(block.register_forward_pre_hook(lambda *_: check_cancel()))
        yield adapter
        check_cancel()
        # Reject source/weight changes made during this call; never return an
        # apparently reusable latent for an identity that no longer exists.
        inspect_assets(source_directory, model_directory, check_cancel)
    finally:
        original_error = sys.exc_info()[1]
        for hook in hooks:
            hook.remove()
        try:
            if adapter is not None:
                try:
                    adapter.model.to(device="cpu")
                    adapter.device = torch.device("cpu")
                except Exception as cleanup_error:
                    if original_error is None:
                        raise
                    if hasattr(original_error, "add_note"):
                        original_error.add_note(f"Owned adapter CPU cleanup also failed: {cleanup_error}")
        finally:
            for key in reversed(loaded):
                sys.modules.pop(key, None)
