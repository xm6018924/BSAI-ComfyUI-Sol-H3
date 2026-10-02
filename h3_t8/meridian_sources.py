"""Lazy source-bound upstream math and authorized geometry dependencies.

No external imports at node registration; no weights/code copied from gated Omega.
"""

from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
import threading

from .meridian_checkpoint_io import file_sha

REVISION = "2083d059d8544ff7eaaf86966b83e4964a904737"
FILES = {
    "geometry": "d8e9832b23deb5801f73e1dbe54b9e8a9e4abc8e87b036af54b2c9507ce8ca74",
    "path": "344d0bb0dac712266846de4614a73260016c989b9f8cb8441a5b966c471b06fc",
}
_IMPORT_LOCK = threading.RLock()


def source_module(root, name, check=lambda: None):
    if name not in FILES:
        raise ValueError("Unknown Meridian source module")
    path = Path(root).resolve(strict=True) / "recam" / f"{name}.py"
    check()
    actual = file_sha(path, cancelled=lambda: (check(), False)[1])
    if actual != FILES[name]:
        raise ValueError(
            f"Meridian {name}.py differs from qualified revision {REVISION}; audit before use"
        )
    namespace = "_t8_meridian_" + name + "_" + actual
    with _IMPORT_LOCK:
        if namespace not in sys.modules:
            spec = importlib.util.spec_from_file_location(namespace, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[namespace] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[namespace]
                raise
        return sys.modules[namespace]


@contextmanager
def omega_imports(root):
    """Restore sys.path, never silently switch an already loaded Omega package."""
    root = Path(root).resolve(strict=True)
    expected = root / "vggt_omega" / "__init__.py"
    if not expected.is_file():
        raise ValueError("Omega source directory must contain vggt_omega/__init__.py")
    with _IMPORT_LOCK:
        current = sys.modules.get("vggt_omega")
        if (
            current is not None
            and Path(current.__file__).resolve() != expected.resolve()
        ):
            raise ValueError(
                "Different Omega source already imported; use a clean process for this source"
            )
        previous = sys.path.copy()
        sys.path.insert(0, str(root))
        try:
            yield
        finally:
            sys.path[:] = previous


def resolve_directory(value, env, default):
    path = Path(value or os.environ.get(env) or default).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(
            f"Directory not found: {path}. Set {env} or the advanced node path."
        )
    return path


def resolve_file(value, env, default):
    path = Path(value or os.environ.get(env) or default).expanduser().resolve()
    if not path.is_file():
        raise ValueError(
            f"File not found: {path}. Set {env} or the advanced node path."
        )
    return path
