"""Local file identities shared by the isolated workers."""
import hashlib
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    path = Path(path)
    before = path.stat()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError('Input changed during hashing')
    return digest.hexdigest()
