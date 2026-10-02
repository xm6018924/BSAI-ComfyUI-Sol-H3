"""Bounded file inventories for prepared model/teacher/import directories."""
import os
from pathlib import Path

IGNORED_DIRS = {'.git', '.cache', '__pycache__'}
MAX_FILES = 50000


def linked(path):
    return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0) & 0x400)


def absolute_path(value):
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError('Prepared paths must be nonempty absolute local paths')
    path = Path(value)
    if '..' in path.parts or '.' in path.parts:
        raise ValueError('Prepared paths cannot contain traversal components')
    return str(path)


def inventory(directory):
    root = Path(absolute_path(str(directory)))
    if not root.is_dir() or linked(root):
        raise ValueError('Prepared inventory requires a real directory, not a link')
    paths = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in IGNORED_DIRS)
        for name in directories:
            child = Path(current) / name
            if linked(child):
                raise ValueError('Linked directory in prepared inventory')
        for name in sorted(filenames):
            if name.endswith(('.pyc', '.pyo')):
                continue
            path = Path(current) / name
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('Linked or invalid file in prepared inventory')
            paths.append(str(path))
            if len(paths) > MAX_FILES:
                raise ValueError('Prepared directory inventory is too large')
    return sorted(paths)
