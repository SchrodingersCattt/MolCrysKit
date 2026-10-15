"""Pinned Open Babel comparator metadata and availability probes."""
from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

VERSION = "3.2.1"
TAG = "openbabel-3-2-1"
COMMIT = "0e94434fa75c9f61095023e3c12e0d5f2ac035ff"
SOURCE_URL = "https://github.com/openbabel/openbabel/tree/openbabel-3-2-1"


def baseline_metadata() -> dict[str, object]:
    return {"name": "Open Babel", "version": VERSION, "tag": TAG, "commit": COMMIT, "source_url": SOURCE_URL, "production_dependency": False}


def native_binary() -> str | None:
    return os.environ.get("OBABEL_BIN") or shutil.which("obabel")


def python_bindings_available() -> bool:
    
    try:
        return importlib.util.find_spec("openbabel.pybel") is not None
    except ModuleNotFoundError:
        return False


def recipe_path() -> Path:
    return Path(__file__).with_name("README.md")


__all__ = ["VERSION", "TAG", "COMMIT", "SOURCE_URL", "baseline_metadata", "native_binary", "python_bindings_available", "recipe_path"]
