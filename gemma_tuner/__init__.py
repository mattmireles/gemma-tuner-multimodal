"""Gemma macOS Tuner package namespace."""

import warnings as _warnings

# PyTorch is a required runtime dependency but is excluded from pyproject.toml
# [project.dependencies] because the install URL varies by platform:
#   macOS MPS/CPU : pip install torch torchaudio
#   CUDA          : pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
# Without this check a user who does `pip install gemma-macos-tuner` gets a
# confusing AttributeError or ModuleNotFoundError deep inside a submodule.
try:
    import torch as _torch  # noqa: F401
except ImportError:
    _warnings.warn(
        "PyTorch is not installed. Install it from https://pytorch.org/ "
        "for your platform, then optionally run: pip install gemma-macos-tuner[torch]",
        ImportWarning,
        stacklevel=2,
    )

__all__ = [
    "core",
    "models",
    "scripts",
    "utils",
    "wizard",
]
