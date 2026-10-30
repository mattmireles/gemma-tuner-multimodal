"""Content-addressed package support for the native Gemma 4 runtime."""

from .packer import pack_runtime, verify_runtime_package

__all__ = ["pack_runtime", "verify_runtime_package"]
