"""Safe I/O utilities for secure model loading and deserialization.

Provides hardened wrappers around PyTorch and pickle operations to prevent
arbitrary code execution via malicious serialized data. These utilities
enforce safe loading patterns by default.

Security considerations:
- torch.load defaults to weights_only=True (PyTorch 2.6+ compatible)
- pickle operations are restricted to basic types
- Path traversal attempts are detected and blocked

Called by:
- utils/dataset_prep.py for local audio path validation
- Future extensions that need safer checkpoint/config loading

Design rationale:
- Explicit safe defaults over implicit trust
- Fail-closed: if safety cannot be verified, operation fails
- Backward compatible: optional allow_unsafe=True for legacy checkpoints
  with appropriate warnings
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def safe_torch_load(
    f: str | os.PathLike | bytes,
    *,
    map_location: Any = None,
    weights_only: bool = True,
    allow_unsafe: bool = False,
) -> Any:
    """Load PyTorch checkpoint with safe defaults.

    Wrapper around torch.load that defaults to weights_only=True for
    security. Prevents arbitrary code execution from malicious pickle
    payloads in checkpoint files.

    Args:
        f: File path or file-like object to load from
        map_location: Device mapping for loading (e.g., 'cpu', 'cuda:0')
        weights_only: If True (default), only load tensor data and
            basic Python types. Prevents code execution.
        allow_unsafe: If True, allows pickle loading with a security
            warning. Use only for legacy checkpoints you trust.

    Returns:
        Loaded checkpoint data (typically dict containing state_dict)

    Raises:
        RuntimeError: If loading fails or unsafe mode is attempted
            without explicit allow_unsafe=True

    Security notes:
        - weights_only=True is the safe default (PyTorch 2.6+ behavior)
        - Legacy checkpoints that require pickle may fail; use
          allow_unsafe=True only for trusted sources
        - If you encounter "Attempting to deserialize object..." errors,
          the checkpoint may contain non-tensor data requiring unsafe mode

    Example:
        # Safe loading (recommended)
        checkpoint = safe_torch_load("model.pt", map_location="cpu")

        # Legacy checkpoint from trusted source
        checkpoint = safe_torch_load(
            "legacy_model.pt",
            weights_only=False,
            allow_unsafe=True
        )
    """
    import torch

    if not weights_only and not allow_unsafe:
        raise RuntimeError(
            "Attempting to load checkpoint with weights_only=False without "
            "explicit allow_unsafe=True. This is a security risk. "
            "If you trust this checkpoint, add allow_unsafe=True."
        )

    if not weights_only and allow_unsafe:
        logger.warning(
            "Loading checkpoint with weights_only=False - "
            "executing arbitrary pickle code. Ensure you trust this source: %s",
            f,
        )

    try:
        return torch.load(f, map_location=map_location, weights_only=weights_only)
    except RuntimeError as e:
        if "weights_only" in str(e) or "torch.load" in str(e):
            logger.error(
                "Failed to load checkpoint with weights_only=%s. The checkpoint may contain non-tensor data. Error: %s",
                weights_only,
                e,
            )
        raise


def safe_pickle_load(f: str | os.PathLike | bytes) -> Any:
    """Load pickle data with restricted unpickler.

    Uses pickle with a restricted unpickler that only allows basic types.
    Prevents arbitrary code execution from malicious pickle payloads.

    Args:
        f: File path or file-like object to load from

    Returns:
        Deserialized Python object (restricted to safe types)

    Raises:
        pickle.UnpicklingError: If unsafe pickle content is detected
        RuntimeError: If loading fails

    Security notes:
        - Only allows basic Python types (str, int, float, list, dict, tuple)
        - Blocks custom class instantiation that could execute code
        - For config files, prefer JSON over pickle

    Example:
        data = safe_pickle_load("data.pkl")
    """

    class RestrictedUnpickler(pickle.Unpickler):
        """Unpickler that restricts safe types only."""

        SAFE_MODULES = frozenset(
            [
                "builtins",
                "collections",
                "collections.abc",
                "copyreg",
                "__builtin__",
            ]
        )
        SAFE_TYPES = frozenset(
            [
                "str",
                "int",
                "float",
                "bool",
                "list",
                "tuple",
                "dict",
                "set",
                "frozenset",
                "bytes",
                "bytearray",
                "NoneType",
            ]
        )

        def find_class(self, module: str, name: str) -> Any:
            """Override to only allow safe classes."""
            if module not in self.SAFE_MODULES:
                raise pickle.UnpicklingError(f"Blocked unpickling of non-whitelisted module: {module}.{name}")
            if name not in self.SAFE_TYPES:
                raise pickle.UnpicklingError(f"Blocked unpickling of non-whitelisted type: {module}.{name}")
            return super().find_class(module, name)

    if isinstance(f, (str, os.PathLike)):
        with open(f, "rb") as file:
            return RestrictedUnpickler(file).load()
    else:
        return RestrictedUnpickler(f).load()


def validate_safe_path(
    path: str,
    *,
    base_dir: str | None = None,
    allow_symlinks: bool = False,
    max_symlinks: int = 5,
) -> Path:
    """Validate that a path is safe (no traversal, proper symlinks).

    Checks that the resolved path stays within base_dir (if specified)
    and doesn't use symlink tricks for traversal attacks. When no base_dir
    is provided, the function still normalizes the path and applies the
    symlink policy, but it does not confine absolute paths to the current
    working directory.

    Args:
        path: Input path to validate
        base_dir: Optional base directory that path must resolve within
        allow_symlinks: Whether symlinks are permitted in the path
        max_symlinks: Maximum symlink hops allowed (prevents symlink loops)

    Returns:
        Resolved Path object that passed validation

    Raises:
        ValueError: If path traversal detected or symlinks disallowed
        RuntimeError: If too many symlinks encountered

    Security notes:
        - Resolves all .. components to prevent directory traversal
        - Checks final resolved path is within base_dir when one is provided
        - Symlinks are followed and validated to prevent symlink attacks
        - Used by audio loading and file access utilities

    Example:
        safe_path = validate_safe_path(
            "../../etc/passwd",
            base_dir="/data/audio"
        )  # Raises ValueError

        safe_path = validate_safe_path(
            "audio/file.wav",
            base_dir="/data"
        )  # Returns Path("/data/audio/file.wav")
    """
    base_path = Path(base_dir).resolve() if base_dir else None
    resolution_root = base_path or Path.cwd().resolve()

    input_path = Path(path)

    # Resolve relative paths against the explicit base_dir when provided,
    # otherwise preserve the existing cwd-relative behavior for callers that
    # already pass local relative paths.
    candidate_path = input_path if input_path.is_absolute() else resolution_root / input_path

    current = Path(candidate_path.anchor) if candidate_path.is_absolute() else Path()
    parts = candidate_path.parts[1:] if candidate_path.is_absolute() else candidate_path.parts

    # When a confinement root is provided, reject symlinks inside that controlled
    # subtree. Without a base_dir we only reject a symlinked final path, because
    # absolute system paths may legitimately traverse OS-level symlink prefixes
    # such as /var -> /private/var on macOS.
    if not allow_symlinks:
        if base_path is not None and candidate_path.is_relative_to(base_path):
            for part in parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError(f"Symlink detected in path (symlinks disallowed): {current}")
        elif candidate_path.is_symlink():
            raise ValueError(f"Symlink detected in path (symlinks disallowed): {candidate_path}")

    # Resolve the full path
    try:
        resolved = candidate_path.resolve()
    except RuntimeError as e:
        raise RuntimeError(f"Failed to resolve path {path}: {e}") from e

    # Check for symlink loops
    if allow_symlinks and base_path is not None and candidate_path.is_relative_to(base_path):
        symlink_count = 0
        current_check = Path(candidate_path.anchor) if candidate_path.is_absolute() else Path()

        for part in parts:
            current_check = current_check / part
            while current_check.is_symlink() and symlink_count < max_symlinks:
                symlink_count += 1
                target = current_check.readlink()
                current_check = target if target.is_absolute() else current_check.parent / target

        if symlink_count >= max_symlinks:
            raise RuntimeError(f"Too many symlinks (> {max_symlinks}) in path: {path}")

    if base_path is not None:
        try:
            resolved.relative_to(base_path)
        except ValueError:
            raise ValueError(
                f"Path traversal detected: {path} resolves to {resolved} which is outside base directory {base_path}"
            ) from None

    return resolved
