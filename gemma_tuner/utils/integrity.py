"""Checkpoint integrity verification utilities.

Provides hash-based integrity checking for model checkpoints and exported artifacts
to detect corruption during save/load operations and ensure reproducibility.

Called by:
- models/gemma/finetune.py after training completes
- scripts/export.py after model export

Design rationale:
- SHA256 for file-level integrity (standard, available in stdlib)
- JSON manifests for human readability and tooling integration
- Atomic writes (temp file + rename) to prevent partial manifest corruption
- Separate from encryption/signing — this is integrity only, not authenticity
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Integrity manifest filename
INTEGRITY_MANIFEST_FILENAME = ".integrity.json"

# File extensions that should be included in integrity manifests
_CHECKPOINT_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".json", ".txt", ".md"}


def _compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Compute hash of file contents using specified algorithm.

    Args:
        file_path: Path to file to hash
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hex digest of file contents
    """
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_directory_integrity(
    directory: str,
    include_patterns: set[str] | None = None,
    exclude_patterns: set[str] | None = None,
) -> dict[str, Any]:
    """Compute integrity manifest for all files in a directory.

    Walks the directory recursively, computes SHA256 hashes for all files
    matching include_patterns (default: checkpoint file extensions),
    and returns a manifest dictionary.

    Args:
        directory: Root directory to scan
        include_patterns: File extensions to include (default: _CHECKPOINT_EXTENSIONS)
        exclude_patterns: File/directory names to exclude (default: None)

    Returns:
        Dictionary with:
        - version: manifest format version
        - algorithm: hash algorithm used
        - root: directory path
        - files: dict of relative paths -> {size, hash}
        - total_files: count of files hashed
        - total_bytes: sum of file sizes

    Called by:
    - create_integrity_manifest() for manifest generation
    - verify_directory_integrity() for comparison
    """
    if include_patterns is None:
        include_patterns = _CHECKPOINT_EXTENSIONS

    if exclude_patterns is None:
        exclude_patterns = {".git", "__pycache__", ".cache", "node_modules"}

    root_path = Path(directory).resolve()
    manifest: dict[str, Any] = {
        "version": "1.0",
        "algorithm": "sha256",
        "root": str(root_path),
        "files": {},
        "total_files": 0,
        "total_bytes": 0,
    }

    if not root_path.exists():
        logger.warning("Directory does not exist: %s", directory)
        return manifest

    for file_path in root_path.rglob("*"):
        # Skip directories and excluded patterns
        if file_path.is_dir():
            continue

        # Check for excluded directory components
        relative_parts = file_path.relative_to(root_path).parts
        if any(part in exclude_patterns for part in relative_parts):
            continue

        # Check extension
        if file_path.suffix not in include_patterns:
            continue

        # Compute hash and size
        try:
            file_hash = _compute_file_hash(str(file_path))
            file_size = file_path.stat().st_size

            rel_path = str(file_path.relative_to(root_path))
            manifest["files"][rel_path] = {
                "size": file_size,
                "hash": file_hash,
            }
            manifest["total_files"] += 1
            manifest["total_bytes"] += file_size
        except (OSError, IOError) as e:
            logger.warning("Failed to hash %s: %s", file_path, e)
            continue

    return manifest


def create_integrity_manifest(
    directory: str,
    manifest_path: str | None = None,
    include_metadata: bool = True,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Create and save integrity manifest for a directory.

    Computes file hashes and writes a JSON manifest. The manifest is written
    atomically (temp file + rename) to prevent corruption.

    Args:
        directory: Directory to create manifest for
        manifest_path: Output path (default: {directory}/.integrity.json)
        include_metadata: Whether to include additional metadata
        metadata: Optional additional metadata to include in manifest

    Returns:
        Path to created manifest file

    Side effects:
        - Creates JSON manifest file
        - Creates parent directories if needed

    Called by:
    - finetune.py after training completes
    - export.py after model export
    """
    directory = os.path.abspath(directory)

    if manifest_path is None:
        manifest_path = os.path.join(directory, INTEGRITY_MANIFEST_FILENAME)

    manifest = compute_directory_integrity(directory)

    if include_metadata:
        manifest["metadata"] = metadata or {}
        manifest["metadata"]["manifest_created"] = _get_timestamp()

    # Atomic write: temp file + rename
    temp_path = manifest_path + ".tmp"
    try:
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        with open(temp_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        os.replace(temp_path, manifest_path)

        logger.info(
            "Created integrity manifest: %s (%d files, %d bytes)",
            manifest_path,
            manifest["total_files"],
            manifest["total_bytes"],
        )
        return manifest_path
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def verify_directory_integrity(
    directory: str,
    manifest_path: str | None = None,
) -> tuple[bool, list[str]]:
    """Verify directory contents against an integrity manifest.

    Recomputes hashes for all files listed in the manifest and compares
    against stored values. Returns success status and list of any failures.

    Args:
        directory: Directory to verify
        manifest_path: Path to manifest (default: {directory}/.integrity.json)

    Returns:
        Tuple of (success: bool, failures: list of error messages)

    Called by:
    - Loading functions that want to verify checkpoints before use
    - CI/CD pipelines for artifact verification
    """
    directory = os.path.abspath(directory)

    if manifest_path is None:
        manifest_path = os.path.join(directory, INTEGRITY_MANIFEST_FILENAME)

    if not os.path.exists(manifest_path):
        return False, [f"Integrity manifest not found: {manifest_path}"]

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Corrupt integrity manifest: {e}"]

    failures = []
    files_checked = 0

    for rel_path, expected in manifest.get("files", {}).items():
        file_path = os.path.join(directory, rel_path)

        if not os.path.exists(file_path):
            failures.append(f"Missing file: {rel_path}")
            continue

        try:
            actual_hash = _compute_file_hash(file_path)
            actual_size = os.path.getsize(file_path)

            if actual_hash != expected.get("hash"):
                failures.append(f"Hash mismatch: {rel_path}")

            if actual_size != expected.get("size"):
                failures.append(f"Size mismatch: {rel_path}")

            files_checked += 1
        except (OSError, IOError) as e:
            failures.append(f"Cannot read {rel_path}: {e}")

    # Check for extra files not in manifest (exclude the manifest itself)
    current_manifest = compute_directory_integrity(directory)
    current_files = set(current_manifest.get("files", {}).keys())
    expected_files = set(manifest.get("files", {}).keys())

    extra_files = current_files - expected_files
    manifest_filename = Path(manifest_path).name
    for extra in extra_files:
        # Skip the integrity manifest file itself - it won't be in its own listing
        if extra == manifest_filename or extra.endswith(INTEGRITY_MANIFEST_FILENAME):
            continue
        failures.append(f"Extra file not in manifest: {extra}")

    if failures:
        logger.warning(
            "Integrity check failed for %s: %d issues found (%d files checked)",
            directory,
            len(failures),
            files_checked,
        )
        return False, failures

    logger.info("Integrity check passed: %s (%d files verified)", directory, files_checked)
    return True, []


def _get_timestamp() -> str:
    """Return ISO format timestamp."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def quick_integrity_check(directory: str) -> bool:
    """Quick integrity check that logs warnings but doesn't return details.

    Convenience function for load-time verification that should warn but
    not block on missing manifests (backward compatibility).

    Args:
        directory: Directory to check

    Returns:
        True if manifest exists and verification passes, False otherwise
    """
    manifest_path = os.path.join(directory, INTEGRITY_MANIFEST_FILENAME)

    if not os.path.exists(manifest_path):
        logger.debug("No integrity manifest found for %s (skipping check)", directory)
        return False

    success, failures = verify_directory_integrity(directory, manifest_path)

    if not success:
        for failure in failures[:5]:  # Log first 5 failures
            logger.warning("Integrity check: %s", failure)
        if len(failures) > 5:
            logger.warning("... and %d more issues", len(failures) - 5)

    return success
