"""Tests for checkpoint integrity verification utilities.

Covers:
- File hash computation
- Integrity manifest creation and verification
- Corruption detection
- Edge cases (empty dirs, missing files, etc.)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from gemma_tuner.utils.integrity import (
    INTEGRITY_MANIFEST_FILENAME,
    _compute_file_hash,
    compute_directory_integrity,
    create_integrity_manifest,
    quick_integrity_check,
    verify_directory_integrity,
)


class TestComputeFileHash:
    """Tests for _compute_file_hash function."""

    def test_hash_consistency(self):
        """Same file content should produce same hash."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            path = f.name

        try:
            hash1 = _compute_file_hash(path)
            hash2 = _compute_file_hash(path)
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex length
        finally:
            os.unlink(path)

    def test_different_content_different_hash(self):
        """Different content should produce different hashes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
            f1.write("content A")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
            f2.write("content B")
            path2 = f2.name

        try:
            hash1 = _compute_file_hash(path1)
            hash2 = _compute_file_hash(path2)
            assert hash1 != hash2
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_binary_file_hashing(self):
        """Should handle binary files correctly."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            f.write(b"\x00\x01\x02\x03\xff\xfe")
            path = f.name

        try:
            hash_val = _compute_file_hash(path)
            assert len(hash_val) == 64
            assert all(c in "0123456789abcdef" for c in hash_val)
        finally:
            os.unlink(path)


class TestComputeDirectoryIntegrity:
    """Tests for compute_directory_integrity function."""

    def test_empty_directory(self):
        """Empty directory should return empty manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = compute_directory_integrity(tmpdir)
            assert manifest["version"] == "1.0"
            assert manifest["algorithm"] == "sha256"
            assert manifest["total_files"] == 0
            assert manifest["total_bytes"] == 0
            assert manifest["files"] == {}

    def test_filters_by_extension(self):
        """Should only include files with checkpoint extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different extensions
            Path(tmpdir, "model.safetensors").write_text("model weights")
            Path(tmpdir, "config.json").write_text('{"key": "value"}')
            Path(tmpdir, "readme.md").write_text("# README")
            Path(tmpdir, "script.py").write_text("print('hello')")  # Should be excluded
            Path(tmpdir, "data.csv").write_text("a,b,c")  # Should be excluded

            manifest = compute_directory_integrity(tmpdir)

            assert "model.safetensors" in manifest["files"]
            assert "config.json" in manifest["files"]
            assert "readme.md" in manifest["files"]
            assert "script.py" not in manifest["files"]
            assert "data.csv" not in manifest["files"]
            assert manifest["total_files"] == 3

    def test_nested_directories(self):
        """Should recursively process nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            nested = Path(tmpdir, "subdir", "nested")
            nested.mkdir(parents=True)
            Path(nested, "weights.safetensors").write_text("weights")

            manifest = compute_directory_integrity(tmpdir)

            assert "subdir/nested/weights.safetensors" in manifest["files"]
            assert manifest["total_files"] == 1

    def test_excludes_special_dirs(self):
        """Should exclude .git, __pycache__, etc."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, ".git").mkdir()
            Path(tmpdir, "__pycache__").mkdir()
            Path(tmpdir, ".git", "config").write_text("git config")
            Path(tmpdir, "__pycache__", "file.pyc").write_text("compiled")
            Path(tmpdir, "valid.json").write_text('{"valid": true}')

            manifest = compute_directory_integrity(tmpdir)

            assert ".git/config" not in manifest["files"]
            assert "__pycache__/file.pyc" not in manifest["files"]
            assert "valid.json" in manifest["files"]

    def test_file_size_tracking(self):
        """Should track total bytes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "x" * 1000
            Path(tmpdir, "file.json").write_text(content)

            manifest = compute_directory_integrity(tmpdir)

            assert manifest["total_bytes"] == 1000
            assert manifest["files"]["file.json"]["size"] == 1000


class TestCreateIntegrityManifest:
    """Tests for create_integrity_manifest function."""

    def test_creates_manifest_file(self):
        """Should create .integrity.json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").write_text("weights")

            manifest_path = create_integrity_manifest(tmpdir)

            assert Path(manifest_path).exists()
            assert Path(manifest_path).name == INTEGRITY_MANIFEST_FILENAME

    def test_manifest_content(self):
        """Manifest should contain correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").write_text("weights")

            create_integrity_manifest(tmpdir)

            with open(Path(tmpdir, INTEGRITY_MANIFEST_FILENAME)) as f:
                manifest = json.load(f)

            assert manifest["version"] == "1.0"
            assert manifest["algorithm"] == "sha256"
            assert "root" in manifest
            assert "files" in manifest
            assert "total_files" in manifest
            assert "total_bytes" in manifest

    def test_includes_metadata(self):
        """Should include metadata when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").write_text("weights")

            metadata = {"model_id": "test-model", "dtype": "bfloat16"}
            create_integrity_manifest(tmpdir, metadata=metadata)

            with open(Path(tmpdir, INTEGRITY_MANIFEST_FILENAME)) as f:
                manifest = json.load(f)

            assert manifest["metadata"]["model_id"] == "test-model"
            assert manifest["metadata"]["dtype"] == "bfloat16"
            assert "manifest_created" in manifest["metadata"]

    def test_atomic_write(self):
        """Should use atomic write pattern (no partial files)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").write_text("weights")

            manifest_path = create_integrity_manifest(tmpdir)

            # Should not leave temp files
            temp_files = list(Path(tmpdir).glob("*.tmp"))
            assert len(temp_files) == 0

            # Manifest should be valid JSON
            with open(manifest_path) as f:
                json.load(f)  # Should not raise


class TestVerifyDirectoryIntegrity:
    """Tests for verify_directory_integrity function."""

    def test_verification_success(self):
        """Should pass for unmodified directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").write_text("weights")
            create_integrity_manifest(tmpdir)

            success, failures = verify_directory_integrity(tmpdir)

            assert success is True
            assert failures == []

    def test_detects_missing_file(self):
        """Should fail if file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").write_text("weights")
            create_integrity_manifest(tmpdir)

            # Delete the file
            Path(tmpdir, "model.safetensors").unlink()

            success, failures = verify_directory_integrity(tmpdir)

            assert success is False
            assert any("Missing file" in f for f in failures)

    def test_detects_corrupted_file(self):
        """Should fail if file is modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").write_text("original weights")
            create_integrity_manifest(tmpdir)

            # Modify the file
            Path(tmpdir, "model.safetensors").write_text("corrupted weights")

            success, failures = verify_directory_integrity(tmpdir)

            assert success is False
            assert any("Hash mismatch" in f for f in failures)

    def test_detects_extra_file(self):
        """Should warn about extra files not in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").write_text("weights")
            create_integrity_manifest(tmpdir)

            # Add extra file
            Path(tmpdir, "extra.json").write_text("extra data")

            success, failures = verify_directory_integrity(tmpdir)

            assert success is False
            assert any("Extra file" in f for f in failures)

    def test_missing_manifest(self):
        """Should fail gracefully if manifest doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            success, failures = verify_directory_integrity(tmpdir)

            assert success is False
            assert any("not found" in f.lower() for f in failures)


class TestQuickIntegrityCheck:
    """Tests for quick_integrity_check function."""

    def test_returns_true_for_valid(self):
        """Should return True for valid directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").write_text("weights")
            create_integrity_manifest(tmpdir)

            result = quick_integrity_check(tmpdir)

            assert result is True

    def test_returns_false_for_no_manifest(self):
        """Should return False if no manifest exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").write_text("weights")

            result = quick_integrity_check(tmpdir)

            assert result is False

    def test_returns_false_for_corruption(self):
        """Should return False for corrupted directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").write_text("weights")
            create_integrity_manifest(tmpdir)

            # Corrupt the file
            Path(tmpdir, "model.safetensors").write_text("corrupted")

            result = quick_integrity_check(tmpdir)

            assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
