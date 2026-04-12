#!/usr/bin/env python3
"""
Tests for NVIDIA Granary Dataset Integration

This module provides comprehensive tests for the Granary dataset integration including:
- Configuration loading with audio source extraction
- Granary-specific configuration validation
- Path resolution for varying directory structures
- Preparation script functionality and error handling

Test coverage focuses on the new functionality added for Granary support while
ensuring compatibility with existing configuration and dataset loading systems.
"""

import configparser
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gemma_tuner.core.config import ConfigConstants, load_model_dataset_config, load_profile_config

# Mark entire module as slow: it imports heavy dependencies and patches HF APIs.
pytestmark = pytest.mark.slow

# Lazy import heavy deps to avoid failing fast CI when excluded with -m "not slow"
try:
    from gemma_tuner.scripts.prepare_granary import resolve_granary_audio_path, validate_granary_config
except Exception as e:  # pragma: no cover - only triggers in minimal envs
    pytest.skip(f"Granary dependencies unavailable: {e}", allow_module_level=True)


def make_cfg(sections: dict) -> configparser.ConfigParser:
    """Helper function to create ConfigParser from dictionary."""
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    for sec, vals in sections.items():
        cfg[sec] = {k: str(v) for k, v in vals.items()}
    return cfg


class TestAudioSourceExtraction:
    """Test audio source extraction from dataset configuration sections."""

    def test_audio_source_extraction_basic(self):
        """Test basic audio source extraction from dataset configuration."""
        cfg = make_cfg(
            {
                "DEFAULT": {},
                "model:gemma-4-e2b-it": {"base_model": "google/gemma-4-E2B-it", "group": "gemma"},
                "dataset:granary-en": {
                    "hf_name": "nvidia/Granary",
                    "hf_subset": "en",
                    "local_path": "data/datasets/granary-en",
                    "text_column": "text",
                    "train_split": "train",
                    "validation_split": "validation",
                    "max_label_length": "256",
                    "max_duration": "30.0",
                    "audio_source_voxpopuli": "/path/to/voxpopuli",
                    "audio_source_ytc": "/path/to/ytc",
                    "audio_source_librilight": "/path/to/librilight",
                },
                "group:gemma": {},
            }
        )

        config = load_model_dataset_config(cfg, "gemma-4-e2b-it", "granary-en")

        # Verify audio sources are extracted
        assert "audio_sources" in config
        assert config["audio_sources"]["voxpopuli"] == "/path/to/voxpopuli"
        assert config["audio_sources"]["ytc"] == "/path/to/ytc"
        assert config["audio_sources"]["librilight"] == "/path/to/librilight"

    def test_audio_source_extraction_empty(self):
        """Test behavior when no audio sources are configured."""
        cfg = make_cfg(
            {
                "DEFAULT": {},
                "model:gemma-4-e2b-it": {"base_model": "google/gemma-4-E2B-it", "group": "gemma"},
                "dataset:regular-dataset": {
                    "source": "regular-dataset",
                    "text_column": "text",
                    "train_split": "train",
                    "validation_split": "validation",
                    "max_label_length": "256",
                    "max_duration": "30.0",
                },
                "group:gemma": {},
            }
        )

        config = load_model_dataset_config(cfg, "gemma-4-e2b-it", "regular-dataset")

        # Verify no audio_sources key is added when none exist
        assert "audio_sources" not in config

    def test_audio_source_extraction_partial(self):
        """Test audio source extraction with only some sources configured."""
        cfg = make_cfg(
            {
                "DEFAULT": {},
                "model:gemma-4-e2b-it": {"base_model": "google/gemma-4-E2B-it", "group": "gemma"},
                "dataset:granary-partial": {
                    "hf_name": "nvidia/Granary",
                    "text_column": "text",
                    "train_split": "train",
                    "validation_split": "validation",
                    "max_label_length": "256",
                    "max_duration": "30.0",
                    "audio_source_voxpopuli": "/path/to/voxpopuli",
                    "regular_config": "value",  # Non audio-source key
                },
                "group:gemma": {},
            }
        )

        config = load_model_dataset_config(cfg, "gemma-4-e2b-it", "granary-partial")

        # Verify only configured audio sources are extracted
        assert "audio_sources" in config
        assert config["audio_sources"]["voxpopuli"] == "/path/to/voxpopuli"
        assert len(config["audio_sources"]) == 1
        assert "ytc" not in config["audio_sources"]

    def test_audio_source_extraction_profile_config(self):
        """Test audio source extraction works through profile configuration loading."""
        cfg = make_cfg(
            {
                "DEFAULT": {},
                "model:gemma-4-e2b-it": {"base_model": "google/gemma-4-E2B-it", "group": "gemma"},
                "dataset:granary-en": {
                    "hf_name": "nvidia/Granary",
                    "text_column": "text",
                    "train_split": "train",
                    "validation_split": "validation",
                    "max_label_length": "256",
                    "max_duration": "30.0",
                    "audio_source_voxpopuli": "/path/to/voxpopuli",
                },
                "profile:test-granary": {
                    "model": "gemma-4-e2b-it",
                    "dataset": "granary-en",
                    "per_device_train_batch_size": "16",
                    "num_train_epochs": "3",
                    "logging_steps": "10",
                    "save_steps": "50",
                    "save_total_limit": "2",
                    "gradient_accumulation_steps": "1",
                },
                "group:gemma": {},
            }
        )

        config = load_profile_config(cfg, "test-granary")

        # Verify audio sources are available in profile config
        assert "audio_sources" in config
        assert config["audio_sources"]["voxpopuli"] == "/path/to/voxpopuli"


class TestGranaryConfigValidation:
    """Test Granary-specific configuration validation."""

    def test_validate_granary_config_valid(self):
        """Test validation passes for valid Granary configuration."""
        config = {
            "hf_name": "nvidia/Granary",
            "hf_subset": "en",
            "local_path": "data/datasets/granary-en",
            "audio_sources": {
                "voxpopuli": "/tmp/test/voxpopuli",
                "ytc": "/tmp/test/ytc",
                "librilight": "/tmp/test/librilight",
            },
        }

        # Create temporary directories to simulate downloaded corpora
        with tempfile.TemporaryDirectory() as temp_dir:
            for corpus in ["voxpopuli", "ytc", "librilight"]:
                corpus_path = Path(temp_dir) / corpus
                corpus_path.mkdir()
                config["audio_sources"][corpus] = str(corpus_path)

            # Should not raise any exceptions
            validate_granary_config(config)

    def test_validate_granary_config_missing_hf_keys(self):
        """Test validation fails for missing HuggingFace configuration keys."""
        config = {
            "hf_name": "nvidia/Granary",
            # Missing hf_subset and local_path
            "audio_sources": {"voxpopuli": "/path/to/voxpopuli"},
        }

        with pytest.raises(ValueError, match="Missing required Granary configuration keys"):
            validate_granary_config(config)

    def test_validate_granary_config_wrong_hf_name(self):
        """Test validation fails for incorrect HuggingFace dataset name."""
        config = {
            "hf_name": "wrong/dataset",
            "hf_subset": "en",
            "local_path": "data/datasets/granary-en",
            "audio_sources": {"voxpopuli": "/path/to/voxpopuli"},
        }

        with pytest.raises(ValueError, match="Invalid hf_name"):
            validate_granary_config(config)

    def test_validate_granary_config_no_audio_sources(self):
        """Test validation fails when no audio sources are configured."""
        config = {
            "hf_name": "nvidia/Granary",
            "hf_subset": "en",
            "local_path": "data/datasets/granary-en",
            # No audio_sources
        }

        with pytest.raises(ValueError, match="No audio sources configured"):
            validate_granary_config(config)

    def test_validate_granary_config_missing_required_corpora(self):
        """Test validation fails when required corpora are missing."""
        config = {
            "hf_name": "nvidia/Granary",
            "hf_subset": "en",
            "local_path": "data/datasets/granary-en",
            "audio_sources": {
                "voxpopuli": "/path/to/voxpopuli"
                # Missing ytc and librilight
            },
        }

        with pytest.raises(ValueError, match="Missing required audio corpora"):
            validate_granary_config(config)

    def test_validate_granary_config_nonexistent_directories(self):
        """Test validation fails when audio source directories don't exist."""
        config = {
            "hf_name": "nvidia/Granary",
            "hf_subset": "en",
            "local_path": "data/datasets/granary-en",
            "audio_sources": {
                "voxpopuli": "/nonexistent/path/voxpopuli",
                "ytc": "/nonexistent/path/ytc",
                "librilight": "/nonexistent/path/librilight",
            },
        }

        with pytest.raises(FileNotFoundError, match="Audio source directory not found"):
            validate_granary_config(config)


class TestPathResolution:
    """Test audio path resolution for Granary's varying directory structures."""

    def test_resolve_granary_audio_path_direct(self):
        """Test direct path resolution when file exists at expected location."""
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_sources = {"voxpopuli": temp_dir}

            # Create test audio file
            audio_file = Path(temp_dir) / "test_audio.flac"
            audio_file.touch()

            result = resolve_granary_audio_path("test_audio.flac", "voxpopuli", audio_sources)

            assert result == audio_file
            assert result.exists()

    def test_resolve_granary_audio_path_corpus_prefix(self):
        """Test path resolution when relative path includes corpus prefix."""
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_sources = {"voxpopuli": temp_dir}

            # Create test audio file without corpus prefix in actual path
            audio_file = Path(temp_dir) / "fr" / "audio.flac"
            audio_file.parent.mkdir(parents=True)
            audio_file.touch()

            # Relative path includes corpus prefix
            result = resolve_granary_audio_path("voxpopuli/fr/audio.flac", "voxpopuli", audio_sources)

            assert result == audio_file
            assert result.exists()

    def test_resolve_granary_audio_path_filename_only(self):
        """Test path resolution falls back to filename-only matching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_sources = {"librilight": temp_dir}

            # Create test audio file with just filename
            audio_file = Path(temp_dir) / "audio_sample.flac"
            audio_file.touch()

            # Relative path has complex structure but only filename matters
            result = resolve_granary_audio_path("deep/nested/path/audio_sample.flac", "librilight", audio_sources)

            assert result == audio_file
            assert result.exists()

    def test_resolve_granary_audio_path_not_found(self):
        """Test path resolution returns None when file cannot be found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_sources = {"ytc": temp_dir}

            # Don't create any files
            result = resolve_granary_audio_path("nonexistent.flac", "ytc", audio_sources)

            assert result is None

    def test_resolve_granary_audio_path_yodas_not_implemented(self):
        """Test YODAS corpus handling (currently returns None with warning)."""
        audio_sources = {"yodas": "/path/to/yodas"}

        result = resolve_granary_audio_path("yodas/audio.flac", "yodas", audio_sources)

        # Currently not implemented, should return None
        assert result is None

    def test_resolve_granary_audio_path_unknown_source(self):
        """Test handling of unknown corpus sources."""
        audio_sources = {"voxpopuli": "/path/to/voxpopuli"}

        result = resolve_granary_audio_path("audio.flac", "unknown_corpus", audio_sources)

        assert result is None


class TestGranaryConfigConstants:
    """Test Granary-specific configuration constants."""

    def test_granary_constants_exist(self):
        """Test that all required Granary constants are defined."""
        assert hasattr(ConfigConstants, "GRANARY_AUDIO_SOURCE_PREFIX")
        assert hasattr(ConfigConstants, "GRANARY_SUPPORTED_CORPORA")
        assert hasattr(ConfigConstants, "GRANARY_REQUIRED_KEYS")
        assert hasattr(ConfigConstants, "GRANARY_MINIMUM_EXTERNAL_CORPORA")

    def test_granary_supported_corpora(self):
        """Test Granary supported corpora definitions."""
        corpora = ConfigConstants.GRANARY_SUPPORTED_CORPORA

        # Check all expected corpora are defined
        expected_corpora = {"voxpopuli", "ytc", "librilight", "yodas"}
        assert set(corpora.keys()) == expected_corpora

        # Check all have descriptions
        for corpus, description in corpora.items():
            assert isinstance(description, str)
            assert len(description) > 0

    def test_granary_minimum_external_corpora(self):
        """Test minimum external corpora requirements."""
        minimum_corpora = ConfigConstants.GRANARY_MINIMUM_EXTERNAL_CORPORA

        # Should require the three external downloads (not YODAS)
        expected_minimum = {"voxpopuli", "ytc", "librilight"}
        assert minimum_corpora == expected_minimum

    def test_granary_required_keys(self):
        """Test Granary required configuration keys."""
        required_keys = ConfigConstants.GRANARY_REQUIRED_KEYS

        # Check essential HuggingFace configuration keys
        assert "hf_name" in required_keys
        assert "hf_subset" in required_keys
        assert "local_path" in required_keys


class TestGranaryIntegration:
    """Integration tests for the complete Granary workflow."""

    @patch("gemma_tuner.scripts.prepare_granary.load_dataset")
    @patch("gemma_tuner.scripts.prepare_granary.load_profile_config")
    def test_prepare_granary_basic_workflow(self, mock_load_config, mock_load_dataset):
        """Test basic Granary preparation workflow with mocked dependencies."""
        # Mock configuration loading
        mock_config = {
            "hf_name": "nvidia/Granary",
            "hf_subset": "en",
            "local_path": "data/datasets/granary-en",
            "audio_sources": {"voxpopuli": "/tmp/voxpopuli", "ytc": "/tmp/ytc", "librilight": "/tmp/librilight"},
        }
        mock_load_config.return_value = mock_config

        # Mock HuggingFace dataset
        mock_dataset_item = {
            "utt_id": "test_001",
            "dataset_source": "voxpopuli",
            "audio_filepath": "test_audio.flac",
            "answer": "test transcription",
            "source_lang": "en",
            "duration": 5.0,
        }
        mock_dataset = [mock_dataset_item]
        mock_load_dataset.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock audio file and all required corpus directories
            for corpus in ["voxpopuli", "ytc", "librilight"]:
                corpus_dir = Path(temp_dir) / corpus
                corpus_dir.mkdir()
                mock_config["audio_sources"][corpus] = str(corpus_dir)

            # Create test audio file in voxpopuli directory
            audio_file = Path(temp_dir) / "voxpopuli" / "test_audio.flac"
            audio_file.touch()

            mock_config["local_path"] = temp_dir

            # Import and test preparation function
            try:
                from gemma_tuner.scripts.prepare_granary import prepare_granary

                # Should complete without errors and return manifest path
                result = prepare_granary("granary-en")
                assert result is not None
                assert "granary_en_prepared.csv" in result
            except (ImportError, ModuleNotFoundError) as e:
                # Allow for import/dependency issues in test environment
                pytest.skip(f"Skipping integration test due to missing dependencies: {e}")
            except Exception as e:
                # For other errors, check if it's a configuration issue we expect
                if "No module named" in str(e) or "datasets" in str(e):
                    pytest.skip(f"Skipping due to missing HuggingFace datasets: {e}")
                else:
                    raise

    def test_audio_source_prefix_constant(self):
        """Test audio source prefix constant is used correctly."""
        prefix = ConfigConstants.GRANARY_AUDIO_SOURCE_PREFIX
        assert prefix == "audio_source_"

        # Test prefix recognition in configuration
        test_key = f"{prefix}voxpopuli"
        assert test_key.startswith(prefix)

        extracted_name = test_key[len(prefix) :]
        assert extracted_name == "voxpopuli"


# Integration test with real configuration loading
def test_end_to_end_config_integration():
    """Test end-to-end configuration integration with audio sources."""
    # Create a minimal config that includes audio sources
    cfg_dict = {
        "DEFAULT": {"num_train_epochs": "3"},
        "group:gemma": {"dtype": "float32"},
        "model:gemma-4-e2b-it": {
            "base_model": "google/gemma-4-E2B-it",
            "group": "gemma",
            "per_device_train_batch_size": "16",
        },
        "dataset:granary-test": {
            "hf_name": "nvidia/Granary",
            "hf_subset": "en",
            "local_path": "data/datasets/granary-test",
            "text_column": "text",
            "train_split": "train",
            "validation_split": "validation",
            "max_label_length": "256",
            "max_duration": "30.0",
            "audio_source_voxpopuli": "/path/to/voxpopuli",
            "audio_source_ytc": "/path/to/ytc",
            "audio_source_librilight": "/path/to/librilight",
        },
    }

    cfg = make_cfg(cfg_dict)

    # Test model+dataset loading
    config = load_model_dataset_config(cfg, "gemma-4-e2b-it", "granary-test")

    # Verify all configuration is properly merged
    assert config["base_model"] == "google/gemma-4-E2B-it"
    assert config["hf_name"] == "nvidia/Granary"
    assert config["hf_subset"] == "en"
    assert config["dtype"] == "float32"  # From group
    assert config["per_device_train_batch_size"] == 16  # Type coerced

    # Verify audio sources are extracted
    assert "audio_sources" in config
    audio_sources = config["audio_sources"]
    assert audio_sources["voxpopuli"] == "/path/to/voxpopuli"
    assert audio_sources["ytc"] == "/path/to/ytc"
    assert audio_sources["librilight"] == "/path/to/librilight"
    assert len(audio_sources) == 3
