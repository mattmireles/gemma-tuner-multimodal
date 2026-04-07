"""
Configuration Testing Suite for Gemma Fine-Tuning System

This module provides comprehensive test coverage for the configuration management
system, ensuring robust profile validation, type coercion, and error handling
across different configuration scenarios.

Key testing areas:
- Profile configuration validation and schema enforcement
- Type coercion for numeric and boolean configuration values
- Default value application and inheritance behavior
- Error handling for missing or invalid configuration parameters
- Configuration loading and parsing across different INI formats

Called by:
- pytest test discovery for automated configuration testing
- CI/CD pipelines for configuration validation before deployment
- Development workflows for configuration schema verification
- Quality assurance testing for configuration system reliability

Calls to:
- core.config.load_profile_config() for profile loading functionality testing
- core.config._validate_profile_config() for validation logic verification
- core.config.ConfigConstants for configuration schema reference
- configparser.ConfigParser for INI file format handling and testing
- pytest fixtures and assertions for test execution and validation

Testing patterns:
- Fixture-based test data generation for repeatable test scenarios
- Parametric testing for configuration variations and edge cases
- Exception testing for error condition validation and handling
- Type validation testing for configuration value coercion and conversion
- Integration testing for end-to-end configuration loading workflows

Configuration validation strategy:
- Required key presence verification across all profile types
- Type safety enforcement for numeric and boolean configuration values
- Default value application testing for optional configuration parameters
- Cross-validation testing for dependent configuration parameters
- Error message clarity verification for debugging and troubleshooting
"""

import configparser

import pytest

from gemma_tuner.core.config import ConfigConstants, _validate_profile_config, load_profile_config
from gemma_tuner.core.profile_config import ProfileConfig


def make_cfg(sections: dict) -> configparser.ConfigParser:
    """
    Creates a ConfigParser instance from dictionary-based section definitions.

    This utility function provides a clean interface for creating test configuration
    objects from nested dictionaries, enabling concise test data definition and
    repeatable test scenario construction.

    Called by:
    - test_validate_profile_defaults_and_types() for profile validation testing
    - test_load_profile_config_missing_profile_raises() for error condition testing
    - Other configuration test functions requiring structured INI data

    Dictionary structure transformation:
    - Outer keys become INI section names (e.g., "profile:test", "DEFAULT")
    - Inner dictionaries become key-value pairs within each section
    - All values converted to strings for INI format compatibility
    - Preserves nested structure for complex configuration scenarios

    Args:
        sections: Nested dictionary defining INI structure
            Format: {"section_name": {"key": value, ...}, ...}
            Example: {"profile:test": {"model": "gemma-4-e2b-it", "batch_size": 4}}

    Returns:
        ConfigParser instance ready for configuration testing
        All values converted to strings for INI compatibility
        Section structure preserved for test validation

    Example usage:
        cfg = make_cfg({
            "profile:test": {
                "model": "gemma-4-e2b-it",
                "batch_size": 4,
                "gradient_checkpointing": True
            },
            "DEFAULT": {
                "language_mode": "strict"
            }
        })
    """
    cfg = configparser.ConfigParser()
    for sec, vals in sections.items():
        cfg[sec] = {k: str(v) for k, v in vals.items()}
    return cfg


def test_validate_profile_defaults_and_types():
    """
    Tests profile configuration validation, default application, and type coercion.

    This test verifies that the configuration validation system correctly applies
    default values for optional parameters and performs proper type conversion
    from string INI values to appropriate Python types.

    Validation behavior tested:
    - Default value application for missing optional configuration keys
    - Type coercion from INI string format to Python native types
    - Required key presence verification without missing key errors
    - Configuration schema enforcement across standard profile requirements

    Called by:
    - pytest discovery during automated test suite execution
    - Configuration validation testing during development workflows
    - CI/CD pipelines for configuration system regression testing

    Type coercion validation:
    - String "128" → int 128 for max_label_length numeric parameter
    - String "30.0" → float 30.0 for max_duration floating-point parameter
    - String "true" → bool True for gradient_checkpointing boolean parameter
    - String preservation for text-based configuration parameters

    Default value verification:
    - language_mode defaults to "strict" when not explicitly specified
    - Other optional parameters receive appropriate defaults per ConfigConstants
    - Required parameters must be present or validation raises ValueError
    """
    conf = {
        "model": "gemma-4-e2b-it",
        "dataset": "dummy",
        "base_model": "google/gemma-4-E2B-it",
        "train_split": "train",
        "validation_split": "validation",
        "text_column": "text",
        "max_label_length": "128",
        "max_duration": "30.0",
        "per_device_train_batch_size": "4",
        "num_train_epochs": "2",
        "logging_steps": "10",
        "save_steps": "50",
        "save_total_limit": "2",
        "gradient_accumulation_steps": "1",
        "gradient_checkpointing": "true",
    }
    _validate_profile_config(conf, ConfigConstants.REQUIRED_PROFILE_KEYS)

    # Default value application verification
    assert conf["language_mode"] == "strict"

    # Type coercion verification for different Python types
    assert isinstance(conf["max_label_length"], int)
    assert isinstance(conf["max_duration"], float)
    assert isinstance(conf["gradient_checkpointing"], bool)


def test_validate_image_modality_profile():
    """Image modality accepts allowed token budgets and requires prompt for VQA."""
    base = {
        "model": "gemma-4-e2b-it",
        "dataset": "dummy",
        "base_model": "google/gemma-4-E2B-it",
        "train_split": "train",
        "validation_split": "validation",
        "text_column": "caption",
        "max_label_length": 128,
        "max_duration": 30.0,
        "per_device_train_batch_size": 1,
        "num_train_epochs": 1,
        "logging_steps": 1,
        "save_steps": 50,
        "save_total_limit": 1,
        "gradient_accumulation_steps": 1,
        "modality": "image",
        "image_sub_mode": "caption",
        "image_token_budget": 280,
    }
    _validate_profile_config(dict(base), ConfigConstants.REQUIRED_PROFILE_KEYS)

    vqa = {**base, "image_sub_mode": "vqa", "prompt_column": "question"}
    _validate_profile_config(dict(vqa), ConfigConstants.REQUIRED_PROFILE_KEYS)

    with pytest.raises(ValueError, match=r"image_token_budget"):
        bad_budget = {**base, "image_token_budget": 999}
        _validate_profile_config(bad_budget, ConfigConstants.REQUIRED_PROFILE_KEYS)

    with pytest.raises(ValueError, match=r"prompt_column"):
        vqa_no_prompt = {**base, "image_sub_mode": "vqa"}
        _validate_profile_config(vqa_no_prompt, ConfigConstants.REQUIRED_PROFILE_KEYS)


def test_load_profile_config_missing_profile_raises():
    """
    Tests that loading non-existent profiles raises appropriate ValueError exceptions.

    This test ensures robust error handling when attempting to load profile
    configurations that don't exist in the provided configuration file. It
    validates that the configuration system fails fast with clear error messages
    rather than silently using incorrect defaults.

    Error handling validation:
    - Missing profile section raises ValueError with descriptive message
    - Available profiles are properly identified in configuration
    - Error occurs before any configuration processing or validation
    - Exception provides sufficient context for debugging configuration issues

    Called by:
    - pytest discovery during automated error handling validation
    - Configuration system testing for robustness verification
    - Development workflows ensuring proper error condition handling
    - CI/CD pipelines for configuration error detection and reporting

    Test scenario:
    - Configuration contains "profile:foo" but not "profile:bar"
    - Attempt to load "bar" profile should fail immediately
    - ValueError exception should be raised with appropriate error message
    - No configuration processing should occur for missing profiles

    This validates defensive programming practices and helps prevent
    silent configuration errors that could lead to training failures.
    """
    cfg = make_cfg(
        {
            "DEFAULT": {},
            "profile:foo": {"model": "gemma-4-e2b-it", "dataset": "ds"},
        }
    )
    with pytest.raises(ValueError, match=r"not found in config"):
        load_profile_config(cfg, "bar")


def test_load_profile_config_required_keys_enforced():
    # Minimal hierarchical config with missing required key should raise
    cfg = make_cfg(
        {
            "DEFAULT": {},
            "model:gemma-4-e2b-it": {"group": "gemma", "base_model": "google/gemma-4-E2B-it"},
            "dataset:dummy": {
                "source": "dummy_source",
                "text_column": "text",
                "max_label_length": "128",
                "max_duration": "30",
                "train_split": "train",
                "validation_split": "validation",
            },
            "profile:p": {"model": "gemma-4-e2b-it", "dataset": "dummy"},
        }
    )

    # Missing training hyperparameters trigger required check
    with pytest.raises(ValueError, match=r"Missing required config keys"):
        load_profile_config(cfg, "p")


# ── ProfileConfig dataclass tests ──


class TestProfileConfig:
    """Tests for the ProfileConfig typed container and its dict-compatible interface."""

    def test_from_dict_known_keys(self):
        """Known keys are assigned to typed dataclass fields."""
        d = {"model": "gemma-4", "dataset": "librispeech", "max_label_length": 128}
        pc = ProfileConfig.from_dict(d)
        assert pc.model == "gemma-4"
        assert pc.dataset == "librispeech"
        assert pc.max_label_length == 128

    def test_from_dict_unknown_keys_go_to_extras(self):
        """Unknown keys are accessible via dict interface but stored in _extras."""
        d = {"model": "gemma-4", "some_custom_key": "custom_value"}
        pc = ProfileConfig.from_dict(d)
        assert pc["some_custom_key"] == "custom_value"
        assert "some_custom_key" in pc

    def test_dict_style_read_write(self):
        """Dict-style [] access works for both known and unknown keys."""
        pc = ProfileConfig(model="gemma-4")
        assert pc["model"] == "gemma-4"
        pc["model"] = "gemma-5"
        assert pc.model == "gemma-5"
        pc["new_key"] = 42
        assert pc["new_key"] == 42

    def test_get_with_default(self):
        """.get() returns default when key is missing."""
        pc = ProfileConfig()
        assert pc.get("model_name_or_path") is None
        assert pc.get("nonexistent", "fallback") == "fallback"

    def test_contains(self):
        """'in' operator works for known and unknown keys."""
        pc = ProfileConfig.from_dict({"model": "gemma-4", "extra": True})
        assert "model" in pc
        assert "extra" in pc
        assert "nonexistent_xyz" not in pc

    def test_update(self):
        """.update() merges keys into the config."""
        pc = ProfileConfig(model="old")
        pc.update({"model": "new", "custom": 123})
        assert pc.model == "new"
        assert pc["custom"] == 123

    def test_setdefault(self):
        """.setdefault() only sets if key is missing from extras."""
        pc = ProfileConfig()
        pc.setdefault("new_key", "value")
        assert pc["new_key"] == "value"
        pc.setdefault("new_key", "other")
        assert pc["new_key"] == "value"

    def test_pop(self):
        """.pop() removes and returns extras keys."""
        pc = ProfileConfig.from_dict({"extra_key": "val"})
        assert pc.pop("extra_key") == "val"
        assert "extra_key" not in pc
        assert pc.pop("missing", "default") == "default"

    def test_keys_values_items(self):
        """keys(), values(), items() include both known and extras."""
        pc = ProfileConfig.from_dict({"model": "gemma-4", "custom": True})
        k = pc.keys()
        assert "model" in k
        assert "custom" in k
        items = dict(pc.items())
        assert items["model"] == "gemma-4"
        assert items["custom"] is True

    def test_iteration_enables_dict_conversion(self):
        """dict(config) produces a plain dict with all keys."""
        pc = ProfileConfig.from_dict({"model": "gemma-4", "extra": 99})
        d = dict(pc)
        assert d["model"] == "gemma-4"
        assert d["extra"] == 99
        assert isinstance(d, dict)

    def test_to_dict(self):
        """.to_dict() returns a plain dict."""
        pc = ProfileConfig.from_dict({"model": "gemma-4", "extra": True})
        d = pc.to_dict()
        assert isinstance(d, dict)
        assert d["model"] == "gemma-4"
        assert d["extra"] is True

    def test_load_profile_config_returns_profile_config(self):
        """load_profile_config() returns a ProfileConfig, not a plain dict."""
        cfg = make_cfg(
            {
                "DEFAULT": {
                    "learning_rate": "1e-4",
                },
                "model:gemma-4-e2b-it": {
                    "group": "gemma",
                    "base_model": "google/gemma-4-E2B-it",
                },
                "group:gemma": {},
                "dataset:dummy": {
                    "source": "dummy_source",
                    "text_column": "text",
                    "max_label_length": "128",
                    "max_duration": "30",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "profile:test": {
                    "model": "gemma-4-e2b-it",
                    "dataset": "dummy",
                    "per_device_train_batch_size": "4",
                    "num_train_epochs": "2",
                    "logging_steps": "10",
                    "save_steps": "50",
                    "save_total_limit": "2",
                    "gradient_accumulation_steps": "1",
                },
            }
        )
        result = load_profile_config(cfg, "test")
        assert isinstance(result, ProfileConfig)
        assert result.model == "gemma-4-e2b-it"
        assert result["dataset"] == "dummy"
        assert isinstance(result.max_label_length, int)
        assert result.max_label_length == 128

    def test_profile_overrides_default_num_train_epochs(self):
        """Explicit [profile] values must win over [DEFAULT] for the same key (wizard epochs)."""
        cfg = make_cfg(
            {
                "DEFAULT": {
                    "num_train_epochs": "3",
                    "logging_steps": "25",
                    "save_steps": "1000",
                    "save_total_limit": "1",
                    "gradient_accumulation_steps": "1",
                    "learning_rate": "1e-5",
                    "warmup_steps": "50",
                    "output_dir": "output",
                },
                "model:gemma-4-e2b-it": {
                    "group": "gemma",
                    "base_model": "google/gemma-4-E2B-it",
                },
                "group:gemma": {},
                "dataset:dummy": {
                    "source": "dummy_source",
                    "text_column": "text",
                    "max_label_length": "128",
                    "max_duration": "30",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "profile:test": {
                    "model": "gemma-4-e2b-it",
                    "dataset": "dummy",
                    "per_device_train_batch_size": "4",
                    "num_train_epochs": "7",
                    "logging_steps": "10",
                    "save_steps": "50",
                    "save_total_limit": "2",
                    "gradient_accumulation_steps": "1",
                },
            }
        )
        result = load_profile_config(cfg, "test")
        assert result.num_train_epochs == 7
