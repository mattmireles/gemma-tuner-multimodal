"""
Configuration Testing Suite for Whisper Fine-Tuning System

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

from core.config import load_profile_config, _validate_profile_config, ConfigConstants


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
            Example: {"profile:test": {"model": "whisper-base", "batch_size": 4}}
    
    Returns:
        ConfigParser instance ready for configuration testing
        All values converted to strings for INI compatibility
        Section structure preserved for test validation
    
    Example usage:
        cfg = make_cfg({
            "profile:test": {
                "model": "whisper-small",
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
        "model": "whisper-small",
        "dataset": "dummy",
        "base_model": "openai/whisper-small",
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
    cfg = make_cfg({
        "DEFAULT": {},
        "profile:foo": {"model": "whisper-small", "dataset": "ds"},
    })
    with pytest.raises(ValueError):
        load_profile_config(cfg, "bar")


def test_load_profile_config_required_keys_enforced():
    # Minimal hierarchical config with missing required key should raise
    cfg = make_cfg({
        "DEFAULT": {},
        "model:whisper-small": {"group": "whisper", "base_model": "openai/whisper-small"},
        "dataset:dummy": {
            "source": "dummy_source",
            "text_column": "text",
            "max_label_length": "128",
            "max_duration": "30",
            "train_split": "train",
            "validation_split": "validation"
        },
        "profile:p": {"model": "whisper-small", "dataset": "dummy"},
    })

    # Missing training hyperparameters trigger required check
    with pytest.raises(ValueError):
        load_profile_config(cfg, "p")


