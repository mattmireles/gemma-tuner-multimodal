"""
Configuration Management System for Gemma Fine-Tuning

This module provides hierarchical configuration loading and validation for the
Gemma fine-tuning pipeline. It implements a sophisticated configuration merge
strategy that allows for flexible profile-based training setups with sensible
defaults and overrides.

Key responsibilities:
- Hierarchical configuration loading from INI files
- Profile-based configuration management
- Model and dataset configuration merging
- Configuration validation and semantic checking
- Default value propagation and override handling

Called by:
- main.py:main() for loading training profiles (lines 209, 214)
- scripts/finetune.py for direct model+dataset configuration
- scripts/evaluate.py for evaluation configuration loading
- scripts/blacklist.py for blacklist generation configuration

Configuration hierarchy (lowest to highest precedence):
1. [DEFAULT] section - Base defaults for all configurations
2. [dataset_defaults] section - Dataset-specific defaults
3. [group:{group_name}] section - Model group defaults (e.g., Gemma, distil)
4. [model:{model_name}] section - Specific model configuration
5. [dataset:{dataset_name}] section - Specific dataset configuration
6. [profile:{profile_name}] section - Profile-specific overrides (highest precedence)

Example configuration flow:
- Profile "gemma-3n-audioset" requested
- Loads DEFAULT → dataset_defaults → group:gemma → model:gemma-3n-e4b-it → dataset:librispeech → profile:gemma-3n-audioset
- Each level can override previous values
- Final config contains merged values with profile having ultimate precedence
"""

import configparser
from typing import Dict

from gemma_tuner.core.profile_config import ProfileConfig
from gemma_tuner.utils.device import to_bool


# Configuration validation and processing constants
class ConfigConstants:
    """Named constants for configuration validation, type coercion, and default values.

    This class centralizes all configuration-related constants to avoid magic values
    throughout the codebase. Constants are organized by category for clarity.
    """

    # Required configuration keys for training profiles
    # These keys must be present after configuration merging for training to proceed
    REQUIRED_PROFILE_KEYS = [
        "model",
        "dataset",
        "base_model",
        "train_split",
        "validation_split",
        "text_column",
        "max_label_length",
        "max_duration",
        "per_device_train_batch_size",
        "num_train_epochs",
        "logging_steps",
        "save_steps",
        "save_total_limit",
        "gradient_accumulation_steps",
    ]

    # Required keys for direct model+dataset configuration (reduced set)
    # Used when bypassing profile system for direct model+dataset training
    REQUIRED_MODEL_DATASET_KEYS = [
        "base_model",
        "train_split",
        "validation_split",
        "text_column",
        "max_label_length",
        "max_duration",
    ]

    # INI file section name prefixes for hierarchical configuration
    # These prefixes determine configuration precedence during merge operations
    PROFILE_PREFIX = "profile:"  # Highest precedence - specific training profiles
    MODEL_PREFIX = "model:"  # Model-specific configuration (batch sizes, dtype)
    DATASET_PREFIX = "dataset:"  # Dataset-specific configuration (columns, splits)
    GROUP_PREFIX = "group:"  # Model group configuration (e.g., gemma)

    # Special section names with specific merge behavior
    DEFAULT_SECTION = "DEFAULT"  # Global defaults, lowest precedence
    DATASET_DEFAULTS_SECTION = "dataset_defaults"  # Common dataset processing defaults

    # Validation thresholds for semantic correctness
    MIN_MAX_DURATION = 0.0  # Minimum valid audio duration (seconds)
    MIN_MAX_LABEL_LENGTH = 0  # Minimum valid transcription length (tokens)
    MIN_BATCH_SIZE = 1  # Minimum training batch size
    MIN_TRAIN_EPOCHS = 1  # Minimum number of training epochs
    MIN_LEARNING_RATE = 1e-7  # Minimum learning rate to prevent underflow
    MAX_LEARNING_RATE = 1.0  # Maximum learning rate to prevent instability

    # Type coercion mapping for configuration values
    # INI files store all values as strings; these define target types
    INT_COERCION_KEYS = {
        "max_label_length",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "num_train_epochs",
        "logging_steps",
        "save_steps",
        "save_total_limit",
        "gradient_accumulation_steps",
        "preprocessing_num_workers",
        "dataloader_num_workers",
        "lora_r",
        "lora_alpha",
        "warmup_steps",
        "max_samples",
        "max_seq_length",
    }

    FLOAT_COERCION_KEYS = {
        "max_duration",
        "validation_wer_threshold",
        "wer_threshold",
        "lora_dropout",
        "learning_rate",
        "weight_decay",
        "temperature",
        "distillation_temperature",
        "distillation_alpha",
        "kl_weight",
        "sample_validation_rate",
    }

    BOOL_COERCION_KEYS = {
        "force_languages",
        "streaming",
        "bf16",
        "fp16",
        "gradient_checkpointing",
        "enable_8bit",
        "streaming_enabled",
        "load_validation",
        "visualize",
        "concatenate_audio",
        "use_peft",
        "skip_audio_validation",
    }

    # List coercion mapping (key -> delimiter)
    # Used for comma-separated values in INI files
    LIST_COERCION_MAPPING = {
        "lora_target_modules": ",",
        "languages": ",",
    }

    # Default fallback values for common configuration keys
    # Applied when keys are missing or empty
    FALLBACK_DEFAULTS = {
        "language_mode": "strict",  # Language handling mode for multilingual models
        "force_languages": False,  # Whether to force language detection
        "languages": "all",  # Default language set for training
        "streaming_enabled": False,  # Dataset streaming for large datasets
        "preprocessing_num_workers": 0,  # CPU workers for data preprocessing
        "dataloader_num_workers": 4,  # CPU workers for data loading
        "gradient_checkpointing": False,  # Memory vs compute tradeoff
        "bf16": False,  # BFloat16 mixed precision training
        "fp16": False,  # Float16 mixed precision training
        "skip_audio_validation": False,  # Skip audio file existence checks (faster but risky)
        "sample_validation_rate": 1.0,  # Fraction of audio files to validate (0.0-1.0)
        # Text vs audio fine-tuning (defaults preserve existing audio-only behavior)
        "modality": "audio",
        "text_sub_mode": "instruction",
        "prompt_column": None,
        "max_seq_length": 2048,
    }

    # Granary Dataset Integration Constants
    # NVIDIA Granary dataset requires external audio corpus downloads
    # These constants define the supported corpora and validation requirements
    GRANARY_AUDIO_SOURCE_PREFIX = "audio_source_"  # Prefix for audio source configuration keys

    # Supported Granary audio corpora with descriptions
    # Used for validation and user guidance in preparation scripts
    GRANARY_SUPPORTED_CORPORA = {
        "voxpopuli": "VoxPopuli multilingual parliamentary speeches dataset",
        "ytc": "YouTube Commons dataset with diverse content",
        "librilight": "LibriLight large-scale English audiobook dataset",
        "yodas": "YODAS dataset (included in HuggingFace download)",
    }

    # Granary dataset configuration validation requirements
    # Keys that must be present for Granary dataset preparation
    GRANARY_REQUIRED_KEYS = {
        "hf_name": "nvidia/Granary",  # HuggingFace dataset identifier
        "hf_subset": "",  # Language subset (e.g., "en", "es")
        "local_path": "",  # Local preparation output directory
    }

    # Granary audio source validation patterns
    # Minimum required external corpora (YODAS is included in HF download)
    GRANARY_MINIMUM_EXTERNAL_CORPORA = {"voxpopuli", "ytc", "librilight"}


def load_profile_config(cfg: configparser.ConfigParser, profile_name: str) -> "ProfileConfig":
    """
    Loads and merges hierarchical configuration for a named training profile.

    This function implements the core configuration loading strategy for profile-based
    training. It merges configuration values from multiple levels following a strict
    precedence hierarchy, ensuring that specific overrides take precedence over defaults.
    The hierarchical approach allows for DRY configuration management where common
    settings are defined once and specialized where needed.

    Called by:
    - main.py:main() when using --profile argument for finetune operation (line 225)
    - main.py:main() when using --profile argument for evaluate operation (line 318)
    - main.py:main() when using --profile argument for blacklist operation (line 446)
    - wizard.py:generate_profile_config() for wizard-generated profiles (line 1012)
    - Batch training automation scripts using profile-based workflows
    - CI/CD pipelines executing profile-based training runs

    Calls to:
    - load_model_dataset_config() for the core hierarchical merge logic (line 144)
    - _validate_profile_config() for semantic validation of merged configuration (line 150)
    - configparser.ConfigParser methods for section existence and value retrieval

    Configuration merge order (each level overrides previous):
    1. [DEFAULT] - Global defaults for all configurations (learning rate, epochs, etc.)
    2. [dataset_defaults] - Common dataset processing parameters (text columns, durations)
    3. [group:{group}] - Model family defaults (dtype, attention implementation)
    4. [model:{model}] - Specific model configuration (base model, batch sizes)
    5. [dataset:{dataset}] - Specific dataset configuration (splits, languages)
    6. [profile:{profile}] - Profile-specific overrides (highest precedence)

    Profile resolution workflow:
    1. Verify profile section exists in config.ini (raises ValueError if missing)
    2. Start with DEFAULT section values (always present in ConfigParser)
    3. Layer dataset_defaults if present (common preprocessing settings)
    4. Extract model and dataset names from profile section
    5. Delegate core merge logic to load_model_dataset_config() for consistency
    6. Apply profile section overrides as final layer (highest precedence)
    7. Validate final configuration for required keys and semantic correctness
    8. Return fully resolved configuration ready for training pipeline

    State management:
    - No internal state maintained (stateless function)
    - Returns new dictionary without modifying input ConfigParser
    - Type coercion applied in-place to returned dictionary

    Args:
        cfg (configparser.ConfigParser): Loaded configuration file parser containing
                                        all sections from config.ini
        profile_name (str): Name of the profile to load, must match [profile:name] section
                           Examples: "gemma-3n-audioset", "medium-lora-data3"

    Returns:
        Dict: Merged configuration dictionary with all resolved values, type-coerced,
              and validated for completeness. Ready for direct use by training scripts.

    Raises:
        ValueError: If profile section not found in config.ini, or if configuration
                   validation fails (missing required keys, invalid value ranges)
        configparser.NoOptionError: If required model/dataset keys missing from profile

    Example:
        >>> cfg = configparser.ConfigParser()
        >>> cfg.read("config.ini")
        >>> config = load_profile_config(cfg, "gemma-3n-custom")
        >>> print(config["base_model"])  # "google/gemma-3n-e4b-it" (from model section)
        >>> print(config["learning_rate"]) # 1e-5 (from DEFAULT section)
        >>> print(config["batch_size"])    # 32 (from profile section override)
    """
    section = f"{ConfigConstants.PROFILE_PREFIX}{profile_name}"
    if not cfg.has_section(section):
        raise ValueError(f"Profile '{profile_name}' not found in config.ini.")

    # Extract model and dataset names from the profile section, then delegate the
    # full DEFAULT → dataset_defaults → group → model → dataset merge to
    # load_model_dataset_config(). The previous code built an `out` dict from layers
    # 1-2 that was immediately overwritten by the assignment below (dead code).
    model_name = cfg.get(section, "model")
    dataset_name = cfg.get(section, "dataset")

    out: Dict = _load_model_dataset_config_dict(cfg, model_name, dataset_name)

    # Layer 6: Profile overrides (highest precedence) — own keys only, no DEFAULT bleed.
    if cfg.has_section(section):
        out.update(_section_own_keys(cfg, section))

    # Validate the merged configuration
    _validate_profile_config(out, required_keys=ConfigConstants.REQUIRED_PROFILE_KEYS)
    return ProfileConfig.from_dict(out)


def _section_own_keys(cfg: configparser.ConfigParser, section: str) -> Dict:
    """Return only the keys explicitly set in a section, excluding DEFAULT bleed.

    configparser makes every key from [DEFAULT] appear in every section via
    cfg[section] and cfg.options(section). Using raw cfg[section] in a merge
    chain causes earlier-layer overrides to be silently reset when a later
    layer's section happens to inherit the same key from DEFAULT.

    This helper returns only the keys that are genuinely local to the section
    by subtracting cfg.defaults() from cfg.options(section). The returned dict
    is safe to use in out.update() at any merge layer.

    Called by:
    - _load_model_dataset_config_dict() for group, model, and dataset layers
    - load_profile_config() already uses the same pattern inline for layer 6
    """
    defaults = set(cfg.defaults().keys())
    return {k: cfg[section][k] for k in cfg.options(section) if k not in defaults}


def load_model_dataset_config(cfg: configparser.ConfigParser, model_name: str, dataset_name: str) -> "ProfileConfig":
    """Public wrapper that returns a ProfileConfig. See _load_model_dataset_config_dict for full docs."""
    out = _load_model_dataset_config_dict(cfg, model_name, dataset_name)
    return ProfileConfig.from_dict(out)


def _load_model_dataset_config_dict(cfg: configparser.ConfigParser, model_name: str, dataset_name: str) -> Dict:
    """
    Loads and merges configuration for direct model+dataset training without profiles.

    This function provides an alternative configuration loading strategy for cases where
    users want to directly specify a model and dataset combination without defining a
    named profile. It's particularly useful for ad-hoc training runs, experimentation,
    and programmatic training workflows where profile creation overhead is undesirable.

    Called by:
    - main.py:main() when parsing model+dataset combination ("model+dataset" format) (line 312)
    - load_profile_config() for the core hierarchical merge logic (line 144)
    - wizard.py:generate_profile_config() for wizard-generated configurations (line 1012)
    - Testing scripts bypassing the profile system for isolated model/dataset testing
    - Batch processing scripts iterating over model/dataset combinations
    - Research workflows experimenting with different model/dataset pairings

    Calls to:
    - _validate_profile_config() for semantic validation with relaxed key requirements (line 234)
    - configparser.ConfigParser methods for section existence checks and value retrieval

    Configuration merge order (each layer overrides previous):
    1. [DEFAULT] - Global defaults (learning rates, epochs, output directories)
    2. [dataset_defaults] - Common dataset processing parameters (text columns, durations)
    3. [group:{group}] - Model group configuration extracted from model's "group" attribute
    4. [model:{model}] - Model-specific configuration (base model path, batch sizes, dtype)
    5. [dataset:{dataset}] - Dataset-specific configuration (splits, languages) - highest precedence

    Direct configuration workflow:
    1. Validate both model and dataset sections exist in config.ini
    2. Initialize output dictionary with DEFAULT section (global defaults)
    3. Layer dataset_defaults if section exists (common preprocessing settings)
    4. Extract model's group from [model:name] section and apply [group:name] if exists
    5. Apply complete [model:name] section configuration
    6. Apply [dataset:name] configuration as final override layer
    7. Validate merged configuration with relaxed key requirements (no profile keys needed)
    8. Return fully merged configuration ready for training

    State management:
    - Stateless function with no internal state persistence
    - Creates new dictionary without modifying input ConfigParser
    - Type coercion and validation applied to returned dictionary

    Args:
        cfg (configparser.ConfigParser): Loaded configuration parser containing all
                                        sections from config.ini file
        model_name (str): Name matching [model:name] section in config.ini
                         Examples: "gemma-3n-e4b-it", "gemma-3n-lora", "distil-small.en"
        dataset_name (str): Name matching [dataset:name] section in config.ini
                           Examples: "librispeech", "custom-data", "bq_gemma_finetuning_20250813"

    Returns:
        Dict: Fully merged configuration dictionary with hierarchical overrides applied,
              type-coerced values, and validation completed. Ready for direct use by
              model-specific training implementations.

    Raises:
        ValueError: If model section [model:name] or dataset section [dataset:name] not
                   found in config.ini, or if merged configuration fails validation
        configparser.NoOptionError: If model section missing required "group" attribute
                                    needed for group configuration resolution

    Example:
        >>> cfg = configparser.ConfigParser()
        >>> cfg.read("config.ini")
        >>> config = load_model_dataset_config(cfg, "gemma-3n", "custom-dataset")
        >>> # Results in merged config with Gemma group defaults, gemma-3n model
        >>> # settings, and custom-dataset overrides
        >>> print(config["base_model"])     # "google/gemma-3n-e4b-it" (from model)
        >>> print(config["text_column"])    # "transcript" (from dataset)
        >>> print(config["learning_rate"])  # 1e-5 (from DEFAULT)
    """
    model_section = f"{ConfigConstants.MODEL_PREFIX}{model_name}"
    dataset_section = f"{ConfigConstants.DATASET_PREFIX}{dataset_name}"

    if not cfg.has_section(model_section):
        raise ValueError(f"Model '{model_name}' not found in config.ini.")
    if not cfg.has_section(dataset_section):
        raise ValueError(f"Dataset '{dataset_name}' not found in config.ini.")

    out: Dict = {}

    # Layer 1: Global defaults (always present)
    out.update(cfg[ConfigConstants.DEFAULT_SECTION])

    # Layer 2: Dataset processing defaults
    if cfg.has_section(ConfigConstants.DATASET_DEFAULTS_SECTION):
        out.update(cfg[ConfigConstants.DATASET_DEFAULTS_SECTION])

    # Layer 3: Model group configuration — own keys only, no DEFAULT bleed.
    # Use fallback=None so a missing "group" key returns None instead of raising
    # configparser.NoOptionError (which callers don't catch — they only catch ValueError).
    group_name = cfg.get(model_section, "group", fallback=None)
    if group_name is not None:
        group_section = f"{ConfigConstants.GROUP_PREFIX}{group_name}"
        if cfg.has_section(group_section):
            out.update(_section_own_keys(cfg, group_section))

    # Layer 4: Model-specific configuration — own keys only, no DEFAULT bleed.
    out.update(_section_own_keys(cfg, model_section))

    # Layer 5: Dataset-specific configuration — own keys only, no DEFAULT bleed.
    out.update(_section_own_keys(cfg, dataset_section))

    # Extract audio source paths for multi-corpus datasets like Granary
    # This enables external audio corpus integration by parsing audio_source_* keys
    # Example: audio_source_voxpopuli = /path/to/voxpopuli → {"voxpopuli": "/path/to/voxpopuli"}
    audio_sources = {}
    for key, value in _section_own_keys(cfg, dataset_section).items():
        if key.startswith("audio_source_"):
            # Extract source name: "audio_source_voxpopuli" → "voxpopuli"
            source_name = key[len("audio_source_") :]
            if source_name:  # Ensure source name is not empty
                audio_sources[source_name] = value

    # Add audio_sources dictionary to configuration if any sources were found
    # This is used by dataset preparation scripts like scripts/prepare_granary.py
    # for validating and mapping external audio corpus paths
    if audio_sources:
        out["audio_sources"] = audio_sources

    # Validate with relaxed requirements (no profile-specific keys needed)
    _validate_profile_config(out, required_keys=ConfigConstants.REQUIRED_MODEL_DATASET_KEYS)
    return out


def _validate_profile_config(conf: Dict, required_keys: list[str]) -> None:
    """
    Validates merged configuration for completeness and semantic correctness.

    This function performs comprehensive validation of configuration dictionaries after
    hierarchical merging. It handles both structural validation (required keys) and
    semantic validation (value correctness), along with type coercion from INI string
    values to appropriate Python types. This is the final step before configuration
    is passed to training implementations.

    Called by:
    - load_profile_config() after profile configuration merge (line 150)
    - load_model_dataset_config() after model+dataset configuration merge (line 234)

    Calls to:
    - ConfigConstants class for validation thresholds and type coercion mappings
    - Internal _parse_bool() helper for boolean value parsing from INI strings

    Validation workflow:
    1. Apply fallback defaults for missing configuration keys
    2. Type coercion from INI strings to proper Python types (int, float, bool, list)
    3. Structural validation - ensure all required keys are present
    4. Semantic validation - verify values are within valid ranges and logically sound
    5. Training-specific validations - LoRA parameters, data splits, learning rates

    Type coercion performed:
    - Integers: batch sizes, epochs, steps, workers, LoRA parameters
    - Floats: durations, thresholds, learning rates, dropout values
    - Booleans: feature flags, training options (handles "true"/"1"/"yes" variations)
    - Lists: comma-separated values like LoRA target modules, languages

    Semantic validations performed:
    - max_duration > 0.0 (audio clips must have positive duration)
    - max_label_length > 0 (transcriptions must have positive token length)
    - batch_size >= 1 (training requires at least 1 sample per batch)
    - num_train_epochs >= 1 (training requires at least 1 complete epoch)
    - learning_rate within reasonable bounds (prevents training instability)
    - LoRA parameters within valid ranges (rank > 0, dropout [0,1))
    - train_split and validation_split not empty (data splits must be specified)

    State modification:
    - Modifies conf dictionary in-place with type-coerced values
    - Adds fallback defaults for missing keys
    - Does not modify original ConfigParser or INI file

    Args:
        conf (Dict): Configuration dictionary to validate and modify in-place.
                    Contains merged values from hierarchical configuration loading.
        required_keys (list[str]): List of configuration keys that must be present
                                  after merging. Different for profile vs model+dataset.

    Raises:
        ValueError: If validation fails with detailed error message indicating:
                   - Missing required configuration keys
                   - Invalid value types or ranges
                   - Semantic inconsistencies in configuration
                   - Training parameter conflicts

    Side Effects:
        - Modifies conf dictionary in-place with type-coerced values
        - Adds default values for missing optional configuration keys

    Example:
        >>> conf = {"max_duration": "30.0", "learning_rate": "1e-5", "gradient_checkpointing": "true"}
        >>> _validate_profile_config(conf, ["max_duration", "learning_rate"])
        >>> # conf now contains: {"max_duration": 30.0, "learning_rate": 1e-5, "gradient_checkpointing": True}
        >>> print(type(conf["max_duration"]))  # <class 'float'>
    """
    # Apply fallback defaults for missing or empty keys
    # This ensures consistent configuration state regardless of INI completeness
    for key, default_value in ConfigConstants.FALLBACK_DEFAULTS.items():
        conf.setdefault(key, default_value)

    # Integer coercion with validation for numeric configuration values
    # These values control training behavior and resource allocation
    for key in ConfigConstants.INT_COERCION_KEYS:
        if key in conf and conf[key] is not None and conf[key] != "":
            # Only wrap type conversion in try/except — not the range checks.
            # This mirrors the float coercion block pattern below.
            try:
                if not isinstance(conf[key], int):
                    conf[key] = int(conf[key])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid integer for {key}: {conf[key]!r}") from e
            # Semantic range validation — separate from type coercion above
            if (
                key in ("per_device_train_batch_size", "per_device_eval_batch_size")
                and conf[key] < ConfigConstants.MIN_BATCH_SIZE
            ):
                raise ValueError(f"{key} must be >= {ConfigConstants.MIN_BATCH_SIZE}, got {conf[key]}")
            if key == "num_train_epochs" and conf[key] < ConfigConstants.MIN_TRAIN_EPOCHS:
                raise ValueError(f"num_train_epochs must be >= {ConfigConstants.MIN_TRAIN_EPOCHS}, got {conf[key]}")

    # Float coercion with range validation for hyperparameters
    # These values affect training stability and convergence
    for key in ConfigConstants.FLOAT_COERCION_KEYS:
        if key in conf and conf[key] is not None and conf[key] != "":
            # Only wrap the float conversion in try/except — not the range checks.
            # If we wrapped the range checks too, a valid float like 1e-8 would produce
            # "Invalid float for learning_rate: 1e-8 - learning_rate too small: ..."
            # which implies the value isn't a float when it actually is.
            try:
                if not isinstance(conf[key], float):
                    conf[key] = float(conf[key])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid float for {key}: {conf[key]!r}") from e
            # Semantic range validation — separate from type coercion above
            if key == "learning_rate":
                if conf[key] < ConfigConstants.MIN_LEARNING_RATE:
                    raise ValueError(f"learning_rate too small: {conf[key]} < {ConfigConstants.MIN_LEARNING_RATE}")
                if conf[key] > ConfigConstants.MAX_LEARNING_RATE:
                    raise ValueError(f"learning_rate too large: {conf[key]} > {ConfigConstants.MAX_LEARNING_RATE}")

    # Boolean coercion for feature flags and training options
    # Handles various string representations of boolean values in INI files
    for key in ConfigConstants.BOOL_COERCION_KEYS:
        if key in conf and conf[key] is not None and conf[key] != "":
            conf[key] = to_bool(conf[key])

    # List coercion for comma-separated configuration values
    # Common for target modules, languages, and other multi-value settings
    for key, separator in ConfigConstants.LIST_COERCION_MAPPING.items():
        if key in conf and isinstance(conf[key], str):
            # Split by separator, strip whitespace, filter empty values
            conf[key] = [item.strip() for item in conf[key].split(separator) if item.strip()]

    # Check for missing required keys
    missing = [k for k in required_keys if k not in conf]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    # Semantic validation with type coercion
    if "max_duration" in conf:
        # Convert to float if needed and validate range
        md = float(conf["max_duration"]) if not isinstance(conf["max_duration"], float) else conf["max_duration"]
        if md <= ConfigConstants.MIN_MAX_DURATION:
            raise ValueError(f"max_duration must be > {ConfigConstants.MIN_MAX_DURATION}, got {md}")

    if "max_label_length" in conf:
        # Convert to int if needed and validate range
        ml = (
            int(conf["max_label_length"]) if not isinstance(conf["max_label_length"], int) else conf["max_label_length"]
        )
        if ml <= ConfigConstants.MIN_MAX_LABEL_LENGTH:
            raise ValueError(f"max_label_length must be > {ConfigConstants.MIN_MAX_LABEL_LENGTH}, got {ml}")

    # LoRA parameter validation for Parameter-Efficient Fine-Tuning (PEFT)
    # Validates LoRA configuration when PEFT methods are enabled
    if any(lora_key in conf for lora_key in ("lora_r", "lora_alpha", "lora_dropout", "lora_target_modules")):
        if "lora_r" in conf:
            lora_r = int(conf["lora_r"])
            if lora_r <= 0:
                raise ValueError(f"lora_r must be a positive integer, got {lora_r}")
            if lora_r > 1024:  # Reasonable upper bound for LoRA rank
                raise ValueError(f"lora_r too large ({lora_r}), consider values <= 1024 for efficiency")

        if "lora_alpha" in conf:
            lora_alpha = int(conf["lora_alpha"])
            if lora_alpha <= 0:
                raise ValueError(f"lora_alpha must be a positive integer, got {lora_alpha}")

        if "lora_dropout" in conf:
            lora_dropout = float(conf["lora_dropout"])
            if lora_dropout < 0.0 or lora_dropout >= 1.0:
                raise ValueError(f"lora_dropout must be in range [0.0, 1.0), got {lora_dropout}")

        if "lora_target_modules" in conf and isinstance(conf["lora_target_modules"], list):
            # Validate that target modules are not empty
            if not conf["lora_target_modules"]:
                raise ValueError("lora_target_modules cannot be empty list when LoRA is enabled")

    # Modality / text-mode keys (defaults applied via FALLBACK_DEFAULTS)
    if "modality" in conf and conf["modality"] is not None:
        modality = str(conf["modality"]).strip().lower()
        conf["modality"] = modality
        if modality not in ("audio", "text"):
            raise ValueError(f"modality must be 'audio' or 'text', got {modality!r}")
    if "text_sub_mode" in conf and conf["text_sub_mode"] is not None:
        sub = str(conf["text_sub_mode"]).strip().lower()
        conf["text_sub_mode"] = sub
        if sub not in ("instruction", "completion"):
            raise ValueError(f"text_sub_mode must be 'instruction' or 'completion', got {sub!r}")
    if "prompt_column" in conf:
        pc = conf["prompt_column"]
        if pc is None or (isinstance(pc, str) and not str(pc).strip()):
            conf["prompt_column"] = None
        elif isinstance(pc, str):
            conf["prompt_column"] = pc.strip()

    if "max_seq_length" in conf and conf["max_seq_length"] is not None and conf["max_seq_length"] != "":
        msl = int(conf["max_seq_length"]) if not isinstance(conf["max_seq_length"], int) else conf["max_seq_length"]
        if msl < 1:
            raise ValueError(f"max_seq_length must be >= 1, got {msl}")

    # Validate data splits are specified
    for split_key in ("train_split", "validation_split"):
        if split_key in conf and not str(conf[split_key]).strip():
            raise ValueError(f"{split_key} cannot be empty - must specify data split name")
