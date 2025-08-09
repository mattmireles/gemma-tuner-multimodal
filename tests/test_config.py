import configparser
import pytest

from core.config import load_profile_config, _validate_profile_config, ConfigConstants


def make_cfg(sections: dict) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    for sec, vals in sections.items():
        cfg[sec] = {k: str(v) for k, v in vals.items()}
    return cfg


def test_validate_profile_defaults_and_types():
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

    # Defaults
    assert conf["language_mode"] == "strict"
    # Type coercion
    assert isinstance(conf["max_label_length"], int)
    assert isinstance(conf["max_duration"], float)
    assert isinstance(conf["gradient_checkpointing"], bool)


def test_load_profile_config_missing_profile_raises():
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


