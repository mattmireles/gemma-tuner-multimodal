from whisper_tuner.core.config import ConfigConstants, _validate_profile_config


def test_validate_profile_config_applies_fallbacks():
    conf = {
        "max_duration": "30.0",
        "max_label_length": "64",
        "num_train_epochs": "3",
        "per_device_train_batch_size": "4",
        "train_split": "train",
        "validation_split": "validation",
        "text_column": "text",
    }
    _validate_profile_config(conf, ["max_duration", "max_label_length"])
    # Ensure fallbacks are present and coerced types exist
    assert conf.get("language_mode") in ("strict", "mixed", "override:" + "") or isinstance(conf.get("language_mode"), str)
    assert isinstance(conf["max_duration"], float)
    assert isinstance(conf["max_label_length"], int)
