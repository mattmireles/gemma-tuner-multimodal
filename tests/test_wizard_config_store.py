import configparser

from gemma_tuner.wizard import config as wizard_config
from gemma_tuner.wizard import config_store, granary


def test_add_dataset_to_config_sets_required_defaults(tmp_path, monkeypatch):
    config_path = tmp_path / "config.ini"
    config_path.write_text("")
    monkeypatch.setattr(config_store, "_CONFIG_INI", config_path)

    wizard_config._add_dataset_to_config("bq-sample", "text_perfect")

    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(config_path)

    assert cfg.get("dataset:bq-sample", "source") == "bq-sample"
    assert cfg.get("dataset:bq-sample", "text_column") == "text_perfect"
    assert cfg.get("dataset:bq-sample", "train_split") == "train"
    assert cfg.get("dataset:bq-sample", "validation_split") == "validation"


def test_config_module_reexports_split_helpers():
    assert wizard_config._read_config is config_store._read_config
    assert wizard_config.setup_granary_dataset is granary.setup_granary_dataset
