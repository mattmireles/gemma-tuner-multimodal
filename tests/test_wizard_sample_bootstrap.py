"""Bundled sample-text config bootstrap (tests config_store only; no torch/wizard UI)."""

import configparser
from pathlib import Path

from gemma_tuner.wizard.config_store import ensure_bundled_sample_config_sections


def test_config_ini_example_defines_sample_sections():
    root = Path(__file__).resolve().parent.parent
    example = root / "config" / "config.ini.example"
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(example)
    assert cfg.has_section("dataset:sample-text")
    assert cfg.has_section("profile:sample-text")
    assert cfg.get("profile:sample-text", "modality") == "text"


def test_ensure_bundled_sample_config_sections_copies_from_example(tmp_path):
    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    (repo / "data" / "datasets" / "sample-text").mkdir(parents=True)

    project_root = Path(__file__).resolve().parent.parent
    example_src = project_root / "config" / "config.ini.example"
    (repo / "config" / "config.ini.example").write_text(example_src.read_text(encoding="utf-8"))

    cfg_ini = repo / "config" / "config.ini"
    cfg_ini.write_text(
        "[DEFAULT]\nx = 1\n\n[model:gemma-3n-e2b-it]\nbase_model = google/gemma-3n-E2B-it\n",
        encoding="utf-8",
    )

    assert ensure_bundled_sample_config_sections(
        config_ini=cfg_ini,
        sample_dataset_name="sample-text",
    )

    out = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    out.read(cfg_ini)
    assert out.has_section("dataset:sample-text")
    assert out.has_section("profile:sample-text")
    assert out.get("dataset:sample-text", "source") == "sample-text"
    assert out.get("profile:sample-text", "text_column") == "response"
