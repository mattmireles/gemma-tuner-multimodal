#!/usr/bin/env python3

"""Shared config.ini access helpers for the interactive wizard.

This module owns persistence details for the wizard's `config.ini` mutations so
dataset import/setup flows can share the same read/write path without growing
`wizard/config.py` into the write path for every concern.
"""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Optional

# Absolute path to the repository root so config.ini and data/ references work
# regardless of the caller's working directory.
# Layout: gemma_tuner/wizard/config_store.py -> three parents up = repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_INI = _PROJECT_ROOT / "config" / "config.ini"


def _read_config_path(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(str(path))
    return cfg


def _read_config() -> configparser.ConfigParser:
    return _read_config_path(_CONFIG_INI)


def _write_config_to_path(path: Path, cfg: configparser.ConfigParser) -> None:
    import os as _os

    # Write with owner-only permissions (0o600) so GCP project IDs stored by
    # the wizard are not world-readable on shared systems.
    fd = _os.open(str(path), _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC, 0o600)
    with _os.fdopen(fd, "w") as f:
        cfg.write(f)


def _write_config(cfg: configparser.ConfigParser) -> None:
    _write_config_to_path(_CONFIG_INI, cfg)


def ensure_bundled_sample_config_sections(
    *,
    config_ini: Path,
    sample_dataset_name: str,
) -> bool:
    """Copy ``[dataset:*]`` / ``[profile:*]`` for the bundled sample from ``config.ini.example``.

    Used when ``data/datasets/<name>/`` exists but the user's ``config.ini`` predates
    those sections. Returns ``True`` if ``config_ini`` was written.
    """
    repo_root = config_ini.parent.parent
    sample_dir = repo_root / "data" / "datasets" / sample_dataset_name
    if not sample_dir.exists() or not config_ini.exists():
        return False

    example_path = config_ini.with_name("config.ini.example")
    example_cfg = configparser.ConfigParser()
    if example_path.exists():
        example_cfg.read(str(example_path))

    cfg = _read_config_path(config_ini)
    dataset_section = f"dataset:{sample_dataset_name}"
    profile_section = f"profile:{sample_dataset_name}"
    changed = False

    if not cfg.has_section(dataset_section):
        cfg.add_section(dataset_section)
        if example_cfg.has_section(dataset_section):
            for key, value in example_cfg.items(dataset_section):
                cfg.set(dataset_section, key, value)
        else:
            cfg.set(dataset_section, "source", sample_dataset_name)
            cfg.set(dataset_section, "train_split", "train")
            cfg.set(dataset_section, "validation_split", "validation")
        changed = True

    if not cfg.has_section(profile_section) and example_cfg.has_section(profile_section):
        cfg.add_section(profile_section)
        for key, value in example_cfg.items(profile_section):
            cfg.set(profile_section, key, value)
        changed = True

    if changed:
        _write_config_to_path(config_ini, cfg)
    return changed


def _add_dataset_to_config(dataset_name: str, text_column: str) -> None:
    """Ensure `[dataset:dataset_name]` exists with source and text_column."""
    cfg = _read_config()
    section = f"dataset:{dataset_name}"
    if not cfg.has_section(section):
        cfg.add_section(section)
    cfg.set(section, "source", dataset_name)
    if text_column:
        cfg.set(section, "text_column", text_column)

    # BQ-created datasets have standard train/validation splits.
    # This ensures they are always present for the config validator.
    if not cfg.has_option(section, "train_split"):
        cfg.set(section, "train_split", "train")
    if not cfg.has_option(section, "validation_split"):
        cfg.set(section, "validation_split", "validation")

    _write_config(cfg)


def _update_bq_defaults(project_id: Optional[str], dataset_id: Optional[str]) -> None:
    cfg = _read_config()
    section = "bigquery"
    if not cfg.has_section(section):
        cfg.add_section(section)
    if project_id:
        cfg.set(section, "last_project_id", project_id)
    if dataset_id:
        cfg.set(section, "last_dataset_id", dataset_id)
    _write_config(cfg)
