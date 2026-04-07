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


def _read_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(str(_CONFIG_INI))
    return cfg


def _write_config(cfg: configparser.ConfigParser) -> None:
    import os as _os

    # Write with owner-only permissions (0o600) so GCP project IDs stored by
    # the wizard are not world-readable on shared systems.
    fd = _os.open(str(_CONFIG_INI), _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC, 0o600)
    with _os.fdopen(fd, "w") as f:
        cfg.write(f)


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
