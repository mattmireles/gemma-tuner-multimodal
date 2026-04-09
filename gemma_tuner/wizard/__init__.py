"""Wizard package for interactive training configuration.

Re-exports public API for backward compatibility. Imports are lazy so submodules
like ``config_store`` can load without pulling Rich/questionary (used by tests and
lightweight callers).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "wizard_main",
    "WizardConstants",
    "TrainingMethod",
    "ModelSpecs",
    "get_wizard_device_info",
    "detect_datasets",
]


def __getattr__(name: str) -> Any:
    if name == "wizard_main":
        from gemma_tuner.wizard.runner import wizard_main

        return wizard_main
    if name in (
        "WizardConstants",
        "TrainingMethod",
        "ModelSpecs",
        "detect_datasets",
        "get_wizard_device_info",
    ):
        from gemma_tuner.wizard import base

        return getattr(base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
