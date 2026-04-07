"""Wizard package for interactive training configuration.

Re-exports public API for backward compatibility.
"""

from gemma_tuner.wizard.base import ModelSpecs, TrainingMethod, WizardConstants, detect_datasets, get_wizard_device_info
from gemma_tuner.wizard.runner import wizard_main

__all__ = [
    "wizard_main",
    "WizardConstants",
    "TrainingMethod",
    "ModelSpecs",
    "get_wizard_device_info",
    "detect_datasets",
]
