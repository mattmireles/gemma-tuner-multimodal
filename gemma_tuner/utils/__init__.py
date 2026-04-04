"""Utilities package for dataset, device, and helper functions.

Called by:
- Training and evaluation modules needing dataset loading helpers
- CLI entrypoints that need device detection or memory helpers

Exports:
- dataset_utils: high-level dataset/patch loading helpers
- device: device detection, synchronization, cache management
"""

__all__ = [
    "dataset_utils",
    "device",
]
