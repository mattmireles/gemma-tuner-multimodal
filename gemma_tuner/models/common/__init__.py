"""Common components shared across Gemma model trainers.

Provides LLM-ready building blocks used by all training variants:
- args: shared training argument builders and worker-count logic
- collators: batched data collation for standard, LoRA, and distill
- metrics: standardized WER/CER compute utilities
- visualizer: lightweight callback to stream metrics to the visualizer

Called by:
- models.gemma.finetune
- models.gemma.finetune
- models.gemma.finetune
"""

__all__ = [
    "args",
    "collators",
    "metrics",
    "visualizer",
]
