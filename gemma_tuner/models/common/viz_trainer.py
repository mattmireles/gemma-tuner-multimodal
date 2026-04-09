"""Hugging Face Trainer subclass that retains last batch/outputs for the live dashboard."""

from __future__ import annotations

import logging
from typing import Any, Optional

from transformers import Trainer

logger = logging.getLogger(__name__)


class GemmaVizTrainer(Trainer):
    """
    When ``visualize=True``, requests ``output_attentions=True`` on the forward pass and
    stores the last ``outputs`` and a small audio/vision batch snapshot for the Socket.IO
    visualizer. Falls back to a plain forward if attentions are unsupported (e.g. some
    checkpointed / fused paths).
    """

    def __init__(self, *args: Any, visualize: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._visualize = bool(visualize)
        self._viz_last_outputs: Optional[Any] = None
        self._viz_last_batch: Optional[dict[str, Any]] = None

    @staticmethod
    def _snapshot_batch_for_viz(batch: dict[str, Any]) -> dict[str, Any]:
        """Keep tensors needed for mel / vision panels (shallow copy of selected keys)."""
        if not batch:
            return {}
        out: dict[str, Any] = {}
        for k in ("input_features", "audio_values", "pixel_values"):
            if k in batch and batch[k] is not None:
                out[k] = batch[k]
        return out

    def compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[Any] = None,
    ):
        if not self._visualize:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        inputs_with = dict(inputs)
        inputs_with["output_attentions"] = True
        self._viz_last_batch = self._snapshot_batch_for_viz(inputs_with)

        try:
            loss, outputs = super().compute_loss(
                model,
                inputs_with,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        except Exception as e:
            logger.debug("Viz forward with output_attentions failed; retrying without: %s", e)
            self._viz_last_batch = self._snapshot_batch_for_viz(inputs)
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )

        self._viz_last_outputs = outputs
        if return_outputs:
            return loss, outputs
        return loss
