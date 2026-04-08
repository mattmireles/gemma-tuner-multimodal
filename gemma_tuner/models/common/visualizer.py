from __future__ import annotations

import logging

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class VisualizerTrainerCallback(TrainerCallback):
    """
    Lightweight callback that streams training metrics to the built-in visualizer.

    Usage: add to a HuggingFace Trainer when profile_config['visualize'] is True.
    """

    def __init__(self, update_every_steps: int = 10):
        self.update_every_steps = max(1, int(update_every_steps))
        self._initialized = False
        self._last_logged_step = -1

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            trainer = kwargs.get("trainer")
            if trainer is not None:
                model = getattr(trainer, "model", None)
        self.initialize(model)

    def initialize(self, model) -> None:
        # Lazily initialize the visualizer with the model and device.
        # HF Trainer passes 'model' as a kwarg (not 'trainer').
        if model is None:
            return
        try:
            from gemma_tuner.visualizer import init_visualizer

            init_visualizer(model, model.device)
            self._initialized = True
        except Exception as e:
            logger.warning("Visualizer init failed (training continues without it): %s", e)
            self._initialized = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not self._initialized:
            return
        try:
            from gemma_tuner.visualizer import get_visualizer

            viz = get_visualizer()
            if viz:
                viz.update_epoch(state.epoch)
        except Exception:
            pass

    def push_epoch(self, epoch: float) -> None:
        if not self._initialized:
            return
        try:
            from gemma_tuner.visualizer import get_visualizer

            viz = get_visualizer()
            if viz:
                viz.update_epoch(epoch)
        except Exception:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._initialized or not logs:
            return
        # Throttle updates
        if state.global_step is None:
            return
        if state.global_step == self._last_logged_step:
            return
        if state.global_step % self.update_every_steps != 0:
            return

        self._last_logged_step = state.global_step

        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")

        loss = logs.get("loss") or logs.get("train_loss")
        try:
            lr = optimizer.param_groups[0]["lr"] if optimizer and optimizer.param_groups else 0.0
        except Exception:
            lr = 0.0

        # Compute grad norm if available
        grad_norm_sq = 0.0
        try:
            if model is not None:
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        grad_norm_sq += float(param_norm.item()) ** 2
            grad_norm = grad_norm_sq**0.5
        except Exception:
            grad_norm = 0.0

        try:
            from gemma_tuner.visualizer import get_visualizer

            viz = get_visualizer()
            if viz:
                viz.update_training_step(
                    loss=float(loss) if loss is not None else 0.0,
                    learning_rate=float(lr),
                    optimizer=optimizer,
                    global_step=int(state.global_step),
                )
        except Exception as e:
            logger.debug("Visualizer on_log push failed: %s", e)

    def push_training_step(self, *, loss: float, learning_rate: float, batch=None, outputs=None, optimizer=None):
        if not self._initialized:
            return
        try:
            from gemma_tuner.visualizer import get_visualizer

            viz = get_visualizer()
            if viz:
                viz.update_training_step(
                    loss=float(loss),
                    learning_rate=float(learning_rate),
                    batch=batch,
                    outputs=outputs,
                    optimizer=optimizer,
                )
        except Exception as e:
            logger.debug("Visualizer push_training_step failed: %s", e)
