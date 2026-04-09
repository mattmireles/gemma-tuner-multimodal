from __future__ import annotations

import logging

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class VisualizerTrainerCallback(TrainerCallback):
    """
    Lightweight callback that streams training metrics to the built-in visualizer.

    Usage: add to a HuggingFace Trainer when profile_config['visualize'] is True.
    After constructing ``Trainer``, call ``bind_trainer(trainer)`` so the callback can
    read the last batch/outputs from ``GemmaVizTrainer`` for attention / mel / token panels.
    """

    def __init__(self, update_every_steps: int = 10):
        # Mirrors profile logging_steps (for discoverability); emission rate follows HF on_log.
        self.update_every_steps = max(1, int(update_every_steps))
        self._initialized = False
        self._last_logged_step = -1
        self._trainer = None

    def bind_trainer(self, trainer) -> None:
        """Set the Trainer instance (call once after ``Trainer(...)`` construction)."""
        self._trainer = trainer

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
        if state.global_step is None:
            return
        # Hugging Face only invokes on_log on its logging schedule (logging_steps,
        # logging_first_step, etc.). Do NOT gate again with global_step % update_every_steps —
        # that drops logging_first_step (e.g. step 1) and any step that is not an exact
        # multiple, so the dashboard stays frozen while the terminal still prints metrics.
        if state.global_step == self._last_logged_step:
            return

        train_loss = logs.get("loss") or logs.get("train_loss")
        if train_loss is None:
            return

        self._last_logged_step = state.global_step

        optimizer = kwargs.get("optimizer")

        loss = train_loss
        try:
            lr = optimizer.param_groups[0]["lr"] if optimizer and optimizer.param_groups else 0.0
        except Exception:
            lr = 0.0

        # HF Trainer computes the gradient norm BEFORE zeroing gradients and
        # writes it to logs["grad_norm"]. We must read it here — by the time
        # on_log fires, optimizer.zero_grad() has already run, so trying to
        # re-compute the norm by walking model.parameters() downstream would
        # see every p.grad as None and return a constant 0.0 (which is what
        # was happening before this fix: "signal strength stays at 0").
        grad_norm = logs.get("grad_norm")

        try:
            from gemma_tuner.visualizer import get_visualizer

            viz = get_visualizer()
            if viz:
                batch = None
                outputs = None
                tr = self._trainer
                if tr is not None:
                    batch = getattr(tr, "_viz_last_batch", None)
                    outputs = getattr(tr, "_viz_last_outputs", None)
                viz.update_training_step(
                    loss=float(loss) if loss is not None else 0.0,
                    learning_rate=float(lr),
                    optimizer=optimizer,
                    batch=batch,
                    outputs=outputs,
                    global_step=int(state.global_step),
                    gradient_norm=float(grad_norm) if grad_norm is not None else None,
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
