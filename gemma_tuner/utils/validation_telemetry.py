"""Fail-closed, token-weighted loss telemetry for the corrected Gemma smoke.

This is opt-in. Historical profiles retain their original Trainer behavior.
No prompt, target, row identifier, owner, or image path is written to the ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import TrainerCallback


def assistant_token_stats(
    logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[float, int]:
    """Return summed causal NLL and count for unmasked assistant tokens.

    Gemma's completion-only forward may return only the final K logits. The
    preceding position is retained, so the labels align with logits[:, :-1].
    """
    if logits.ndim != 3 or labels.ndim != 2 or attention_mask.shape != labels.shape:
        raise ValueError("assistant loss requires [B,K,V] logits and [B,L] labels/mask")
    kept = int(logits.shape[1])
    if kept < 2 or kept > labels.shape[1]:
        raise ValueError("completion-only logits are incompatible with labels")
    shifted_labels = labels[:, -(kept - 1) :]
    scored = (shifted_labels != -100) & attention_mask[:, -(kept - 1) :].bool()
    count = int(scored.sum().item())
    if count <= 0:
        raise RuntimeError("validation batch has zero scored assistant tokens")
    selected_logits = logits[:, :-1, :][scored].float()
    selected_labels = shifted_labels[scored].to(selected_logits.device)
    loss_sum = float(F.cross_entropy(selected_logits, selected_labels, reduction="sum").item())
    if not math.isfinite(loss_sum):
        raise RuntimeError("non-finite assistant-token loss")
    return loss_sum, count


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split_identity(
    train_ds: Any, validation_ds: Any, *, train_rows: int, validation_rows: int
) -> dict[str, str]:
    """Reject missing, duplicate, or train/validation-overlapping identities."""
    if train_ds is None or validation_ds is None or not len(train_ds) or not len(validation_ds):
        raise ValueError("strict telemetry requires nonempty train and validation datasets")
    if train_rows < 1 or validation_rows < 1 or train_rows > len(train_ds) or validation_rows > len(validation_ds):
        raise ValueError("strict telemetry fixed subset sizes must be positive and available")
    identities = []
    content_hashes = []
    for split, dataset in (("train", train_ds), ("validation", validation_ds)):
        seen: set[str] = set()
        ids: set[str] = set()
        owners: set[str] = set()
        images: set[str] = set()
        hashes: set[str] = set()
        for row in dataset:
            for key in ("id", "owner_id", "image_path"):
                value = row.get(key)
                if value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip():
                    raise ValueError(f"strict telemetry {split} lacks {key}")
            identifier = str(row["id"])
            if identifier in seen:
                raise ValueError(f"strict telemetry {split} has duplicate IDs")
            seen.add(identifier)
            ids.add(identifier)
            owners.add(str(row["owner_id"]))
            path = Path(str(row["image_path"])).resolve()
            if not path.is_file():
                raise ValueError(f"strict telemetry {split} image is missing")
            images.add(str(path))
            hashes.add(_sha256_file(path))
        identities.append((ids, owners, images))
        content_hashes.append(hashes)
    for index, name in enumerate(("ID", "owner", "image path")):
        if identities[0][index] & identities[1][index]:
            raise ValueError(f"strict telemetry train/validation {name} overlap")
    # Different paths can point to identical screenshots. The whole loaded
    # split, not merely the fixed telemetry rows, must remain image-disjoint.
    if content_hashes[0] & content_hashes[1]:
        raise ValueError("strict telemetry train/validation image hashes overlap")
    selected_hashes = []
    for dataset, count in ((train_ds, train_rows), (validation_ds, validation_rows)):
        hashes = set()
        for index in range(count):
            path = Path(str(dataset[index]["image_path"]))
            if not path.is_file():
                raise ValueError("strict telemetry selected image is missing")
            hashes.add(_sha256_file(path))
        selected_hashes.append(hashes)
    return {
        "train_ids_sha256": hashlib.sha256("\n".join(sorted(identities[0][0])).encode()).hexdigest(),
        "validation_ids_sha256": hashlib.sha256("\n".join(sorted(identities[1][0])).encode()).hexdigest(),
        "fixed_train_images_sha256": hashlib.sha256("\n".join(sorted(selected_hashes[0])).encode()).hexdigest(),
        "fixed_validation_images_sha256": hashlib.sha256("\n".join(sorted(selected_hashes[1])).encode()).hexdigest(),
        "train_image_hashes_sha256": hashlib.sha256("\n".join(sorted(content_hashes[0])).encode()).hexdigest(),
        "validation_image_hashes_sha256": hashlib.sha256("\n".join(sorted(content_hashes[1])).encode()).hexdigest(),
    }


class ValidationTelemetry(TrainerCallback):
    """Append exact fixed-subset loss and optimizer metrics, rejecting gaps."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        train_ds: Any,
        validation_ds: Any,
        collator: Any,
        train_rows: int,
        validation_rows: int,
        interval_steps: int,
        save_steps: int,
        stop_after_step: int,
        full_validation_at_stop: bool,
        provenance: dict[str, str],
        exposure_ledger: Any,
        mode_panels: dict[str, Any] | None = None,
        start_step: int = 0,
    ) -> None:
        if min(interval_steps, save_steps, stop_after_step) < 1:
            raise ValueError("strict telemetry intervals and stop step must be positive")
        self.path = Path(output_dir) / "validation_telemetry.jsonl"
        self.train_ds = train_ds
        self.validation_ds = validation_ds
        self.collator = collator
        self.train_rows = train_rows
        self.validation_rows = validation_rows
        self.interval_steps = interval_steps
        self.save_steps = save_steps
        self.stop_after_step = stop_after_step
        self.full_validation_at_stop = full_validation_at_stop
        self.provenance = dict(provenance)
        self.exposure_ledger = exposure_ledger
        self.mode_panels = dict(mode_panels or {})
        self.start_step = int(start_step)
        if self.start_step < 0 or self.start_step >= self.stop_after_step:
            raise ValueError("strict telemetry start step must precede stop step")
        self.rows: dict[tuple[str, int], dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                key = (row["event"], int(row["step"]))
                if key in self.rows or row.get("provenance") != self.provenance:
                    raise RuntimeError("strict telemetry resume ledger has duplicate or changed provenance")
                self.rows[key] = row

    def _append(self, row: dict[str, Any]) -> None:
        row = {"schema_version": "gemma_corrected_validation_telemetry_v1", "provenance": self.provenance, **row}
        key = (row["event"], int(row["step"]))
        previous = self.rows.get(key)
        if previous is not None:
            if previous != row:
                raise RuntimeError(f"strict telemetry metric changed on resume at {key}")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
        self.rows[key] = row

    def _evaluate(self, model: Any, dataset: Any, count: int) -> tuple[float, int]:
        prior_mode = model.training
        model.eval()
        loss_sum = 0.0
        tokens = 0
        try:
            with torch.no_grad():
                for index in range(count):
                    batch = self.collator([dataset[index]])
                    device = next(model.parameters()).device
                    prepared = {
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    }
                    labels = prepared.pop("labels")
                    attention = prepared.get("attention_mask")
                    if attention is None:
                        raise RuntimeError("strict telemetry batch lacks attention mask")
                    outputs = model(**prepared)
                    subtotal, denominator = assistant_token_stats(outputs.logits, labels, attention)
                    loss_sum += subtotal
                    tokens += denominator
        finally:
            model.train(prior_mode)
        if tokens <= 0 or not math.isfinite(loss_sum):
            raise RuntimeError("strict telemetry evaluation is missing or non-finite")
        return loss_sum, tokens

    def record_eval(self, model: Any, step: int, *, full_validation: bool = False) -> None:
        exposures = len(self.exposure_ledger.rows)
        if step > self.start_step and exposures < step - self.start_step:
            raise RuntimeError("strict telemetry exposure ledger did not advance")
        for event, dataset, count in (
            ("fixed_train_eval", self.train_ds, self.train_rows),
            ("fixed_validation_eval", self.validation_ds, self.validation_rows),
        ):
            loss_sum, tokens = self._evaluate(model, dataset, count)
            self._append(
                {
                    "event": event,
                    "step": step,
                    "exposures": exposures,
                    "loss_sum": loss_sum,
                    "scored_tokens": tokens,
                    "loss": loss_sum / tokens,
                }
            )
        if full_validation:
            loss_sum, tokens = self._evaluate(model, self.validation_ds, len(self.validation_ds))
            self._append(
                {
                    "event": "full_validation_eval",
                    "step": step,
                    "exposures": exposures,
                    "examples": len(self.validation_ds),
                    "loss_sum": loss_sum,
                    "scored_tokens": tokens,
                    "loss": loss_sum / tokens,
                }
            )
        if self.mode_panels and (step == self.start_step or step % self.save_steps == 0 or step >= self.stop_after_step):
            for mode, dataset in sorted(self.mode_panels.items()):
                loss_sum, tokens = self._evaluate(model, dataset, len(dataset))
                self._append(
                    {
                        "event": f"mode_validation_eval:{mode}",
                        "step": step,
                        "exposures": exposures,
                        "examples": len(dataset),
                        "loss_sum": loss_sum,
                        "scored_tokens": tokens,
                        "loss": loss_sum / tokens,
                    }
                )

    def _require_evals(self, step: int) -> None:
        required = {self.start_step, step}
        required.update(range(self.start_step + self.interval_steps, step + 1, self.interval_steps))
        required.update(range(self.start_step + self.save_steps, step + 1, self.save_steps))
        for expected in required:
            for event in ("fixed_train_eval", "fixed_validation_eval"):
                if (event, expected) not in self.rows:
                    raise RuntimeError(f"strict telemetry missing {event} at step {expected}")
        if self.mode_panels:
            mode_steps = {self.start_step, step}
            mode_steps.update(range(self.start_step + self.save_steps, step + 1, self.save_steps))
            for expected in mode_steps:
                for mode in self.mode_panels:
                    event = f"mode_validation_eval:{mode}"
                    if (event, expected) not in self.rows:
                        raise RuntimeError(f"strict telemetry missing {event} at step {expected}")
        if (
            self.full_validation_at_stop
            and step >= self.stop_after_step
            and ("full_validation_eval", self.stop_after_step) not in self.rows
        ):
            raise RuntimeError("strict telemetry missing full validation at stop step")

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001, ARG002
        if int(state.global_step) == self.start_step and (
            "fixed_validation_eval", self.start_step
        ) not in self.rows:
            model = kwargs.get("model")
            if model is None:
                raise RuntimeError("strict telemetry callback did not receive model at resumed start")
            self.record_eval(model, self.start_step)
        self._require_evals(int(state.global_step))
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001, ARG002
        step = int(state.global_step)
        if step % self.interval_steps == 0 or step % self.save_steps == 0 or step >= self.stop_after_step:
            model = kwargs.get("model")
            if model is None:
                raise RuntimeError("strict telemetry callback did not receive model")
            self.record_eval(model, step, full_validation=self.full_validation_at_stop and step == self.stop_after_step)
        # This ledger owns token-weighted validation. HF Trainer's default
        # eval_loss averages batch means and would duplicate a costly full pass.
        control.should_evaluate = False
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001, ARG002
        logs = logs or {}
        if "loss" not in logs:
            return control
        step = int(state.global_step)
        exposures = len(self.exposure_ledger.rows)
        if exposures < step - self.start_step:
            raise RuntimeError("strict telemetry optimization step lacks committed exposures")
        for required in ("learning_rate", "grad_norm"):
            if required not in logs:
                raise RuntimeError(f"strict telemetry missing {required} at step {step}")
        row = {"event": "optimization", "step": step, "exposures": exposures, "stochastic_loss": float(logs["loss"])}
        for key in ("learning_rate", "grad_norm", "epoch"):
            if key in logs:
                row[key] = float(logs[key])
        if not all(math.isfinite(value) for key, value in row.items() if isinstance(value, float)):
            raise RuntimeError("strict telemetry non-finite optimization metric")
        self._append(row)
        return control

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001, ARG002
        step = int(state.global_step)
        self._require_evals(step)
        if ("optimization", step) not in self.rows:
            raise RuntimeError(f"strict telemetry missing optimization loss at checkpoint step {step}")
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001, ARG002
        step = int(state.global_step)
        self._require_evals(step)
        for expected in range(self.start_step + 1, step + 1):
            if ("optimization", expected) not in self.rows:
                raise RuntimeError(f"strict telemetry missing optimization loss at step {expected}")
        # The immutable checkpoint marker already binds every checkpoint byte.
        # Link that tree digest into the telemetry ledger after Trainer saves.
        from gemma_tuner.utils.checkpoints import verify_complete_checkpoint

        for marker in sorted(self.path.parent.glob("checkpoint-*/.complete.json")):
            manifest = verify_complete_checkpoint(marker.parent)
            checkpoint_step = int(manifest["global_step"])
            if checkpoint_step <= step:
                eval_row = self.rows.get(("fixed_validation_eval", checkpoint_step))
                if eval_row is None:
                    raise RuntimeError("strict telemetry checkpoint lacks validation metric")
                self._append(
                    {
                        "event": "checkpoint",
                        "step": checkpoint_step,
                        "tree_sha256": manifest["tree_sha256"],
                        "exposures": eval_row["exposures"],
                    }
                )
        return control
