"""Regression tests for Plan 30's mandatory train/validation telemetry."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import TrainingArguments

from gemma_tuner.models.gemma.finetune import (
    CompletionOnlyTrainer,
    _validate_strict_telemetry_config,
    completion_only_causal_loss,
)
from gemma_tuner.utils.exposure_ledger import ExposureCommitCallback, ExposureLedger, ExposureTrackingCollator
from gemma_tuner.utils.validation_telemetry import (
    ValidationTelemetry,
    assistant_token_stats,
    validate_split_identity,
)


def test_token_weighted_partial_batch_aggregation_ignores_prompt_and_padding() -> None:
    torch.manual_seed(3)
    logits = torch.randn(2, 5, 7)
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, -100, -100, 4, -100]])
    attention = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    combined_sum, combined_count = assistant_token_stats(logits, labels, attention)
    a_sum, a_count = assistant_token_stats(logits[:1], labels[:1], attention[:1])
    b_sum, b_count = assistant_token_stats(logits[1:], labels[1:], attention[1:])
    assert combined_count == a_count + b_count == 4
    assert combined_sum == pytest.approx(a_sum + b_sum)
    assert combined_sum / combined_count != pytest.approx((a_sum / a_count + b_sum / b_count) / 2)
    assert combined_sum / combined_count == pytest.approx(
        float(completion_only_causal_loss(logits, labels, attention).item())
    )


def test_strict_profile_rejects_disabled_validation_before_model_load(monkeypatch) -> None:
    import transformers

    config = {
        "require_validation_telemetry": "true",
        "required_transformers_version": transformers.__version__,
        "modality": "image",
        "load_validation": "false",
        "completion_only_logits": "true",
        "record_exposures": "true",
        "eval_strategy": "steps",
        "logging_steps": "1",
        "telemetry_interval_steps": "1",
        "telemetry_train_rows": "1",
        "telemetry_validation_rows": "1",
        "stop_after_step": "1",
    }
    with pytest.raises(ValueError, match="load_validation=true"):
        _validate_strict_telemetry_config(config)
    config["load_validation"] = "true"
    assert _validate_strict_telemetry_config(config)
    config["required_transformers_version"] = "0.0.0"
    with pytest.raises(RuntimeError, match="Transformers 0.0.0"):
        _validate_strict_telemetry_config(config)


def _dataset(tmp_path):
    images = []
    for index in range(3):
        path = tmp_path / f"image-{index}.png"
        path.write_bytes(f"image-{index}".encode())
        images.append(path)
    train = [{"id": "t0", "owner_id": "owner-t", "image_path": str(images[0])}]
    validation = [{"id": "v0", "owner_id": "owner-v", "image_path": str(images[1])}]
    return train, validation


def test_split_identity_rejects_id_owner_and_image_overlap(tmp_path) -> None:
    train, validation = _dataset(tmp_path)
    provenance = validate_split_identity(train, validation, train_rows=1, validation_rows=1)
    assert len(provenance["train_ids_sha256"]) == 64
    for key, value, expected in (
        ("id", "t0", "ID overlap"),
        ("owner_id", "owner-t", "owner overlap"),
        ("image_path", train[0]["image_path"], "image path overlap"),
    ):
        changed = [dict(validation[0], **{key: value})]
        with pytest.raises(ValueError, match=expected):
            validate_split_identity(train, changed, train_rows=1, validation_rows=1)
    copied = tmp_path / "same-image-different-path.png"
    copied.write_bytes((tmp_path / "image-0.png").read_bytes())
    changed = [dict(validation[0], image_path=str(copied))]
    with pytest.raises(ValueError, match="image hashes overlap"):
        validate_split_identity(train, changed, train_rows=1, validation_rows=1)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.randn(7, 7))

    def forward(self, input_ids, attention_mask, logits_to_keep):
        return SimpleNamespace(logits=self.table[input_ids][:, -logits_to_keep:, :])


def _collate(rows):
    token = 2 if rows[0]["id"].startswith("t") else 3
    return {
        "input_ids": torch.tensor([[0, 1, token, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, token, 4]]),
        "logits_to_keep": 3,
    }


def _telemetry(tmp_path, *, stop=2):
    train, validation = _dataset(tmp_path)
    return ValidationTelemetry(
        tmp_path,
        train_ds=train,
        validation_ds=validation,
        collator=_collate,
        train_rows=1,
        validation_rows=1,
        interval_steps=1,
        save_steps=2,
        stop_after_step=stop,
        full_validation_at_stop=True,
        provenance={"config_sha256": "a" * 64},
        exposure_ledger=SimpleNamespace(rows=[]),
    )


def test_step_zero_and_interval_emit_finite_token_weighted_losses(tmp_path) -> None:
    telemetry = _telemetry(tmp_path)
    model = ToyModel()
    telemetry.record_eval(model, 0)
    state = SimpleNamespace(global_step=0)
    control = SimpleNamespace(should_evaluate=True)
    telemetry.on_train_begin(None, state, control)
    for step in (1, 2):
        state.global_step = step
        telemetry.exposure_ledger.rows.append({"exposure": step})
        telemetry.on_step_end(None, state, control, model=model)
        assert not control.should_evaluate
        telemetry.on_log(None, state, control, logs={"loss": 1.2, "learning_rate": 1e-4, "grad_norm": 0.3})
    telemetry.on_save(None, state, control)
    telemetry.on_train_end(None, state, control)
    rows = [json.loads(line) for line in telemetry.path.read_text().splitlines()]
    assert len(rows) == 9  # 3 fixed-train + 3 fixed-val + 2 optimization + full-val
    for row in rows:
        assert "input_ids" not in row and "owner_id" not in row and "image_path" not in row
        if row["event"].endswith("_eval"):
            assert row["scored_tokens"] == 2
            assert row["loss"] == pytest.approx(row["loss_sum"] / row["scored_tokens"])


def test_resume_requires_exact_step_metrics_and_unchanged_provenance(tmp_path) -> None:
    telemetry = _telemetry(tmp_path)
    telemetry.record_eval(ToyModel(), 0)
    resumed = _telemetry(tmp_path)
    with pytest.raises(RuntimeError, match="missing fixed_train_eval at step 1"):
        resumed.on_train_begin(None, SimpleNamespace(global_step=1), SimpleNamespace())
    with pytest.raises(RuntimeError, match="changed provenance"):
        ValidationTelemetry(
            tmp_path,
            train_ds=[],
            validation_ds=[],
            collator=_collate,
            train_rows=1,
            validation_rows=1,
            interval_steps=1,
            save_steps=2,
            stop_after_step=2,
            full_validation_at_stop=True,
            provenance={"config_sha256": "b" * 64},
            exposure_ledger=SimpleNamespace(rows=[]),
        )


def test_checkpoint_rejects_missing_optimization_metric(tmp_path) -> None:
    telemetry = _telemetry(tmp_path, stop=1)
    model = ToyModel()
    telemetry.record_eval(model, 0)
    state = SimpleNamespace(global_step=1)
    control = SimpleNamespace(should_evaluate=True)
    telemetry.exposure_ledger.rows.append({"exposure": 1})
    telemetry.on_step_end(None, state, control, model=model)
    with pytest.raises(RuntimeError, match="missing optimization loss"):
        telemetry.on_save(None, state, control)


def test_actual_trainer_emits_step_zero_train_validation_and_per_step_metrics(tmp_path) -> None:
    train, validation = _dataset(tmp_path)
    second = tmp_path / "image-2.png"
    train.append({"id": "t1", "owner_id": "owner-t", "image_path": str(second)})
    exposures = ExposureLedger(tmp_path / "exposures.jsonl")
    telemetry = ValidationTelemetry(
        tmp_path,
        train_ds=train,
        validation_ds=validation,
        collator=_collate,
        train_rows=1,
        validation_rows=1,
        interval_steps=1,
        save_steps=2,
        stop_after_step=2,
        full_validation_at_stop=True,
        provenance={"config_sha256": "c" * 64},
        exposure_ledger=exposures,
    )
    args = TrainingArguments(
        output_dir=str(tmp_path / "trainer"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=1e-3,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=2,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )
    model = ToyModel()
    trainer = CompletionOnlyTrainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=ExposureTrackingCollator(_collate, exposures),
        callbacks=[ExposureCommitCallback(exposures), telemetry],
    )
    telemetry.record_eval(model, 0)
    trainer.train()
    assert len(exposures.rows) == 2
    rows = {(row["event"], row["step"]): row for row in map(json.loads, telemetry.path.read_text().splitlines())}
    for step in (0, 1, 2):
        assert rows[("fixed_train_eval", step)]["scored_tokens"] == 2
        assert rows[("fixed_validation_eval", step)]["scored_tokens"] == 2
    assert rows[("optimization", 1)]["exposures"] == 1
    assert rows[("optimization", 2)]["exposures"] == 2
    assert rows[("full_validation_eval", 2)]["scored_tokens"] == 2


def test_plan31_mode_panels_emit_token_denominators_and_fail_on_missing_mode(tmp_path) -> None:
    train, validation = _dataset(tmp_path)
    panels = {mode: [dict(validation[0], input_mode=mode)] for mode in
              ("full", "no_ocr", "no_quadrants", "no_system", "image_instruction", "image_only")}
    telemetry = ValidationTelemetry(
        tmp_path, train_ds=train, validation_ds=validation, collator=_collate,
        train_rows=1, validation_rows=1, interval_steps=1, save_steps=2,
        stop_after_step=2, full_validation_at_stop=True,
        provenance={"config_sha256": "d" * 64},
        exposure_ledger=SimpleNamespace(rows=[]), mode_panels=panels,
    )
    model = ToyModel()
    telemetry.record_eval(model, 0)
    mode_rows = [row for row in map(json.loads, telemetry.path.read_text().splitlines())
                 if row["event"].startswith("mode_validation_eval:")]
    assert len(mode_rows) == 6
    assert all(row["examples"] == 1 and row["scored_tokens"] == 2
               and row["loss"] == pytest.approx(row["loss_sum"] / 2) for row in mode_rows)
    telemetry.rows.pop(("mode_validation_eval:image_only", 0))
    with pytest.raises(RuntimeError, match="missing mode_validation_eval:image_only at step 0"):
        telemetry._require_evals(0)


def test_plan31_resumed_start_records_mode_baseline_after_checkpoint_load(tmp_path) -> None:
    train, validation = _dataset(tmp_path)
    telemetry = ValidationTelemetry(
        tmp_path, train_ds=train, validation_ds=validation, collator=_collate,
        train_rows=1, validation_rows=1, interval_steps=22, save_steps=88,
        stop_after_step=440, start_step=352, full_validation_at_stop=True,
        provenance={"config_sha256": "e" * 64},
        exposure_ledger=SimpleNamespace(rows=[]),
        mode_panels={"image_only": [dict(validation[0], input_mode="image_only")]},
    )
    model = ToyModel()
    telemetry.on_train_begin(None, SimpleNamespace(global_step=352), SimpleNamespace(), model=model)
    assert ("mode_validation_eval:image_only", 352) in telemetry.rows
    assert telemetry.rows[("mode_validation_eval:image_only", 352)]["exposures"] == 0
    with pytest.raises(RuntimeError, match="missing fixed_train_eval at step 374"):
        telemetry._require_evals(374)
