from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import PretrainedConfig, PreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

from gemma_tuner.utils.checkpoints import (
    ImmutableCheckpointCallback,
    StopAfterStepCallback,
    seal_checkpoint,
    verify_complete_checkpoint,
)


def write_checkpoint(root: Path, step: int = 2) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text('{"peft_type":"LORA"}')
    save_file({"adapter": torch.ones(2)}, checkpoint / "adapter_model.safetensors")
    for name in ("optimizer.pt", "rng_state.pth", "scheduler.pt", "training_args.bin"):
        torch.save({"step": step}, checkpoint / name)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return checkpoint


def test_checkpoint_seal_is_atomic_and_content_bound(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path)
    manifest = seal_checkpoint(checkpoint)
    assert verify_complete_checkpoint(checkpoint) == manifest
    assert manifest["global_step"] == 2
    assert not (checkpoint / ".complete.json.tmp").exists()
    with (checkpoint / "optimizer.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="differ"):
        verify_complete_checkpoint(checkpoint)


def test_checkpoint_rejects_truncated_torch_state(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path)
    (checkpoint / "optimizer.pt").write_bytes(b"not a torch zip")
    with pytest.raises(ValueError, match="optimizer.pt"):
        seal_checkpoint(checkpoint)


def test_checkpoint_rejects_wrong_step(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path, step=3)
    (checkpoint / "trainer_state.json").write_text('{"global_step":2}')
    with pytest.raises(ValueError, match="global_step"):
        seal_checkpoint(checkpoint)


def test_callback_seals_saved_checkpoint(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path, step=4)
    args = type("Args", (), {"output_dir": str(tmp_path)})()
    state = type("State", (), {"global_step": 4})()
    control = object()
    assert ImmutableCheckpointCallback().on_save(args, state, control) is control
    assert verify_complete_checkpoint(checkpoint)["global_step"] == 4


def test_planned_stop_requests_save_and_training_stop_at_boundary() -> None:
    callback = StopAfterStepCallback(3)
    state = type("State", (), {"global_step": 2})()
    control = type("Control", (), {"should_save": False, "should_training_stop": False})()
    callback.on_step_end(None, state, control)
    assert control.should_save is False
    assert control.should_training_stop is False
    state.global_step = 3
    callback.on_step_end(None, state, control)
    assert control.should_save is True
    assert control.should_training_stop is True


class _TinyConfig(PretrainedConfig):
    model_type = "tiny-resume-test"

    def __init__(self, vocab_size: int = 16, hidden_size: int = 8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class _TinyCausalLM(PreTrainedModel):
    config_class = _TinyConfig

    def __init__(self, config: _TinyConfig) -> None:
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, labels=None, **kwargs):
        del kwargs
        logits = self.lm_head(torch.tanh(self.q_proj(self.embed(input_ids))))
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, self.config.vocab_size),
            labels[:, 1:].reshape(-1),
        )
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}


class _TinyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = torch.tensor([1, 2 + index % 5, 3, 4], dtype=torch.long)
        return {"input_ids": tokens, "labels": tokens.clone()}


def _tiny_lora_model() -> nn.Module:
    torch.manual_seed(123)
    base = _TinyCausalLM(_TinyConfig())
    return get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj"],
            task_type="CAUSAL_LM",
        ),
    )


def _training_args(output: Path, max_steps: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        learning_rate=1e-3,
        lr_scheduler_type="constant",
        save_strategy="steps",
        save_steps=2,
        save_total_limit=3,
        logging_steps=1,
        report_to=[],
        seed=77,
        data_seed=77,
        full_determinism=True,
    )


def _adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters() if "lora_" in name}


@pytest.mark.integration
def test_interrupted_resume_matches_uninterrupted_weights_and_row_order(
    tmp_path: Path,
) -> None:
    dataset = _TinyDataset()
    uninterrupted_model = _tiny_lora_model()
    uninterrupted = Trainer(
        model=uninterrupted_model,
        args=_training_args(tmp_path / "uninterrupted", 4),
        train_dataset=dataset,
        callbacks=[ImmutableCheckpointCallback()],
    )
    uninterrupted.train()

    resumed_root = tmp_path / "resumed"
    first_model = _tiny_lora_model()
    first = Trainer(
        model=first_model,
        args=_training_args(resumed_root, 2),
        train_dataset=dataset,
        callbacks=[ImmutableCheckpointCallback()],
    )
    first.train()
    checkpoint = resumed_root / "checkpoint-2"
    verify_complete_checkpoint(checkpoint)

    resumed_model = _tiny_lora_model()
    resumed = Trainer(
        model=resumed_model,
        args=_training_args(resumed_root, 4),
        train_dataset=dataset,
        callbacks=[ImmutableCheckpointCallback()],
    )
    resumed.train(resume_from_checkpoint=str(checkpoint))
    verify_complete_checkpoint(resumed_root / "checkpoint-4")

    expected = _adapter_state(uninterrupted_model)
    actual = _adapter_state(resumed_model)
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    uninterrupted_losses = [item["loss"] for item in uninterrupted.state.log_history if "loss" in item]
    resumed_losses = [item["loss"] for item in resumed.state.log_history if "loss" in item]
    assert resumed_losses == uninterrupted_losses
