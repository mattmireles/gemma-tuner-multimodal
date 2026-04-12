"""Gemma 4 training path smoke: same shape as test_smoke_gemma3n, forces family GEMMA_4 for loader/patch."""

from __future__ import annotations

import configparser
import json
import math
import shutil
from importlib import metadata
from pathlib import Path

import pytest
from packaging.version import Version

from gemma_tuner.models.gemma.family import GemmaFamily

TINY_GEMMA = "fxmarty/tiny-random-GemmaForCausalLM"


@pytest.mark.integration
@pytest.mark.skipif(
    Version(metadata.version("transformers")) < Version("5.5.0"),
    reason="Gemma 4 training path requires transformers>=5.5 (requirements/requirements-gemma4.txt)",
)
def test_text_completion_smoke_two_steps_gemma4_family(tmp_path, monkeypatch):
    """Forces ``detect_family`` → GEMMA_4 so ``apply_clippable_linear_patch`` runs before load."""
    import gemma_tuner.utils.dataset_utils as du
    from gemma_tuner.core.config import load_profile_config
    from gemma_tuner.models.gemma import finetune as gemma_finetune

    monkeypatch.setattr(gemma_finetune, "detect_family", lambda _mid: GemmaFamily.GEMMA_4)

    repo_root = Path(__file__).resolve().parent.parent
    csv_src = repo_root / "tests" / "data" / "text_completion_tiny.csv"
    data_dir = tmp_path / "data" / "datasets" / "text-smoke-g4"
    data_dir.mkdir(parents=True)
    shutil.copy(csv_src, data_dir / "train.csv")
    shutil.copy(csv_src, data_dir / "validation.csv")

    ini = f"""[DEFAULT]
num_train_epochs = 1
logging_steps = 1
save_steps = 100
save_total_limit = 1
gradient_accumulation_steps = 1
learning_rate = 1e-4
warmup_steps = 0
output_dir = output

[dataset_defaults]
text_column = text
max_label_length = 128
max_duration = 30.0
id_column = id
streaming_enabled = false
preprocessing_num_workers = 0
dataloader_num_workers = 0

[group:gemma]
dtype = float32
attn_implementation = eager
optim = adamw_torch

[model:gemma-4-e2b-it]
base_model = {TINY_GEMMA}
group = gemma
per_device_train_batch_size = 1
per_device_eval_batch_size = 1

[dataset:text-smoke-g4]
source = text-smoke-g4
train_split = train
validation_split = validation
text_column = text
max_label_length = 128
max_duration = 30.0

[profile:text-smoke-g4]
model = gemma-4-e2b-it
dataset = text-smoke-g4
train_split = train
validation_split = validation
text_column = text
modality = text
text_sub_mode = completion
max_seq_length = 64
max_steps = 2
num_train_epochs = 1
logging_steps = 1
save_steps = 100
save_total_limit = 1
gradient_accumulation_steps = 1
learning_rate = 1e-4
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
lora_r = 4
lora_alpha = 8
lora_dropout = 0.05
eval_strategy = no
load_validation = false
"""
    (tmp_path / "config.ini").write_text(ini, encoding="utf-8")

    cwd = Path.cwd()
    du._config = None
    try:
        monkeypatch.chdir(tmp_path)
        du._config = None
        cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        cfg.read("config.ini")
        profile = load_profile_config(cfg, "text-smoke-g4")
        out = tmp_path / "out_smoke_gemma4"
        out.mkdir(parents=True)
        gemma_finetune.main(profile, str(out))
    finally:
        monkeypatch.chdir(cwd)
        du._config = None

    results_path = out / "train_results.json"
    assert results_path.is_file(), f"missing {results_path}"
    metrics = json.loads(results_path.read_text(encoding="utf-8"))
    loss = metrics.get("train_loss")
    assert loss is not None
    assert math.isfinite(float(loss))
