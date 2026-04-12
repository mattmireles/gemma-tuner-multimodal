"""Gated integration: real Gemma 3n multimodal checkpoint, image caption + VQA, ≥10 train steps.

Verifies end-to-end: ``AutoProcessor`` vision path, ``apply_chat_template`` with image content
blocks, ``apply_image_token_budget_to_processor``, ``inject_mm_token_type_ids``, LoRA, and
finite loss — i.e. the train/serve ``image_token_budget`` contract on real tensors.

**Requires Hugging Face access** to ``google/gemma-3n-E4B-it`` (gated model):

- Set ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN``, or run ``huggingface-cli login`` first.
- Without credentials, tests skip (they do not fail CI).

Run manually::

    HF_TOKEN=... pytest tests/test_smoke_image_multimodal.py -m integration -v

Optional in CI: add ``HF_TOKEN`` as a secret and run the integration job on ``main`` / release.
"""

from __future__ import annotations

import configparser
import json
import math
import os
import shutil
from pathlib import Path

import pytest

GEMMA_3N_MULTIMODAL = "google/gemma-3n-E4B-it"


def _hf_auth_available() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return get_token() is not None
    except Exception:
        return False


requires_hf_gemma = pytest.mark.skipif(
    not _hf_auth_available(),
    reason="Gated Gemma 3n multimodal: set HF_TOKEN or run huggingface-cli login",
)


@pytest.mark.integration
@requires_hf_gemma
def test_image_caption_multimodal_smoke_ten_steps(tmp_path, monkeypatch):
    """Caption mode: 10 optimizer steps, finite train_loss on real vision tower."""
    import gemma_tuner.utils.dataset_utils as du
    from gemma_tuner.core.config import load_profile_config
    from gemma_tuner.models.gemma import finetune as gemma_finetune

    repo_root = Path(__file__).resolve().parent.parent
    fixture_src = repo_root / "tests" / "data" / "image_caption_tiny"
    data_dir = tmp_path / "data" / "datasets" / "img-mm-caption"
    shutil.copytree(fixture_src, data_dir)

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
text_column = caption
max_label_length = 256
max_duration = 30.0
id_column = id
streaming_enabled = false
preprocessing_num_workers = 0
dataloader_num_workers = 0

[group:gemma]
dtype = float32
attn_implementation = eager
optim = adamw_torch

[model:gemma-3n-e2b-it]
base_model = {GEMMA_3N_MULTIMODAL}
group = gemma
per_device_train_batch_size = 1
per_device_eval_batch_size = 1

[dataset:img-mm-caption]
source = img-mm-caption
train_split = train
validation_split = validation
text_column = caption
max_label_length = 256
max_duration = 30.0

[profile:img-mm-caption]
model = gemma-3n-e2b-it
dataset = img-mm-caption
train_split = train
validation_split = validation
text_column = caption
modality = image
image_sub_mode = caption
image_path_column = image_path
image_token_budget = 70
max_steps = 10
num_train_epochs = 1
logging_steps = 1
save_steps = 100
save_total_limit = 1
gradient_accumulation_steps = 1
learning_rate = 1e-4
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
eval_strategy = no
load_validation = false
"""
    (tmp_path / "config.ini").write_text(ini, encoding="utf-8")

    cwd = Path.cwd()
    du._config = None
    out = tmp_path / "out_smoke_img_mm_caption"
    try:
        monkeypatch.chdir(tmp_path)
        du._config = None
        cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        cfg.read("config.ini")
        profile = load_profile_config(cfg, "img-mm-caption")
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


@pytest.mark.integration
@requires_hf_gemma
def test_image_vqa_multimodal_smoke_ten_steps(tmp_path, monkeypatch):
    """VQA mode: question + answer columns, 10 steps, finite loss."""
    import csv

    import gemma_tuner.utils.dataset_utils as du
    from gemma_tuner.core.config import load_profile_config
    from gemma_tuner.models.gemma import finetune as gemma_finetune

    repo_root = Path(__file__).resolve().parent.parent
    fixture_src = repo_root / "tests" / "data" / "image_caption_tiny"
    data_dir = tmp_path / "data" / "datasets" / "img-mm-vqa"
    shutil.copytree(fixture_src, data_dir)

    # VQA schema: question + answer (reuse same images as caption fixture)
    train_csv = data_dir / "train.csv"
    rows = []
    with open(train_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "image_path", "question", "answer"])
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "id": row["id"],
                    "image_path": row["image_path"],
                    "question": "Describe this image in one short phrase.",
                    "answer": row["caption"],
                }
            )

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
text_column = answer
max_label_length = 256
max_duration = 30.0
id_column = id
streaming_enabled = false
preprocessing_num_workers = 0
dataloader_num_workers = 0

[group:gemma]
dtype = float32
attn_implementation = eager
optim = adamw_torch

[model:gemma-3n-e2b-it]
base_model = {GEMMA_3N_MULTIMODAL}
group = gemma
per_device_train_batch_size = 1
per_device_eval_batch_size = 1

[dataset:img-mm-vqa]
source = img-mm-vqa
train_split = train
validation_split = validation
text_column = answer
max_label_length = 256
max_duration = 30.0

[profile:img-mm-vqa]
model = gemma-3n-e2b-it
dataset = img-mm-vqa
train_split = train
validation_split = validation
text_column = answer
prompt_column = question
modality = image
image_sub_mode = vqa
image_path_column = image_path
image_token_budget = 70
max_steps = 10
num_train_epochs = 1
logging_steps = 1
save_steps = 100
save_total_limit = 1
gradient_accumulation_steps = 1
learning_rate = 1e-4
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
eval_strategy = no
load_validation = false
"""
    (tmp_path / "config.ini").write_text(ini, encoding="utf-8")

    cwd = Path.cwd()
    du._config = None
    out = tmp_path / "out_smoke_img_mm_vqa"
    try:
        monkeypatch.chdir(tmp_path)
        du._config = None
        cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        cfg.read("config.ini")
        profile = load_profile_config(cfg, "img-mm-vqa")
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
