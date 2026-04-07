"""Offline smoke: image modality dataset load + collator batch (no full finetune — HF tiny stubs lack vision processors)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from tests.test_gemma_image_collator import _FakeImageProcessor


def test_image_modality_dataset_and_collator_batch(tmp_path):
    """Validates CSV schema, path resolution pattern, and DataCollatorGemmaImage batching."""
    import gemma_tuner.utils.dataset_utils as du
    from gemma_tuner.models.common.collators import DataCollatorGemmaImage, apply_image_token_budget_to_processor
    from gemma_tuner.models.gemma.family import GemmaFamily
    from gemma_tuner.utils.dataset_utils import load_dataset_split

    repo_root = Path(__file__).resolve().parent.parent
    fixture_src = repo_root / "tests" / "data" / "image_caption_tiny"
    data_dir = tmp_path / "data" / "datasets" / "image-smoke"
    shutil.copytree(fixture_src, data_dir)

    ini = tmp_path / "config.ini"
    ini.write_text(
        """
[dataset:image-smoke]
source = image-smoke-src
text_column = caption
train_split = train
validation_split = validation
max_label_length = 64
max_duration = 30.0
""",
        encoding="utf-8",
    )

    cwd = Path.cwd()
    du._config = None
    try:
        os.chdir(tmp_path)
        du._config = None
        ds, _src = load_dataset_split(
            split="train",
            dataset_config={
                "name": "image-smoke",
                "text_column": "caption",
                "modality": "image",
                "image_sub_mode": "caption",
                "image_path_column": "image_path",
            },
            streaming_enabled=False,
        )
    finally:
        os.chdir(cwd)
        du._config = None

    assert len(ds) >= 1
    row = ds[0]
    dataset_dir = str(data_dir)
    path_val = row["image_path"]
    if path_val and not Path(path_val).is_absolute():
        path_val = str(Path(dataset_dir) / path_val)

    proc = _FakeImageProcessor()
    apply_image_token_budget_to_processor(proc, 70)
    collator = DataCollatorGemmaImage(
        processor=proc,
        text_column="caption",
        family=GemmaFamily.GEMMA_3N,
        image_path_column="image_path",
        image_token_budget=70,
        sub_mode="caption",
    )
    batch = collator(
        [
            {
                "id": row["id"],
                "image_path": path_val,
                "caption": row["caption"],
            }
        ]
    )
    assert "train_loss" not in batch
    assert "input_ids" in batch and "labels" in batch
