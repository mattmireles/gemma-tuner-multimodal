"""Check Transformers' actual sequential resume batch boundary."""

from __future__ import annotations

import torch
from torch import nn
from transformers import Trainer, TrainingArguments

from gemma_tuner.utils.exposure_ledger import ExposureCommitCallback, ExposureTrackingCollator
from gemma_tuner.utils.plan31_exposure_ledger import Plan31ExposureLedger
from tests.test_plan31_exposure_ledger import fixture as plan31_fixture


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 8)
        self.head = nn.Linear(8, 8)

    def forward(self, input_ids, labels):
        logits = self.head(self.embedding(input_ids))
        return {"loss": nn.functional.cross_entropy(logits.view(-1, 8), labels.view(-1)),
                "logits": logits}


class TracingCollator:
    def __init__(self):
        self.seen = []

    def __call__(self, rows):
        self.seen.extend(row["id"] for row in rows)
        ids = torch.tensor([[row["token"] for row in rows]]).T
        return {"input_ids": ids, "labels": ids.clone()}


def test_sequential_resume_collator_skips_are_visible(tmp_path):
    features, schedule_path = plan31_fixture(tmp_path)
    data = [{**feature, "token": index + 1} for index, feature in enumerate(features)]
    first_collator = TracingCollator()
    first = Trainer(
        model=TinyModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path / "first"), per_device_train_batch_size=1,
            gradient_accumulation_steps=1, max_steps=4,
            save_strategy="steps", save_steps=4, logging_steps=1,
            train_sampling_strategy="sequential", report_to=[], remove_unused_columns=False,
        ),
        train_dataset=data, data_collator=first_collator,
    )
    first.train()
    second_collator = TracingCollator()
    ledger = Plan31ExposureLedger(
        tmp_path / "resumed-exposures.jsonl", schedule_path=schedule_path,
        train_rows=data, start_ordinal=2, end_ordinal=3,
    )
    second = Trainer(
        model=TinyModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path / "second"), per_device_train_batch_size=1,
            gradient_accumulation_steps=1, max_steps=6,
            save_strategy="no", logging_steps=1,
            train_sampling_strategy="sequential", report_to=[], remove_unused_columns=False,
        ),
        train_dataset=data, data_collator=ExposureTrackingCollator(second_collator, ledger),
        callbacks=[ExposureCommitCallback(ledger)],
    )
    second.train(resume_from_checkpoint=str(tmp_path / "first" / "checkpoint-4"))
    assert second.state.global_step == 6
    # The skipped first row is not collated on this pinned Trainer path; the exposure
    # recorder receives exactly the next two source rows.
    assert second_collator.seen == ["row-2", "row-3"]
    assert ledger.verify_complete()["exposures"] == 2
