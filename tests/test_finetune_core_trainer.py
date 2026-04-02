import numpy as np
from types import SimpleNamespace


def test_build_trainer_smoke(monkeypatch):
    # Lazy import
    from whisper_tuner.models.whisper.finetune_core.trainer import build_trainer

    # Minimal stubs
    class _Tok:
        pad_token_id = 0
        def pad(self, items, return_tensors="pt"):
            import torch
            max_len = max(len(x["input_ids"]) for x in items)
            ids = []
            mask = []
            for x in items:
                seq = x["input_ids"]
                pad = [self.pad_token_id] * (max_len - len(seq))
                ids.append(seq + pad)
                mask.append([1]*len(seq) + [0]*len(pad))
            return SimpleNamespace(input_ids=np.array(ids), attention_mask=np.array(mask))

    class _FE:
        model_input_names = ["input_features"]
        def pad(self, items, return_tensors="pt"):
            import numpy as np
            max_len = max(len(x["input_features"]) for x in items)
            feats = []
            for x in items:
                f = x["input_features"]
                f = f + [[0.0, 0.0]] * (max_len - len(f))
                feats.append(f)
            return {"input_features": np.array(feats)}

    class _Proc:
        tokenizer = _Tok()
        feature_extractor = _FE()
        def batch_decode(self, ids, skip_special_tokens=True):
            return ["hello"] * len(ids)

    class _Model:
        def __init__(self):
            self.config = SimpleNamespace()

    # Tiny dataset
    vectorized = {
        "train": [{"input_features": [[0.0, 0.0]], "labels": [1, 2, 3]}]
    }

    # Minimal TrainingArguments-like namespace
    training_args = SimpleNamespace(
        output_dir="/tmp",
        per_device_train_batch_size=1,
        batch_eval_metrics=False,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        seed=42,
        full_determinism=False,
        accelerator_config=SimpleNamespace(
            gradient_accumulation_kwargs=None,
            to_dict=lambda: {
                "non_blocking": False,
                "split_batches": False,
                "dispatch_batches": None,
                "even_batches": False,
                "use_seedable_sampler": False,
                "gradient_accumulation_kwargs": None,
            },
        ),
        gradient_accumulation_steps=1,
        deepspeed_plugin=None,
        save_only_model=False,
        skip_memory_metrics=True,
        get_process_log_level=lambda: 30,  # logging.WARNING
    )

    # Provide a dummy trainer to avoid heavy Transformers-side requirements
    class DummyTrainer:
        def __init__(
            self,
            args,
            model,
            train_dataset,
            eval_dataset=None,
            processing_class=None,
            data_collator=None,
            compute_metrics=None,
            callbacks=None,
            **kwargs,
        ):
            self.args = args
            self.model = model
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset

    trainer = build_trainer(
        training_args=training_args,
        model=_Model(),
        processor=_Proc(),
        tokenizer=_Tok(),
        feature_extractor=_FE(),
        vectorized_datasets=vectorized,
        callbacks=[],
        trainer_class=DummyTrainer,
        trainer_kwargs=None,
    )

    # Must have attributes Trainer typically has
    assert hasattr(trainer, "args")
    assert trainer.train_dataset is not None
