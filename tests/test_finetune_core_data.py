import types
import numpy as np
from datasets import Dataset, DatasetDict


def test_preprocess_datasets_minimal(monkeypatch):
    # Lazy import to allow monkeypatch
    from whisper_tuner.models.whisper.finetune_core import data as data_mod

    # Stub audio loader to avoid real IO
    monkeypatch.setattr(
        data_mod, "load_audio_local_or_gcs", lambda src, sampling_rate: np.zeros(int(sampling_rate), dtype=np.float32)
    )
    # Stub label encoder
    monkeypatch.setattr(data_mod, "encode_labels", lambda tokenizer, text, max_len: np.array([1, 2, 3], dtype=np.int64))
    # Stub language resolver
    monkeypatch.setattr(
        data_mod,
        "resolve_language",
        lambda language_mode, sample_language, forced_language=None: (sample_language or "en", "transcribe"),
    )

    # Dummy feature extractor and processor
    class _FE:
        sampling_rate = 16000
        model_input_names = ["input_features"]

        class _Out:
            def __init__(self):
                self.input_features = [[0.0, 0.0]]

        def __call__(self, audio, sampling_rate):
            return self._Out()

    class _Processor:
        def __init__(self):
            self.tokenizer = object()

    feature_extractor = _FE()
    processor = _Processor()

    # Minimal dataset
    train = Dataset.from_dict(
        {
            "audio_path": ["/dev/null"],
            "text": ["hello"],
            "language": ["en"],
            "id": ["row-1"],
        }
    )
    raw = DatasetDict({"train": train})

    profile_config = {
        "language_mode": "strict",
        "languages": "en",
        "text_column": "text",
        "max_label_length": 16,
    }

    processed = data_mod.preprocess_datasets(
        raw_datasets=raw,
        feature_extractor=feature_extractor,
        processor=processor,
        profile_config=profile_config,
        effective_workers=1,
    )

    assert "train" in processed
    assert len(processed["train"]) == 1
    example = processed["train"][0]
    for key in ("input_features", "labels", "language", "id"):
        assert key in example
