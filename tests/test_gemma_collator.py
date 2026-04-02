import types
import torch
from whisper_tuner.models.gemma.finetune import DataCollatorGemmaAudio, GemmaTrainingConstants

class DummyProcessor:
    def __init__(self):
        class Tok:
            pad_token_id = 0
        self.tokenizer = Tok()
        self.sampling_rate = 16000
    def __call__(self, messages=None, audios=None, return_tensors=None, padding=None, text=None):
        # Return minimal tensors to mimic processor output
        batch = len(messages) if messages is not None else len(text)
        input_ids = torch.ones((batch, 5), dtype=torch.long)
        attention_mask = torch.ones((batch, 5), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_collator_produces_labels_and_ids():
    proc = DummyProcessor()
    collator = DataCollatorGemmaAudio(processor=proc, text_column="text")
    batch = [
        {"audio_path": "/path/a.wav", "text": "hello"},
        {"audio_path": "/path/b.wav", "text": "world"},
    ]
    # Monkeypatch audio loader to avoid file I/O
    import whisper_tuner.models.gemma.finetune as finetune_mod
    def fake_loader(path, sampling_rate=None):
        return [0.0] * 16000
    finetune_mod.load_audio_local_or_gcs = fake_loader

    out = collator(batch)
    assert "input_ids" in out and "attention_mask" in out and "labels" in out
    assert out["labels"].shape == out["input_ids"].shape
    # Ensure PAD becomes IGNORE id where applicable
    mask_zeros = (out["attention_mask"] == 0)
    if mask_zeros.any():
        assert (out["labels"][mask_zeros] == GemmaTrainingConstants.IGNORE_TOKEN_ID).all()
