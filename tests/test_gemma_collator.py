import torch

from gemma_tuner.models.common.collators import DataCollatorGemmaAudio
from gemma_tuner.models.gemma.constants import GemmaTrainingConstants


class DummyProcessor:
    def __init__(self):
        class Tok:
            pad_token_id = 0
            bos_token_id = 0  # Matches the zeros in dummy input_ids
            unk_token_id = 3

            def convert_tokens_to_ids(self, token):
                # Return unk_token_id for any special token — signals "not found"
                # so the collator skips prompt masking gracefully.
                return self.unk_token_id

            def encode(self, text, add_special_tokens=False):
                return [99]

        self.tokenizer = Tok()
        self.sampling_rate = 16000

    def __call__(self, messages=None, audios=None, return_tensors=None, padding=None, text=None):
        # Return minimal tensors to mimic processor output.
        # Attention mask includes zeros at the end to exercise PAD-masking logic.
        batch = len(messages) if messages is not None else len(text)
        input_ids = torch.zeros((batch, 5), dtype=torch.long)
        attention_mask = torch.ones((batch, 5), dtype=torch.long)
        # Set last two positions to zero to simulate padding
        attention_mask[:, -2:] = 0
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_collator_produces_labels_and_ids():
    proc = DummyProcessor()
    collator = DataCollatorGemmaAudio(processor=proc, text_column="text")
    batch = [
        {"audio_path": "/path/a.wav", "text": "hello"},
        {"audio_path": "/path/b.wav", "text": "world"},
    ]
    # Monkeypatch audio loader to avoid file I/O.
    # DataCollatorGemmaAudio lives in collators.py and imports load_audio_local_or_gcs
    # locally inside __call__ via `from gemma_tuner.utils.dataset_prep import ...`.
    # Patching the module attribute before the call makes the local import pick up
    # the fake — patch the source module, not the consumer module.
    import gemma_tuner.utils.dataset_prep as dataset_prep_mod

    def fake_loader(path, sampling_rate=None):
        return [0.0] * 16000

    dataset_prep_mod.load_audio_local_or_gcs = fake_loader

    out = collator(batch)
    assert "input_ids" in out and "attention_mask" in out and "labels" in out
    assert out["labels"].shape == out["input_ids"].shape
    # Ensure PAD becomes IGNORE id where applicable
    mask_zeros = out["attention_mask"] == 0
    assert mask_zeros.any(), "Test expects some zero entries in attention_mask"
    assert (out["labels"][mask_zeros] == GemmaTrainingConstants.IGNORE_TOKEN_ID).all()
