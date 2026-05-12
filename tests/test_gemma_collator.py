import pytest
import torch
from PIL import Image as PILImage

from gemma_tuner.models.common.collators import DataCollatorGemmaAudio, DataCollatorGemmaAudioVisual
from gemma_tuner.models.gemma.constants import GemmaTrainingConstants
from gemma_tuner.models.gemma.family import GemmaFamily


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

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        if tokenize:
            raise AssertionError("DataCollatorGemmaAudio uses tokenize=False then processor(text=..., audio=...)")
        if not conversation:
            return []
        first = conversation[0]
        batch_size = 1 if isinstance(first, dict) else len(conversation)
        return ["dummy prompt"] * batch_size

    def __call__(self, text=None, audio=None, images=None, return_tensors=None, padding=None, **kwargs):
        # Return minimal tensors to mimic processor output.
        # Attention mask includes zeros at the end to exercise PAD-masking logic.
        if text is None:
            raise ValueError("expected text")
        batch = len(text) if isinstance(text, list) else 1
        input_ids = torch.zeros((batch, 5), dtype=torch.long)
        attention_mask = torch.ones((batch, 5), dtype=torch.long)
        # Set last two positions to zero to simulate padding
        attention_mask[:, -2:] = 0
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_collator_produces_labels_and_ids():
    proc = DummyProcessor()
    collator = DataCollatorGemmaAudio(processor=proc, text_column="text", family=GemmaFamily.GEMMA_3N)
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
    assert "token_type_ids" in out and "mm_token_type_ids" in out
    assert torch.equal(out["token_type_ids"], torch.zeros_like(out["input_ids"]))
    assert torch.equal(out["mm_token_type_ids"], torch.zeros_like(out["input_ids"]))
    # Ensure PAD becomes IGNORE id where applicable
    mask_zeros = out["attention_mask"] == 0
    assert mask_zeros.any(), "Test expects some zero entries in attention_mask"
    assert (out["labels"][mask_zeros] == GemmaTrainingConstants.IGNORE_TOKEN_ID).all()


def test_audio_collator_gemma4_injects_mm_token_type_ids():
    proc = DummyProcessor()
    collator = DataCollatorGemmaAudio(processor=proc, text_column="text", family=GemmaFamily.GEMMA_4)
    import gemma_tuner.utils.dataset_prep as dataset_prep_mod

    def fake_loader(path, sampling_rate=None):
        return [0.0] * 16000

    dataset_prep_mod.load_audio_local_or_gcs = fake_loader

    out = collator([{"audio_path": "/path/a.wav", "text": "hello"}])
    assert "token_type_ids" in out and "mm_token_type_ids" in out
    assert torch.equal(out["token_type_ids"], torch.zeros_like(out["input_ids"]))
    assert torch.equal(out["mm_token_type_ids"], torch.zeros_like(out["input_ids"]))


class DummyAudioVisualProcessor(DummyProcessor):
    def __init__(self):
        super().__init__()
        self.last_audio = None
        self.last_images = None
        self.last_text = None

    def __call__(self, text=None, audio=None, images=None, return_tensors=None, padding=None, **kwargs):
        if audio is None or images is None:
            raise ValueError("expected both audio and images")
        self.last_audio = audio
        self.last_images = images
        self.last_text = text
        return super().__call__(text=text, return_tensors=return_tensors, padding=padding, **kwargs)


def test_audiovisual_collator_produces_labels_and_ids(tmp_path):
    proc = DummyAudioVisualProcessor()
    collator = DataCollatorGemmaAudioVisual(
        processor=proc,
        text_column="text",
        family=GemmaFamily.GEMMA_4,
        image_path_column="image_path",
    )
    image_path = tmp_path / "sample.png"
    PILImage.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path)

    import gemma_tuner.utils.dataset_prep as dataset_prep_mod

    def fake_loader(path, sampling_rate=None):
        return [0.0] * 16000

    dataset_prep_mod.load_audio_local_or_gcs = fake_loader

    out = collator(
        [{"audio_path": "/path/a.wav", "image_path": str(image_path), "text": "scene: people talking in a kitchen"}]
    )
    assert "input_ids" in out and "attention_mask" in out and "labels" in out
    assert out["labels"].shape == out["input_ids"].shape
    assert "token_type_ids" in out and "mm_token_type_ids" in out
    assert torch.equal(out["token_type_ids"], torch.zeros_like(out["input_ids"]))
    assert torch.equal(out["mm_token_type_ids"], torch.zeros_like(out["input_ids"]))


def test_audiovisual_collator_batch_passes_per_sample_audio_and_images(tmp_path):
    proc = DummyAudioVisualProcessor()
    collator = DataCollatorGemmaAudioVisual(
        processor=proc,
        text_column="text",
        family=GemmaFamily.GEMMA_4,
        image_path_column="image_path",
    )
    paths = []
    for i in range(3):
        p = tmp_path / f"av_{i}.png"
        PILImage.new("RGB", (8, 8), color=(0, 0, i * 80)).save(p)
        paths.append(str(p))

    import gemma_tuner.utils.dataset_prep as dataset_prep_mod

    def fake_loader(path, sampling_rate=None):
        return [0.0] * 16000

    dataset_prep_mod.load_audio_local_or_gcs = fake_loader

    collator([{"audio_path": f"/path/{i}.wav", "image_path": paths[i], "text": f"sample {i}"} for i in range(3)])
    # Per-sample audio is a flat list of length batch; images is a list-per-sample wrapping.
    assert isinstance(proc.last_audio, list) and len(proc.last_audio) == 3
    assert isinstance(proc.last_images, list) and len(proc.last_images) == 3
    for inner in proc.last_images:
        assert isinstance(inner, list) and len(inner) == 1
    assert isinstance(proc.last_text, list) and len(proc.last_text) == 3


def test_audiovisual_collator_missing_audio_path_raises(tmp_path):
    proc = DummyAudioVisualProcessor()
    collator = DataCollatorGemmaAudioVisual(
        processor=proc,
        text_column="text",
        family=GemmaFamily.GEMMA_4,
        image_path_column="image_path",
    )
    p = tmp_path / "x.png"
    PILImage.new("RGB", (8, 8), color=(1, 2, 3)).save(p)

    import gemma_tuner.utils.dataset_prep as dataset_prep_mod

    def fake_loader(path, sampling_rate=None):
        return [0.0] * 16000

    dataset_prep_mod.load_audio_local_or_gcs = fake_loader

    with pytest.raises(KeyError, match=r"no audio path"):
        collator([{"image_path": str(p), "text": "no audio key here"}])
