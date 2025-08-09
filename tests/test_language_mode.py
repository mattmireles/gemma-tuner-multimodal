import types

from core.inference import generate


class DummyProcessor:
    def get_decoder_prompt_ids(self, language, task):
        return [(0, 42)]


class DummyGenConfig:
    def __init__(self):
        self.forced_decoder_ids = None


class DummyModel:
    def __init__(self):
        self.generation_config = DummyGenConfig()
        self._last_input = None

    def generate(self, input_features=None, **kwargs):
        self._last_input = input_features
        # Return a simple tensor-like object with cpu().numpy()
        class _T:
            def __init__(self):
                pass
            def cpu(self):
                return self
            def numpy(self):
                return [[1, 2, 3]]
        return _T()


def test_language_mode_mixed_sets_no_forced_ids():
    m = DummyModel()
    p = DummyProcessor()
    out = generate(m, p, input_features=[0.0], language_mode="mixed")
    assert m.generation_config.forced_decoder_ids is None
    assert out == [[1, 2, 3]]


def test_language_mode_strict_uses_batch_language():
    m = DummyModel()
    p = DummyProcessor()
    generate(m, p, input_features=[0.0], language_mode="strict", batch_language="en")
    assert m.generation_config.forced_decoder_ids == [(0, 42)]


def test_language_mode_override_sets_ids():
    m = DummyModel()
    p = DummyProcessor()
    generate(m, p, input_features=[0.0], language_mode="override:es")
    assert m.generation_config.forced_decoder_ids == [(0, 42)]


