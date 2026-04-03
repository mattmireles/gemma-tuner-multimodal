from whisper_tuner.core.inference import generate


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
        self._captured_forced_ids = "UNSET"

    def generate(self, input_features=None, forced_decoder_ids="UNSET", **kwargs):
        self._last_input = input_features
        self._captured_forced_ids = forced_decoder_ids

        # Return a simple tensor-like object with cpu().numpy()
        class _T:
            def cpu(self):
                return self

            def numpy(self):
                return [[1, 2, 3]]

        return _T()


def test_language_mode_mixed_sets_no_forced_ids():
    m = DummyModel()
    p = DummyProcessor()
    out = generate(m, p, input_features=[0.0], language_mode="mixed")
    # Mixed mode: forced_decoder_ids kwarg should be None
    assert m._captured_forced_ids is None
    assert out == [[1, 2, 3]]


def test_language_mode_strict_uses_batch_language():
    m = DummyModel()
    p = DummyProcessor()
    generate(m, p, input_features=[0.0], language_mode="strict", batch_language="en")
    # Strict mode: forced_decoder_ids kwarg should contain the language token
    assert m._captured_forced_ids == [(0, 42)]


def test_language_mode_override_sets_ids():
    m = DummyModel()
    p = DummyProcessor()
    generate(m, p, input_features=[0.0], language_mode="override:es")
    # Override mode: forced_decoder_ids kwarg should contain the forced language token
    assert m._captured_forced_ids == [(0, 42)]
