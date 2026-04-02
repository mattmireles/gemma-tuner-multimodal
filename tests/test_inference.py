import numpy as np

from whisper_tuner.core.inference import prepare_features, decode_and_score


class DummyFE:
    sampling_rate = 16000
    def __call__(self, audio, sampling_rate):
        # Return structure with input_features[0]
        class _R:
            def __init__(self, arr):
                self.input_features = [arr]
        # Fake mel features as a small array
        feats = np.zeros((80, 10), dtype=np.float32)
        return _R(feats)


def test_prepare_features_with_array_and_path(tmp_path, monkeypatch):
    # Array input
    fe = DummyFE()
    arr = np.zeros(16000, dtype=np.float32)
    out = prepare_features(arr, fe)
    assert isinstance(out, np.ndarray)

    # Path input: write a short wav via librosa soundfile fallback
    import soundfile as sf
    wav_path = tmp_path / "a.wav"
    sf.write(str(wav_path), arr, 16000)
    # Monkeypatch librosa.load to avoid heavy processing; just read back
    import librosa as _lib
    def fake_load(path, sr):
        data, _ = sf.read(str(path))
        return data, sr
    monkeypatch.setattr(_lib, "load", fake_load)
    out2 = prepare_features(str(wav_path), fe)
    assert isinstance(out2, np.ndarray)


def test_decode_and_score_basic():
    # With empty strings, returns Nones for metrics
    wer, cer, pred, label, n_pred, n_label = decode_and_score(None, ["", "hi"], ["ref", ""], normalizer=lambda x: x)
    assert wer is None and cer is None

    # With simple strings, uses provided normalizer
    wer, cer, pred, label, n_pred, n_label = decode_and_score(None, ["a"], ["a"], normalizer=lambda x: x)
    assert wer == 0.0 and cer == 0.0


