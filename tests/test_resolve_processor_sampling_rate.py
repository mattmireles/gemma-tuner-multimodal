"""Tests for resolve_processor_sampling_rate (gemma.constants)."""

from gemma_tuner.models.gemma.constants import AudioProcessingConstants, resolve_processor_sampling_rate


class _ProcSr:
    sampling_rate = 8000


class _FE:
    sampling_rate = 24000


class _ProcFe:
    feature_extractor = _FE()


def test_hint_overrides_processor():
    assert resolve_processor_sampling_rate(_ProcSr(), hint=48000) == 48000


def test_processor_sampling_rate():
    assert resolve_processor_sampling_rate(_ProcSr()) == 8000


def test_feature_extractor_fallback():
    assert resolve_processor_sampling_rate(_ProcFe()) == 24000


def test_default_when_missing():
    class _Empty:
        pass

    assert resolve_processor_sampling_rate(_Empty()) == AudioProcessingConstants.DEFAULT_SAMPLING_RATE
