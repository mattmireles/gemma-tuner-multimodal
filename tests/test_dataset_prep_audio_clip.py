"""Tests for load_audio_local_or_gcs float clipping to [-1, 1]."""

import numpy as np

from gemma_tuner.utils.dataset_prep import load_audio_local_or_gcs


def test_raw_array_clips_out_of_range_samples():
    out = load_audio_local_or_gcs(np.array([1.5, -2.0, 0.5], dtype=np.float32), 16000)
    assert out.dtype == np.float32
    assert np.allclose(out, np.array([1.0, -1.0, 0.5], dtype=np.float32))


def test_hf_audio_dict_clips_when_no_resample():
    # Same source/target rate avoids librosa.resample (optional resampy in some envs).
    d = {"array": np.array([2.0, -1.5, 0.25], dtype=np.float32), "sampling_rate": 16000}
    out = load_audio_local_or_gcs(d, 16000)
    assert out.dtype == np.float32
    assert np.allclose(out, np.array([1.0, -1.0, 0.25], dtype=np.float32))
