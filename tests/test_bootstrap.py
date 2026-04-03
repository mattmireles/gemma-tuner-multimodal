import importlib
import os
import platform
import sys


def test_bootstrap_sets_mps_env(monkeypatch):
    # Simulate Apple Silicon environment
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    # Clear any existing values
    for key in ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO"):
        monkeypatch.delenv(key, raising=False)

    # Reload module to re-run bootstrap side effects
    sys.modules.pop("whisper_tuner.core.bootstrap", None)
    import whisper_tuner.core.bootstrap

    importlib.reload(whisper_tuner.core.bootstrap)

    high = float(os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0"))
    low = float(os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0"))

    assert high == 0.9, f"Expected high watermark 0.9, got {high}"
    assert low == 0.7, f"Expected low watermark 0.7, got {low}"
    assert 0.0 < low < high < 1.0
