import os
import sys
import importlib
import platform


def test_bootstrap_sets_mps_env(monkeypatch):
    # Simulate Apple Silicon environment
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    # Clear any existing values
    for key in ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO"):
        monkeypatch.delenv(key, raising=False)

    # Reload module to re-run bootstrap side effects
    sys.modules.pop("whisper_tuner.core.bootstrap", None)
    import whisper_tuner.core.bootstrap  # noqa: F401
    importlib.reload(whisper_tuner.core.bootstrap)

    high = float(os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0"))
    low = float(os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0"))

    assert 0.0 < low < high < 1.0
