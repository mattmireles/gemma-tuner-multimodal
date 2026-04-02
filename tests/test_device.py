import logging
from unittest.mock import patch

from whisper_tuner.utils.device import apply_device_defaults


def test_apply_device_defaults_mps_warnings(caplog):
    """Verify that warnings are issued for suboptimal MPS settings."""
    with patch("whisper_tuner.utils.device.get_device") as mock_get_device:
        # Simulate running on an MPS device
        mock_get_device.return_value.type = "mps"

        # Case 1: Incorrect attn_implementation
        profile_config = {"attn_implementation": "sdpa", "gradient_checkpointing": False}
        with caplog.at_level(logging.WARNING):
            apply_device_defaults(profile_config)
            assert "Overriding attn_implementation to 'eager'" in caplog.text
            assert profile_config["attn_implementation"] == "eager"

        caplog.clear()

        # Case 2: Gradient checkpointing enabled
        profile_config = {"gradient_checkpointing": True}
        with caplog.at_level(logging.WARNING):
            apply_device_defaults(profile_config)
            assert "Gradient checkpointing is enabled" in caplog.text

        caplog.clear()

        # Case 3: Incorrect dtype
        profile_config = {"dtype": "float16"}
        with caplog.at_level(logging.WARNING):
            apply_device_defaults(profile_config)
            assert "Overriding dtype to 'float32'" in caplog.text
            assert profile_config["dtype"] == "float32"
