import logging
from unittest.mock import patch

from gemma_tuner.utils.device import apply_device_defaults


def test_apply_device_defaults_mps_warnings(caplog):
    """Verify MPS defaults fill unset values without stomping explicit choices."""
    with patch("gemma_tuner.utils.device.get_device") as mock_get_device:
        # Simulate running on an MPS device
        mock_get_device.return_value.type = "mps"

        # Case 1: Unset values receive MPS-friendly defaults.
        profile_config = {"gradient_checkpointing": False}
        with caplog.at_level(logging.INFO):
            apply_device_defaults(profile_config)
            assert "defaulting attn_implementation to 'eager'" in caplog.text
            assert profile_config["attn_implementation"] == "eager"
            assert profile_config["dtype"] == "float32"

        caplog.clear()

        # Case 2: Explicit profile values are preserved on MPS.
        profile_config = {"attn_implementation": "sdpa", "dtype": "float16", "gradient_checkpointing": False}
        with caplog.at_level(logging.INFO):
            apply_device_defaults(profile_config)
            assert "keeping explicit attn_implementation='sdpa'" in caplog.text
            assert "keeping explicit dtype='float16'" in caplog.text
            assert profile_config["attn_implementation"] == "sdpa"
            assert profile_config["dtype"] == "float16"

        caplog.clear()

        # Case 3: Gradient checkpointing still warns.
        profile_config = {"gradient_checkpointing": True}
        with caplog.at_level(logging.WARNING):
            apply_device_defaults(profile_config)
            assert "Gradient checkpointing is enabled" in caplog.text

        caplog.clear()

        # Case 4: fp16 still self-corrects because it is unsupported on MPS here.
        profile_config = {"fp16": True}
        with caplog.at_level(logging.WARNING):
            apply_device_defaults(profile_config)
            assert "fp16 mixed precision is not supported on MPS" in caplog.text
            assert profile_config["fp16"] is False
