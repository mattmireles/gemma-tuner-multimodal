"""Plan 32 smoke must validate its string-valued INI booleans and telemetry."""

import pytest

from tools.run_plan32_warmstart_smoke import validate_smoke_profile


def _profile():
    return {
        "require_validation_telemetry": "true",
        "record_exposures": "true",
        "load_validation": "true",
        "input_mode_column": None,
        "max_samples": "1",
        "max_steps": "1",
        "stop_after_step": "1",
        "telemetry_train_rows": "1",
        "telemetry_validation_rows": "1",
        "required_transformers_version": "5.5.2",
        "plan32_epoch": "0",
    }


def test_smoke_profile_accepts_string_valued_ini_booleans():
    validate_smoke_profile(_profile())


@pytest.mark.parametrize("key", ["require_validation_telemetry", "record_exposures", "load_validation"])
def test_smoke_profile_rejects_disabled_required_boolean(key):
    profile = _profile()
    profile[key] = "false"
    with pytest.raises(ValueError, match="one row/step"):
        validate_smoke_profile(profile)


def test_smoke_profile_rejects_success_lineage_mode():
    profile = _profile()
    profile["input_mode_column"] = "input_mode"
    with pytest.raises(ValueError, match="one row/step"):
        validate_smoke_profile(profile)
