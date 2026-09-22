import importlib.util
import csv
import json
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools" / "prepare_tt_telepathic_sft.py"
SPEC = importlib.util.spec_from_file_location("prepare_plan29", MODULE_PATH)
prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(prepare)


def test_build_plan29_projection_preserves_sealed_contract(tmp_path: Path) -> None:
    source = Path(__file__).parents[2] / "perfect-dictator" / "data" / "datasets" / "tt_screenshot_plan28_sft_v1" / "private" / "plan29-sft-v1"
    result = prepare.build_plan29_projection(source, tmp_path / "staging")
    assert result["written"] == {"full": {"train": 2812, "validation": 252}}
    assert result["sealed_test"] == {"count": 232, "staged": False}
    assert len(result["validation_order_ids"]) == 252
    profiles = (tmp_path / "staging" / "profiles.ini").read_text()
    assert "[profile:telepathic-plan29-r64-one-epoch]" in profiles
    assert "save_steps = 88" in profiles
    assert "gradient_accumulation_steps = 8" in profiles
    assert "lora_r = 64" in profiles
    with (tmp_path / "staging" / "full" / "train.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2812
    assert all(row["prompt"].count("<first_pass_screenshot_ocr>") == 1 for row in rows)
    assert all(row["prompt"].count("</first_pass_screenshot_ocr>") == 1 for row in rows)
