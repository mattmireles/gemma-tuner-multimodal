"""Contract guards for Plan 32's single-candidate validator."""

import csv
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from gemma_tuner.models.common.plan31_input_modes import OCR_CLOSE, OCR_OPEN
from tools import run_plan32_literal_eval as evaluator
from tools.run_plan32_literal_eval import frozen_rows, parse_vote, prompt_sections, render_judge, verify_model


def test_ane_supervisor_runs_exactly_one_generation_per_child(tmp_path, monkeypatch):
    ledger = tmp_path / "generations.jsonl"
    evaluator.append_jsonl(ledger, {"id": "existing"})
    launches = []

    class Child:
        returncode = 0

        def __init__(self, _command, *, env):
            assert env[evaluator.GUARDED_CHILD_ENV] == "1"
            assert env[evaluator.ROW_CHILD_ENV] == "1"
            launches.append(env)
            evaluator.append_jsonl(ledger, {"id": f"new-{len(launches)}"})

        def poll(self):
            return self.returncode

    monkeypatch.setattr(evaluator.subprocess, "Popen", Child)
    monkeypatch.setitem(
        sys.modules,
        "tools.plan32_ane_vision",
        SimpleNamespace(
            host_pressure_snapshot=lambda: {
                "free_memory_percent": 75.0,
                "swap_used_mb": 100.0,
            },
            pressure_violation=lambda *_args, **_kwargs: None,
        ),
    )

    assert evaluator.supervise_ane_child(
        generation_path=ledger,
        target_generations=3,
        max_swap_used_mb=4096,
        max_swap_growth_mb=2048,
        min_free_memory_percent=3,
    ) == 0
    assert len(launches) == 2
    assert len(evaluator.read_jsonl(ledger)) == 3


def test_only_three_way_status_and_feedback_are_accepted():
    assert parse_vote('{"status":"APPROVED","feedback":"Grounded."}') == {"status": "APPROVED", "feedback": "Grounded."}
    for bad in (
        '{"grade":"Excellent","feedback":"Grounded."}',
        '{"status":"Good","feedback":"Grounded."}',
        '{"status":"REJECTED","feedback":""}',
        '{"status":"NEEDS_REVIEW","feedback":"Unclear.","winner":"A"}',
    ):
        with pytest.raises(ValueError):
            parse_vote(bad)


def test_prompt_rejects_hash_drift(tmp_path):
    path = tmp_path / "judge.md"
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        prompt_sections(path, "0" * 64)


def test_render_has_one_candidate_and_ocr_once():
    row = {
        "system_prompt": "Original task system text",
        "prompt": "Return valid JSON only." + OCR_OPEN + "OCR WORD" + OCR_CLOSE,
    }
    user = """<instructions>${canonical_full_system_instruction}
${canonical_full_user_message}</instructions>
<ocr>${structured_ocr}</ocr>
<output>${primary_analyst_output}</output>"""
    rendered = render_judge("Judge system text", user, row, json.dumps({"context_analysis": {}}))
    assert rendered.count("OCR WORD") == 1
    assert rendered.count("Original task system text") == 1
    assert rendered.count("<output>") == 1
    assert "${" not in rendered


def test_render_accepts_literal_dollar_brace_in_candidate_but_not_in_template():
    row = {"system_prompt": "S", "prompt": "Return valid JSON only." + OCR_OPEN + "O" + OCR_CLOSE}
    user = "${canonical_full_system_instruction}${canonical_full_user_message}${structured_ocr}${primary_analyst_output}"
    assert "${HOME}" in render_judge("J", user, row, "echo ${HOME}")
    with pytest.raises(ValueError, match="unfilled"):
        render_judge("J", user + "${stray}", row, "x")


def test_quantized_model_must_be_decoder_only_with_exact_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda _name: {"mlx": "0.32.2", "mlx-vlm": "0.7.1"}[_name])
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"synthetic")
    (model / "config.json").write_text(
        json.dumps({"quantization": {"group_size": 64, "bits": 6, "mode": "affine"}}), encoding="utf-8"
    )
    receipt = {
        "schema_version": "plan32_decoder_quantization_v1",
        "bits": 6,
        "quantized_language_modules": 345,
        "quantized_nonlanguage_modules": 0,
        "output_sha256": {"model.safetensors": hashlib.sha256(b"synthetic").hexdigest()},
    }
    (model / "quantization-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert verify_model(model)
    receipt["quantized_nonlanguage_modules"] = 1
    (model / "quantization-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="decoder-only"):
        verify_model(model)


def test_latency_subset_verifies_only_selected_images_and_all_projection_files(tmp_path):
    panel = tmp_path / "validation-60.jsonl"
    rows = []
    with panel.open("w", encoding="utf-8") as handle:
        for index in range(60):
            payload = b"image" + str(index).encode()
            if index < 20:
                (tmp_path / f"{index}.png").write_bytes(payload)
            item = {"id": str(index), "image_sha256": hashlib.sha256(payload).hexdigest()}
            handle.write(json.dumps(item) + "\n")
            rows.append({"id": str(index), "image_path": f"{index}.png"})
    csv_path = tmp_path / "validation-panel-full.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image_path"])
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "epoch": 2,
        "fresh_panel_rows": 60,
        "outputs_sha256": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in ("validation-60.jsonl", "validation-panel-full.csv")
        },
    }
    receipt_path = tmp_path / "projection.receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert len(frozen_rows(tmp_path, digest, subset_only=True, panel_rows=20)) == 20
    (tmp_path / "0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="screenshot hash changed"):
        frozen_rows(tmp_path, digest, subset_only=True, panel_rows=20)
