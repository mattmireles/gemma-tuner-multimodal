"""Tests for the private fixture and reference receipt contracts."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

TOOLS = Path(__file__).parents[2] / "tools" / "runtime"
SPEC = importlib.util.spec_from_file_location("reference_common", TOOLS / "reference_common.py")
assert SPEC and SPEC.loader
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)
sys.modules["reference_common"] = common

MLX_SPEC = importlib.util.spec_from_file_location("benchmark_mlx_w6", TOOLS / "benchmark_mlx_w6.py")
assert MLX_SPEC and MLX_SPEC.loader
mlx_benchmark = importlib.util.module_from_spec(MLX_SPEC)
MLX_SPEC.loader.exec_module(mlx_benchmark)

FREEZE_SPEC = importlib.util.spec_from_file_location("freeze_phase0_fixtures", TOOLS / "freeze_phase0_fixtures.py")
assert FREEZE_SPEC and FREEZE_SPEC.loader
freeze_fixtures = importlib.util.module_from_spec(FREEZE_SPEC)
FREEZE_SPEC.loader.exec_module(freeze_fixtures)

REVIEW_SPEC = importlib.util.spec_from_file_location("build_image_review", TOOLS / "build_image_review.py")
assert REVIEW_SPEC and REVIEW_SPEC.loader
image_review = importlib.util.module_from_spec(REVIEW_SPEC)
REVIEW_SPEC.loader.exec_module(image_review)

IDLE_SPEC = importlib.util.spec_from_file_location("run_idle_gated", TOOLS / "run_idle_gated.py")
assert IDLE_SPEC and IDLE_SPEC.loader
idle_gate = importlib.util.module_from_spec(IDLE_SPEC)
IDLE_SPEC.loader.exec_module(idle_gate)

SCORE_SPEC = importlib.util.spec_from_file_location("score_image_review", TOOLS / "score_image_review.py")
assert SCORE_SPEC and SCORE_SPEC.loader
score_image_review = importlib.util.module_from_spec(SCORE_SPEC)
SCORE_SPEC.loader.exec_module(score_image_review)


def write_contract(root: Path, mode: str = "image_to_text") -> tuple[Path, Path, dict[str, object]]:
    private_root = root / "private"
    private_file = private_root / mode / "fixture.json"
    private_file.parent.mkdir(parents=True)
    media = private_file.parent / "input.bin"
    media.write_bytes(b"private input")
    request = {"messages": [{"role": "user", "content": "private prompt"}]}
    fixture_id = common.sha256_text(f"fixture:{mode}")
    private = {
        "schema_version": common.PRIVATE_SCHEMA,
        "fixture_id": fixture_id,
        "mode": mode,
        "request": request,
        "media": {"image_0": media.name},
    }
    private_file.write_text(json.dumps(private), encoding="utf-8")
    row = {
        "schema_version": common.MANIFEST_SCHEMA,
        "fixture_id": fixture_id,
        "mode": mode,
        "private_ref": f"{mode}/fixture.json",
        "request_sha256": common.sha256_text(common.canonical_json(request)),
        "input_sha256": {"image_0": common.sha256_file(media)},
        "expected": {"output_sha256": common.sha256_text("expected"), "quality_floor": "human_approved"},
        "generation": {"max_new_tokens": 64, "temperature": 0.0},
    }
    manifest = root / "manifest.jsonl"
    manifest.write_text(common.canonical_json(row) + "\n", encoding="utf-8")
    return manifest, private_root, row


def write_text_contract(root: Path) -> tuple[Path, Path, dict[str, object]]:
    private_root = root / "private"
    private_file = private_root / "text_to_text" / "fixture.json"
    private_file.parent.mkdir(parents=True)
    request = {"messages": [{"role": "user", "content": "private prompt"}]}
    fixture_id = common.sha256_text("fixture:text_to_text")
    private = {
        "schema_version": common.PRIVATE_SCHEMA,
        "fixture_id": fixture_id,
        "mode": "text_to_text",
        "request": request,
        "media": {},
    }
    private_file.write_text(json.dumps(private), encoding="utf-8")
    row = {
        "schema_version": common.MANIFEST_SCHEMA,
        "fixture_id": fixture_id,
        "mode": "text_to_text",
        "private_ref": "text_to_text/fixture.json",
        "request_sha256": common.sha256_text(common.canonical_json(request)),
        "input_sha256": {"request": common.sha256_text(common.canonical_json(request))},
        "expected": {"output_sha256": common.sha256_text("expected"), "quality_floor": "human_approved"},
        "generation": {"max_new_tokens": 64, "temperature": 0.0},
    }
    manifest = root / "manifest.jsonl"
    manifest.write_text(common.canonical_json(row) + "\n", encoding="utf-8")
    return manifest, private_root, row


def test_manifest_resolves_private_data_without_tracking_it(tmp_path: Path) -> None:
    manifest, private_root, row = write_contract(tmp_path)
    loaded = common.load_manifest(manifest)
    assert loaded == [row]
    private = common.load_private_fixture(row, private_root)
    assert private["request"]["messages"][0]["content"] == "private prompt"
    assert private["_resolved_media"]["image_0"].read_bytes() == b"private input"
    assert "private prompt" not in manifest.read_text(encoding="utf-8")


def test_private_request_hash_drift_fails_closed(tmp_path: Path) -> None:
    manifest, private_root, row = write_contract(tmp_path)
    private_path = private_root / str(row["private_ref"])
    payload = json.loads(private_path.read_text(encoding="utf-8"))
    payload["request"]["messages"][0]["content"] = "changed"
    private_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="request hash mismatch"):
        common.load_private_fixture(common.load_manifest(manifest)[0], private_root)


def test_private_media_hash_drift_fails_closed(tmp_path: Path) -> None:
    manifest, private_root, row = write_contract(tmp_path)
    media = (private_root / str(row["private_ref"])).parent / "input.bin"
    media.write_bytes(b"changed")
    with pytest.raises(ValueError, match="media hash mismatch"):
        common.load_private_fixture(common.load_manifest(manifest)[0], private_root)


def test_text_fixture_request_hash_is_its_private_input(tmp_path: Path) -> None:
    manifest, private_root, row = write_text_contract(tmp_path)
    loaded = common.load_manifest(manifest)[0]
    private = common.load_private_fixture(loaded, private_root)
    assert private["_resolved_media"] == {}
    assert loaded["input_sha256"]["request"] == row["request_sha256"]


def test_freeze_text_preserves_exact_request_envelope(tmp_path: Path) -> None:
    source = tmp_path / "text.csv"
    request = {
        "messages": [
            {"role": "system", "content": "exact system"},
            {"role": "user", "content": "exact rendered context plus transcript"},
        ]
    }
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "request_json", "expected_output"])
        writer.writeheader()
        writer.writerow(
            {
                "id": "fixture-1",
                "request_json": common.canonical_json(request),
                "expected_output": "exact product response",
            }
        )

    private_root = tmp_path / "private"
    manifest_root = tmp_path / "fixtures"
    rows = freeze_fixtures.freeze_text(source, 1, private_root, manifest_root)

    private = common.load_private_fixture(rows[0], private_root)
    assert private["request"] == request
    assert private["expected_output"] == "exact product response"
    tracked_manifest = (manifest_root / "text" / "manifest.jsonl").read_text(encoding="utf-8")
    assert "exact rendered context" not in tracked_manifest


def test_freeze_text_rejects_invented_column_contract(tmp_path: Path) -> None:
    source = tmp_path / "text.csv"
    source.write_text(
        "id,system_prompt,screenshot_description,raw_asr,clean_transcript,rewrite\n1,system,screen,raw,clean,rewrite\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected_output.*request_json|request_json.*expected_output"):
        freeze_fixtures.freeze_text(source, 1, tmp_path / "private", tmp_path / "fixtures")


def test_freeze_image_accepts_exact_gold_ledger_without_rewriting_output(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (4, 4), color="white").save(image_path)
    source = tmp_path / "image.jsonl"
    source_row = {
        "example_id": "plan29-example-1",
        "image_path": str(image_path),
        "system_prompt": "exact system",
        "user_prompt": "exact user",
        "gold_response": "exact gold response",
    }
    source.write_text(common.canonical_json(source_row) + "\n", encoding="utf-8")

    private_root = tmp_path / "private"
    manifest_root = tmp_path / "fixtures"
    rows = freeze_fixtures.freeze_image(source, 1, private_root, manifest_root)

    private = common.load_private_fixture(rows[0], private_root)
    assert private["request"]["messages"] == [
        {"role": "system", "content": "exact system"},
        {"role": "user", "content": "exact user"},
    ]
    assert private["expected_output"] == "exact gold response"
    assert rows[0]["expected"]["output_sha256"] == common.sha256_text("exact gold response")


def test_image_review_slot_assignment_is_deterministic_and_mixed() -> None:
    fixtures = [common.sha256_text(f"fixture-{index}") for index in range(20)]
    first = [image_review._slot_sources("seed", fixture) for fixture in fixtures]
    second = [image_review._slot_sources("seed", fixture) for fixture in fixtures]
    assert first == second
    assert set(first) == {("reference", "candidate"), ("candidate", "reference")}


def test_image_review_scoring_unblinds_and_checks_noninferiority() -> None:
    mapping = {
        "schema_version": "gemma4-e4b-image-review-map-v1",
        "review_id": "review-1",
        "items": [
            {
                "pair_id": "pair-1",
                "slots": {"A": "candidate", "B": "reference"},
            }
        ],
    }
    review = {
        "schema_version": "gemma4-e4b-image-review-v1",
        "review_id": "review-1",
        "results": [
            {
                "pair_id": "pair-1",
                "outputs": {
                    "A": {"grade": "Good", "material_omission": False, "notes": ""},
                    "B": {"grade": "NeedsImprovement", "material_omission": True, "notes": ""},
                },
                "preference": "A",
            }
        ],
    }
    summary = score_image_review.score_review(review, mapping)
    assert summary["arms"]["candidate"]["good_or_better"] == 1
    assert summary["arms"]["reference"]["good_or_better"] == 0
    assert summary["preferences"] == {"candidate": 1, "reference": 0, "tie": 0}
    assert summary["matched_product_noninferiority"]["pass"]


def test_image_review_scoring_rejects_incomplete_review() -> None:
    mapping = {
        "schema_version": "gemma4-e4b-image-review-map-v1",
        "review_id": "review-1",
        "items": [{"pair_id": "pair-1", "slots": {"A": "candidate", "B": "reference"}}],
    }
    review = {
        "schema_version": "gemma4-e4b-image-review-v1",
        "review_id": "review-1",
        "results": [
            {
                "pair_id": "pair-1",
                "outputs": {
                    "A": {"grade": "", "material_omission": False},
                    "B": {"grade": "Good", "material_omission": False},
                },
                "preference": "Tie",
            }
        ],
    }
    with pytest.raises(ValueError, match="grade is missing"):
        score_image_review.score_review(review, mapping)


def test_idle_gate_fails_on_each_declared_stop_condition() -> None:
    passing = {
        "cpu_idle_percent": 95.0,
        "swap_in_delta_bytes": 0,
        "swap_out_delta_bytes": 0,
        "cpu_offenders": [],
        "other_ml_jobs": [],
        "thermal": {"pass": True},
    }
    assert idle_gate.sample_passes(passing, 90.0)
    for field, value in (
        ("cpu_idle_percent", 89.9),
        ("swap_in_delta_bytes", 1),
        ("swap_out_delta_bytes", 1),
        ("cpu_offenders", [{"pid": 1}]),
        ("other_ml_jobs", [{"pid": 2}]),
        ("thermal", {"pass": False}),
    ):
        failing = dict(passing)
        failing[field] = value
        assert not idle_gate.sample_passes(failing, 90.0)


def test_idle_gate_ml_job_detection_uses_executable_tokens_not_repo_path() -> None:
    assert idle_gate._looks_like_ml_job("python3.11", ["/tmp/venv/bin/python", "/tmp/benchmark_mlx_w6.py"])
    assert idle_gate._looks_like_ml_job("kokoro-worker", ["/usr/local/bin/kokoro-worker"])
    assert not idle_gate._looks_like_ml_job(
        "codex", ["/Applications/Codex.app/codex", "--cwd", "/tmp/gemma-tuner-multimodal"]
    )


def test_manifest_rejects_absolute_private_path(tmp_path: Path) -> None:
    manifest, _, row = write_contract(tmp_path)
    row["private_ref"] = "/tmp/private.json"
    manifest.write_text(common.canonical_json(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="safe path"):
        common.load_manifest(manifest)


def test_stage_summary_reports_p50_and_p95() -> None:
    rows = [{stage: float(index) for stage in common.STAGES} for index in (1, 2, 3, 4, 5)]
    summary = common.summarize_stage_rows(rows)
    assert set(summary) == set(common.STAGES)
    assert summary["end_to_end"]["p50_ms"] == 3.0
    assert summary["end_to_end"]["p95_ms"] == pytest.approx(4.8)


def test_normalized_error_rates_ignore_case_and_punctuation() -> None:
    assert common.normalized_error_rates("Hello, WORLD!", "hello world") == {"wer": 0.0, "cer": 0.0}
    rates = common.normalized_error_rates("one two three", "one too three")
    assert rates["wer"] == pytest.approx(1 / 3)
    assert 0 < rates["cer"] < rates["wer"]


def test_mlx_timing_keeps_encoder_inside_first_token_boundary() -> None:
    class Result:
        prompt_tokens = 100
        prompt_tps = 50.0
        generation_tokens = 20
        generation_tps = 10.0

    timings = mlx_benchmark.result_timings_ms(Result(), elapsed_seconds=4.5)
    assert timings == {
        "other_prepost_ms": 500.0,
        "encoder_and_prefill_ms": 2000.0,
        "first_token_ms": 2000.0,
        "decode_ms": 2000.0,
        "end_to_end_ms": 4500.0,
    }
