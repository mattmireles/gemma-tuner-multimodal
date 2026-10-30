"""Plan 33 evaluation panels are frozen, disjoint, and bound to a model receipt."""

import hashlib
import json

import pytest

from tools import run_plan33_eval as evaluation

RECEIPT = evaluation.PROJECTION / "projection.receipt.json"
needs_private_projection = pytest.mark.skipif(not RECEIPT.exists(), reason="private Plan 33 projection absent")


def _receipt_sha():
    return hashlib.sha256(RECEIPT.read_bytes()).hexdigest()


@needs_private_projection
def test_exposed20_is_first_twenty_validation_rows_in_every_mode():
    rows = evaluation.panel_rows("exposed20", _receipt_sha())
    assert len(rows) == 20 * len(evaluation.MODES)
    by_mode = {}
    for mode, row, image_sha in rows:
        assert row["input_mode"] == mode and len(image_sha) == 64
        by_mode.setdefault(mode, []).append(row["id"])
    assert list(by_mode) == list(evaluation.MODES)
    assert all(ids == by_mode["full"] for ids in by_mode.values())


@needs_private_projection
def test_selection_panel_is_disjoint_from_exposed20():
    exposed = {row["id"] for _, row, _ in evaluation.panel_rows("exposed20", _receipt_sha())}
    selection = evaluation.panel_rows("selection60", _receipt_sha())
    assert len(selection) == 360
    assert not exposed & {row["id"] for _, row, _ in selection}


def test_unknown_panel_and_changed_projection_fail_closed():
    with pytest.raises(ValueError):
        evaluation.panel_rows("test", "0" * 64)


def test_model_without_plan33_receipt_or_with_changed_bytes_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluation.verify_model(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    receipt = {"schema_version": "plan33_mlx_model_v1", "precision": "bfloat16",
               "files_sha256": {"config.json": "0" * 64}}
    (tmp_path / "plan33-mlx-receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluation.verify_model(tmp_path)


@pytest.mark.parametrize("raw", [
    '{"status": "APPROVED", "feedback": "ok"}',
    '```json\n{"status": "APPROVED", "feedback": "ok"}\n```',
    '  ```\n{"status": "APPROVED", "feedback": "ok"}\n```\n',
])
def test_vote_parser_accepts_bare_or_single_fenced_json(raw):
    assert evaluation.parse_plan33_vote(raw) == {"status": "APPROVED", "feedback": "ok"}


@pytest.mark.parametrize("raw", [
    'Verdict: ```json\n{"status": "APPROVED", "feedback": "ok"}\n```',
    '```json\n{"status": "MAYBE", "feedback": "ok"}\n```',
    '```json\n{"status": "APPROVED"}\n```',
])
def test_vote_parser_still_rejects_prose_bad_status_or_missing_feedback(raw):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        evaluation.parse_plan33_vote(raw)
