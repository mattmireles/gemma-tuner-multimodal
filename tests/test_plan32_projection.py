import json
import hashlib
from pathlib import Path

import pytest

from gemma_tuner.utils.plan32_projection import _target_contract, _verify_validation_image_hashes


def _target():
    return {
        "context_analysis": {
            "type": "document",
            "active_conversation": {
                "participants": [],
                "current_topic": {"summary": "A document is open.", "existing_text": ""},
            },
            "technical_context": {"language": "English", "unusual_terms": []},
        }
    }


def test_plan32_literal_target_contract_accepts_empty_values():
    _target_contract(json.dumps(_target()))


@pytest.mark.parametrize("field", ["notes", "tone", "started_at", "key_terms", "definitions", "usage_examples"])
def test_plan32_literal_target_contract_rejects_dropped_fields(field):
    value = _target()
    value["context_analysis"][field] = "must not be retained"
    with pytest.raises(ValueError, match="unexpected top-level context fields"):
        _target_contract(json.dumps(value))


def test_plan32_literal_target_contract_requires_exact_messages_array():
    value = _target()
    value["context_analysis"]["active_conversation"]["participants"] = [
        {"name": "Speaker", "role": "participant", "recent_messages": "not an array"}
    ]
    with pytest.raises(ValueError, match="participant literal fields changed"):
        _target_contract(json.dumps(value))


def test_plan32_validation_screenshots_must_match_frozen_source_hash(tmp_path):
    image = tmp_path / "screen.png"
    image.write_bytes(b"frozen screenshot bytes")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    rows = [{"id": "row-1", "image_path": "screen.png"}]
    _verify_validation_image_hashes(tmp_path, rows, {"row-1": digest})

    image.write_bytes(b"changed screenshot bytes")
    with pytest.raises(ValueError, match="validation screenshot differs"):
        _verify_validation_image_hashes(tmp_path, rows, {"row-1": digest})
