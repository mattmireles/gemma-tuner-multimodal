import io
import json
from pathlib import Path

from whisper_tuner.utils.gemma_dataset_prep import prepare_gemma_jsonl, _build_messages


def test_prepare_gemma_jsonl_writes_records(tmp_path: Path):
    # Create a tiny CSV
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("audio_path,text\n/path/a.wav,hello\n/path/b.wav,world\n")
    out_path = tmp_path / "train_gemma.jsonl"

    rc = prepare_gemma_jsonl(str(csv_path), str(out_path), text_column="text")
    assert rc == 0
    assert out_path.exists()

    lines = out_path.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert set(first.keys()) == {"audio_path", "messages"}
    assert first["audio_path"] == "/path/a.wav"
    # Message structure
    assert isinstance(first["messages"], list) and len(first["messages"]) == 2
    assert first["messages"][0]["role"] == "user"
    assert first["messages"][1]["role"] == "assistant"


def test_build_messages_shape():
    msgs = _build_messages("hi")
    assert isinstance(msgs, list) and len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert any(c.get("type") == "audio" for c in msgs[0]["content"])