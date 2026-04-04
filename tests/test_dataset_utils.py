import os

import pandas as pd

from whisper_tuner.utils.dataset_utils import load_dataset_split


def _write_csv(path, rows):
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def test_patch_precedence_override_then_protect_then_blacklist(tmp_path):
    # Create minimal dataset files
    data_dir = tmp_path / "data" / "datasets" / "toy"
    os.makedirs(data_dir, exist_ok=True)
    _write_csv(
        str(data_dir / "train.csv"),
        [
            {"id": 1, "text": "orig1"},
            {"id": 2, "text": "orig2"},
            {"id": 3, "text": "orig3"},
        ],
    )

    # Fake config.ini pointing to source "toy_source"
    cfg_path = tmp_path / "config.ini"
    cfg_path.write_text("""
[dataset:toy]
source = toy_source
text_column = text
train_split = train
validation_split = validation
max_label_length = 64
max_duration = 30
""")

    # Patches directory
    patches_root = tmp_path / "data_patches" / "toy_source"
    # Override: change id=1 text to fixed1
    _write_csv(
        str(patches_root / "override_text_perfect" / "o.csv"), [{"id": 1, "text": "fixed1", "text_perfect": "fixed1"}]
    )
    # Protect id=2
    _write_csv(str(patches_root / "do_not_blacklist" / "p.csv"), [{"id": 2}])
    # Blacklist id=2 and id=3 (2 should be rescued)
    _write_csv(str(patches_root / "delete" / "b.csv"), [{"id": 2}, {"id": 3}])

    # Monkeypatch cwd so dataset_utils reads our config.ini
    cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        ds, source = load_dataset_split(
            split="train",
            dataset_config={"name": "toy", "text_column": "text"},
            max_samples=None,
            patches_dir="data_patches/",
            streaming_enabled=False,
        )
    finally:
        os.chdir(cwd)

    # Expect id=3 removed; id=2 kept; id=1 text overridden
    ids = set(int(r["id"]) for r in ds)
    assert ids == {1, 2}
    row1 = next(r for r in ds if int(r["id"]) == 1)
    assert row1["text"] == "fixed1"


def test_streaming_blacklist_respected(tmp_path):
    # Prepare tiny dataset
    data_dir = tmp_path / "data" / "datasets" / "toy"
    os.makedirs(data_dir, exist_ok=True)
    _write_csv(
        str(data_dir / "train.csv"),
        [
            {"id": 10, "text": "a"},
            {"id": 11, "text": "b"},
        ],
    )

    # config.ini
    cfg_path = tmp_path / "config.ini"
    cfg_path.write_text("""
[dataset:toy]
source = toy_source
text_column = text
train_split = train
validation_split = validation
max_label_length = 64
max_duration = 30
""")

    # Patches: blacklist id=11
    patches_root = tmp_path / "data_patches" / "toy_source"
    _write_csv(str(patches_root / "delete" / "b.csv"), [{"id": 11}])

    cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        ds, _ = load_dataset_split(
            split="train",
            dataset_config={"name": "toy", "text_column": "text"},
            streaming_enabled=True,
        )
        # Materialize a small number of samples from streaming
        samples = list(ds)
    finally:
        os.chdir(cwd)

    ids = [int(r["id"]) for r in samples]
    assert ids == [10]


def test_granary_dataset_accepts_source_type_without_source(tmp_path):
    data_dir = tmp_path / "data" / "datasets" / "granary-en"
    os.makedirs(data_dir, exist_ok=True)
    _write_csv(
        str(data_dir / "train.csv"),
        [
            {"id": 21, "text": "hello granary"},
        ],
    )

    cfg_path = tmp_path / "config.ini"
    cfg_path.write_text("""
[dataset:granary-en]
source_type = granary
text_column = text
train_split = train
validation_split = validation
max_label_length = 64
max_duration = 30
""")

    cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        ds, source = load_dataset_split(
            split="train",
            dataset_config={"name": "granary-en", "text_column": "text"},
            streaming_enabled=False,
        )
    finally:
        os.chdir(cwd)

    assert source == "granary-en"
    assert len(ds) == 1
    assert ds[0]["text"] == "hello granary"


def test_bigquery_dataset_uses_prepared_fallback_adapter(tmp_path):
    data_dir = tmp_path / "data" / "datasets" / "analytics-en"
    os.makedirs(data_dir, exist_ok=True)
    _write_csv(
        str(data_dir / "analytics-en_prepared.csv"),
        [
            {"id": 31, "text": "hello bigquery"},
        ],
    )

    cfg_path = tmp_path / "config.ini"
    cfg_path.write_text("""
[dataset:analytics-en]
source_type = bigquery
text_column = text
train_split = train
validation_split = validation
max_label_length = 64
max_duration = 30
""")

    cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        ds, source = load_dataset_split(
            split="train",
            dataset_config={"name": "analytics-en", "text_column": "text"},
            streaming_enabled=False,
        )
    finally:
        os.chdir(cwd)

    assert source == "analytics-en"
    assert len(ds) == 1
    assert ds[0]["text"] == "hello bigquery"


def test_patch_ids_are_normalized_consistently(tmp_path):
    data_dir = tmp_path / "data" / "datasets" / "toy"
    os.makedirs(data_dir, exist_ok=True)
    _write_csv(
        str(data_dir / "train.csv"),
        [
            {"id": 2, "text": "keep me"},
            {"id": 3, "text": "remove me"},
        ],
    )

    cfg_path = tmp_path / "config.ini"
    cfg_path.write_text("""
[dataset:toy]
source = toy_source
text_column = text
train_split = train
validation_split = validation
max_label_length = 64
max_duration = 30
""")

    patches_root = tmp_path / "data_patches" / "toy_source"
    _write_csv(str(patches_root / "do_not_blacklist" / "protected.csv"), [{"id": "2.0"}])
    _write_csv(str(patches_root / "delete" / "blacklist.csv"), [{"id": 2}, {"id": "3.0"}])

    cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        ds, _ = load_dataset_split(
            split="train",
            dataset_config={"name": "toy", "text_column": "text"},
            streaming_enabled=False,
        )
    finally:
        os.chdir(cwd)

    ids = sorted(int(row["id"]) for row in ds)
    assert ids == [2]
