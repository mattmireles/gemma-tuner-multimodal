import json
import os
import tempfile

from core.runs import (
    get_next_run_id,
    create_run_directory,
    mark_run_as_completed,
    RunConstants,
)


def test_get_next_run_id_and_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        a = get_next_run_id(tmp)
        b = get_next_run_id(tmp)
        assert b == a + 1


def test_create_run_directory_and_completion_marker():
    with tempfile.TemporaryDirectory() as tmp:
        run_id = 7
        run_dir = create_run_directory(
            output_dir=tmp,
            profile_name="profileA",
            run_id=run_id,
            run_type=RunConstants.RUN_TYPE_FINETUNING,
        )
        assert os.path.isdir(run_dir)

        meta_path = os.path.join(run_dir, RunConstants.METADATA_FILE)
        assert os.path.isfile(meta_path)

        # Mark as completed writes marker and updates metadata
        mark_run_as_completed(run_dir)
        assert os.path.isfile(os.path.join(run_dir, RunConstants.COMPLETION_MARKER))

        with open(meta_path, "r") as f:
            meta = json.load(f)
        assert meta.get("status") == RunConstants.STATUS_COMPLETED
        assert meta.get("end_time") is not None


