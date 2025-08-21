from typer.testing import CliRunner

from cli_typer import app


def test_distributed_train_dry_run():
    runner = CliRunner()
    res = runner.invoke(app, ["distributed-train", "--dry-run"])  # Should not import distributed.*
    assert res.exit_code == 0
    assert "Dry run" in res.stdout
    assert "step 10/10" in res.stdout
