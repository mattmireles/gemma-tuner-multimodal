from typer.testing import CliRunner

from whisper_tuner.cli_typer import app


def test_system_check_help_and_run():
    runner = CliRunner()
    # Help
    res = runner.invoke(app, ["system-check", "--help"])  # Typer prints help
    assert res.exit_code == 0
    assert "System Check" in res.stdout or "system-check" in res.stdout

    # Run (should not error even if torch missing in some envs)
    res = runner.invoke(app, ["system-check"])  # Executes diagnostics
    assert res.exit_code == 0
    assert "✅ system-check completed" in res.stdout
