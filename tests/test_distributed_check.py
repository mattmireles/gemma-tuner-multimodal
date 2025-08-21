from typer.testing import CliRunner

from cli_typer import app


def test_distributed_check_help():
    runner = CliRunner()
    result = runner.invoke(app, ["distributed-check", "--help"])
    assert result.exit_code == 0
    # Typer rich help may not include docstring text; ensure command name is present
    assert "distributed-check" in result.stdout
