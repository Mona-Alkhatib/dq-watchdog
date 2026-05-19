from typer.testing import CliRunner

from watchdog.cli import app

runner = CliRunner()


def test_cli_help_lists_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("generate", "train", "eval", "serve", "drift"):
        assert cmd in result.stdout.lower()
