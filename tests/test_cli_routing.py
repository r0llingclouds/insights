from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner


def test_root_routes_quoted_query_to_do_help(tmp_path: Path) -> None:
    # Importing CLI requires optional deps (dotenv, typer, etc.) which are present in the app env.
    from insights.cli import app

    runner = CliRunner()
    # First arg includes spaces (how a quoted shell string arrives).
    res = runner.invoke(
        app,
        ["--app-dir", str(tmp_path), "chat on 2e999822e29a431ea789221123fd3425", "--help"],
    )
    assert res.exit_code == 0
    # Help should be for `do` (natural-language agent), not "No such command".
    assert "No such command" not in res.output
    assert "Natural-language query" in res.output or "natural-language" in res.output.lower()


