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


def test_sources_json_includes_token_count(tmp_path: Path) -> None:
    from insights.cli import app
    from insights.storage.db import Database
    from insights.storage.models import SourceKind
    from insights.utils.tokens import estimate_tokens

    # Create one cached source+document in the tmp app dir DB.
    db = Database.open(tmp_path / "insights.db")
    try:
        s = db.upsert_source(kind=SourceKind.FILE, locator=str(tmp_path / "a.txt"), title="a")
        sv = db.create_source_version(
            source_id=s.id,
            content_hash="deadbeef",
            extractor="docling",
            status="ok",
            error=None,
        )
        text = "hello world"
        db.upsert_document(
            source_version_id=sv.id,
            markdown=text,
            plain_text=text,
            token_count=estimate_tokens(text),
        )
    finally:
        db.close()

    runner = CliRunner()
    res = runner.invoke(app, ["--app-dir", str(tmp_path), "sources", "--json"])
    assert res.exit_code == 0
    # Rich prints a Python-ish structure; just assert the key exists.
    assert "token_count" in res.output


