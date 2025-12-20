from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Paths:
    app_dir: Path
    db_path: Path
    cache_dir: Path


def default_app_dir() -> Path:
    return Path.home() / ".insights"


def ensure_dirs(paths: Paths) -> None:
    paths.app_dir.mkdir(parents=True, exist_ok=True)
    paths.cache_dir.mkdir(parents=True, exist_ok=True)


def load_env(app_dir: Path) -> None:
    """
    Load environment variables from:
    - cwd/.env
    - ~/.insights/.env
    """
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    load_dotenv(dotenv_path=app_dir / ".env", override=False)


def resolve_paths(db: Path | None = None, app_dir: Path | None = None) -> Paths:
    resolved_app_dir = (app_dir or default_app_dir()).expanduser().resolve()
    resolved_db = (db or (resolved_app_dir / "insights.db")).expanduser().resolve()
    cache_dir = resolved_app_dir / "cache"
    paths = Paths(app_dir=resolved_app_dir, db_path=resolved_db, cache_dir=cache_dir)
    ensure_dirs(paths)
    load_env(resolved_app_dir)
    return paths


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is required")
    return value


