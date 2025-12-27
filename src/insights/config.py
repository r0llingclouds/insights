from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Paths:
    app_dir: Path
    db_path: Path
    cache_dir: Path


@dataclass(frozen=True, slots=True)
class DefaultsConfig:
    provider: str = "anthropic"
    model: str | None = None
    max_context_tokens: int = 12000
    max_output_tokens: int = 800
    temperature: float = 0.2
    stream: bool = True


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "claude-sonnet-4-5-20250929"
    max_steps: int = 10


@dataclass(frozen=True, slots=True)
class AppConfig:
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


def _parse_defaults(data: dict[str, Any]) -> DefaultsConfig:
    return DefaultsConfig(
        provider=data.get("provider", "anthropic"),
        model=data.get("model"),
        max_context_tokens=data.get("max_context_tokens", 12000),
        max_output_tokens=data.get("max_output_tokens", 800),
        temperature=data.get("temperature", 0.2),
        stream=data.get("stream", True),
    )


def _parse_agent(data: dict[str, Any]) -> AgentConfig:
    return AgentConfig(
        model=data.get("model", "claude-sonnet-4-5-20250929"),
        max_steps=data.get("max_steps", 10),
    )


def load_config(app_dir: Path | None = None) -> AppConfig:
    """
    Load configuration from TOML files.

    Search order (later overrides earlier):
    1. ~/.config/insights/config.toml
    2. <app_dir>/config.toml
    3. ./insights.toml
    """
    config_paths = [
        Path.home() / ".config" / "insights" / "config.toml",
        (app_dir or default_app_dir()) / "config.toml",
        Path.cwd() / "insights.toml",
    ]

    merged: dict[str, Any] = {"defaults": {}, "agent": {}}

    for path in config_paths:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = tomllib.load(f)
                if "defaults" in data:
                    merged["defaults"].update(data["defaults"])
                if "agent" in data:
                    merged["agent"].update(data["agent"])
            except Exception:
                # Skip invalid config files
                pass

    return AppConfig(
        defaults=_parse_defaults(merged.get("defaults", {})),
        agent=_parse_agent(merged.get("agent", {})),
    )


def default_app_dir() -> Path:
    # Keep things simple: default to a stable, user-visible location.
    return Path.home() / "Documents" / "insights"


def ensure_dirs(paths: Paths) -> None:
    paths.app_dir.mkdir(parents=True, exist_ok=True)
    paths.cache_dir.mkdir(parents=True, exist_ok=True)


def load_env(app_dir: Path) -> None:
    """
    Load environment variables from:
    - cwd/.env
    - ~/Documents/insights/.env (or the resolved app_dir)
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


