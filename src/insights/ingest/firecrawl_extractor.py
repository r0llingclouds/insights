from __future__ import annotations

from typing import Any

from insights.config import require_env


def _extract_markdown(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    md = getattr(value, "markdown", None)
    if isinstance(md, str) and md.strip():
        return md
    if isinstance(value, dict):
        if isinstance(value.get("markdown"), str):
            return value["markdown"]
        data = value.get("data")
        if isinstance(data, dict) and isinstance(data.get("markdown"), str):
            return data["markdown"]
        # Sometimes libraries wrap in {"data": {"content": "..."}}
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            return data["content"]
    return None


def extract_markdown_with_firecrawl(url: str) -> str:
    """
    Convert a URL to Markdown using Firecrawl.

    Supports both older and newer Firecrawl Python clients by trying multiple
    imports/call patterns.
    """
    api_key = require_env("FIRECRAWL_API_KEY")

    constructors: list[tuple[str, Any]] = []
    try:
        from firecrawl import FirecrawlApp  # type: ignore

        constructors.append(("FirecrawlApp", FirecrawlApp))
    except ImportError:
        pass
    try:
        from firecrawl import Firecrawl  # type: ignore

        constructors.append(("Firecrawl", Firecrawl))
    except ImportError:
        pass

    if not constructors:
        raise RuntimeError(
            "Firecrawl backend requires the `firecrawl` Python package. "
            "Install it (e.g. `uv sync`) and set FIRECRAWL_API_KEY."
        )

    last_error: Exception | None = None
    for name, ctor in constructors:
        # Instantiate client (support both ctor(api_key=...) and ctor(api_key)).
        try:
            client = ctor(api_key=api_key)
        except TypeError:
            try:
                client = ctor(api_key)
            except Exception as e:
                last_error = e
                continue
        except Exception as e:
            last_error = e
            continue

        # Method differences across versions:
        # - newer docs often show scrape_url(...)
        # - older versions may only expose scrape(...)
        methods: list[tuple[str, Any]] = []
        if hasattr(client, "scrape_url"):
            methods.append(("scrape_url", getattr(client, "scrape_url")))
        if hasattr(client, "scrape"):
            methods.append(("scrape", getattr(client, "scrape")))
        if not methods:
            last_error = AttributeError(f"{name} client has no scrape_url() or scrape() method")
            continue

        # Signature differences: some accept params={...}, some accept formats=[...].
        call_variants = [
            {"params": {"formats": ["markdown"]}},
            {"formats": ["markdown"]},
            # Some clients support these options; harmless if they don't (TypeError caught).
            {"params": {"formats": ["markdown"], "onlyMainContent": True}},
            {"formats": ["markdown"], "onlyMainContent": True},
        ]

        for method_name, method in methods:
            for kwargs in call_variants:
                try:
                    result = method(url, **kwargs)
                except TypeError:
                    continue
                except Exception as e:
                    last_error = e
                    continue

                md = _extract_markdown(result)
                if md:
                    return md

            last_error = RuntimeError(f"Firecrawl {method_name}() returned no markdown content")

    raise RuntimeError(f"Firecrawl extraction failed: {last_error}") from last_error


