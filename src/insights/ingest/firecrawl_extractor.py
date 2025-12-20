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

    # Newer client (common in `firecrawl-py`): FirecrawlApp.scrape_url(...)
    try:
        from firecrawl import FirecrawlApp  # type: ignore

        app = FirecrawlApp(api_key=api_key)
        try:
            result = app.scrape_url(url, params={"formats": ["markdown"]})
        except TypeError:
            # Some versions accept formats directly.
            result = app.scrape_url(url, formats=["markdown"])
        md = _extract_markdown(result)
        if md:
            return md
        raise RuntimeError("Firecrawl returned no markdown content")
    except ImportError:
        pass

    # Older client / alternative wrapper: Firecrawl.scrape(...)
    try:
        from firecrawl import Firecrawl  # type: ignore

        client = Firecrawl(api_key=api_key)
        try:
            result = client.scrape(url, formats=["markdown"])
        except TypeError:
            result = client.scrape(url, params={"formats": ["markdown"]})
        md = _extract_markdown(result)
        if md:
            return md
        raise RuntimeError("Firecrawl returned no markdown content")
    except ImportError as e:
        raise RuntimeError(
            "Firecrawl backend requires the `firecrawl` Python package. "
            "Install it (e.g. `uv sync`) and set FIRECRAWL_API_KEY."
        ) from e


