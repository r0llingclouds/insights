from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from insights.storage.models import SourceKind


@dataclass(frozen=True, slots=True)
class DetectedSource:
    kind: SourceKind
    locator: str
    display_title: str | None


def is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    # Remove a single trailing slash for consistency (except root).
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    # Keep query because it may materially change content.
    query = parsed.query
    return f"{scheme}://{netloc}{path}" + (f"?{query}" if query else "")


def extract_youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""

    if host in {"youtu.be"}:
        vid = path.lstrip("/").split("/")[0]
        return vid or None

    if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        if path == "/watch":
            qs = parse_qs(parsed.query)
            vid = (qs.get("v") or [None])[0]
            return vid
        # shorts, live, embed
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"shorts", "live", "embed"}:
            return parts[1]
    return None


def detect_source(value: str, *, forced_type: str | None = None) -> DetectedSource:
    """
    Detect a source kind and stable locator.

    forced_type:
      - None or 'auto'
      - 'file' | 'url' | 'youtube'
    """
    ft = (forced_type or "auto").lower()
    if ft not in {"auto", "file", "url", "youtube"}:
        raise ValueError("forced_type must be one of: auto, file, url, youtube")

    if ft in {"url", "youtube"} or (ft == "auto" and is_url(value)):
        url = canonicalize_url(value)
        if ft == "youtube" or (ft == "auto" and extract_youtube_video_id(url)):
            vid = extract_youtube_video_id(url)
            if not vid:
                raise ValueError("Could not parse YouTube video id from URL")
            return DetectedSource(kind=SourceKind.YOUTUBE, locator=vid, display_title=None)
        return DetectedSource(kind=SourceKind.URL, locator=url, display_title=None)

    # file
    path = Path(value).expanduser().resolve()
    return DetectedSource(kind=SourceKind.FILE, locator=str(path), display_title=path.name)


