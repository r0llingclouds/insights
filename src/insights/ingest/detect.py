from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from insights.ingest.tweet_extractor import is_tweet_url
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
    scheme = (parsed.scheme or "").lower()
    if not scheme:
        raise ValueError("URL must include a scheme (http/https)")

    # Normalize internationalized domains to ASCII (punycode) so:
    # - https://マリウス.com/... and https://xn--gckvb8fzb.com/... map to the same locator.
    host = (parsed.hostname or "").rstrip(".")
    if not host:
        host = parsed.netloc.rstrip(".")
    try:
        host_ascii = host.encode("idna").decode("ascii")
    except UnicodeError:
        host_ascii = host
    host_ascii = host_ascii.lower()

    port = parsed.port
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    if port and not default_port:
        netloc = f"{host_ascii}:{port}"
    else:
        netloc = host_ascii

    path = parsed.path or "/"
    # Remove a single trailing slash for consistency (except root).
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    # Keep query because it may materially change content.
    query = parsed.query
    return f"{scheme}://{netloc}{path}" + (f"?{query}" if query else "")


def is_linkedin_url(url: str) -> bool:
    """Check if URL is a LinkedIn post/article URL."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return host in {"linkedin.com", "www.linkedin.com"} or host.endswith(".linkedin.com")


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
      - 'file' | 'url' | 'youtube' | 'tweet' | 'linkedin'
    """
    ft = (forced_type or "auto").lower()
    if ft not in {"auto", "file", "url", "youtube", "tweet", "linkedin"}:
        raise ValueError("forced_type must be one of: auto, file, url, youtube, tweet, linkedin")

    if ft in {"url", "youtube", "tweet", "linkedin"} or (ft == "auto" and is_url(value)):
        url = canonicalize_url(value)
        # Check for tweet first (before generic URL)
        if ft == "tweet" or (ft == "auto" and is_tweet_url(url)):
            if not is_tweet_url(url):
                raise ValueError("Could not parse tweet URL")
            return DetectedSource(kind=SourceKind.TWEET, locator=url, display_title=None)
        # Check for LinkedIn (before generic URL)
        if ft == "linkedin" or (ft == "auto" and is_linkedin_url(url)):
            if not is_linkedin_url(url):
                raise ValueError("Could not parse LinkedIn URL")
            return DetectedSource(kind=SourceKind.LINKEDIN, locator=url, display_title=None)
        if ft == "youtube" or (ft == "auto" and extract_youtube_video_id(url)):
            vid = extract_youtube_video_id(url)
            if not vid:
                raise ValueError("Could not parse YouTube video id from URL")
            return DetectedSource(kind=SourceKind.YOUTUBE, locator=vid, display_title=None)
        return DetectedSource(kind=SourceKind.URL, locator=url, display_title=None)

    # file
    path = Path(value).expanduser().resolve()
    return DetectedSource(kind=SourceKind.FILE, locator=str(path), display_title=path.name)


