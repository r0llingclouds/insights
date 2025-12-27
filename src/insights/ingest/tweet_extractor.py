"""Tweet extraction utilities for Twitter/X posts.

Supports:
- twitterapi.io API (requires TWITTERAPI_KEY env var)
- oEmbed fallback (publish.twitter.com/oembed - free, no auth)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

TWITTERAPI_BASE = "https://api.twitterapi.io"

_TWEET_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?(?:twitter\.com|x\.com)/(?P<user>[^/]+)/status/(?P<id>\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class TweetData:
    """Extracted tweet content and metadata."""

    tweet_id: str
    text: str
    author_name: str
    author_username: str
    created_at: datetime | None
    likes: int | None
    retweets: int | None
    url: str


def extract_tweet_id(url: str) -> str | None:
    """Extract tweet ID from a Twitter/X status URL."""
    match = _TWEET_URL_PATTERN.search(url)
    if match:
        return match.group("id")
    return None


def is_tweet_url(url: str) -> bool:
    """Check if URL is a Twitter/X status URL."""
    return extract_tweet_id(url) is not None


def fetch_tweet_twitterapi(tweet_id: str) -> TweetData | None:
    """Fetch tweet using twitterapi.io API.

    Requires TWITTERAPI_KEY environment variable.
    Pricing: ~$0.15 per 1000 tweets.
    """
    api_key = os.getenv("TWITTERAPI_KEY")
    if not api_key:
        return None

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                f"{TWITTERAPI_BASE}/twitter/tweets",
                params={"tweet_ids": tweet_id},
                headers={"X-API-Key": api_key},
            )
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") != "success" or not data.get("tweets"):
            return None

        tweet = data["tweets"][0]
        author = tweet.get("author", {})

        created_at = None
        if tweet.get("createdAt"):
            try:
                created_at = datetime.fromisoformat(
                    tweet["createdAt"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        username = author.get("userName", "")
        return TweetData(
            tweet_id=tweet_id,
            text=tweet.get("text", ""),
            author_name=author.get("name", ""),
            author_username=username,
            created_at=created_at,
            likes=tweet.get("likeCount"),
            retweets=tweet.get("retweetCount"),
            url=f"https://x.com/{username}/status/{tweet_id}",
        )
    except Exception as e:
        logger.warning("twitterapi.io error: %s", e)
        return None


def fetch_tweet_oembed(url: str) -> TweetData | None:
    """Fetch tweet using Twitter's oEmbed endpoint (free, no auth required).

    Note: oEmbed provides limited metadata (no likes/retweets/created_at).
    """
    tweet_id = extract_tweet_id(url)
    if not tweet_id:
        return None

    oembed_url = f"https://publish.twitter.com/oembed?url={quote(url, safe='')}"

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(oembed_url)
            resp.raise_for_status()
            data = resp.json()

        html = data.get("html", "")
        author_name = data.get("author_name", "")
        author_url = data.get("author_url", "")

        # Extract username from author_url (e.g., https://twitter.com/username)
        author_username = ""
        if author_url:
            parts = author_url.rstrip("/").split("/")
            if parts:
                author_username = parts[-1]

        # Extract text from blockquote HTML
        text = ""
        text_match = re.search(r"<p[^>]*>(.*?)</p>", html, re.DOTALL | re.IGNORECASE)
        if text_match:
            # Remove HTML tags and unescape entities
            raw_text = re.sub(r"<[^>]+>", "", text_match.group(1))
            text = unescape(raw_text).strip()

        return TweetData(
            tweet_id=tweet_id,
            text=text,
            author_name=author_name,
            author_username=author_username,
            created_at=None,
            likes=None,
            retweets=None,
            url=url,
        )
    except Exception as e:
        logger.warning("oEmbed error: %s", e)
        return None


def fetch_tweet(url: str) -> TweetData:
    """Fetch tweet with fallback strategy.

    Tries twitterapi.io first (if TWITTERAPI_KEY is set), falls back to oEmbed.

    Raises:
        RuntimeError: If tweet cannot be fetched from any source.
    """
    tweet_id = extract_tweet_id(url)
    if not tweet_id:
        raise ValueError(f"Could not extract tweet ID from URL: {url}")

    # Try twitterapi.io first if API key is available
    if os.getenv("TWITTERAPI_KEY"):
        logger.info("Fetching tweet via twitterapi.io...")
        data = fetch_tweet_twitterapi(tweet_id)
        if data:
            return data
        logger.warning("twitterapi.io failed, falling back to oEmbed")

    # Fallback to oEmbed
    logger.info("Fetching tweet via oEmbed...")
    data = fetch_tweet_oembed(url)
    if data:
        return data

    raise RuntimeError(f"Failed to fetch tweet: {url}")
