from __future__ import annotations

import logging
from pathlib import Path

from insights.config import require_env

logger = logging.getLogger(__name__)


def download_youtube_audio(*, video_id: str, cache_dir: Path, refresh: bool) -> tuple[Path, str | None]:
    """
    Download best available audio for a YouTube video into cache_dir.

    Returns (audio_path, title).
    """
    target_dir = (cache_dir / "youtube" / video_id).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if not refresh:
        existing = sorted(target_dir.glob(f"{video_id}.*"))
        if existing:
            # Prefer common audio extensions.
            for ext in (".m4a", ".mp3", ".wav", ".mp4", ".aac", ".flac", ".ogg", ".webm"):
                for p in existing:
                    if p.suffix.lower() == ext:
                        return p, None
            return existing[0], None

    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        from yt_dlp import YoutubeDL  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("yt-dlp is required for YouTube ingestion. Install dependencies with `uv sync`.") from e

    # Prefer m4a (often accepted by STT services), fall back to bestaudio.
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(target_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title") if isinstance(info, dict) else None
        filename = ydl.prepare_filename(info)
        path = Path(filename).resolve()
        if not path.exists():
            # Some versions may return a different path; fallback to scanning.
            existing = sorted(target_dir.glob(f"{video_id}.*"))
            if not existing:
                raise RuntimeError("yt-dlp reported success but audio file was not found in cache dir")
            path = existing[0].resolve()
        return path, str(title) if title else None


def transcribe_with_assemblyai(*, audio_path: Path) -> str:
    """
    Transcribe an audio file using AssemblyAI.
    """
    api_key = require_env("ASSEMBLYAI_API_KEY")
    try:
        import assemblyai as aai  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "assemblyai package is required for transcription. Install dependencies with `uv sync`."
        ) from e

    aai.settings.api_key = api_key
    # Keep config conservative; user can extend later.
    config = aai.TranscriptionConfig(speech_models=["universal"])
    transcript = aai.Transcriber(config=config).transcribe(str(audio_path))
    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")
    if not transcript.text:
        raise RuntimeError("Transcription succeeded but returned empty text")
    return transcript.text


