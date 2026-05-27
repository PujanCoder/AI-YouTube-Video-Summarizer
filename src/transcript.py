"""
transcript.py
=============
Handles all YouTube transcript extraction logic.

Supports:
- Multiple URL formats (watch, youtu.be, embed, shorts)
- Multi-language fallback
- Timestamps preservation
- Graceful error messages
"""

import re
from typing import Optional
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from youtube_transcript_api.formatters import TextFormatter


# ── Regex patterns for all common YouTube URL formats ──
_YT_URL_PATTERNS = [
    r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})",
    r"^([a-zA-Z0-9_-]{11})$",  # bare video ID
]


def extract_video_id(url: str) -> str:
    """
    Extract the 11-character YouTube video ID from any supported URL format.

    Supported formats:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - https://www.youtube.com/shorts/VIDEO_ID
        - VIDEO_ID (bare)

    Raises:
        ValueError: If no valid video ID is found.
    """
    url = url.strip()
    for pattern in _YT_URL_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(
        "❌ Could not extract a video ID from the URL. "
        "Please paste a valid YouTube link (e.g. https://www.youtube.com/watch?v=...)."
    )


def get_transcript(video_id: str, preferred_language: str = "en") -> tuple[list[dict], str]:
    """
    Fetch the transcript for a YouTube video.

    Strategy:
        1. Try the preferred language.
        2. Fall back to any available manually-created transcript.
        3. Fall back to any auto-generated transcript.

    Args:
        video_id:           11-character YouTube video ID.
        preferred_language: ISO 639-1 language code (default 'en').

    Returns:
        A tuple of:
            - transcript_raw (list of dicts with keys: text, start, duration)
            - transcript_text (plain string, joined sentences)

    Raises:
        RuntimeError: On API errors or missing transcripts.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # ── 1. Try preferred language (manual first, then generated) ──
        try:
            transcript_obj = transcript_list.find_transcript([preferred_language])
        except NoTranscriptFound:
            # ── 2. Fall back to any manual transcript ──
            try:
                transcript_obj = next(
                    t for t in transcript_list if not t.is_generated
                )
            except StopIteration:
                # ── 3. Fall back to any auto-generated ──
                transcript_obj = next(iter(transcript_list))

        # If the found transcript isn't in the preferred language, translate
        if transcript_obj.language_code != preferred_language:
            try:
                transcript_obj = transcript_obj.translate(preferred_language)
            except Exception:
                pass  # Use original language if translation fails

        transcript_raw  = transcript_obj.fetch()
        transcript_text = format_transcript_text(transcript_raw)
        return transcript_raw, transcript_text

    except TranscriptsDisabled:
        raise RuntimeError(
            "❌ Transcripts are disabled for this video. "
            "The creator has turned off captions."
        )
    except VideoUnavailable:
        raise RuntimeError(
            "❌ This video is unavailable (private, deleted, or age-restricted)."
        )
    except Exception as e:
        raise RuntimeError(f"❌ Could not fetch transcript: {e}")


def format_transcript_text(transcript_raw: list[dict]) -> str:
    """
    Convert raw transcript segments into a single clean string.

    Args:
        transcript_raw: List of {'text': str, 'start': float, 'duration': float}.

    Returns:
        A single string with all segments joined by spaces.
    """
    segments = [seg.get("text", "").strip() for seg in transcript_raw]
    # Remove music / inaudible markers
    cleaned = [s for s in segments if s and not s.startswith("[")]
    return " ".join(cleaned)


def get_transcript_with_timestamps(transcript_raw: list[dict], interval_secs: int = 60) -> list[dict]:
    """
    Group transcript segments into minute-level chunks with timestamps.

    Useful for building a rough chapter outline from the raw transcript.

    Args:
        transcript_raw:  Raw segments.
        interval_secs:   How many seconds per chunk (default 60).

    Returns:
        List of {'timestamp': 'MM:SS', 'text': str} dicts.
    """
    chunks = []
    current_chunk  = {"start": 0, "text": []}

    for seg in transcript_raw:
        start = seg.get("start", 0)
        text  = seg.get("text", "").strip()
        if not text or text.startswith("["):
            continue

        if start - current_chunk["start"] >= interval_secs and current_chunk["text"]:
            minutes  = int(current_chunk["start"]) // 60
            seconds  = int(current_chunk["start"]) % 60
            timestamp = f"{minutes:02d}:{seconds:02d}"
            chunks.append({"timestamp": timestamp, "text": " ".join(current_chunk["text"])})
            current_chunk = {"start": start, "text": [text]}
        else:
            current_chunk["text"].append(text)

    # Append final chunk
    if current_chunk["text"]:
        minutes   = int(current_chunk["start"]) // 60
        seconds   = int(current_chunk["start"]) % 60
        timestamp = f"{minutes:02d}:{seconds:02d}"
        chunks.append({"timestamp": timestamp, "text": " ".join(current_chunk["text"])})

    return chunks
