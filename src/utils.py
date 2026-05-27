"""
utils.py
========
Miscellaneous helper functions used across the application.
"""

import re
import requests


def get_youtube_thumbnail(video_id: str | None) -> str | None:
    """
    Return the highest-resolution thumbnail URL available for a video.

    Tries resolutions in descending order:
        maxresdefault → sddefault → hqdefault

    Args:
        video_id: 11-character YouTube video ID, or None.

    Returns:
        URL string of the best available thumbnail, or None on failure.
    """
    if not video_id:
        return None

    resolutions = ["maxresdefault", "sddefault", "hqdefault"]
    for res in resolutions:
        url = f"https://img.youtube.com/vi/{video_id}/{res}.jpg"
        try:
            resp = requests.head(url, timeout=4)
            if resp.status_code == 200:
                return url
        except requests.RequestException:
            continue
    return None


def format_seconds(total_seconds: float) -> str:
    """
    Convert a float number of seconds to a human-readable MM:SS string.

    Args:
        total_seconds: Duration in seconds (may be float).

    Returns:
        Formatted string like '04:32'.

    Examples:
        >>> format_seconds(272.4)
        '04:32'
        >>> format_seconds(3661)
        '61:01'
    """
    total_seconds = int(total_seconds)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def sanitize_filename(name: str) -> str:
    """
    Strip characters that are illegal or problematic in file names.

    Replaces any character that is not alphanumeric, a hyphen,
    or an underscore with an underscore.

    Args:
        name: Raw string to sanitize.

    Returns:
        Safe filename-compatible string (max 80 chars).

    Examples:
        >>> sanitize_filename("Hello World! (2024)")
        'Hello_World__2024_'
    """
    sanitized = re.sub(r"[^\w\-]", "_", name)
    return sanitized[:80]


def truncate_text(text: str, max_chars: int = 4000, suffix: str = "…") -> str:
    """
    Truncate text to a maximum character count, appending a suffix.

    Args:
        text:      Input text string.
        max_chars: Maximum number of characters to keep.
        suffix:    String appended when text is truncated.

    Returns:
        Original text (if short enough) or truncated version.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def word_count(text: str) -> int:
    """Return the approximate word count of a string."""
    return len(text.split())


def estimate_reading_time(text: str, wpm: int = 200) -> str:
    """
    Estimate how long it would take to read a block of text.

    Args:
        text: Input string.
        wpm:  Words per minute reading speed (default 200).

    Returns:
        Human-readable string like '3 min read'.
    """
    words   = word_count(text)
    minutes = max(1, round(words / wpm))
    return f"{minutes} min read"
