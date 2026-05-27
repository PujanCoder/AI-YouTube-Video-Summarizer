"""
cache_manager.py
================
Handles two independent persistence layers:

1. Result Cache  – JSON files keyed by video_id + model + length.
                   Prevents redundant API calls for the same video/settings.

2. Summary History – A JSONL log of past summaries shown in the sidebar.
                     Keeps the last 50 entries.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime

# ── Directory layout ──────────────────────────────
_BASE_DIR      = Path("data")
_CACHE_DIR     = _BASE_DIR / "cache"
_HISTORY_FILE  = _BASE_DIR / "history.jsonl"
_MAX_HISTORY   = 50        # Maximum history entries kept on disk

# Ensure directories exist at import time
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_BASE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────
def _cache_path(cache_key: str) -> Path:
    """
    Return the file path for a given cache key.

    We hash the key so the filename is always filesystem-safe
    regardless of what characters the key contains.
    """
    hashed = hashlib.md5(cache_key.encode()).hexdigest()
    return _CACHE_DIR / f"{hashed}.json"


# ─────────────────────────────────────────────────
# Result Cache API
# ─────────────────────────────────────────────────
def get_cached_result(cache_key: str) -> dict | None:
    """
    Load a previously saved summarisation result from disk.

    Args:
        cache_key: Composite key string (video_id + model + length).

    Returns:
        The cached dict (with 'summaries' and 'transcript' keys),
        or None if no cache entry exists.
    """
    path = _cache_path(cache_key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupt file – treat as cache miss
        return None


def save_to_cache(cache_key: str, data: dict) -> None:
    """
    Persist a summarisation result to the local cache.

    Args:
        cache_key: Composite key string.
        data:      Dict to serialize (must be JSON-serialisable).
    """
    path = _cache_path(cache_key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass  # Non-fatal – just skip caching


def clear_cache() -> int:
    """
    Delete all cached result files.

    Returns:
        Number of files deleted.
    """
    deleted = 0
    for f in _CACHE_DIR.glob("*.json"):
        try:
            f.unlink()
            deleted += 1
        except OSError:
            pass
    return deleted


def get_cache_size() -> int:
    """Return the number of cached result files."""
    return len(list(_CACHE_DIR.glob("*.json")))


# ─────────────────────────────────────────────────
# Summary History API
# ─────────────────────────────────────────────────
def load_history() -> list[dict]:
    """
    Load all history entries from the JSONL log file.

    Returns:
        List of history entry dicts, oldest first.
        Empty list if the file doesn't exist or is unreadable.
    """
    if not _HISTORY_FILE.exists():
        return []
    entries = []
    try:
        with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines
    except OSError:
        return []
    return entries


def save_history_entry(entry: dict) -> None:
    """
    Append one entry to the history log, then trim to _MAX_HISTORY.

    If the same video_id already exists, the old entry is replaced
    (update-in-place semantics) to avoid duplicates.

    Args:
        entry: Dict with at minimum keys: video_id, url, title, timestamp.
    """
    existing  = load_history()
    video_id  = entry.get("video_id", "")

    # Remove any pre-existing entry for the same video
    existing  = [e for e in existing if e.get("video_id") != video_id]

    # Append the new entry
    existing.append(entry)

    # Keep only the most recent _MAX_HISTORY entries
    trimmed   = existing[-_MAX_HISTORY:]

    # Rewrite the file
    try:
        with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
            for e in trimmed:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    except OSError:
        pass  # Non-fatal


def delete_history_entry(video_id: str) -> bool:
    """
    Remove a specific entry from the history log by video_id.

    Args:
        video_id: 11-character YouTube video ID.

    Returns:
        True if an entry was found and removed, False otherwise.
    """
    existing = load_history()
    filtered = [e for e in existing if e.get("video_id") != video_id]
    if len(filtered) == len(existing):
        return False  # Nothing was removed

    try:
        with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
            for e in filtered:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return True
    except OSError:
        return False


def clear_history() -> None:
    """Delete the entire history log file."""
    try:
        _HISTORY_FILE.unlink(missing_ok=True)
    except OSError:
        pass
