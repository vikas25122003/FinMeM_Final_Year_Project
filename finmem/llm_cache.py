"""
LLM Response Cache — Disk-based caching for LLM API calls.

Caches LLM responses to disk based on prompt hash.
Same prompt + same model → cached response (zero API calls, zero cost).

Enable:  LLM_CACHE=true  (default: true)
Disable: LLM_CACHE=false
Dir:     LLM_CACHE_DIR=./cache/llm (default)
"""

import hashlib
import json
import os
import logging
from typing import Optional, List, Any

logger = logging.getLogger(__name__)

CACHE_DIR = os.getenv("LLM_CACHE_DIR", "./cache/llm")
_cache_enabled = None  # lazy init


def _is_enabled() -> bool:
    """Check if caching is enabled (cached for performance)."""
    global _cache_enabled
    if _cache_enabled is None:
        _cache_enabled = os.getenv("LLM_CACHE", "true").lower().strip() in ("true", "1", "yes")
        if _cache_enabled:
            logger.info(f"[LLM Cache] ENABLED — dir: {CACHE_DIR}")
        else:
            logger.info("[LLM Cache] DISABLED")
    return _cache_enabled


def cache_key(messages: List[Any], model_id: str) -> str:
    """Generate a deterministic cache key from messages + model."""
    # Normalize messages to a stable JSON representation
    normalized = []
    for m in messages:
        if hasattr(m, "role") and hasattr(m, "content"):
            normalized.append({"r": m.role, "c": m.content})
        elif isinstance(m, dict):
            normalized.append({"r": m.get("role", ""), "c": m.get("content", "")})
        else:
            normalized.append({"r": "user", "c": str(m)})

    raw = json.dumps(normalized, sort_keys=True, ensure_ascii=False) + "|" + model_id
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def read_cache(key: str) -> Optional[str]:
    """Read cached response. Returns None on miss."""
    if not _is_enabled():
        return None

    path = os.path.join(CACHE_DIR, f"{key}.txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"[LLM Cache] HIT {key[:8]}...")
            return content
        except Exception as e:
            logger.warning(f"[LLM Cache] Read error: {e}")
    return None


def write_cache(key: str, response: str) -> None:
    """Write response to cache."""
    if not _is_enabled():
        return

    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, f"{key}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(response)
        logger.debug(f"[LLM Cache] STORED {key[:8]}...")
    except Exception as e:
        logger.warning(f"[LLM Cache] Write error: {e}")


def cache_stats() -> dict:
    """Get cache statistics."""
    if not os.path.exists(CACHE_DIR):
        return {"entries": 0, "size_mb": 0.0}

    entries = [f for f in os.listdir(CACHE_DIR) if f.endswith(".txt")]
    total_size = sum(
        os.path.getsize(os.path.join(CACHE_DIR, f)) for f in entries
    )
    return {
        "entries": len(entries),
        "size_mb": round(total_size / (1024 * 1024), 2),
    }
