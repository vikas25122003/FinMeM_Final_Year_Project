"""
Objective 2 — Step 1: Reflection Logger

Appends a JSONL record for every trading reflection to
./logs/reflections/YYYY-MM-DD.jsonl. These logs become the
training data for the importance classifier.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Default log directory (overridable via env var)
_LOG_DIR = os.getenv("REFLECTION_LOG_DIR", "./logs/reflections")


def _ensure_log_dir(log_dir: Optional[str] = None) -> str:
    """Create the reflection log directory if it doesn't exist."""
    d = log_dir or _LOG_DIR
    os.makedirs(d, exist_ok=True)
    return d


def log_reflection(
    date: str,
    ticker: str,
    decision: str,
    memory_ids_used: List[int],
    rationale: str,
    cumulative_return: float = 0.0,
    log_dir: Optional[str] = None,
) -> None:
    """Append a single reflection record to the daily JSONL file.

    Args:
        date:              Trading date (YYYY-MM-DD).
        ticker:            Stock ticker symbol.
        decision:          BUY / SELL / HOLD.
        memory_ids_used:   List of memory IDs the LLM used.
        rationale:         LLM's explanation text.
        cumulative_return: Portfolio cumulative return at decision time.
        log_dir:           Override log directory (default: from env).
    """
    d = _ensure_log_dir(log_dir)

    record = {
        "date": str(date),
        "ticker": ticker.upper(),
        "decision": decision.upper(),
        "memory_ids_used": [int(mid) for mid in (memory_ids_used or [])],
        "rationale": str(rationale)[:2000],  # Cap to 2KB
        "cumulative_return": float(cumulative_return),
        "logged_at": datetime.utcnow().isoformat() + "Z",
    }

    filename = f"{date}.jsonl"
    filepath = os.path.join(d, filename)

    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.debug(f"[Obj2] Logged reflection for {ticker} on {date}")
    except Exception as exc:
        logger.warning(f"[Obj2] Failed to log reflection: {exc}")


def load_all_reflections(log_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read all JSONL files in the log directory and return a flat list.

    Returns:
        List of reflection dictionaries, sorted by date.
    """
    d = _ensure_log_dir(log_dir)
    reflections: List[Dict[str, Any]] = []

    for filename in sorted(os.listdir(d)):
        if not filename.endswith(".jsonl"):
            continue
        filepath = os.path.join(d, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        reflections.append(json.loads(line))
        except Exception as exc:
            logger.warning(f"[Obj2] Error reading {filepath}: {exc}")

    logger.info(f"[Obj2] Loaded {len(reflections)} total reflection records")
    return reflections
