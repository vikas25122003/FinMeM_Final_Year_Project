"""
Objective 2 — Step 2: Weak Label Generator

For each reflection record, fetches next-day return via yfinance
and assigns a binary label:
    next_day_return > +0.5%  →  label = 1  (profitable memory)
    next_day_return < -0.5%  →  label = 0  (unprofitable memory)
    otherwise               →  skipped     (ambiguous)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def generate_labels(
    reflections: List[Dict[str, Any]],
    pos_threshold: Optional[float] = None,
    neg_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Generate weak labels for each reflection record using next-day returns.

    Args:
        reflections:   List of reflection dicts from logger.load_all_reflections().
        pos_threshold: Positive return threshold (default from env: 0.005 = +0.5%).
        neg_threshold: Negative return threshold (default from env: -0.005 = -0.5%).

    Returns:
        List of labelled dicts: {memory_id, ticker, date, label, return_pct}.
    """
    import yfinance as yf

    if pos_threshold is None:
        pos_threshold = float(os.getenv("LABEL_THRESHOLD_POS", "0.005"))
    if neg_threshold is None:
        neg_threshold = float(os.getenv("LABEL_THRESHOLD_NEG", "-0.005"))

    labelled: List[Dict[str, Any]] = []
    stats = {"positive": 0, "negative": 0, "skipped": 0, "error": 0}

    for record in reflections:
        ticker = record.get("ticker", "")
        date_str = record.get("date", "")
        memory_ids = record.get("memory_ids_used", [])

        if not ticker or not date_str:
            stats["error"] += 1
            continue

        try:
            ref_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            # Fetch 5 trading days starting from the decision date
            start = ref_date
            end = ref_date + timedelta(days=7)

            data = yf.download(
                ticker, start=str(start), end=str(end),
                progress=False, auto_adjust=True,
            )

            if data is None or len(data) < 2:
                stats["skipped"] += 1
                continue

            # Compute next-day return
            close_prices = data["Close"].values.flatten()
            if len(close_prices) < 2:
                stats["skipped"] += 1
                continue

            next_day_return = (close_prices[1] - close_prices[0]) / close_prices[0]

            # Apply thresholds
            if next_day_return > pos_threshold:
                label = 1
                stats["positive"] += 1
            elif next_day_return < neg_threshold:
                label = 0
                stats["negative"] += 1
            else:
                stats["skipped"] += 1
                continue

            # Create a labelled sample for each memory ID used
            for mid in memory_ids:
                labelled.append({
                    "memory_id": int(mid),
                    "ticker": ticker,
                    "date": date_str,
                    "label": label,
                    "return_pct": float(next_day_return * 100),
                    "decision": record.get("decision", "HOLD"),
                })

            # Also create a record for the overall reflection
            if not memory_ids:
                labelled.append({
                    "memory_id": -1,
                    "ticker": ticker,
                    "date": date_str,
                    "label": label,
                    "return_pct": float(next_day_return * 100),
                    "decision": record.get("decision", "HOLD"),
                })

        except Exception as exc:
            logger.warning(f"[Obj2] Label generation failed for {ticker}/{date_str}: {exc}")
            stats["error"] += 1

    logger.info(
        f"[Obj2] Labels generated: {stats['positive']} positive, "
        f"{stats['negative']} negative, {stats['skipped']} skipped, "
        f"{stats['error']} errors"
    )
    print(
        f"Labels generated: {stats['positive']} positive, "
        f"{stats['negative']} negative, {stats['skipped']} skipped"
    )

    return labelled
