"""
Objective 3 — Step 2: Cross-Ticker Memory Retrieval

Queries memory stores of correlated tickers and formats the results
into a prompt block for the LLM.
"""

import os
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_cross_ticker_memories(
    target_ticker: str,
    query_embedding: np.ndarray,
    corr_matrix: pd.DataFrame,
    chroma_stores: Optional[Dict[str, Any]] = None,
    brain_db: Optional[Any] = None,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Retrieve memories from correlated tickers.

    Args:
        target_ticker:  The ticker we're making a decision for.
        query_embedding: Query vector (from the target ticker's character string).
        corr_matrix:    DataFrame of pairwise correlations.
        chroma_stores:  Dict of ChromaDB collections (optional, for future use).
        brain_db:       BrainDB instance (for direct memory query).
        top_k:          Number of memories per correlated ticker.
        threshold:      Minimum |correlation| to include a ticker.

    Returns:
        List of dicts: {text, source_ticker, corr_weight, layer}.
        Sorted descending by corr_weight.
    """
    if top_k is None:
        top_k = int(os.getenv("CROSS_TICKER_TOP_K", "2"))
    if threshold is None:
        threshold = float(os.getenv("CORRELATION_THRESHOLD", "0.6"))

    cross_memories: List[Dict[str, Any]] = []

    try:
        # Get target ticker's row from correlation matrix
        if target_ticker not in corr_matrix.index:
            logger.debug(f"[Obj3] {target_ticker} not in correlation matrix")
            return []

        corr_row = corr_matrix.loc[target_ticker].drop(target_ticker, errors="ignore")

        # Filter to correlated tickers above threshold
        qualifying = corr_row[corr_row.abs() >= threshold].sort_values(
            ascending=False, key=abs
        )

        if qualifying.empty:
            logger.debug(f"[Obj3] No tickers above correlation threshold {threshold} for {target_ticker}")
            return []

        logger.info(
            f"[Obj3] Cross-ticker candidates for {target_ticker}: "
            f"{dict(qualifying.items())}"
        )

        for other_ticker, corr_val in qualifying.items():
            try:
                # Try ChromaDB first (if available)
                if chroma_stores and f"{other_ticker}_shallow" in chroma_stores:
                    collection = chroma_stores[f"{other_ticker}_shallow"]
                    results = collection.query(
                        query_embeddings=[query_embedding.flatten().tolist()],
                        n_results=top_k,
                    )
                    if results and results.get("documents"):
                        for doc in results["documents"][0]:
                            cross_memories.append({
                                "text": doc,
                                "source_ticker": str(other_ticker),
                                "corr_weight": float(abs(corr_val)),
                                "layer": "shallow",
                            })
                # Fall back to BrainDB direct query
                elif brain_db is not None:
                    texts, ids = brain_db.short_term_memory.query(
                        query_embedding=query_embedding,
                        top_k=top_k,
                        symbol=str(other_ticker),
                    )
                    for text in texts:
                        cross_memories.append({
                            "text": text,
                            "source_ticker": str(other_ticker),
                            "corr_weight": float(abs(corr_val)),
                            "layer": "shallow",
                        })
                else:
                    logger.debug(
                        f"[Obj3] No memory store for {other_ticker} — skipping"
                    )

            except Exception as exc:
                logger.warning(
                    f"[Obj3] Failed to query {other_ticker} memories: {exc}"
                )
                continue

    except Exception as exc:
        logger.warning(f"[Obj3] Cross-ticker retrieval failed: {exc}")
        return []

    # Sort by correlation weight (descending)
    cross_memories.sort(key=lambda x: x["corr_weight"], reverse=True)

    logger.info(f"[Obj3] Retrieved {len(cross_memories)} cross-ticker memories")
    return cross_memories


def format_cross_ticker_prompt_block(
    cross_memories: List[Dict[str, Any]],
) -> str:
    """Format cross-ticker memories into a prompt section for the LLM.

    Args:
        cross_memories: List of dicts from get_cross_ticker_memories().

    Returns:
        Formatted string to inject into the LLM prompt.
        Empty string if no cross memories.
    """
    if not cross_memories:
        return ""

    lines = [
        "\n=== Cross-Asset Signals (correlation-weighted) ===",
    ]

    for mem in cross_memories:
        ticker = mem.get("source_ticker", "???")
        corr = mem.get("corr_weight", 0.0)
        text = mem.get("text", "")[:200]  # Truncate to 200 chars
        lines.append(f"• {ticker} (corr={corr:.2f}): {text}")

    lines.append(
        "\nNote: Above signals are from correlated assets and may indicate "
        "sector-level trends. Use them as supporting context."
    )

    return "\n".join(lines) + "\n"
