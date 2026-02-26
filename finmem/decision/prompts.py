"""
Prompt Templates for FinMEM Reflection — Paper-Faithful

Paper structure:
- Character profile embedded in prompts
- Memories from all layers with IDs for pivotal identification
- Train mode: future price given, reflect on predictive memories
- Test mode: no future data, generate buy/hold/sell decision
"""

# ── Train mode prompts ────────────────────────────────────────────────

train_investment_info_prefix = (
    "Today's date is {cur_date}. You are analyzing the stock {symbol}. "
    "The actual next-day price change is: {future_record}. "
    "Based on the memories below, reflect on what information was most useful "
    "for predicting this price movement.\n\n"
)

train_prompt = """You are a professional financial analyst doing a training exercise.
Given the investment information and memories below, analyze what happened and why.

{investment_info}

Respond with a JSON object containing:
1. "summary_reason": A detailed reflection on why the price moved as it did, 
   and which memories were most predictive.
2. "short_memory_ids": List of short-term memory IDs that were PIVOTAL for understanding this movement (integers).
3. "mid_memory_ids": List of mid-term memory IDs that were PIVOTAL (integers).
4. "long_memory_ids": List of long-term memory IDs that were PIVOTAL (integers).
5. "reflection_memory_ids": List of reflection memory IDs that were PIVOTAL (integers).

IMPORTANT: Only include memory IDs that were truly pivotal for investment success — 
these memories will receive a bonus in priority for future retrieval.

Respond ONLY with valid JSON, no other text."""

# ── Test mode prompts ─────────────────────────────────────────────────

test_investment_info_prefix = (
    "Today's date is {cur_date}. You are analyzing the stock {symbol} "
    "and need to make an investment decision.\n\n"
)

test_prompt = """You are a professional financial analyst making a trading decision.
Based on the investment information and memories below, decide whether to buy, hold, or sell.

{investment_info}

Respond with a JSON object containing:
1. "investment_decision": One of "buy", "hold", or "sell".
2. "summary_reason": A detailed explanation of your decision, referencing 
   specific memories that influenced it.
3. "confidence": A float between 0.0 and 1.0 indicating your confidence.
4. "short_memory_ids": List of short-term memory IDs that were PIVOTAL for this decision (integers).
5. "mid_memory_ids": List of mid-term memory IDs that were PIVOTAL (integers).
6. "long_memory_ids": List of long-term memory IDs that were PIVOTAL (integers).
7. "reflection_memory_ids": List of reflection memory IDs that were PIVOTAL (integers).

IMPORTANT: Only include memory IDs that were truly pivotal for investment success — 
these memories will receive a bonus in priority for future retrieval.

Respond ONLY with valid JSON, no other text."""

test_sentiment_explanation = (
    "\n(Consider: What is the overall sentiment from recent news? "
    "Is there a clear positive or negative trend?)\n"
)

test_momentum_explanation = (
    "\nMomentum Signal:\n"
    "The 3-day cumulative return for this stock provides a momentum signal. "
    "Consider this alongside the memory-based analysis.\n"
)
