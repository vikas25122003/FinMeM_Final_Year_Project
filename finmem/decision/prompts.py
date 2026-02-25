"""
Prompt Templates for FinMEM Reflection

Train and test mode prompts for the LLM-based reflection mechanism.
Based on the reference implementation's prompting strategy.
"""

# ─── Memory ID description prompts ──────────────────────────────────────

short_memory_id_desc = (
    "Select the most relevant short-term memory IDs that influenced your analysis. "
    "These are recent news and daily market events."
)

mid_memory_id_desc = (
    "Select the most relevant mid-term memory IDs that influenced your analysis. "
    "These are quarterly filings and weekly trends."
)

long_memory_id_desc = (
    "Select the most relevant long-term memory IDs that influenced your analysis. "
    "These are annual filings and fundamental data."
)

reflection_memory_id_desc = (
    "Select the most relevant reflection memory IDs that influenced your analysis. "
    "These are past trading reflections and lessons learned."
)

# ─── Train mode prompts ────────────────────────────────────────────────

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
2. "short_memory_ids": List of short-term memory IDs that were most relevant (integers).
3. "mid_memory_ids": List of mid-term memory IDs that were most relevant (integers).
4. "long_memory_ids": List of long-term memory IDs that were most relevant (integers).
5. "reflection_memory_ids": List of reflection memory IDs that were most relevant (integers).

Respond ONLY with valid JSON, no other text."""

train_trade_reason_summary = (
    "Provide a comprehensive summary of why the stock moved in this direction, "
    "referencing specific memories that were most predictive."
)

# ─── Test mode prompts ─────────────────────────────────────────────────

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
4. "short_memory_ids": List of short-term memory IDs that influenced your decision (integers).
5. "mid_memory_ids": List of mid-term memory IDs that influenced your decision (integers).
6. "long_memory_ids": List of long-term memory IDs that influenced your decision (integers).
7. "reflection_memory_ids": List of reflection memory IDs that influenced your decision (integers).

Respond ONLY with valid JSON, no other text."""

test_invest_action_choice = (
    "Choose one of: 'buy', 'sell', or 'hold'. "
    "Consider the overall market sentiment, trends, and fundamentals."
)

test_trade_reason_summary = (
    "Provide a comprehensive explanation for your trading decision, "
    "referencing specific short-term, mid-term, long-term, and reflection memories."
)

test_sentiment_explanation = (
    "\n(Consider: What is the overall sentiment from recent news? "
    "Is there a clear positive or negative trend?)\n"
)

test_momentum_explanation = (
    "\nMomentum Signal:\n"
    "The 3-day cumulative return for this stock provides a momentum signal. "
    "Consider this alongside the memory-based analysis.\n"
)
