"""
Agentic FinMEM — Objective 2: Learned Importance Scoring via Reflection Feedback

Replaces FinMem's random v_E initialization (coin-flip between 40/60/80)
with a supervised LogisticRegression classifier trained on the agent's own
trading history. Closes the feedback loop between memory importance and
trading outcomes.

Modules:
    logger     — Logs every reflection decision to JSONL files
    labeller   — Generates weak labels using next-day returns (yfinance)
    trainer    — Builds features, trains LogisticRegression classifier
    inference  — Runtime v_E prediction (replaces random assignment)
"""
