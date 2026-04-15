"""
Agentic FinMEM — Objective 3: Portfolio-Aware Cross-Ticker Memory Contextualization

Extends FinMem from a single-stock agent to a portfolio-aware system by:
1. Computing rolling Pearson correlation between tickers
2. Injecting cross-asset memory signals weighted by correlation
3. Applying a concentration guard to prevent over-allocation to
   highly correlated positions

Modules:
    matrix     — Rolling Pearson correlation matrix computation
    retrieval  — Cross-ticker memory query and prompt formatting
    guard      — Portfolio concentration guard (BUY override logic)
"""
