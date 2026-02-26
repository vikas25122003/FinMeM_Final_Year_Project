# 🧠 FinMEM — LLM Trading Agent with Layered Memory & Character Design

**A paper-faithful Python implementation of the FinMEM trading agent for automated stock trading using Large Language Models with a cognitive memory architecture.**

> Based on: *"FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design"* — [arXiv:2311.13743](https://arxiv.org/abs/2311.13743)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2311.13743-b31b1b.svg)](https://arxiv.org/abs/2311.13743)

---

## 📌 Project Overview

FinMEM is an **LLM-powered autonomous trading agent** that mimics human cognitive processes for financial decision-making. Unlike traditional algorithmic trading (rule-based) or Deep Reinforcement Learning approaches (PPO, DQN), FinMEM uses:

1. **Layered Memory** — A 4-tier memory system (short/mid/long/reflection) inspired by human cognitive architecture
2. **Self-Adaptive Character** — Dynamically switches between risk-seeking and risk-averse modes based on recent performance
3. **Working Memory Operations** — Summarization, Observation, and Reflection pipelines powered by LLMs
4. **Memory Promotion/Demotion** — Important memories automatically "jump" to deeper layers, stale ones decay

### Why This Matters

Traditional trading bots use fixed rules. DRL agents need millions of training episodes. FinMEM leverages the **reasoning capabilities of LLMs** combined with a structured memory system to make informed trading decisions — much closer to how a human analyst thinks.

---

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          FinMEM Trading Agent                             │
├──────────────┬────────────────────────────────┬───────────────────────────┤
│  PROFILING   │       LAYERED MEMORY           │    DECISION MODULE        │
│  MODULE      │                                │                           │
│              │  ┌──────────────────────────┐  │  Working Memory Ops:      │
│ Self-Adaptive│  │ SHORT-TERM (Q=14 days)   │←─│  1. Summarization (LLM)   │
│ Character    │  │ Daily news summaries,    │  │  2. Observation  (LLM)    │
│              │  │ price observations       │  │  3. Reflection   (LLM)    │
│ Switches:    │  ├──────────────────────────┤  │                           │
│ risk_seeking │  │ MID-TERM (Q=90 days)     │  │  Train: Reflect with      │
│     ↕        │  │ 10-Q quarterly filings   │  │  future price labels      │
│ risk_averse  │  ├──────────────────────────┤  │                           │
│              │  │ LONG-TERM (Q=365 days)   │  │  Test: Buy/Hold/Sell      │
│ Based on     │  │ 10-K annual reports      │  │  decisions with momentum  │
│ 3-day return │  ├──────────────────────────┤  │                           │
│              │  │ REFLECTION               │  │  Guardrails AI:           │
│              │  │ Past trading insights    │  │  LLM identifies pivotal   │
│              │  └──────────────────────────┘  │  memories → +0.05 bonus   │
├──────────────┴────────────────────────────────┴───────────────────────────┤
│                    Market Environment (Day-by-Day Simulation)              │
│              Yahoo Finance │ SEC EDGAR Filings │ Google/Finnhub News       │
├───────────────────────────────────────────────────────────────────────────┤
│              Portfolio (Single-Share Trading) + Feedback Loop              │
│              5 Metrics: Sharpe │ Volatility │ Drawdown │ CR │ B&H         │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Paper-Faithful Implementation Details

### Memory Scoring (Exact Match to Paper)

| Component | Formula | Description |
|-----------|---------|-------------|
| **Recency Decay** | `S_recency = e^(-δ / Q_l)` | Q_l = 14 (short), 90 (mid), 365 (long) days |
| **Importance Decay** | `S_importance = v × α_l^δ` | α_l = 0.9, 0.967, 0.988 per layer |
| **Importance Init** | `v ∈ {0.4, 0.6, 0.8}` | Probabilistic sampling, layer-dependent |
| **Compound Score** | `γ = S_recency + S_relevancy + S_importance` | Pure additive sum (paper §3.2) |
| **Similarity** | FAISS `IndexFlatIP` | Cosine similarity via normalized inner product |

### Memory Promotion (Jump Mechanism)

```
Short ──→ Mid ──→ Long          (importance ≥ threshold → promote)
Short ←── Mid ←── Long          (importance < threshold → demote)
         ↑                      On promotion: recency resets to 1.0
         └── LLM Promotion: pivotal memories get +0.05 bonus
```

### Self-Adaptive Character (Paper §3.1)

```python
if 3_day_cumulative_return >= 0:
    character = "risk_seeking"   # Confident → aggressive trades
else:
    character = "risk_averse"    # Losing → conservative/defensive
```

### Three Working Memory Operations (Paper §3.3)

| Step | Operation | What It Does |
|------|-----------|--------------|
| 1 | **Summarization** | LLM condenses raw news into key financial insights |
| 2 | **Observation** | LLM analyzes price patterns, momentum, support/resistance |
| 3 | **Reflection** | LLM queries all 4 memory layers, makes buy/hold/sell decision |

### Evaluation Metrics (Paper §4)

| Metric | Formula |
|--------|---------|
| Cumulative Return | `(V_final - V_initial) / V_initial` |
| Sharpe Ratio | `mean(daily_returns) / std(daily_returns) × √252` |
| Annualized Volatility | `std(daily_returns) × √252` |
| Daily Volatility | `std(daily_returns)` |
| Max Drawdown | `max((peak - trough) / peak)` |

---

## ✨ Key Features

- 🧠 **4-Layer Memory System** — Short / Mid / Long / Reflection with FAISS vector search
- 🔄 **Memory Promotion & Demotion** — Automatic "jump" mechanism based on importance thresholds
- 📉 **Exponential Decay** — Paper-exact `e^(-δ/Q_l)` recency scoring
- 🎭 **Self-Adaptive Character** — Dynamic risk mode switching on 3-day returns
- 🔍 **3 Working Memory Operations** — Summarize → Observe → Reflect pipeline
- ⭐ **LLM-Based Promotion** — Guardrails AI equivalent: pivotal memories get boosted
- 📰 **Real News Integration** — Google News RSS + Finnhub API
- 📄 **SEC EDGAR Filings** — Real 10-K and 10-Q filing text (no API key needed)
- 📊 **5 Paper Metrics + Buy & Hold Baseline** — Complete evaluation framework
- 💰 **Single-Share Trading** — Paper-faithful position sizing
- 💾 **Checkpointing** — Save/load full agent state
- 📅 **Day-by-Day Simulation** — Train & test modes with separate data splits

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/vikas25122003/FinMeM_Final_Year_Project.git
cd FinMeM_Final_Year_Project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the project root:
```env
# Required: LLM access via OpenRouter (get key at https://openrouter.ai/)
OPENROUTER_API_KEY=your_openrouter_key_here

# Optional: Better news coverage via Finnhub (free at https://finnhub.io/)
FINNHUB_API_KEY=your_finnhub_key_here
```

### 3. Run the Agent

```bash
# Train mode — Agent learns from historical data with future price labels
python3 run.py --ticker TSLA --start-date 2025-01-01 --end-date 2025-01-31 --mode train

# Test mode — Agent makes real buy/hold/sell decisions (no future data)
python3 run.py --ticker TSLA --start-date 2025-02-01 --end-date 2025-02-28 --mode test

# Full pipeline: Train → Save → Test
python3 run.py --ticker TSLA --mode train --save-checkpoint data/checkpoints/tsla \
    --start-date 2025-01-01 --end-date 2025-01-31
python3 run.py --ticker TSLA --mode test --checkpoint data/checkpoints/tsla \
    --start-date 2025-02-01 --end-date 2025-02-28
```

### 4. Enable FAISS (Optional)
```bash
# FAISS is installed but disabled by default (compatibility issues on some platforms)
# To enable FAISS vector search backend:
export FINMEM_USE_FAISS=1
python3 run.py --ticker TSLA --mode train
```

---

## 📊 Example Output

```
============================================================
  FinMEM Trading Simulation (Paper-Faithful)
  Ticker: TSLA | Mode: train
  Period: 2025-01-01 → 2025-01-31
  Capital: $100,000.00
  Character: Self-Adaptive (paper default)
============================================================

  Building dataset from Yahoo Finance...
  Fetched 10-K: 5043 chars (SEC EDGAR)
  Fetched 10-Q: 5043 chars (SEC EDGAR)
  Built dataset: 20 days

  Day 1: 2025-01-02 | $379.28 | [risk_seeking]
  Day 4: 2025-01-07 | $394.36 | Character switched: risk_seeking → risk_averse (-3.78%)
  Day 6: 2025-01-10 | $394.74 | Character switched: risk_averse → risk_seeking (+2.27%)
  ...

============================================================
  📊 Results Summary
============================================================
  Period:        2025-01-01 → 2025-01-31
  Days:          19
  Initial:       $100,000.00
  Final:         $100,111.21
  Return:        $111.21 (+0.11%)

  📈 Paper Metrics (FinMEM vs Buy & Hold):
  ──────────────────────────────────────────────────
  Metric                          FinMEM          B&H
  Cum. Return (%)                  0.11%        5.54%
  Sharpe Ratio                    5.2428       1.6356
  Ann. Volatility                 0.0030       0.5455
  Daily Volatility              0.000187     0.034364
  Max Drawdown (%)                 0.02%        9.14%

  Memory Stats:  short: 19, mid: 12, long: 16, reflection: 19
  Trades: 16 (single-share)
============================================================
```

> **Note**: FinMEM trades conservatively (1 share at a time) — lower returns but **much better risk-adjusted performance** (Sharpe: 5.24 vs 1.64, Max Drawdown: 0.02% vs 9.14%).

---

## 📋 CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ticker, -t` | Stock symbol (TSLA, AAPL, MSFT, etc.) | `AAPL` |
| `--mode, -m` | `train` (with labels) or `test` (blind) | `train` |
| `--risk` | `conservative`, `moderate`, `aggressive` | `moderate` |
| `--capital, -c` | Initial portfolio capital ($) | `100000` |
| `--start-date, -s` | Simulation start (YYYY-MM-DD) | 30 days ago |
| `--end-date, -e` | Simulation end (YYYY-MM-DD) | today |
| `--dataset, -d` | Pre-built dataset pickle path | auto-build |
| `--checkpoint, -ckp` | Load checkpoint from path | — |
| `--save-checkpoint` | Save checkpoint after run | — |
| `--top-k` | Memories retrieved per layer (cognitive span) | `5` |
| `--quiet, -q` | Suppress output | `false` |
| `--verbose, -v` | Debug logging | `false` |

---

## 📁 Project Structure

```
FinMeM/
├── run.py                              # CLI entry point
├── requirements.txt                    # Dependencies
├── .env                                # API keys (not tracked)
├── .gitignore
│
├── finmem/
│   ├── config.py                       # All paper parameters (Q_l, α_l, thresholds)
│   ├── llm_client.py                   # OpenRouter API client
│   │
│   ├── memory/                         # 📦 Layered Memory System
│   │   ├── embeddings.py              # sentence-transformers (all-MiniLM-L6-v2)
│   │   ├── memory_functions.py        # Paper formulas: decay, scoring, importance
│   │   └── layered_memory.py          # MemoryDB + BrainDB (4-layer orchestrator + FAISS)
│   │
│   ├── decision/                       # 🤖 Decision / Working Memory
│   │   ├── prompts.py                 # Train/test prompt templates
│   │   └── reflection.py             # 3 ops: summarize_news → observe_price → reflect
│   │
│   ├── profiling/                      # 🎭 Self-Adaptive Profiling
│   │   └── agent_profile.py           # Dynamic risk_seeking ↔ risk_averse switching
│   │
│   ├── data/                           # 📊 Data Pipeline
│   │   ├── build_dataset.py           # Build datasets (Yahoo + SEC + News)
│   │   ├── sec_filings.py            # SEC EDGAR 10-K/10-Q fetcher
│   │   ├── price_fetcher.py          # Yahoo Finance via yfinance
│   │   ├── news_fetcher.py           # Google News RSS
│   │   └── finnhub_news.py           # Finnhub news API
│   │
│   ├── evaluation/                     # 📈 Paper Metrics
│   │   ├── __init__.py
│   │   └── metrics.py                # 5 metrics + Buy & Hold baseline
│   │
│   └── simulation/                     # 🔄 Simulation Engine
│       ├── simulator.py               # Main loop: all paper components wired
│       ├── environment.py             # Market environment (day stepper)
│       └── portfolio.py               # Single-share trading + feedback
│
└── tests/                              # Unit tests for paper formulas
```

---

## 🔧 Paper Parameters (config.py)

### Memory Layer Configuration

| Layer | Q_l (Stability) | α_l (Importance Decay) | Jump Up | Jump Down |
|-------|-----------------|----------------------|---------|-----------|
| Short | 14 days | 0.900 | ≥ 0.80 → Mid | — |
| Mid | 90 days | 0.967 | ≥ 0.85 → Long | < 0.10 → Short |
| Long | 365 days | 0.988 | — | < 0.15 → Mid |
| Reflection | 90 days | 0.967 | — | — |

### Trading Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Position Size | 1 share per trade | Paper §4.1 |
| Cognitive Span (top_k) | 5 per layer | Paper §3.3 |
| LLM Promotion Bonus | +0.05 importance | Paper §3.2 |
| Initial Capital | $100,000 | Paper §4.1 |

---

## 🔑 API Keys & Data Sources

| Service | Purpose | Cost | Required? |
|---------|---------|------|-----------|
| [**OpenRouter**](https://openrouter.ai/) | LLM access (GPT-4, DeepSeek, etc.) | Pay-per-token | ✅ Required |
| [**SEC EDGAR**](https://www.sec.gov/edgar) | 10-K/10-Q filing text | Free (no key) | ✅ Auto-fetched |
| [**Yahoo Finance**](https://finance.yahoo.com/) | Stock price data | Free (no key) | ✅ Auto-fetched |
| [**Finnhub**](https://finnhub.io/) | Stock news articles | Free tier | ⬜ Optional |
| **Sentence-Transformers** | Text embeddings (local) | Free (no key) | ✅ Auto-downloaded |

---

## 🧪 Testing

```bash
# Run unit tests
python3 -m pytest tests/ -v

# Quick smoke test (verify all imports work)
python3 -c "from finmem.simulation.simulator import TradingSimulator; print('OK')"
```

---

## 📚 References

- **Paper**: [FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design](https://arxiv.org/abs/2311.13743) (Yu et al., 2023)
- **Reference Implementation**: [pipiku915/FinMem-LLM-StockTrading](https://github.com/pipiku915/FinMem-LLM-StockTrading)
- **OpenRouter API**: [openrouter.ai/docs](https://openrouter.ai/docs)
- **Sentence-Transformers**: [sbert.net](https://www.sbert.net/)
- **SEC EDGAR**: [sec.gov/edgar](https://www.sec.gov/edgar/searchedgar/companysearch)
- **FAISS**: [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

> **CS Final Year Project** by Vikas R M Jaivignesha  
> Department of Computer Science  
> Implementation of research paper arXiv:2311.13743
