# 🧠 FinMEM — LLM Trading Agent with Layered Memory

A Python implementation of the **FinMEM** trading agent based on the research paper:
> *"FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design"* — [arXiv:2311.13743](https://arxiv.org/abs/2311.13743)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

- **🧠 4-Layer Memory System** — Short / Mid / Long / Reflection memory with FAISS-style vector search
- **🔄 Memory Promotion & Demotion** — Memories automatically move between layers based on importance thresholds (the paper's "jump" mechanism)
- **📉 Exponential Decay** — Recency scores decay over time; stale memories are cleaned up automatically
- **🎯 Access Counter Feedback** — Portfolio P&L feeds back into memory importance (profitable memories get boosted)
- **💡 LLM Reflection (Working Memory)** — Queries all 4 layers, sends context to LLM, stores structured reflections
- **📅 Day-by-Day Simulation** — Processes one trading day at a time (train & test modes)
- **💾 Checkpointing** — Save/load full agent state to resume training or switch to test mode
- **📊 Real-Time Data** — Yahoo Finance prices + Finnhub/Google News
- **🎭 Agent Profiling** — Configurable risk tolerance, trading style, and character string

---

## 🏗️ Architecture (from the paper)

```
┌─────────────────────────────────────────────────────────┐
│                    FinMEM Agent                         │
├──────────┬──────────────────────────┬───────────────────┤
│ Profiling│     Layered Memory       │    Decision       │
│  Module  │                          │    Module         │
│          │  ┌──────────────────┐    │                   │
│ Character│  │  Short-term      │←───│  Working Memory   │
│  String  │  │  (News, Prices)  │    │  (LLM Reflection) │
│          │  ├──────────────────┤    │                   │
│ Risk     │  │  Mid-term        │    │  Train: Reflect   │
│ Profile  │  │  (Q Filings)     │    │  with future      │
│          │  ├──────────────────┤    │  record            │
│ Trading  │  │  Long-term       │    │                   │
│ Style    │  │  (Annual/Fundas) │    │  Test: Decide     │
│          │  ├──────────────────┤    │  buy/hold/sell    │
│          │  │  Reflection      │    │  with momentum    │
│          │  │  (Past Insights) │    │                   │
│          │  └──────────────────┘    │                   │
├──────────┴──────────────────────────┴───────────────────┤
│              Market Environment (Day-by-Day)            │
│           Portfolio Tracker + Feedback Loop              │
└─────────────────────────────────────────────────────────┘
```

### Memory Scoring

Each memory has three score components:

```
Compound Score = w_recency × Recency + w_importance × Importance
Final Rank     = w_compound × Compound + w_similarity × Similarity(query)
```

| Component | Mechanism |
|-----------|-----------|
| **Recency** | Exponential decay: `e^(-λ × Δt)`, resets on promotion |
| **Importance** | Initialized per layer, updated by access counter feedback |
| **Similarity** | Cosine similarity via sentence-transformers embeddings |

### Memory Jump (Promotion / Demotion)

| Transition | Condition |
|------------|-----------|
| Short → Mid | `importance ≥ 0.80` |
| Mid → Long | `importance ≥ 0.85` |
| Mid → Short | `importance < 0.10` |
| Long → Mid | `importance < 0.15` |

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file:
```env
# Required: LLM access via OpenRouter
OPENROUTER_API_KEY=your_openrouter_key_here

# Optional: Better news via Finnhub (free at https://finnhub.io/)
FINNHUB_API_KEY=your_finnhub_key_here
```

### 3. Run the Agent
```bash
# Train mode — populate memory + reflect with known future prices
python3 run.py --ticker AAPL -s 2024-01-01 -e 2024-02-01 --mode train

# Test mode — make real buy/hold/sell decisions
python3 run.py --ticker AAPL -s 2024-02-01 -e 2024-03-01 --mode test

# With checkpoint (train, save, then test)
python3 run.py --ticker AAPL --mode train --save-checkpoint data/checkpoints/aapl
python3 run.py --ticker AAPL --mode test  --checkpoint data/checkpoints/aapl
```

---

## 📋 CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ticker, -t` | Stock symbol | `AAPL` |
| `--mode, -m` | `train` or `test` | `train` |
| `--risk` | `conservative`, `moderate`, `aggressive` | `moderate` |
| `--capital, -c` | Initial capital ($) | `100000` |
| `--start-date, -s` | Start date (YYYY-MM-DD) | 30 days ago |
| `--end-date, -e` | End date (YYYY-MM-DD) | today |
| `--dataset, -d` | Path to pre-built dataset pickle | auto-build |
| `--checkpoint, -ckp` | Path to load checkpoint from | — |
| `--save-checkpoint` | Path to save checkpoint after run | — |
| `--top-k` | Cognitive span: memories per layer | `5` |
| `--quiet, -q` | Suppress output | `false` |
| `--verbose, -v` | Enable debug logging | `false` |

---

## 📁 Project Structure

```
FinMeM/
├── run.py                          # CLI entry point
├── requirements.txt                # Dependencies
├── .env                            # API keys (not tracked)
├── .gitignore
└── finmem/
    ├── config.py                   # Per-layer memory config, agent settings
    ├── llm_client.py               # OpenRouter API client
    │
    ├── memory/                     # 📦 Layered Memory System
    │   ├── embeddings.py           # sentence-transformers embeddings
    │   ├── memory_functions.py     # Decay, scoring, importance functions
    │   └── layered_memory.py       # MemoryDB (per-layer) + BrainDB (4-layer orchestrator)
    │
    ├── decision/                   # 🤖 Decision / Working Memory
    │   ├── prompts.py              # Train/test prompt templates
    │   └── reflection.py           # LLM reflection (working memory mechanism)
    │
    ├── profiling/                  # 🎭 Agent Profiling
    │   └── agent_profile.py        # Risk levels, trading styles, character
    │
    ├── data/                       # 📊 Data Pipeline
    │   ├── build_dataset.py        # Build pickle datasets from Yahoo Finance
    │   ├── price_fetcher.py        # Price data via yfinance
    │   ├── news_fetcher.py         # Google News RSS
    │   └── finnhub_news.py         # Finnhub news API
    │
    └── simulation/                 # 🔄 Simulation Engine
        ├── simulator.py            # Main agent: day-by-day step loop
        ├── environment.py          # Market environment (day stepper)
        └── portfolio.py            # Portfolio tracker + feedback
```

---

## 📊 Example Output

```
============================================================
  FinMEM Trading Simulation
  Ticker: AAPL | Mode: train
  Period: 2024-01-01 → 2024-02-01
  Capital: $100,000.00
============================================================

  Day 1: 2024-01-02 | $183.73 | Cash: $100,000.00, Shares: 0.00
  Day 6: 2024-01-09 | $183.24 | Cash: $100,483.50, Shares: 0.00
  ...

============================================================
  Simulation Complete
  Days Processed: 21
  Final Value:    $102,881.51
  Total Return:   $2,881.51 (+2.88%)
  Memory Stats:   {short: 4, mid: 0, long: 0, reflection: 17, total_removed: 20}
============================================================
```

---

## 🔧 Configuration

All settings are in `finmem/config.py`:

### Memory Layer Defaults

| Layer | Decay Rate | Jump Up Threshold | Jump Down Threshold |
|-------|-----------|-------------------|---------------------|
| Short | 0.99 (fast decay) | 0.80 → Mid | — |
| Mid | 0.50 | 0.85 → Long | 0.10 → Short |
| Long | 0.10 (slow decay) | — | 0.15 → Mid |
| Reflection | 0.30 | — | — |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 5 | Memories retrieved per layer (cognitive span) |
| `initial_capital` | $100,000 | Starting portfolio cash |
| `max_position_size` | 20% | Max allocation per trade |
| `look_back_window_size` | 7 | Days for momentum calculation |

---

## 🔑 API Keys

| Service | Purpose | Cost | Link |
|---------|---------|------|------|
| **OpenRouter** | LLM access (DeepSeek, GPT-4, etc.) | Pay-per-token | [openrouter.ai](https://openrouter.ai/) |
| **Finnhub** | Stock news (optional) | Free tier | [finnhub.io](https://finnhub.io/) |

> **Note**: Embeddings use `sentence-transformers/all-MiniLM-L6-v2` locally — no API key needed.

---

## 📚 References

- **Paper**: [FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design](https://arxiv.org/abs/2311.13743)  
- **Reference Implementation**: [pipiku915/FinMem-LLM-StockTrading](https://github.com/pipiku915/FinMem-LLM-StockTrading)
- **OpenRouter API**: [openrouter.ai/docs](https://openrouter.ai/docs)
- **sentence-transformers**: [sbert.net](https://www.sbert.net/)

---

## 📄 License

MIT License
