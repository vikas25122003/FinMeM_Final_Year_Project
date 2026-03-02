# 🧠 FinMEM — LLM Trading Agent with Layered Memory & Character Design

**A paper-faithful Python implementation of the FinMEM trading agent for autonomous stock trading using Large Language Models with a cognitive memory architecture.**

> Based on: *"FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design"* — [arXiv:2311.13743](https://arxiv.org/abs/2311.13743)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2311.13743-b31b1b.svg)](https://arxiv.org/abs/2311.13743)

---

## 📌 Project Overview

FinMEM is an **LLM-powered autonomous trading agent** that mimics human cognitive processes for financial decision-making. It is built faithfully on top of the published research paper, using:

1. **Layered Memory System** — 4-tier memory (short / mid / long / reflection) modelled after human cognitive architecture and the Ebbinghaus forgetting curve
2. **Self-Adaptive Character** — Dynamically switches between risk-seeking and risk-averse modes based on a 3-day rolling return window
3. **Working Memory Operations** — Summarization → Observation → Reflection pipeline driven by LLM calls
4. **Memory Promotion/Demotion** — Important memories automatically "jump" to deeper layers via importance threshold checks

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                TradingSimulator (Orchestrator)               │
│  - Runs day-by-day simulation loop                           │
│  - Coordinates all components                                │
└────────┬────────────────────────────────────────────────────┘
         │
         ├──► BrainDB (4-Layer Memory System)
         │    ├─ Short-term  (Q=14 days,  α=0.900)
         │    ├─ Mid-term    (Q=90 days,  α=0.967)
         │    ├─ Long-term   (Q=365 days, α=0.988)
         │    └─ Reflection  (Q=365 days, α=0.988)
         │
         ├──► AgentProfile (Self-Adaptive Character)
         │    └─ risk_seeking ↔ risk_averse on 3-day return
         │
         ├──► LLMClient (Multi-Provider)
         │    ├─ DeepSeek (native, recommended)
         │    ├─ OpenRouter (multi-model)
         │    └─ AWS Bedrock (Claude, Llama, DeepSeek R1)
         │
         ├──► Portfolio (Position Tracker)
         │    └─ Single-share trading + momentum + feedback
         │
         └──► MarketEnvironment (Data Provider)
              ├─ Yahoo Finance (prices)
              ├─ SEC EDGAR (10-K / 10-Q filings)
              └─ Google News RSS / Finnhub (news)
```

---

## 🔬 Paper-Faithful Implementation Details

### Memory Scoring (Exact Match to Paper)

| Component | Formula | Description |
|-----------|---------|-------------|
| **Recency Decay** | `S_recency = e^(-δ / Q_l)` | Q_l = 14 (short), 90 (mid), 365 (long) |
| **Importance Decay** | `S_importance = v × α_l^δ` | α_l = 0.9, 0.967, 0.988 per layer |
| **Importance Init** | `v ∈ {0.4, 0.6, 0.8}` | Probabilistic sampling at creation |
| **Compound Score** | `γ = S_recency + S_relevancy + S_importance` | Pure additive sum (paper §3.2) |
| **Similarity** | FAISS `IndexFlatIP` or NumPy dot product | Cosine via normalised inner product |

### Memory Promotion (Jump Mechanism)

```
Short ──→ Mid   (importance ≥ 0.80 → promote, recency resets to 1.0)
Mid   ──→ Long  (importance ≥ 0.85 → promote, recency resets to 1.0)
Mid   ←── Short (importance < 0.10 → demote)
Long  ←── Mid   (importance < 0.15 → demote)
```

Additionally, the LLM identifies **pivotal memories** during reflection and boosts their importance by `+0.05`.

### Self-Adaptive Character (Paper §3.1)

```python
if 3_day_cumulative_return >= 0:
    character = "risk_seeking"   # Confident → pursue high-return opportunities
else:
    character = "risk_averse"    # Losing → prioritise capital preservation
```

The current character string is used as the **query for all memory layer retrievals**, ensuring risk mindset is encoded in retrieval context.

### Three Working Memory Operations (Paper §3.3)

| Step | Operation | What It Does |
|------|-----------|--------------|
| 1 | **Summarization** | LLM condenses raw news into key financial insights |
| 2 | **Observation** | LLM analyses price patterns, momentum, support/resistance |
| 3 | **Reflection** | LLM queries all 4 layers, makes buy/hold/sell decision, identifies pivotal memories |

---

## ✨ Feature Summary

- 🧠 **4-Layer Memory System** — Short / Mid / Long / Reflection with FAISS or NumPy vector search
- 🔄 **Memory Promotion & Demotion** — Automatic jump mechanism driven by importance thresholds
- 📉 **Exponential Decay** — Paper-exact `e^(-δ/Q_l)` recency scoring per layer
- 🎭 **Self-Adaptive Character** — Dynamic risk mode switching on 3-day rolling returns
- 🔍 **3 Working Memory Operations** — Summarize → Observe → Reflect pipeline
- ⭐ **LLM-Based Promotion** — Pivotal memories identified via reflection get +0.05 importance bonus
- 📰 **News Integration** — Google News RSS + Finnhub API (optional)
- 📄 **SEC EDGAR Filings** — Real 10-K and 10-Q text (no API key required)
- 📊 **5 Paper Metrics + Buy & Hold Baseline** — Complete evaluation framework
- 💰 **Single-Share Trading** — Paper-faithful position sizing
- 💾 **Checkpointing** — Save and resume full agent state
- 📅 **Day-by-Day Simulation** — Train & test modes with separate data splits

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/vikas25122003/FinMeM_Final_Year_Project.git
cd FinMeM_Final_Year_Project

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

**Option A — DeepSeek (recommended, cheapest)**
```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
DEEPSEEK_MODEL=deepseek-chat
```

**Option B — OpenRouter (multi-model access)**
```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-your-key-here
LLM_MODEL=deepseek/deepseek-chat
```

**Option C — AWS Bedrock (Claude / DeepSeek R1 / Llama)**
```env
LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=us.deepseek.r1-v1:0
```

**Optional — Finnhub news (free tier)**
```env
FINNHUB_API_KEY=your-finnhub-key-here
```

### 3. Run the Agent

```bash
# Train mode — agent learns which memories are predictive (future labels provided)
python3 run.py --ticker TSLA --start-date 2025-01-01 --end-date 2025-01-31 --mode train

# Test mode — agent makes real blind decisions
python3 run.py --ticker TSLA --start-date 2025-02-01 --end-date 2025-02-28 --mode test

# Full pipeline: Train → Save → Load → Test
python3 run.py --ticker TSLA --mode train \
    --start-date 2025-01-01 --end-date 2025-01-31 \
    --save-checkpoint data/checkpoints/tsla

python3 run.py --ticker TSLA --mode test \
    --start-date 2025-02-01 --end-date 2025-02-28 \
    --checkpoint data/checkpoints/tsla
```

### 4. Enable FAISS (Optional)

```bash
# FAISS is installed but disabled by default (platform compatibility)
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

> **Note**: FinMEM trades conservatively (1 share at a time) — lower returns but far better risk-adjusted performance (Sharpe: 5.24 vs 1.64, Max Drawdown: 0.02% vs 9.14%).

---

## 📋 CLI Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--ticker, -t` | Stock symbol (TSLA, AAPL, MSFT, etc.) | `AAPL` |
| `--mode, -m` | `train` (learns from future labels) or `test` (blind decisions) | `train` |
| `--risk` | `conservative` (0.3), `moderate` (0.5), `aggressive` (0.7) | `moderate` |
| `--capital, -c` | Initial portfolio capital ($) | `100000` |
| `--start-date, -s` | Simulation start (YYYY-MM-DD) | 30 days ago |
| `--end-date, -e` | Simulation end (YYYY-MM-DD) | today |
| `--dataset, -d` | Path to a pre-built dataset pickle | auto-build |
| `--checkpoint, -ckp` | Load checkpoint from path | — |
| `--save-checkpoint` | Save checkpoint after run to path | — |
| `--top-k` | Memories retrieved per layer (cognitive span) | `5` |
| `--quiet, -q` | Suppress verbose output | `false` |
| `--verbose, -v` | Enable debug logging | `false` |

---

## 📁 Project Structure

```
FinMeM/
├── run.py                          # CLI entry point (argparse)
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template
├── .env                            # Your secrets (not tracked)
├── .gitignore
│
├── finmem/
│   ├── config.py                   # All paper parameters (Q_l, α_l, thresholds)
│   ├── llm_client.py               # Unified client: OpenRouter / DeepSeek / Bedrock
│   │
│   ├── memory/                     # 📦 Layered Memory System
│   │   ├── embeddings.py           # sentence-transformers (all-MiniLM-L6-v2)
│   │   ├── memory_functions.py     # Paper formulas: decay, scoring, importance
│   │   └── layered_memory.py       # MemoryDB + BrainDB (4-layer orchestrator)
│   │
│   ├── decision/                   # 🤖 Working Memory Operations
│   │   ├── prompts.py              # Train/test prompt templates
│   │   └── reflection.py          # Summarize → Observe → Reflect
│   │
│   ├── profiling/                  # 🎭 Self-Adaptive Profiling
│   │   └── agent_profile.py        # risk_seeking ↔ risk_averse switching
│   │
│   ├── data/                       # 📊 Data Pipeline
│   │   ├── build_dataset.py        # Assembles Yahoo + SEC + News dataset
│   │   ├── sec_filings.py          # SEC EDGAR 10-K / 10-Q fetcher
│   │   ├── price_fetcher.py        # Yahoo Finance via yfinance
│   │   ├── news_fetcher.py         # Google News RSS
│   │   └── finnhub_news.py         # Finnhub API (optional)
│   │
│   ├── evaluation/                 # 📈 Paper Evaluation Metrics
│   │   └── metrics.py              # 5 metrics + Buy & Hold baseline
│   │
│   └── simulation/                 # 🔄 Simulation Engine
│       ├── simulator.py            # Main loop — all components wired
│       ├── environment.py          # Market environment (day stepper)
│       └── portfolio.py            # Trades + momentum + feedback loop
│
├── tests/                          # Unit tests for paper formulas
├── Reference_Papers/               # 20 reference papers used for analysis
├── RESEARCH_ANALYSIS.md            # Cross-paper analysis and gap identification
└── DEEP_ANALYSIS.md                # Technical deep-dive of this implementation
```

---

## 🔧 Paper Parameters (config.py)

### Memory Layer Configuration

| Layer | Q_l (Stability) | α_l (Importance Decay) | Promote When | Demote When |
|-------|-----------------|----------------------|--------------|-------------|
| Short | 14 days | 0.900 | importance ≥ 0.80 → Mid | — |
| Mid | 90 days | 0.967 | importance ≥ 0.85 → Long | importance < 0.10 → Short |
| Long | 365 days | 0.988 | — | importance < 0.15 → Mid |
| Reflection | 365 days | 0.988 | — | — |

### Evaluation Metrics (Paper §4)

| Metric | Formula |
|--------|---------|
| Cumulative Return | `(V_final - V_initial) / V_initial` |
| Sharpe Ratio | `mean(daily_returns) / std(daily_returns) × √252` |
| Annualized Volatility | `std(daily_returns) × √252` |
| Daily Volatility | `std(daily_returns)` |
| Max Drawdown | `max((peak - trough) / peak)` |

### Key Trading Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Position Size | 1 share per trade | Paper §4.1 |
| Cognitive Span (top_k) | 5 memories per layer | Paper §3.3 |
| LLM Promotion Bonus | +0.05 importance | Paper §3.2 |
| Initial Capital | $100,000 | Paper §4.1 |
| Momentum Window | 3 days | Paper §3.1 |

---

## 🔑 API Keys & Data Sources

| Service | Purpose | Cost | Required? |
|---------|---------|------|-----------|
| [DeepSeek](https://platform.deepseek.com/) | LLM (cheapest option) | ~$0.001/K tokens | ✅ Recommended |
| [OpenRouter](https://openrouter.ai/) | Multi-model LLM access | Pay-per-token | ✅ Alternative |
| [AWS Bedrock](https://aws.amazon.com/bedrock/) | Managed LLM (Claude, R1) | AWS pricing | ✅ Alternative |
| [Yahoo Finance](https://finance.yahoo.com/) | Stock price data | Free (no key) | ✅ Auto-fetched |
| [SEC EDGAR](https://www.sec.gov/edgar) | 10-K / 10-Q filing text | Free (no key) | ✅ Auto-fetched |
| [Sentence-Transformers](https://sbert.net/) | Text embeddings (local) | Free (no key) | ✅ Auto-downloaded |
| [Finnhub](https://finnhub.io/) | Enhanced news feed | Free tier | ⬜ Optional |

---

## 🧪 Testing

```bash
# Run unit tests (paper formula verification)
python3 -m pytest tests/ -v

# Quick import smoke test
python3 -c "from finmem.simulation.simulator import TradingSimulator; print('OK')"

# Test LLM connection
python3 -c "from finmem.llm_client import LLMClient; c = LLMClient(); print(c.test_connection())"
```

---

## 📚 References

- **Paper**: [FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design](https://arxiv.org/abs/2311.13743) (Yu et al., 2023)
- **Reference Implementation**: [pipiku915/FinMem-LLM-StockTrading](https://github.com/pipiku915/FinMem-LLM-StockTrading)
- **Ebbinghaus Forgetting Curve**: Cognitive science basis for exponential memory decay
- **FAISS**: [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- **Sentence-Transformers**: [sbert.net](https://www.sbert.net/)
- **SEC EDGAR**: [sec.gov/edgar](https://www.sec.gov/edgar/searchedgar/companysearch)

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

> **CS Final Year Project** — R M Jai Vignesha Vikas (2201107080) · Ramachandiran R (2201107083)  
> Department of Computer Science & Engineering · Puducherry Technological University  
> Guided by: Dr. M. Thenmozhi  
> Implementation of research paper arXiv:2311.13743
