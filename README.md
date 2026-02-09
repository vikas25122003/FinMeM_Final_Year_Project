# FinMEM - LLM Trading Agent with Layered Memory

A Python implementation of the FinMEM trading agent based on the research paper:
*"FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design"*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **🧠 Layered Memory System**: 3-tier memory (shallow/intermediate/deep) with decay rates
- **📊 LLM-Powered Decisions**: BUY/HOLD/SELL recommendations with confidence scores
- **📈 Real-Time Data**: Yahoo Finance prices + Finnhub news (with Google RSS fallback)
- **🎭 Agent Profiling**: Configurable risk tolerance and trading styles
- **🔄 Memory Scoring**: `Score = α·Recency + β·Relevancy + γ·Importance`

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source ../venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file:
```env
# Required: LLM access via OpenRouter
OPENROUTER_API_KEY=your_openrouter_key_here

# Optional: Better news coverage (free at https://finnhub.io/)
FINNHUB_API_KEY=your_finnhub_key_here
```

### 3. Run the Agent
```bash
# Basic usage
python run.py --ticker NVDA --mode test

# With options
python run.py --ticker AAPL --risk aggressive --capital 50000
```

## 📋 CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ticker, -t` | Stock symbol | AAPL |
| `--mode, -m` | `train` or `test` | test |
| `--risk` | `conservative`, `moderate`, `aggressive` | moderate |
| `--capital, -c` | Initial capital | 100000 |
| `--start-date, -s` | Start date (YYYY-MM-DD) | 30 days ago |
| `--end-date, -e` | End date (YYYY-MM-DD) | today |

## 📁 Project Structure

```
FinMeM/
├── run.py                  # CLI entry point
├── requirements.txt        # Dependencies
├── .env                    # API keys (not tracked)
├── .gitignore
└── finmem/
    ├── config.py           # Configuration settings
    ├── llm_client.py       # OpenRouter API client
    ├── profiling/
    │   └── agent_profile.py    # Risk levels, trading styles
    ├── memory/
    │   ├── embeddings.py       # sentence-transformers
    │   └── layered_memory.py   # 3-layer memory system
    ├── decision/
    │   └── decision_engine.py  # LLM decision making
    ├── data/
    │   ├── price_fetcher.py    # Yahoo Finance
    │   ├── news_fetcher.py     # Google News RSS
    │   └── finnhub_news.py     # Finnhub API
    └── simulation/
        └── simulator.py        # Trading simulation
```

## 🏗️ Architecture

Based on the FinMEM paper, the agent has three core modules:

### 1. Profiling Module
- Risk levels: Conservative (0.3), Moderate (0.5), Aggressive (0.7)
- Trading styles: Value, Growth, Momentum, Balanced

### 2. Layered Memory System
| Layer | Time Horizon | Decay Rate | Content |
|-------|-------------|------------|---------|
| Shallow | 3 days | 0.9 | Recent news, daily prices |
| Intermediate | 21 days | 0.7 | Weekly trends, patterns |
| Deep | 90 days | 0.3 | Fundamentals, long-term data |

### 3. Decision Engine
- Retrieves relevant memories using embedding similarity
- Builds context prompt for LLM
- Returns `BUY`/`HOLD`/`SELL` with confidence + reasoning

## 📊 Example Output

```
Decision: BUY NVDA (Confidence: 70%)
Reasoning: NVIDIA has shown strong performance driven by 
positive analyst sentiment regarding OpenAI spending benefits.
The stock demonstrated resilience with upward momentum...

Executed: BUY 73.32 shares @ $190.93
```

## 🔑 API Keys

| Service | Purpose | Link |
|---------|---------|------|
| OpenRouter | LLM access (deepseek-chat) | [openrouter.ai](https://openrouter.ai/) |
| Finnhub | Stock news (optional) | [finnhub.io](https://finnhub.io/) |

## 📚 References

- [FinMEM Paper](https://arxiv.org/abs/2311.13743)
- [OpenRouter API](https://openrouter.ai/docs)
- [sentence-transformers](https://www.sbert.net/)

## 📄 License

MIT License
