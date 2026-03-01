# 🤖 FinMEM Improvement 2: Multi-Agent Architecture

## 📖 Overview
The base FinMEM system relies on a single LLM character (which switches between risk-averse and risk-seeking) to process all information (news, price, SEC filings) and make a decision. 

A **Multi-Agent Architecture** splits this cognitive load among specialized agents. By simulating an institutional trading floor, different agents act as dedicated Analysts, Risk Managers, and Quant Traders. They debate conflicting signals, ensuring that decisions are verified from multiple angles before execution, drastically reducing LLM hallucinations and emotional biases.

---

## 🏗 Architecture Diagram

```mermaid
flowchart TD
    %% Define Styles
    classDef orchestrator fill:#ff9900,stroke:#fff,stroke-width:2px,color:#000;
    classDef worker fill:#4a154b,stroke:#fff,stroke-width:2px,color:#fff;
    classDef memory fill:#003b5c,stroke:#fff,stroke-width:2px,color:#fff;
    classDef market fill:#2b5797,stroke:#fff,stroke-width:2px,color:#fff;

    subgraph Data Sources
        N[Financial News]:::market
        P[Price Tickers & MACD]:::market
        S[SEC Filings 10-K/10-Q]:::market
    end

    subgraph Specialized LLM Agents
        A1(Fundamental Analyst Agent):::worker
        A2(Technical Quant Agent):::worker
        A3(Sentiment / News Agent):::worker
    end

    S --> A1
    P --> A2
    N --> A3

    subgraph Memory & Reflection
        Mem[(Shared Multi-Agent Vector DB)]:::memory
        Mem -.-> A1 & A2 & A3
    end

    subgraph Debate & Execution Layer
        O{Lead Portfolio Manager\n(Orchestrator Agent)}:::orchestrator
        RM(Risk Management Agent):::worker
        
        A1 & A2 & A3 -->|Proposed Trade & Rationale| O
        O <-->|Debate / Critique| A1 & A2 & A3
        O -->|Drafted Trade| RM
        RM -->|Approval / Veto| H[Trade Execution Engine]:::market
    end
```

---

## 🛠 Detailed Approach

### 1. Agent Role Assignment
- **Fundamental Analyst**: Scans the 10-K/10-Q filings and macro-economic factors. Prompted to look strictly for long-term viability, debt ratios, and structural changes.
- **Technical Quant**: Given only historical price arrays, moving averages, and volatility numbers. Prompted to spot momentum breakouts and resistance lines.
- **Sentiment Analyst**: Given live news streams and Twitter sentiment. Evaluates the hype, FUD, and immediate catalyst events.

### 2. The Debate Protocol (Orchestrator Logic)
1. **Parallel Generation**: All 3 worker agents generate an initial decision (`BUY`, `SELL`, `HOLD`) and an explanation.
2. **Debate Phase**: The Lead Portfolio Manager (Orchestrator) compares the arrays. If the Quant says `SELL` but Fundamental says `BUY`, the Orchestrator prompts them to critique each other's rationale.
3. **Consensus Resolution**: After a maximum of 2 debate rounds, the Orchestrator forces a final weighted decision.

### 3. Risk Management Layer
Before the trade hits the API, the **Risk Manager Agent** reviews the current portfolio drawdown, cash reserves, and the confidence score of the Orchestrator's decision. If maximum allocation rules are violated, it vetoes the trade and forces a `HOLD`.

### 4. Shared Multi-Agent Memory
Instead of one isolated memory bank, the system maintains a memory graph. The technical agent only retrieves past technical setup memories, while the Fundamental agent cross-references past fundamental memories. 

---

## 💾 Dataset & Requirements

### Datasets Needed
- **Diverse Modality Data**: The dataset must be cleanly split. Time-synchronized tabular data (OHLCV) for the Quant, and synchronized raw text (Reuters, Bloomberg, SEC) for the Fundamental/Sentiment agents.
- **Debate Benchmarks**: Datasets like `FinQA` or `Pixiu` can be used to evaluate if the specialized agents are accurately interpreting their distinct data streams.

### Tech Stack & Libraries
| Component | Technology / Library |
|-----------|----------------------|
| **Agent Framework** | `langgraph` (for stateful multi-agent loops) or `autogen` (for conversational debate) |
| **LLM Provider** | API with parallel execution support (AWS Bedrock Claude 3 Haiku / DeepSeek API) |
| **Data Models** | `pydantic` (to enforce strict JSON outputs from each agent for the Orchestrator to parse) |
| **Concurrency** | `asyncio`, `aiohttp` (for parallel LLM API calls) |

### System Specifications
- **Concurrency**: Because one "Step" now requires 4-5 parallel LLM API calls, switching to an asynchronous architecture is mandatory to prevent the simulation from becoming excruciatingly slow.
- **Latency**: Using high-speed, low-cost models (like Claude 3 Haiku or DeepSeek V3) is critical here, as token usage and request counts will quadruple compared to base FinMEM.

---

## 🚀 Execution Steps
1. Refactor `llm_client.py` to support `async` network calls via `httpx` or `aiohttp`.
2. Abstract the single `Profile` into `agents/quant_agent.py`, `agents/fund_agent.py`, etc.
3. Build the LangGraph state machine where nodes equal agents and edges equal debate rounds.
4. Update the `Simulator` to await the final Pydantic object from the Lead PM before placing trades.
