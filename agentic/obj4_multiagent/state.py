"""
Objective 4 — AgentState TypedDict

Typed state shared across all LangGraph nodes.
Each node reads from and writes to this state object.
LangGraph automatically manages state transitions.
"""

from typing import TypedDict, Optional, List, Dict, Any


class AgentReport(TypedDict, total=False):
    """Structured report from a specialist agent."""
    direction: str          # "BUY", "HOLD", or "SELL"
    confidence: float       # 0.0 to 1.0
    rationale: str          # Reasoning text
    pivotal_memory_ids: List[int]  # Memory IDs the agent found most useful
    raw_response: str       # Full LLM response for debugging


class DebateState(TypedDict, total=False):
    """State for the Bull/Bear debate."""
    bull_arguments: List[str]
    bear_arguments: List[str]
    debate_history: str
    consensus_direction: str     # "BUY", "HOLD", or "SELL"
    consensus_confidence: float  # 0.0 to 1.0
    debate_summary: str


class AgentState(TypedDict, total=False):
    """
    Complete state for the LangGraph multi-agent pipeline.

    Lifecycle:
        1. Initialized with ticker, date, prices, brain reference
        2. regime_node sets regime, q_values
        3. fundamental/sentiment/technical nodes set their reports (parallel)
        4. debate_node reads all reports, produces debate_state
        5. risk_node applies guards, produces risk assessment
        6. final_node produces final_decision + logs to Obj2
    """
    # ── Inputs (set before graph.invoke) ─────────────────────
    ticker: str
    cur_date: str                  # "YYYY-MM-DD"
    cur_price: float
    price_history: List[float]     # Last 60 daily closes
    brain: Any                     # BrainDB instance (not serializable, passed by ref)
    run_mode: str                  # "train" or "test"
    portfolio_state: Dict[str, Any]  # {cash, shares, position_value, total_value}
    character_string: str          # Agent character for memory queries
    top_k: int                     # Memories to retrieve per layer

    # ── Objective 1: Regime (set by regime_node) ────────────
    regime: str                    # "BULL", "SIDEWAYS", or "CRISIS"
    regime_confidence: Dict[str, float]  # Probabilities per regime
    q_values: Dict[str, float]     # Adaptive Q for each memory layer

    # ── Agent Reports (set by specialist nodes) ─────────────
    fundamental_report: Optional[AgentReport]
    sentiment_report: Optional[AgentReport]
    technical_report: Optional[AgentReport]
    regime_report: Optional[AgentReport]

    # ── Debate (set by debate_node) ─────────────────────────
    debate_state: Optional[DebateState]

    # ── Risk Assessment (set by risk_node) ──────────────────
    risk_assessment: Optional[str]
    concentration_guard_fired: bool
    risk_notes: List[str]

    # ── Final Decision (set by final_node) ──────────────────
    final_decision: Optional[str]     # "BUY", "HOLD", or "SELL"
    final_confidence: Optional[float]
    final_rationale: Optional[str]
    kelly_shares: Optional[int]       # Position size via Kelly criterion
    all_pivotal_ids: List[int]        # Memory IDs from all agents
