"""
Objective 4 — LangGraph StateGraph Definition

Defines the multi-agent pipeline as a LangGraph StateGraph:

    START → regime → [fundamental, sentiment, technical] (parallel)
          → debate → risk → final → END

Parallel execution: After regime_node sets the market regime,
the three specialist agents (fundamental, sentiment, technical)
run in parallel via LangGraph's fan-out mechanism.

Usage:
    from agentic.obj4_multiagent.graph import build_graph
    graph = build_graph()
    result = graph.invoke(initial_state)
"""

import logging
from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import (
    regime_node,
    fundamental_node,
    sentiment_node,
    technical_node,
    debate_node,
    risk_node,
    final_node,
)

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """
    Build and compile the multi-agent trading graph.

    Graph topology:
        START
          │
          ▼
        regime_node
          │
          ├──────────────┬──────────────┐
          ▼              ▼              ▼
        fundamental    sentiment     technical    (parallel)
          │              │              │
          └──────────────┴──────────────┘
                         │
                         ▼
                     debate_node
                         │
                         ▼
                     risk_node
                         │
                         ▼
                     final_node
                         │
                         ▼
                        END

    Returns:
        Compiled StateGraph ready for .invoke()
    """
    # Create the state graph
    g = StateGraph(AgentState)

    # ── Add all 7 nodes ──────────────────────────────────────
    g.add_node("regime", regime_node)
    g.add_node("fundamental", fundamental_node)
    g.add_node("sentiment", sentiment_node)
    g.add_node("technical", technical_node)
    g.add_node("debate", debate_node)
    g.add_node("risk", risk_node)
    g.add_node("final", final_node)

    # ── Wire the edges ───────────────────────────────────────

    # Step 1: Start → Regime (always first)
    g.add_edge(START, "regime")

    # Step 2: Regime → 3 specialists (fan-out for parallel execution)
    g.add_edge("regime", "fundamental")
    g.add_edge("regime", "sentiment")
    g.add_edge("regime", "technical")

    # Step 3: All 3 specialists → Debate (fan-in: waits for all 3)
    g.add_edge("fundamental", "debate")
    g.add_edge("sentiment", "debate")
    g.add_edge("technical", "debate")

    # Step 4: Debate → Risk → Final → END (sequential)
    g.add_edge("debate", "risk")
    g.add_edge("risk", "final")
    g.add_edge("final", END)

    # ── Compile ──────────────────────────────────────────────
    compiled = g.compile()
    logger.info("[Obj4] LangGraph compiled: 7 nodes, fan-out parallel specialists")

    return compiled


def get_graph_diagram() -> str:
    """
    Generate a Mermaid diagram of the graph.

    Usage:
        print(get_graph_diagram())
    """
    graph = build_graph()
    try:
        return graph.get_graph().draw_mermaid()
    except Exception:
        # Fallback: return static diagram
        return """
graph TD
    START --> regime
    regime --> fundamental
    regime --> sentiment
    regime --> technical
    fundamental --> debate
    sentiment --> debate
    technical --> debate
    debate --> risk
    risk --> final
    final --> END
"""
