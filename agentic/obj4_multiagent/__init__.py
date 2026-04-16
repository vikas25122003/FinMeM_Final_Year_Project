"""
Objective 4 — Multi-Agent Memory Architecture (LangGraph)

Fuses FinMEM's layered memory system with TradingAgents' specialist
debate architecture using LangGraph for orchestration.

Components:
    state.py   — AgentState TypedDict (shared across all graph nodes)
    prompts.py — Specialist prompts (adapted from TradingAgents)
    nodes.py   — 7 LangGraph node functions
    graph.py   — LangGraph StateGraph definition
    test_multiagent.py — End-to-end test

Novel Contribution:
    First multi-agent trading framework with persistent layered memory
    per specialist agent, including a novel Regime Agent.
"""

__all__ = ["build_graph", "AgentState"]
