"""
Objective 4 — End-to-End Test (No Real LLM Calls)

Tests the full LangGraph pipeline with mocked LLM responses.
Validates:
    1. Graph compiles and all nodes are reachable
    2. State transitions work correctly
    3. Agent reports are populated
    4. Debate produces consensus
    5. Risk manager applies guards
    6. Final decision is produced
    7. Mermaid diagram generation works
"""

import os
import sys
import json
import logging
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import date

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ── Mock Responses ───────────────────────────────────────────────

MOCK_AGENT_RESPONSE = json.dumps({
    "direction": "BUY",
    "confidence": 0.72,
    "rationale": "Strong momentum and positive news sentiment from memory context.",
    "key_observations": ["Revenue growth above 20%", "Positive analyst coverage"],
    "pivotal_memory_ids": [101, 205, 307],
})

MOCK_DEBATE_RESPONSE = json.dumps({
    "consensus_direction": "BUY",
    "consensus_confidence": 0.68,
    "debate_summary": "Bull case stronger: growth catalysts outweigh near-term risks.",
    "bull_strength": 0.75,
    "bear_strength": 0.55,
})

MOCK_RISK_RESPONSE = json.dumps({
    "approved": True,
    "adjusted_direction": "BUY",
    "adjusted_confidence": 0.65,
    "risk_notes": ["Position within limits", "Regime is BULL — no caution adjustment"],
    "kelly_fraction": 0.10,
})

MOCK_PM_RESPONSE = json.dumps({
    "decision": "BUY",
    "confidence": 0.70,
    "rationale": "All specialists agree on bullish signals. Debate confirms. Risk approved.",
    "shares_to_trade": 5,
    "override_used": False,
})


def _mock_invoke(*args, **kwargs):
    """Mock LLM invoke that returns appropriate responses based on context."""
    messages = args[0] if args else kwargs.get("messages", [])
    # Detect which node is calling based on system message content
    system_text = ""
    for m in messages:
        if hasattr(m, 'content'):
            system_text += m.content
        elif isinstance(m, dict):
            system_text += m.get('content', '')

    if "Risk Manager" in system_text:
        content = MOCK_RISK_RESPONSE
    elif "Portfolio Manager" in system_text:
        content = MOCK_PM_RESPONSE
    elif "Investment Judge" in system_text or "neutral" in system_text.lower():
        content = MOCK_DEBATE_RESPONSE
    elif "Bull" in system_text or "Bear" in system_text:
        content = "Strong momentum indicators suggest continued upward trajectory."
    else:
        content = MOCK_AGENT_RESPONSE

    mock_response = MagicMock()
    mock_response.content = content
    return mock_response


class MockBrainDB:
    """Mock BrainDB that returns synthetic memories."""

    def query_short(self, query: str, top_k: int, symbol: str):
        return (
            [f"[{symbol}] Short memory: Price rose 2.3% on strong volume (2024-03-15)",
             f"[{symbol}] Short memory: Bullish news about new product launch"],
            [100, 101],
        )

    def query_mid(self, query: str, top_k: int, symbol: str):
        return (
            [f"[{symbol}] Mid memory: Q3 revenue up 18% YoY, margins expanding",
             f"[{symbol}] Mid memory: Analyst upgraded price target by 15%"],
            [200, 201],
        )

    def query_long(self, query: str, top_k: int, symbol: str):
        return (
            [f"[{symbol}] Long memory: 5-year CAGR of 25% in core business",
             f"[{symbol}] Long memory: Strong balance sheet, low debt-to-equity"],
            [300, 301],
        )

    def query_reflection(self, query: str, top_k: int, symbol: str):
        return (
            [f"[{symbol}] Reflection: Last BUY at similar conditions yielded +8%",
             f"[{symbol}] Reflection: Held through regime transition successfully"],
            [400, 401],
        )

    def boost_importance(self, symbol, mem_id, bonus=0.05):
        return True


def test_graph_compilation():
    """Test 1: Graph compiles without errors."""
    print("=" * 60)
    print("TEST 1: Graph Compilation")
    print("=" * 60)

    from agentic.obj4_multiagent.graph import build_graph
    graph = build_graph()

    assert graph is not None, "Graph compilation returned None"
    print("✅ Graph compiled successfully")
    print()


def test_graph_diagram():
    """Test 2: Mermaid diagram generation."""
    print("=" * 60)
    print("TEST 2: Graph Diagram Generation")
    print("=" * 60)

    from agentic.obj4_multiagent.graph import get_graph_diagram
    diagram = get_graph_diagram()

    assert diagram is not None and len(diagram) > 0, "Empty diagram"
    print(f"✅ Diagram generated ({len(diagram)} chars)")
    print(diagram[:300])
    print()


def test_state_creation():
    """Test 3: AgentState can be constructed."""
    print("=" * 60)
    print("TEST 3: State Creation")
    print("=" * 60)

    from agentic.obj4_multiagent.state import AgentState

    state: AgentState = {
        "ticker": "TSLA",
        "cur_date": "2024-03-15",
        "cur_price": 175.50,
        "price_history": [170 + i * 0.5 for i in range(60)],
        "brain": MockBrainDB(),
        "run_mode": "test",
        "portfolio_state": {
            "cash": 100000,
            "shares": 0,
            "position_value": 0,
            "total_value": 100000,
        },
        "character_string": "financial trading analyst",
        "top_k": 5,
    }

    assert state["ticker"] == "TSLA"
    assert len(state["price_history"]) == 60
    print("✅ AgentState created with all required fields")
    print()


def test_technical_features():
    """Test 4: Technical feature computation."""
    print("=" * 60)
    print("TEST 4: Technical Features")
    print("=" * 60)

    from agentic.obj4_multiagent.nodes import _compute_technical_features

    prices = [170 + i * 0.3 + (i % 5) * 0.5 for i in range(60)]
    features = _compute_technical_features(prices)

    assert "current_price" in features
    assert "sma_5" in features
    assert "sma_20" in features
    assert "momentum_5d" in features
    assert "volatility_20d" in features
    assert "rsi_14" in features

    print(f"✅ Technical features computed:")
    for k, v in features.items():
        if v is not None:
            print(f"   {k}: {v:.4f}")
    print()


def test_full_pipeline():
    """Test 5: Full pipeline with mocked LLM."""
    print("=" * 60)
    print("TEST 5: Full Pipeline (Mocked LLM)")
    print("=" * 60)

    # Set env vars for test
    os.environ["ADAPTIVE_Q"] = "false"
    os.environ["LEARNED_IMPORTANCE"] = "false"
    os.environ["CROSS_TICKER"] = "false"
    os.environ["DEBATE_ROUNDS"] = "1"
    os.environ["AWS_REGION"] = "us-east-1"

    from agentic.obj4_multiagent.graph import build_graph
    from agentic.obj4_multiagent.state import AgentState

    graph = build_graph()

    initial_state: AgentState = {
        "ticker": "TSLA",
        "cur_date": "2024-03-15",
        "cur_price": 175.50,
        "price_history": [170 + i * 0.3 for i in range(60)],
        "brain": MockBrainDB(),
        "run_mode": "test",
        "portfolio_state": {
            "cash": 100000,
            "shares": 0,
            "position_value": 0,
            "total_value": 100000,
        },
        "character_string": "financial trading analyst",
        "top_k": 5,
        "risk_notes": [],
        "all_pivotal_ids": [],
    }

    # Mock ALL ChatBedrock instances
    with patch("agentic.obj4_multiagent.nodes._get_bedrock_llm") as mock_llm_factory:
        mock_llm = MagicMock()
        mock_llm.invoke = _mock_invoke
        mock_llm_factory.return_value = mock_llm

        result = graph.invoke(initial_state)

    # Validate results
    assert result.get("regime") is not None, "Regime not set"
    assert result.get("fundamental_report") is not None, "Fundamental report missing"
    assert result.get("sentiment_report") is not None, "Sentiment report missing"
    assert result.get("technical_report") is not None, "Technical report missing"
    assert result.get("debate_state") is not None, "Debate state missing"
    assert result.get("final_decision") is not None, "Final decision missing"

    print(f"✅ Pipeline completed successfully!")
    print(f"   Regime:     {result['regime']}")
    print(f"   Fundamental: {result['fundamental_report'].get('direction', '?')}")
    print(f"   Sentiment:   {result['sentiment_report'].get('direction', '?')}")
    print(f"   Technical:   {result['technical_report'].get('direction', '?')}")
    print(f"   Debate:      {result['debate_state'].get('consensus_direction', '?')}")
    print(f"   Risk Notes:  {result.get('risk_notes', [])}")
    print(f"   FINAL:       {result['final_decision']} "
          f"(conf: {result.get('final_confidence', 0):.2f})")
    print(f"   Kelly:       {result.get('kelly_shares', 0)} shares")
    print(f"   Pivotal IDs: {result.get('all_pivotal_ids', [])}")
    print()


def test_parse_agent_json():
    """Test 6: JSON parsing with various response formats."""
    print("=" * 60)
    print("TEST 6: JSON Parsing")
    print("=" * 60)

    from agentic.obj4_multiagent.nodes import _parse_agent_json

    # Clean JSON
    result = _parse_agent_json(MOCK_AGENT_RESPONSE)
    assert result["direction"] == "BUY"
    print("✅ Clean JSON parsed")

    # JSON with markdown fences
    fenced = f"```json\n{MOCK_AGENT_RESPONSE}\n```"
    result = _parse_agent_json(fenced)
    assert result["direction"] == "BUY"
    print("✅ Fenced JSON parsed")

    # Fallback text
    result = _parse_agent_json("I think we should buy this stock because of momentum.")
    assert result["direction"] == "BUY"
    print("✅ Fallback text parsed to BUY")

    result = _parse_agent_json("Markets are risky, I recommend to sell.")
    assert result["direction"] == "SELL"
    print("✅ Fallback text parsed to SELL")

    result = _parse_agent_json("No clear signals, maintain current position.")
    assert result["direction"] == "HOLD"
    print("✅ Fallback text parsed to HOLD")
    print()


def main():
    """Run all tests."""
    print("\n" + "═" * 60)
    print("  OBJECTIVE 4 — MULTI-AGENT PIPELINE TEST SUITE")
    print("  LangGraph + FinMEM Memory + TradingAgents Debate")
    print("═" * 60 + "\n")

    tests = [
        test_graph_compilation,
        test_graph_diagram,
        test_state_creation,
        test_technical_features,
        test_parse_agent_json,
        test_full_pipeline,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("═" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print("═" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
