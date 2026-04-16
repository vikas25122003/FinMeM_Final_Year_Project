"""
Objective 4 — LangGraph Node Functions

7 node functions for the multi-agent pipeline:
    1. regime_node      — HMM regime classification (Obj1)
    2. fundamental_node — Long+Mid memory → Claude Haiku
    3. sentiment_node   — Short+Reflection memory → Nova Micro
    4. technical_node   — Short memory + price data → DeepSeek R1
    5. debate_node      — 2-round Bull/Bear debate
    6. risk_node        — Concentration guard (Obj3) + Kelly sizing
    7. final_node       — Portfolio Manager final decision + Obj2 logging

Each node:
    - Reads relevant fields from AgentState
    - Queries specific BrainDB memory layers
    - Calls the appropriate Bedrock model via langchain-aws ChatBedrock
    - Returns a dict of updated state fields
"""

import os
import json
import logging
import math
from typing import Dict, Any, List, Optional

import numpy as np

from .state import AgentState, AgentReport, DebateState
from .prompts import (
    FUNDAMENTAL_SYSTEM_PROMPT,
    SENTIMENT_SYSTEM_PROMPT,
    TECHNICAL_SYSTEM_PROMPT,
    REGIME_SYSTEM_PROMPT,
    BULL_DEBATE_PROMPT,
    BEAR_DEBATE_PROMPT,
    DEBATE_JUDGE_PROMPT,
    RISK_MANAGER_PROMPT,
    PORTFOLIO_MANAGER_PROMPT,
)

logger = logging.getLogger(__name__)

# ── LLM Helpers ─────────────────────────────────────────────────────────

def _get_bedrock_llm(model_env_var: str, fallback_model: str, temperature: float = 0.3):
    """Create a ChatBedrock instance from env var model ID."""
    from langchain_aws import ChatBedrock

    model_id = os.getenv(model_env_var, fallback_model)
    region = os.getenv("AWS_REGION", "us-east-1")

    return ChatBedrock(
        model_id=model_id,
        region_name=region,
        model_kwargs={"max_tokens": 2000, "temperature": temperature},
    )


def _invoke_llm(llm, system_prompt: str, user_prompt: str) -> str:
    """Invoke a ChatBedrock LLM with system + user message."""
    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"LLM invoke failed: {e}")
        return ""


def _parse_agent_json(response: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, handling markdown fences."""
    response = response.strip()
    # Remove markdown code fences
    if response.startswith("```"):
        lines = response.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response = "\n".join(lines)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse agent JSON: {response[:200]}")
        # Fallback: try to extract direction from text
        direction = "HOLD"
        if "buy" in response.lower():
            direction = "BUY"
        elif "sell" in response.lower():
            direction = "SELL"
        return {
            "direction": direction,
            "confidence": 0.5,
            "rationale": response[:500],
            "pivotal_memory_ids": [],
        }


def _format_memories(memories: list, ids: list) -> str:
    """Format retrieved memories into a readable context block."""
    if not memories:
        return "(No memories available for this layer)"
    lines = []
    for i, (mem, mid) in enumerate(zip(memories, ids)):
        lines.append(f"  [Memory #{mid}] {mem[:400]}")
    return "\n".join(lines)


def _compute_technical_features(prices: List[float]) -> Dict[str, Any]:
    """Compute basic technical features from price history."""
    if not prices or len(prices) < 5:
        return {"error": "Insufficient price data"}

    arr = np.array(prices, dtype=float)
    current = arr[-1]
    features = {
        "current_price": float(current),
        "price_5d_ago": float(arr[-5]) if len(arr) >= 5 else None,
        "price_20d_ago": float(arr[-20]) if len(arr) >= 20 else None,
        "sma_5": float(np.mean(arr[-5:])) if len(arr) >= 5 else None,
        "sma_20": float(np.mean(arr[-20:])) if len(arr) >= 20 else None,
        "sma_50": float(np.mean(arr[-50:])) if len(arr) >= 50 else None,
    }

    # Momentum
    if len(arr) >= 5:
        features["momentum_5d"] = float((current - arr[-5]) / arr[-5] * 100)
    if len(arr) >= 20:
        features["momentum_20d"] = float((current - arr[-20]) / arr[-20] * 100)

    # Volatility (annualized)
    if len(arr) >= 20:
        returns = np.diff(np.log(arr[-21:])) if len(arr) >= 21 else np.diff(np.log(arr))
        features["volatility_20d"] = float(np.std(returns) * np.sqrt(252) * 100)

    # RSI approximation (14-day)
    if len(arr) >= 15:
        deltas = np.diff(arr[-15:])
        gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        losses = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0.001
        rs = gains / losses
        features["rsi_14"] = float(100 - (100 / (1 + rs)))

    return features


# ═══════════════════════════════════════════════════════════════════════
#  NODE 1: REGIME (Objective 1 integration)
# ═══════════════════════════════════════════════════════════════════════

def regime_node(state: AgentState) -> Dict[str, Any]:
    """
    Classify market regime using Objective 1's HMM classifier.
    Sets regime, confidence, and adaptive Q values in state.
    """
    logger.info(f"[Obj4] regime_node: {state['ticker']} @ {state['cur_date']}")

    try:
        from agentic.obj1_regime.features import compute_features
        from agentic.obj1_regime.classifier import get_classifier
        from agentic.obj1_regime.q_table import get_all_Q

        mode = os.getenv("ADAPTIVE_Q_MODE", "threshold")
        clf = get_classifier(mode)

        features = compute_features(
            state["ticker"],
            str(state["cur_date"]),
        )
        regime = clf.predict(features)
        regime_proba = clf.predict_proba(features)
        q_values = get_all_Q(regime)

        logger.info(f"[Obj4] Regime: {regime} | Q: {q_values}")

    except Exception as e:
        logger.warning(f"[Obj4] Regime classification failed: {e}")
        regime = "SIDEWAYS"
        regime_proba = {"BULL": 0.33, "SIDEWAYS": 0.34, "CRISIS": 0.33}
        q_values = {"short": 14.0, "mid": 90.0, "long": 365.0}

    return {
        "regime": regime,
        "regime_confidence": regime_proba,
        "q_values": q_values,
    }


# ═══════════════════════════════════════════════════════════════════════
#  NODE 2: FUNDAMENTAL AGENT (Long + Mid memory → Claude Haiku)
# ═══════════════════════════════════════════════════════════════════════

def fundamental_node(state: AgentState) -> Dict[str, Any]:
    """
    Fundamental Agent: Reads Long + Mid memory layers.
    Focus: Financial filings, earnings, balance sheets, ratios.
    Model: Claude Haiku 3.5 (best at document understanding).
    """
    logger.info(f"[Obj4] fundamental_node: {state['ticker']}")

    brain = state.get("brain")
    ticker = state["ticker"]
    top_k = state.get("top_k", 5)
    query = "fundamental financial analysis annual quarterly filings earnings revenue margins growth"

    # Query dedicated memory layers: Long + Mid
    long_mem, long_ids = [], []
    mid_mem, mid_ids = [], []
    if brain:
        try:
            long_mem, long_ids = brain.query_long(query, top_k, ticker)
            mid_mem, mid_ids = brain.query_mid(query, top_k, ticker)
        except Exception as e:
            logger.warning(f"[Obj4] Fundamental memory query failed: {e}")

    memory_context = (
        f"=== LONG-TERM MEMORIES (filings, fundamentals) ===\n"
        f"{_format_memories(long_mem, long_ids)}\n\n"
        f"=== MID-TERM MEMORIES (quarterly observations) ===\n"
        f"{_format_memories(mid_mem, mid_ids)}"
    )

    user_prompt = (
        f"Ticker: {ticker}\n"
        f"Date: {state['cur_date']}\n"
        f"Current Price: ${state.get('cur_price', 0.0):.2f}\n"
        f"Market Regime: {state.get('regime', 'UNKNOWN')}\n\n"
        f"MEMORY CONTEXT:\n{memory_context}\n\n"
        f"Analyze these memories and produce your fundamental assessment."
    )

    llm = _get_bedrock_llm("FUNDAMENTAL_MODEL", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
    response = _invoke_llm(llm, FUNDAMENTAL_SYSTEM_PROMPT, user_prompt)
    report = _parse_agent_json(response)
    report["raw_response"] = response

    logger.info(f"[Obj4] Fundamental: {report.get('direction', '?')} "
                f"(conf: {report.get('confidence', 0):.2f})")

    return {"fundamental_report": report}


# ═══════════════════════════════════════════════════════════════════════
#  NODE 3: SENTIMENT AGENT (Short + Reflection memory → Nova Micro)
# ═══════════════════════════════════════════════════════════════════════

def sentiment_node(state: AgentState) -> Dict[str, Any]:
    """
    Sentiment Agent: Reads Short + Reflection memory layers.
    Focus: News sentiment, social mood, past decision reflections.
    Model: Amazon Nova Micro (fast, cheap for short text).
    """
    logger.info(f"[Obj4] sentiment_node: {state['ticker']}")

    brain = state.get("brain")
    ticker = state["ticker"]
    top_k = state.get("top_k", 5)
    query = "news sentiment market mood analyst opinion social media public perception"

    # Query dedicated memory layers: Short + Reflection
    short_mem, short_ids = [], []
    refl_mem, refl_ids = [], []
    if brain:
        try:
            short_mem, short_ids = brain.query_short(query, top_k, ticker)
            refl_mem, refl_ids = brain.query_reflection(query, top_k, ticker)
        except Exception as e:
            logger.warning(f"[Obj4] Sentiment memory query failed: {e}")

    memory_context = (
        f"=== SHORT-TERM MEMORIES (recent news, daily observations) ===\n"
        f"{_format_memories(short_mem, short_ids)}\n\n"
        f"=== REFLECTION MEMORIES (past decision reviews, lessons learned) ===\n"
        f"{_format_memories(refl_mem, refl_ids)}"
    )

    user_prompt = (
        f"Ticker: {ticker}\n"
        f"Date: {state['cur_date']}\n"
        f"Current Price: ${state.get('cur_price', 0.0):.2f}\n"
        f"Market Regime: {state.get('regime', 'UNKNOWN')}\n\n"
        f"MEMORY CONTEXT:\n{memory_context}\n\n"
        f"Analyze these memories and produce your sentiment assessment."
    )

    llm = _get_bedrock_llm("SENTIMENT_MODEL", "us.amazon.nova-micro-v1:0")
    response = _invoke_llm(llm, SENTIMENT_SYSTEM_PROMPT, user_prompt)
    report = _parse_agent_json(response)
    report["raw_response"] = response

    logger.info(f"[Obj4] Sentiment: {report.get('direction', '?')} "
                f"(conf: {report.get('confidence', 0):.2f})")

    return {"sentiment_report": report}


# ═══════════════════════════════════════════════════════════════════════
#  NODE 4: TECHNICAL AGENT (Short memory + prices → DeepSeek R1)
# ═══════════════════════════════════════════════════════════════════════

def technical_node(state: AgentState) -> Dict[str, Any]:
    """
    Technical Agent: Reads Short memory + computes technical indicators.
    Focus: Price patterns, momentum, RSI, moving averages, volatility.
    Model: DeepSeek R1 (strong quantitative reasoning).
    """
    logger.info(f"[Obj4] technical_node: {state['ticker']}")

    brain = state.get("brain")
    ticker = state["ticker"]
    top_k = state.get("top_k", 3)
    query = "price movement technical trend momentum support resistance volume"

    # Query Short memory only
    short_mem, short_ids = [], []
    if brain:
        try:
            short_mem, short_ids = brain.query_short(query, top_k, ticker)
        except Exception as e:
            logger.warning(f"[Obj4] Technical memory query failed: {e}")

    # Compute technical features from price history
    prices = state.get("price_history", [])
    tech_features = _compute_technical_features(prices)

    memory_context = (
        f"=== SHORT-TERM MEMORIES (recent price observations) ===\n"
        f"{_format_memories(short_mem, short_ids)}"
    )

    tech_summary = "\n".join(f"  {k}: {v}" for k, v in tech_features.items() if v is not None)

    user_prompt = (
        f"Ticker: {ticker}\n"
        f"Date: {state['cur_date']}\n"
        f"Current Price: ${state.get('cur_price', 0.0):.2f}\n"
        f"Market Regime: {state.get('regime', 'UNKNOWN')}\n\n"
        f"TECHNICAL INDICATORS:\n{tech_summary}\n\n"
        f"MEMORY CONTEXT:\n{memory_context}\n\n"
        f"Analyze the technical data and memories, then produce your technical assessment."
    )

    llm = _get_bedrock_llm("TECHNICAL_MODEL", "us.deepseek.r1-v1:0")
    response = _invoke_llm(llm, TECHNICAL_SYSTEM_PROMPT, user_prompt)
    report = _parse_agent_json(response)
    report["raw_response"] = response

    logger.info(f"[Obj4] Technical: {report.get('direction', '?')} "
                f"(conf: {report.get('confidence', 0):.2f})")

    return {"technical_report": report}


# ═══════════════════════════════════════════════════════════════════════
#  NODE 5: DEBATE (Bull/Bear 2-round debate)
# ═══════════════════════════════════════════════════════════════════════

def debate_node(state: AgentState) -> Dict[str, Any]:
    """
    Bull/Bear debate with 2 rounds, adapted from TradingAgents.
    Uses FinMEM reflection memories as "lessons from similar situations."
    """
    logger.info(f"[Obj4] debate_node: {state['ticker']}")

    debate_rounds = int(os.getenv("DEBATE_ROUNDS", "2"))
    llm = _get_bedrock_llm("DEBATE_MODEL", "us.anthropic.claude-3-5-haiku-20241022-v1:0")

    # Gather reports for debate context
    def _report_summary(report: Optional[Dict]) -> str:
        if not report:
            return "(No report available)"
        return f"{report.get('direction', '?')} (conf: {report.get('confidence', 0):.2f}) — {report.get('rationale', 'N/A')}"

    fundamental_str = _report_summary(state.get("fundamental_report"))
    sentiment_str = _report_summary(state.get("sentiment_report"))
    technical_str = _report_summary(state.get("technical_report"))
    regime_str = _report_summary(state.get("regime_report"))

    # Get reflection memories for debate context
    refl_mem_str = "(No reflections)"
    brain = state.get("brain")
    if brain:
        try:
            refl_mem, _ = brain.query_reflection(
                "past trading decisions lessons learned mistakes successes",
                3, state["ticker"]
            )
            if refl_mem:
                refl_mem_str = "\n".join(f"  - {m[:300]}" for m in refl_mem)
        except Exception:
            pass

    bull_arguments = []
    bear_arguments = []
    debate_history = ""

    for round_num in range(debate_rounds):
        logger.info(f"[Obj4] Debate round {round_num + 1}/{debate_rounds}")

        # Bull argues
        bull_prompt = BULL_DEBATE_PROMPT.format(
            ticker=state["ticker"],
            fundamental_report=fundamental_str,
            sentiment_report=sentiment_str,
            technical_report=technical_str,
            regime_report=regime_str,
            regime=state.get("regime", "UNKNOWN"),
            debate_history=debate_history,
            last_bear_argument=bear_arguments[-1] if bear_arguments else "(No bear argument yet)",
            reflection_memories=refl_mem_str,
        )
        bull_response = _invoke_llm(llm, "You are a Bull Analyst.", bull_prompt)
        bull_arguments.append(bull_response)
        debate_history += f"\n\n--- Bull Round {round_num + 1} ---\n{bull_response}"

        # Bear argues
        bear_prompt = BEAR_DEBATE_PROMPT.format(
            ticker=state["ticker"],
            fundamental_report=fundamental_str,
            sentiment_report=sentiment_str,
            technical_report=technical_str,
            regime_report=regime_str,
            regime=state.get("regime", "UNKNOWN"),
            debate_history=debate_history,
            last_bull_argument=bull_arguments[-1],
            reflection_memories=refl_mem_str,
        )
        bear_response = _invoke_llm(llm, "You are a Bear Analyst.", bear_prompt)
        bear_arguments.append(bear_response)
        debate_history += f"\n\n--- Bear Round {round_num + 1} ---\n{bear_response}"

    # Judge produces consensus
    judge_prompt = DEBATE_JUDGE_PROMPT.format(
        ticker=state["ticker"],
        bull_arguments="\n\n".join(f"Round {i+1}: {a}" for i, a in enumerate(bull_arguments)),
        bear_arguments="\n\n".join(f"Round {i+1}: {a}" for i, a in enumerate(bear_arguments)),
        regime=state.get("regime", "UNKNOWN"),
        cur_date=state["cur_date"],
    )
    judge_response = _invoke_llm(llm, "You are a neutral Investment Judge.", judge_prompt)
    judge_result = _parse_agent_json(judge_response)

    debate_state: DebateState = {
        "bull_arguments": bull_arguments,
        "bear_arguments": bear_arguments,
        "debate_history": debate_history,
        "consensus_direction": judge_result.get("consensus_direction", "HOLD"),
        "consensus_confidence": judge_result.get("consensus_confidence", 0.5),
        "debate_summary": judge_result.get("debate_summary", "No summary available"),
    }

    # Also produce a regime_report from the regime context
    regime_report: AgentReport = {
        "direction": state.get("regime", "SIDEWAYS"),
        "confidence": max(state.get("regime_confidence", {}).values(), default=0.5),
        "rationale": f"Market regime is {state.get('regime', 'UNKNOWN')}",
        "pivotal_memory_ids": [],
        "raw_response": "",
    }
    # Map regime to direction
    if state.get("regime") == "BULL":
        regime_report["direction"] = "BUY"
    elif state.get("regime") == "CRISIS":
        regime_report["direction"] = "SELL"
    else:
        regime_report["direction"] = "HOLD"

    logger.info(f"[Obj4] Debate consensus: {debate_state['consensus_direction']} "
                f"(conf: {debate_state['consensus_confidence']:.2f})")

    return {
        "debate_state": debate_state,
        "regime_report": regime_report,
    }


# ═══════════════════════════════════════════════════════════════════════
#  NODE 6: RISK MANAGER (Concentration Guard + Kelly)
# ═══════════════════════════════════════════════════════════════════════

def risk_node(state: AgentState) -> Dict[str, Any]:
    """
    Risk Manager: Applies Objective 3 concentration guard + Kelly sizing.
    In CRISIS regime, reduces confidence by 30%.
    """
    logger.info(f"[Obj4] risk_node: {state['ticker']}")

    debate = state.get("debate_state", {})
    direction = debate.get("consensus_direction", "HOLD")
    confidence = debate.get("consensus_confidence", 0.5)
    regime = state.get("regime", "SIDEWAYS")
    risk_notes = []
    guard_fired = False

    # 1. Regime caution
    if regime == "CRISIS":
        confidence *= 0.7
        risk_notes.append(f"CRISIS regime: confidence reduced 30% → {confidence:.2f}")

    # 2. Concentration guard (Obj3)
    if os.getenv("CROSS_TICKER", "false").lower() == "true" and direction == "BUY":
        try:
            from agentic.obj3_correlation.matrix import compute_correlation_matrix
            corr_matrix = compute_correlation_matrix(reference_date=str(state["cur_date"]))
            threshold = float(os.getenv("CONCENTRATION_THRESHOLD", "0.80"))
            ticker = state["ticker"]

            if ticker in corr_matrix.index:
                row = corr_matrix.loc[ticker].drop(ticker, errors="ignore")
                high_corr = row[row.abs() > threshold]
                if len(high_corr) > 0:
                    guard_fired = True
                    direction = "HOLD"
                    risk_notes.append(
                        f"Concentration guard: {ticker} highly correlated with "
                        f"{', '.join(high_corr.index.tolist())} — BUY→HOLD"
                    )
        except Exception as e:
            risk_notes.append(f"Concentration guard check failed: {e}")

    # 3. Portfolio exposure check
    ps = state.get("portfolio_state", {})
    total = ps.get("total_value", 100000)
    pos_value = ps.get("position_value", 0)
    if total > 0 and pos_value / total > 0.20 and direction == "BUY":
        risk_notes.append(
            f"Position exposure {pos_value/total*100:.1f}% > 20% limit — caution"
        )
        confidence *= 0.8

    # 4. Kelly criterion for position sizing
    win_rate = 0.5 + confidence * 0.15  # Estimate from confidence
    win_loss_ratio = 1.5  # Assume 1.5:1 reward/risk
    kelly_fraction = max(0, (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio)
    kelly_fraction = min(kelly_fraction, 0.25)  # Cap at 25%

    kelly_shares = 0
    if direction == "BUY" and state.get("cur_price", 0) > 0:
        cash = ps.get("cash", 100000)
        kelly_amount = cash * kelly_fraction
        kelly_shares = int(kelly_amount / state["cur_price"])

    # LLM risk assessment (optional, for richer output)
    llm = _get_bedrock_llm("DEBATE_MODEL", "us.anthropic.claude-3-5-haiku-20241022-v1:0", temperature=0.2)
    risk_prompt = RISK_MANAGER_PROMPT.format(
        ticker=state["ticker"],
        consensus_direction=direction,
        consensus_confidence=confidence,
        debate_summary=debate.get("debate_summary", "N/A"),
        regime=regime,
        cash=ps.get("cash", 100000),
        shares=ps.get("shares", 0),
        position_value=pos_value,
        total_value=total,
        guard_status="FIRED — BUY overridden to HOLD" if guard_fired else "OK",
    )
    risk_response = _invoke_llm(llm, "You are a Risk Manager.", risk_prompt)
    risk_result = _parse_agent_json(risk_response)

    # Use LLM adjustments if available
    if risk_result.get("adjusted_direction"):
        direction = risk_result["adjusted_direction"]
    if risk_result.get("adjusted_confidence"):
        confidence = risk_result["adjusted_confidence"]
    if risk_result.get("risk_notes"):
        risk_notes.extend(risk_result["risk_notes"])

    logger.info(f"[Obj4] Risk: {direction} (conf: {confidence:.2f}) | "
                f"Kelly: {kelly_shares} shares | Guard: {guard_fired}")

    return {
        "risk_assessment": risk_response,
        "concentration_guard_fired": guard_fired,
        "risk_notes": risk_notes,
        "final_decision": direction,
        "final_confidence": confidence,
        "kelly_shares": kelly_shares,
    }


# ═══════════════════════════════════════════════════════════════════════
#  NODE 7: PORTFOLIO MANAGER (Final Decision + Obj2 Logging)
# ═══════════════════════════════════════════════════════════════════════

def final_node(state: AgentState) -> Dict[str, Any]:
    """
    Portfolio Manager: Makes the FINAL decision.
    Override rule: If all 4 agents unanimously disagree with debate consensus,
    the unanimous agent view wins.
    Also logs to Objective 2 reflection logger.
    """
    logger.info(f"[Obj4] final_node: {state['ticker']}")

    # Collect all agent directions
    agents = {
        "fundamental": state.get("fundamental_report", {}),
        "sentiment": state.get("sentiment_report", {}),
        "technical": state.get("technical_report", {}),
        "regime": state.get("regime_report", {}),
    }

    agent_directions = {
        name: report.get("direction", "HOLD")
        for name, report in agents.items()
    }

    debate = state.get("debate_state", {})
    consensus = state.get("final_decision", debate.get("consensus_direction", "HOLD"))
    confidence = state.get("final_confidence", debate.get("consensus_confidence", 0.5))

    # Override rule: unanimous agents override debate consensus
    unique_dirs = set(agent_directions.values())
    override_used = False
    if len(unique_dirs) == 1:
        unanimous_dir = unique_dirs.pop()
        if unanimous_dir != consensus:
            logger.info(f"[Obj4] OVERRIDE: All 4 agents say {unanimous_dir}, "
                        f"debate said {consensus} → Using agent unanimous view")
            consensus = unanimous_dir
            override_used = True

    # Portfolio Manager LLM call for final reasoning
    ps = state.get("portfolio_state", {})
    llm = _get_bedrock_llm("PORTFOLIO_MODEL", "us.deepseek.r1-v1:0", temperature=0.2)

    pm_prompt = PORTFOLIO_MANAGER_PROMPT.format(
        ticker=state["ticker"],
        cur_date=state["cur_date"],
        fundamental_direction=agent_directions.get("fundamental", "?"),
        fundamental_confidence=agents.get("fundamental", {}).get("confidence", 0),
        sentiment_direction=agent_directions.get("sentiment", "?"),
        sentiment_confidence=agents.get("sentiment", {}).get("confidence", 0),
        technical_direction=agent_directions.get("technical", "?"),
        technical_confidence=agents.get("technical", {}).get("confidence", 0),
        regime_direction=agent_directions.get("regime", "?"),
        regime_confidence=agents.get("regime", {}).get("confidence", 0),
        consensus_direction=consensus,
        consensus_confidence=confidence,
        debate_summary=debate.get("debate_summary", "N/A"),
        risk_approved=not state.get("concentration_guard_fired", False),
        risk_direction=consensus,
        risk_notes="; ".join(state.get("risk_notes", [])),
        kelly_fraction=state.get("kelly_shares", 0),
        regime=state.get("regime", "UNKNOWN"),
        cash=ps.get("cash", 100000),
        shares=ps.get("shares", 0),
        total_value=ps.get("total_value", 100000),
    )
    pm_response = _invoke_llm(llm, "You are the Portfolio Manager.", pm_prompt)
    pm_result = _parse_agent_json(pm_response)

    final_decision = pm_result.get("decision", consensus)
    final_confidence = pm_result.get("confidence", confidence)
    final_rationale = pm_result.get("rationale", "")

    # Collect all pivotal memory IDs from all agents
    all_ids = []
    for report in agents.values():
        all_ids.extend(report.get("pivotal_memory_ids", []))

    # Log to Objective 2 reflection logger
    if os.getenv("LEARNED_IMPORTANCE", "false").lower() == "true":
        try:
            from agentic.obj2_importance.logger import log_reflection
            log_reflection(
                date=str(state["cur_date"]),
                ticker=state["ticker"],
                decision=final_decision.lower(),
                memory_ids_used=all_ids,
                rationale=final_rationale,
                cumulative_return=0.0,
            )
        except Exception as e:
            logger.warning(f"[Obj4] Obj2 reflection logging failed: {e}")

    logger.info(f"[Obj4] FINAL DECISION: {final_decision} "
                f"(conf: {final_confidence:.2f}) | "
                f"Override: {override_used} | "
                f"Pivotal IDs: {len(all_ids)}")

    return {
        "final_decision": final_decision,
        "final_confidence": final_confidence,
        "final_rationale": final_rationale,
        "kelly_shares": pm_result.get("shares_to_trade", state.get("kelly_shares", 0)),
        "all_pivotal_ids": all_ids,
    }
