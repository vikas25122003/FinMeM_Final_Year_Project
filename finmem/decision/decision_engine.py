"""
Decision Engine Module

Converts memory insights into trading decisions using the LLM.
Based on the Decision-making module from the FinMEM paper.
"""

import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from ..config import DEFAULT_CONFIG
from ..llm_client import LLMClient, ChatMessage
from ..profiling.agent_profile import AgentProfile
from ..memory.layered_memory import LayeredMemory, MemoryLayer


class TradeAction(Enum):
    """Trading actions."""
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


@dataclass
class TradeDecision:
    """A trading decision with reasoning."""
    
    action: TradeAction
    confidence: float  # 0.0 to 1.0
    reasoning: str
    
    # Supporting data
    ticker: str
    timestamp: datetime
    price: Optional[float] = None
    
    # Position sizing (based on confidence and risk profile)
    suggested_size: float = 0.0  # Percentage of portfolio
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "suggested_size": self.suggested_size
        }
    
    def __str__(self) -> str:
        """Human-readable string."""
        return (
            f"{self.action.value} {self.ticker} "
            f"(Confidence: {self.confidence:.0%})\n"
            f"Reasoning: {self.reasoning}"
        )


class DecisionEngine:
    """
    Trading decision engine based on the FinMEM paper.
    
    Retrieves relevant memories from all layers, scores them,
    and uses LLM to generate trading decisions.
    """
    
    def __init__(
        self,
        memory: LayeredMemory,
        llm_client: Optional[LLMClient] = None,
        profile: Optional[AgentProfile] = None
    ):
        """Initialize the decision engine.
        
        Args:
            memory: The layered memory system.
            llm_client: LLM client for decision generation.
            profile: Agent profile for personality/risk settings.
        """
        self.memory = memory
        self.llm = llm_client or LLMClient()
        self.profile = profile or AgentProfile.from_config()
    
    def _build_context_prompt(
        self,
        ticker: str,
        current_price: Optional[float] = None,
        portfolio_context: Optional[str] = None
    ) -> str:
        """Build context prompt from memory layers.
        
        Args:
            ticker: Stock ticker to analyze.
            current_price: Current stock price.
            portfolio_context: Portfolio context string.
            
        Returns:
            Formatted context prompt.
        """
        # Get context from all memory layers
        context = self.memory.get_context_summary(
            ticker=ticker,
            query=f"trading decision analysis for {ticker}",
            max_items_per_layer=5
        )
        
        prompt_parts = [f"## Analysis Context for {ticker}"]
        
        if current_price:
            prompt_parts.append(f"\n**Current Price**: ${current_price:.2f}")
        
        if portfolio_context:
            prompt_parts.append(f"\n**Portfolio Context**: {portfolio_context}")
        
        # Add shallow layer (recent events)
        if context.get("shallow"):
            prompt_parts.append("\n### Recent Events (Last Few Days)")
            for item in context["shallow"]:
                prompt_parts.append(f"- {item}")
        
        # Add intermediate layer (weekly trends)
        if context.get("intermediate"):
            prompt_parts.append("\n### Weekly Trends and Patterns")
            for item in context["intermediate"]:
                prompt_parts.append(f"- {item}")
        
        # Add deep layer (fundamentals)
        if context.get("deep"):
            prompt_parts.append("\n### Fundamental Analysis")
            for item in context["deep"]:
                prompt_parts.append(f"- {item}")
        
        return "\n".join(prompt_parts)
    
    def _parse_decision_response(
        self,
        response: str,
        ticker: str,
        price: Optional[float] = None
    ) -> TradeDecision:
        """Parse LLM response into a TradeDecision.
        
        Args:
            response: Raw LLM response.
            ticker: Stock ticker.
            price: Current price.
            
        Returns:
            Parsed TradeDecision object.
        """
        try:
            # Try to extract JSON from response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            
            data = json.loads(response)
            
            # Parse action
            action_str = data.get("action", "HOLD").upper()
            action = TradeAction[action_str] if action_str in TradeAction.__members__ else TradeAction.HOLD
            
            # Parse confidence
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            
            # Get reasoning
            reasoning = data.get("reasoning", "No reasoning provided.")
            
            # Calculate position size based on confidence and risk profile
            base_size = confidence * 0.2  # Max 20% of portfolio
            suggested_size = base_size * self.profile.get_position_size_factor()
            
            return TradeDecision(
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                ticker=ticker,
                timestamp=datetime.now(),
                price=price,
                suggested_size=suggested_size
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback parsing if JSON fails
            action = TradeAction.HOLD
            if "BUY" in response.upper():
                action = TradeAction.BUY
            elif "SELL" in response.upper():
                action = TradeAction.SELL
            
            return TradeDecision(
                action=action,
                confidence=0.5,
                reasoning=f"Parsed from response: {response[:200]}",
                ticker=ticker,
                timestamp=datetime.now(),
                price=price,
                suggested_size=0.05
            )
    
    def decide(
        self,
        ticker: str,
        current_price: Optional[float] = None,
        portfolio_context: Optional[str] = None
    ) -> TradeDecision:
        """Generate a trading decision for a ticker.
        
        Args:
            ticker: Stock ticker to analyze.
            current_price: Current stock price.
            portfolio_context: Context about current portfolio state.
            
        Returns:
            TradeDecision object with action and reasoning.
        """
        # Build context from memory
        context = self._build_context_prompt(ticker, current_price, portfolio_context)
        
        # Get system prompt from agent profile
        system_prompt = self.profile.get_system_prompt()
        
        # Build decision prompt
        decision_prompt = f"""{context}

## Task

Based on the above context, make a trading decision for {ticker}.

Respond with a JSON object in exactly this format:
{{
    "action": "BUY" or "HOLD" or "SELL",
    "confidence": 0.0 to 1.0,
    "reasoning": "Your detailed explanation..."
}}

Consider:
1. Recent news and events (shallow memory)
2. Weekly trends and patterns (intermediate memory)
3. Fundamental analysis (deep memory)
4. Your risk profile: {self.profile.risk_level.name}

Respond ONLY with the JSON object, no other text."""

        # Call LLM
        response = self.llm.chat_with_system(system_prompt, decision_prompt)
        
        # Parse and return decision
        return self._parse_decision_response(response, ticker, current_price)
    
    def explain_decision(self, decision: TradeDecision) -> str:
        """Generate a detailed explanation for a decision.
        
        Args:
            decision: The trading decision to explain.
            
        Returns:
            Detailed explanation string.
        """
        prompt = f"""Explain this trading decision in detail for a client:

Decision: {decision.action.value} {decision.ticker}
Confidence: {decision.confidence:.0%}
Initial Reasoning: {decision.reasoning}

Provide a clear, professional explanation covering:
1. Key factors that influenced this decision
2. Potential risks to consider
3. Suggested timeframe for this position
4. What signals might change this recommendation

Keep the explanation concise but thorough."""

        return self.llm.chat(prompt)
    
    def reflect_on_outcome(
        self,
        decision: TradeDecision,
        actual_outcome: str,
        profit_loss: float
    ) -> str:
        """Reflect on a past decision's outcome for learning.
        
        This is part of the feedback loop in the FinMEM architecture.
        
        Args:
            decision: The original trading decision.
            actual_outcome: Description of what actually happened.
            profit_loss: Profit/loss from the trade.
            
        Returns:
            Reflection and lessons learned.
        """
        prompt = f"""Reflect on this trading decision and its outcome:

Original Decision: {decision.action.value} {decision.ticker}
Confidence: {decision.confidence:.0%}
Reasoning: {decision.reasoning}

Actual Outcome: {actual_outcome}
Profit/Loss: ${profit_loss:,.2f}

Analyze:
1. Was the decision correct? Why or why not?
2. What signals were missed or misinterpreted?
3. What lessons can be learned for future decisions?

Be specific and actionable in your analysis."""

        reflection = self.llm.chat(prompt)
        
        # Store reflection in deep memory for future learning
        self.memory.add(
            content=f"Reflection on {decision.ticker} trade: {reflection[:500]}",
            source="reflection",
            ticker=decision.ticker,
            importance=0.7,  # Reflections are valuable
            layer=MemoryLayer.DEEP
        )
        
        return reflection
