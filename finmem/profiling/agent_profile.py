"""
Agent Profiling Module

Defines the agent's personality, risk tolerance, and trading style.
Based on the Profiling module from the FinMEM paper.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from ..config import ProfileConfig, DEFAULT_CONFIG


class RiskLevel(Enum):
    """Risk tolerance levels."""
    CONSERVATIVE = 0.3
    MODERATE = 0.5
    AGGRESSIVE = 0.7


class TradingStyle(Enum):
    """Trading styles."""
    VALUE = "value"           # Focus on undervalued stocks
    GROWTH = "growth"         # Focus on high-growth potential
    MOMENTUM = "momentum"     # Follow market trends
    BALANCED = "balanced"     # Mix of strategies


@dataclass
class AgentProfile:
    """
    Agent profile defining personality and trading behavior.
    
    Based on the FinMEM paper's Profiling module which customizes
    the agent's characteristics for improved decision diversity.
    """
    
    name: str = "FinMEM Agent"
    risk_level: RiskLevel = RiskLevel.MODERATE
    trading_style: TradingStyle = TradingStyle.BALANCED
    
    # Dynamic risk adjustment based on market conditions
    adaptive_risk: bool = True
    
    # Personality traits for prompts
    traits: dict = field(default_factory=lambda: {
        "analytical": True,
        "patient": True,
        "disciplined": True
    })
    
    def __post_init__(self):
        """Initialize from config if values are default."""
        pass
    
    @classmethod
    def from_config(cls, config: Optional[ProfileConfig] = None) -> "AgentProfile":
        """Create an AgentProfile from configuration.
        
        Args:
            config: Profile configuration. Uses default if not provided.
            
        Returns:
            Configured AgentProfile instance.
        """
        config = config or DEFAULT_CONFIG.profile
        
        # Map risk tolerance to risk level
        if config.risk_tolerance <= 0.35:
            risk_level = RiskLevel.CONSERVATIVE
        elif config.risk_tolerance <= 0.6:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.AGGRESSIVE
            
        # Map trading style
        style_map = {
            "value": TradingStyle.VALUE,
            "growth": TradingStyle.GROWTH,
            "momentum": TradingStyle.MOMENTUM,
            "balanced": TradingStyle.BALANCED
        }
        trading_style = style_map.get(config.trading_style, TradingStyle.BALANCED)
        
        return cls(
            risk_level=risk_level,
            trading_style=trading_style
        )
    
    def get_system_prompt(self) -> str:
        """Generate a system prompt based on the agent's profile.
        
        Returns:
            System prompt string for LLM context.
        """
        risk_descriptions = {
            RiskLevel.CONSERVATIVE: "conservative and risk-averse",
            RiskLevel.MODERATE: "balanced and measured",
            RiskLevel.AGGRESSIVE: "aggressive and growth-focused"
        }
        
        style_descriptions = {
            TradingStyle.VALUE: "identifying undervalued companies with strong fundamentals",
            TradingStyle.GROWTH: "finding high-growth potential stocks",
            TradingStyle.MOMENTUM: "following market trends and momentum signals",
            TradingStyle.BALANCED: "a balanced approach combining multiple strategies"
        }
        
        return f"""You are {self.name}, a professional financial analyst and trader.

## Your Trading Profile

**Risk Approach**: You are {risk_descriptions[self.risk_level]} in your investment decisions.

**Trading Strategy**: You specialize in {style_descriptions[self.trading_style]}.

## Your Responsibilities

1. Analyze market data, news, and financial information thoroughly
2. Consider both short-term signals and long-term trends
3. Make well-reasoned trading recommendations (BUY, HOLD, or SELL)
4. Always explain your reasoning clearly
5. Account for risk management in every decision

## Your Traits

- Analytical: You rely on data and evidence
- Patient: You don't rush into decisions
- Disciplined: You follow your strategy consistently
"""
    
    def adjust_risk_for_volatility(self, volatility: float) -> float:
        """Adjust risk tolerance based on market volatility.
        
        Args:
            volatility: Current market volatility (0-1 scale).
            
        Returns:
            Adjusted risk tolerance value.
        """
        if not self.adaptive_risk:
            return self.risk_level.value
        
        # Reduce risk tolerance in high volatility
        base_risk = self.risk_level.value
        if volatility > 0.7:
            return max(0.2, base_risk - 0.2)
        elif volatility > 0.5:
            return max(0.3, base_risk - 0.1)
        else:
            return base_risk
    
    def get_position_size_factor(self) -> float:
        """Get position sizing factor based on risk profile.
        
        Returns:
            Multiplier for position sizing (0.5 to 1.5).
        """
        return {
            RiskLevel.CONSERVATIVE: 0.6,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.AGGRESSIVE: 1.4
        }[self.risk_level]


# Pre-configured profiles
CONSERVATIVE_PROFILE = AgentProfile(
    name="Conservative Trader",
    risk_level=RiskLevel.CONSERVATIVE,
    trading_style=TradingStyle.VALUE
)

MODERATE_PROFILE = AgentProfile(
    name="Balanced Trader",
    risk_level=RiskLevel.MODERATE,
    trading_style=TradingStyle.BALANCED
)

AGGRESSIVE_PROFILE = AgentProfile(
    name="Growth Trader",
    risk_level=RiskLevel.AGGRESSIVE,
    trading_style=TradingStyle.GROWTH
)
