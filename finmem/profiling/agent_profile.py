"""
Agent Profiling Module — Paper-Faithful Implementation

Based on the FinMEM paper's Profiling module:
- Three character modes: risk_seeking, risk_averse, self_adaptive
- Self-adaptive: switches risk mode based on 3-day cumulative return
- Dynamic character string: professional background + risk mode + ticker context
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from ..config import ProfileConfig, DEFAULT_CONFIG


class RiskMode(Enum):
    """Paper's three risk character modes."""
    RISK_SEEKING = "risk_seeking"
    RISK_AVERSE = "risk_averse"
    SELF_ADAPTIVE = "self_adaptive"


class TradingStyle(Enum):
    """Trading styles."""
    VALUE = "value"
    GROWTH = "growth"
    MOMENTUM = "momentum"
    BALANCED = "balanced"


@dataclass
class AgentProfile:
    """FinMEM Agent Profile with self-adaptive character setting.
    
    Paper: The agent alternates between risk-seeking and risk-averse
    based on the 3-day cumulative return. When return < 0, it switches
    to risk-averse. When return >= 0, it switches to risk-seeking.
    """
    
    name: str = "FinMEM Agent"
    mode: RiskMode = RiskMode.SELF_ADAPTIVE
    trading_style: TradingStyle = TradingStyle.BALANCED
    
    # Internal state for self-adaptive mode
    current_risk_mode: str = "risk_seeking"  # Tracks current active mode
    
    # Personality traits
    traits: dict = field(default_factory=lambda: {
        "analytical": True,
        "patient": True,
        "disciplined": True,
    })
    
    @classmethod
    def from_config(cls, config: Optional[ProfileConfig] = None) -> "AgentProfile":
        """Create an AgentProfile from configuration."""
        config = config or DEFAULT_CONFIG.profile
        
        # Map risk tolerance to mode
        if config.risk_tolerance <= 0.35:
            mode = RiskMode.RISK_AVERSE
        elif config.risk_tolerance >= 0.65:
            mode = RiskMode.RISK_SEEKING
        else:
            mode = RiskMode.SELF_ADAPTIVE
        
        style_map = {
            "value": TradingStyle.VALUE,
            "growth": TradingStyle.GROWTH,
            "momentum": TradingStyle.MOMENTUM,
            "balanced": TradingStyle.BALANCED,
        }
        trading_style = style_map.get(config.trading_style, TradingStyle.BALANCED)
        
        initial = "risk_seeking" if mode == RiskMode.RISK_SEEKING else (
            "risk_averse" if mode == RiskMode.RISK_AVERSE else "risk_seeking"
        )
        
        return cls(
            mode=mode,
            trading_style=trading_style,
            current_risk_mode=initial,
        )
    
    @classmethod
    def create_self_adaptive(cls) -> "AgentProfile":
        """Create a self-adaptive profile (paper's recommended mode)."""
        return cls(
            name="FinMEM Self-Adaptive Agent",
            mode=RiskMode.SELF_ADAPTIVE,
            trading_style=TradingStyle.BALANCED,
            current_risk_mode="risk_seeking",
        )
    
    def update_character(self, cumulative_3day_return: float) -> None:
        """Update character mode based on 3-day cumulative return.
        
        Paper algorithm:
            If cumulative return of past 3 days < 0:
                Switch to risk_averse
            If cumulative return of past 3 days >= 0:
                Switch to risk_seeking
        
        Only applies in SELF_ADAPTIVE mode.
        
        Args:
            cumulative_3day_return: The sum of returns over the past 3 days.
        """
        if self.mode != RiskMode.SELF_ADAPTIVE:
            return
        
        previous = self.current_risk_mode
        if cumulative_3day_return < 0:
            self.current_risk_mode = "risk_averse"
        else:
            self.current_risk_mode = "risk_seeking"
        
        if previous != self.current_risk_mode:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"Character switched: {previous} → {self.current_risk_mode} "
                f"(3-day return: {cumulative_3day_return:+.4f})"
            )
    
    def get_character_string(self, symbol: str) -> str:
        """Generate dynamic character string for memory retrieval query.
        
        Paper: Character = Professional Background + Risk Inclination + Ticker context.
        This string is used as the retrieval query for all 4 memory layers.
        
        Args:
            symbol: Stock ticker being analyzed.
            
        Returns:
            Character string for memory querying.
        """
        if self.current_risk_mode == "risk_seeking":
            risk_desc = (
                "You are a risk-seeking trader who actively pursues high-return "
                "opportunities. You are willing to accept higher volatility and "
                "potential losses in pursuit of significant gains."
            )
        else:
            risk_desc = (
                "You are a risk-averse trader who prioritizes capital preservation. "
                "You prefer stable, lower-risk investments and are cautious about "
                "entering volatile positions."
            )
        
        return (
            f"A professional financial analyst specializing in stock trading, "
            f"currently analyzing {symbol}. "
            f"Expert in analyzing market trends, financial statements, and news sentiment. "
            f"{risk_desc}"
        )
    
    def get_system_prompt(self) -> str:
        """Generate a system prompt based on the agent's profile."""
        risk_descriptions = {
            "risk_seeking": "aggressive and growth-focused, willing to take calculated risks",
            "risk_averse": "conservative and risk-averse, prioritizing capital preservation",
        }
        
        style_descriptions = {
            TradingStyle.VALUE: "identifying undervalued companies with strong fundamentals",
            TradingStyle.GROWTH: "finding high-growth potential stocks",
            TradingStyle.MOMENTUM: "following market trends and momentum signals",
            TradingStyle.BALANCED: "a balanced approach combining multiple strategies",
        }
        
        risk_text = risk_descriptions.get(
            self.current_risk_mode, "balanced and measured"
        )
        style_text = style_descriptions.get(
            self.trading_style, "a balanced approach"
        )
        
        adaptive_note = ""
        if self.mode == RiskMode.SELF_ADAPTIVE:
            adaptive_note = (
                "\n\n**Note**: Your risk profile adapts dynamically based on recent "
                "market performance. Currently in {} mode.".format(self.current_risk_mode)
            )
        
        return f"""You are {self.name}, a professional financial analyst and trader.

## Your Trading Profile

**Risk Approach**: You are {risk_text} in your investment decisions.{adaptive_note}

**Trading Strategy**: You specialize in {style_text}.

## Your Responsibilities

1. Analyze market data, news, and financial information thoroughly
2. Consider both short-term signals and long-term trends
3. Make well-reasoned trading recommendations (BUY, HOLD, or SELL)
4. Always explain your reasoning clearly
5. Account for risk management in every decision
"""
    
    def get_position_size_factor(self) -> float:
        """Get position sizing factor based on current risk mode."""
        if self.current_risk_mode == "risk_seeking":
            return 1.4
        else:
            return 0.6


# Pre-configured profiles
RISK_SEEKING_PROFILE = AgentProfile(
    name="Risk-Seeking Trader",
    mode=RiskMode.RISK_SEEKING,
    current_risk_mode="risk_seeking",
)

RISK_AVERSE_PROFILE = AgentProfile(
    name="Risk-Averse Trader",
    mode=RiskMode.RISK_AVERSE,
    current_risk_mode="risk_averse",
)

SELF_ADAPTIVE_PROFILE = AgentProfile(
    name="Self-Adaptive Trader",
    mode=RiskMode.SELF_ADAPTIVE,
    current_risk_mode="risk_seeking",
)
