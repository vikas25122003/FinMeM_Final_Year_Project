"""
Portfolio Tracker

Tracks stock positions, cash, P&L, and provides feedback signals
for the memory access counter system.
"""

from datetime import date, datetime
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass, field


@dataclass
class Portfolio:
    """Portfolio tracker with momentum calculation and feedback generation.
    
    Based on the reference implementation's Portfolio class.
    """
    
    symbol: str
    lookback_window_size: int = 7
    
    # State
    cash: float = 100000.0
    shares: float = 0.0
    entry_price: float = 0.0
    
    # Price history
    price_history: List[float] = field(default_factory=list)
    date_history: List[Union[date, datetime]] = field(default_factory=list)
    
    # Action history
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    portfolio_value_history: List[float] = field(default_factory=list)
    
    # Current market price
    current_price: float = 0.0
    current_date: Optional[Union[date, datetime]] = None

    @property
    def position_value(self) -> float:
        """Current value of stock position."""
        return self.shares * self.current_price

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        return self.cash + self.position_value

    @property
    def has_position(self) -> bool:
        """Whether we hold any shares."""
        return self.shares > 0

    def update_market_info(
        self,
        new_market_price_info: float,
        cur_date: Union[date, datetime],
    ) -> None:
        """Update current market price and date.
        
        Args:
            new_market_price_info: Current stock price.
            cur_date: Current date.
        """
        self.current_price = new_market_price_info
        self.current_date = cur_date
        self.price_history.append(new_market_price_info)
        self.date_history.append(cur_date)

    def record_action(self, action: Dict[str, int]) -> None:
        """Record a trading action.
        
        Args:
            action: {"direction": 1 (buy), -1 (sell), 0 (hold)}
        """
        direction = action.get("direction", 0)
        
        trade_record = {
            "date": self.current_date,
            "price": self.current_price,
            "direction": direction,
            "shares_before": self.shares,
            "cash_before": self.cash,
        }
        
        if direction == 1:  # Buy
            # Paper: trade exactly 1 share per decision
            new_shares = 1.0
            invest_amount = self.current_price * new_shares
            if invest_amount <= self.cash and self.current_price > 0:
                self.cash -= invest_amount
                
                # Average into position
                if self.shares > 0:
                    total_value = self.shares * self.entry_price + invest_amount
                    self.shares += new_shares
                    self.entry_price = total_value / self.shares
                else:
                    self.shares = new_shares
                    self.entry_price = self.current_price
                
                trade_record["action"] = "BUY"
                trade_record["shares_traded"] = new_shares
                trade_record["amount"] = invest_amount
            else:
                trade_record["action"] = "HOLD"  # Insufficient funds
                trade_record["shares_traded"] = 0

        elif direction == -1:  # Sell
            # Paper: sell 1 share at a time
            if self.shares >= 1.0:
                shares_to_sell = 1.0
                sell_value = shares_to_sell * self.current_price
                pnl = sell_value - (shares_to_sell * self.entry_price)
                
                trade_record["action"] = "SELL"
                trade_record["shares_traded"] = shares_to_sell
                trade_record["amount"] = sell_value
                trade_record["pnl"] = pnl
                
                self.cash += sell_value
                self.shares -= shares_to_sell
                if self.shares < 0.001:  # Float cleanup
                    self.shares = 0.0
                    self.entry_price = 0.0
            else:
                trade_record["action"] = "HOLD"  # No shares to sell
                trade_record["shares_traded"] = 0

        else:  # Hold
            trade_record["action"] = "HOLD"
            trade_record["shares_traded"] = 0
        
        trade_record["shares_after"] = self.shares
        trade_record["cash_after"] = self.cash
        trade_record["total_value"] = self.total_value
        
        self.action_history.append(trade_record)

    def update_portfolio_series(self) -> None:
        """Update portfolio value history."""
        self.portfolio_value_history.append(self.total_value)

    def get_moment(self, moment_window: int = 3) -> Optional[Dict[str, int]]:
        """Get momentum signal based on recent price changes.
        
        Args:
            moment_window: Number of days to look back.
            
        Returns:
            Dict with "moment": -1 (negative), 0 (zero), 1 (positive),
            or None if insufficient data.
        """
        if len(self.price_history) < moment_window + 1:
            return None
        
        recent_prices = self.price_history[-(moment_window + 1):]
        cumulative_return = recent_prices[-1] - recent_prices[0]
        
        if cumulative_return > 0:
            moment = 1
        elif cumulative_return < 0:
            moment = -1
        else:
            moment = 0
        
        return {"moment": moment}

    def get_feedback_response(self) -> Optional[Dict[str, Any]]:
        """Get feedback for memory access counter updates.
        
        Compares the last two portfolio values to determine
        if the most recent action was profitable.
        
        Returns:
            Dict with "feedback" (+1 or -1) and "date",
            or None if insufficient history.
        """
        if len(self.portfolio_value_history) < 2:
            return None
        
        if len(self.action_history) < 2:
            return None
        
        prev_value = self.portfolio_value_history[-2]
        curr_value = self.portfolio_value_history[-1]
        
        change = curr_value - prev_value
        
        if change == 0:
            return None
        
        feedback = 1 if change > 0 else -1
        prev_action = self.action_history[-2]
        
        return {
            "feedback": feedback,
            "date": prev_action.get("date"),
        }

    def get_summary(self) -> str:
        """Get a text summary of portfolio state."""
        unrealized_pnl = 0.0
        if self.shares > 0 and self.entry_price > 0:
            unrealized_pnl = self.shares * (self.current_price - self.entry_price)
        
        return (
            f"Cash: ${self.cash:,.2f}, "
            f"Shares: {self.shares:.2f}, "
            f"Position Value: ${self.position_value:,.2f}, "
            f"Total: ${self.total_value:,.2f}, "
            f"Unrealized P&L: ${unrealized_pnl:,.2f}"
        )
