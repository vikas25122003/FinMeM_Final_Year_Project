"""
Market Environment

Day-by-day market data stepper.
Provides market info one day at a time: (date, price, filing_k, filing_q, news, record).
"""

import os
import pickle
from datetime import date, datetime
from typing import Dict, Any, List, Optional, Tuple, Union


# Type alias for one day's market info
# (cur_date, cur_price, filing_k, filing_q, news_list, future_record, terminated)
MarketInfoType = Tuple[
    date,                   # Current date
    float,                  # Current price
    Optional[str],          # 10-K filing text (if any)
    Optional[str],          # 10-Q filing text (if any)
    List[str],              # News articles
    Optional[float],        # Next-day price change (train mode)
    bool,                   # Termination flag
]

TerminatedInfoType = Tuple[None, None, None, None, None, None, bool]


class MarketEnvironment:
    """Day-by-day market environment.
    
    Iterates through a date-indexed dataset, returning one day's data per step().
    
    Expected data format:
    {
        date(2024, 1, 1): {
            "price": {symbol: 150.0},
            "filing_k": {symbol: "annual report text..." or ""},
            "filing_q": {symbol: "quarterly report text..." or ""},
            "news": {symbol: ["headline 1", "headline 2", ...]},
        },
        ...
    }
    """

    def __init__(
        self,
        env_data: Dict[date, Dict[str, Any]],
        start_date: date,
        end_date: date,
        symbol: str,
    ):
        """Initialize the market environment.
        
        Args:
            env_data: Date-indexed market data dictionary.
            start_date: First trading date.
            end_date: Last trading date.
            symbol: Stock ticker to trade.
        """
        self.env_data = env_data
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        
        # Filter and sort dates
        all_dates = sorted(env_data.keys())
        self.date_series_full = [
            d for d in all_dates if start_date <= d <= end_date
        ]
        self.date_series = list(self.date_series_full)
        self.simulation_length = len(self.date_series) - 1
        self.cur_date: Optional[date] = None

    def reset(self) -> None:
        """Reset the environment to start."""
        self.date_series = list(self.date_series_full)
        self.cur_date = None

    def step(self) -> Union[MarketInfoType, TerminatedInfoType]:
        """Get the next day's market data.
        
        Returns:
            Tuple of (date, price, filing_k, filing_q, news, future_record, terminated).
            Returns (None, ..., True) when all dates are exhausted.
        """
        try:
            self.cur_date = self.date_series.pop(0)
        except IndexError:
            return None, None, None, None, None, None, True

        # Get future date for computing record
        try:
            future_date = self.date_series[0]
        except IndexError:
            return None, None, None, None, None, None, True

        day_data = self.env_data[self.cur_date]
        future_data = self.env_data[future_date]

        # Extract price
        cur_price = day_data["price"].get(self.symbol, 0.0)
        future_price = future_data["price"].get(self.symbol, 0.0)
        future_record = future_price - cur_price

        # Extract filings
        filing_k = day_data.get("filing_k", {}).get(self.symbol) or None
        filing_q = day_data.get("filing_q", {}).get(self.symbol) or None

        # Extract news
        news_data = day_data.get("news", {}).get(self.symbol, [])
        if isinstance(news_data, str):
            news = [news_data] if news_data else []
        else:
            news = news_data if news_data else []

        return (
            self.cur_date,
            cur_price,
            filing_k,
            filing_q,
            news,
            future_record,
            False,
        )

    @property
    def days_remaining(self) -> int:
        """Number of trading days remaining."""
        return len(self.date_series)

    def save_checkpoint(self, path: str) -> None:
        """Save environment state."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "env.pkl"), "wb") as f:
            pickle.dump({
                "env_data": self.env_data,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "symbol": self.symbol,
                "date_series": self.date_series,
                "cur_date": self.cur_date,
            }, f)

    @classmethod
    def load_checkpoint(cls, path: str) -> "MarketEnvironment":
        """Load environment state."""
        with open(os.path.join(path, "env.pkl"), "rb") as f:
            state = pickle.load(f)
        
        env = cls(
            env_data=state["env_data"],
            start_date=state["start_date"],
            end_date=state["end_date"],
            symbol=state["symbol"],
        )
        env.date_series = state["date_series"]
        env.cur_date = state["cur_date"]
        return env
