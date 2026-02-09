"""
FinMEM Trading Simulator

Simulates trading with the FinMEM agent over a time period.
Supports train mode (populate memory) and test mode (make decisions).
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import json

from ..config import FinMEMConfig, DEFAULT_CONFIG
from ..llm_client import LLMClient
from ..profiling.agent_profile import AgentProfile
from ..memory.layered_memory import LayeredMemory, MemoryLayer
from ..decision.decision_engine import DecisionEngine, TradeDecision, TradeAction
from ..data.price_fetcher import PriceFetcher
from ..data.news_fetcher import NewsFetcher
from ..data.finnhub_news import FinnhubNewsFetcher


def _normalize_datetime(dt: datetime) -> datetime:
    """Convert timezone-aware datetime to timezone-naive (strip tzinfo)."""
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


@dataclass
class Position:
    """A stock position."""
    ticker: str
    shares: float
    entry_price: float
    entry_date: datetime
    
    @property
    def value(self) -> float:
        """Current position value at entry price."""
        return self.shares * self.entry_price


@dataclass 
class Portfolio:
    """Portfolio tracker."""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def total_value(self) -> float:
        """Total portfolio value."""
        position_value = sum(p.value for p in self.positions.values())
        return self.cash + position_value
    
    def get_position_size(self, ticker: str) -> float:
        """Get position size as percentage of portfolio."""
        if ticker not in self.positions:
            return 0.0
        return self.positions[ticker].value / self.total_value


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    decisions: List[TradeDecision]
    trades: List[Dict[str, Any]]
    
    @property
    def total_return(self) -> float:
        """Total return percentage."""
        return ((self.final_value - self.initial_capital) / self.initial_capital) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "final_value": self.final_value,
            "total_return_percent": self.total_return,
            "num_decisions": len(self.decisions),
            "num_trades": len(self.trades)
        }


class TradingSimulator:
    """
    FinMEM Trading Simulator.
    
    Runs the trading agent over a time period in either:
    - Train mode: Populates memory with market data
    - Test mode: Uses memory to make trading decisions
    """
    
    def __init__(self, config: Optional[FinMEMConfig] = None):
        """Initialize the simulator.
        
        Args:
            config: FinMEM configuration.
        """
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components
        self.memory = LayeredMemory(self.config.memory)
        self.llm = LLMClient(self.config.llm)
        self.profile = AgentProfile.from_config(self.config.profile)
        self.decision_engine = DecisionEngine(self.memory, self.llm, self.profile)
        
        # Data fetchers
        self.price_fetcher = PriceFetcher()
        self.finnhub_fetcher = FinnhubNewsFetcher(max_articles=self.config.data.news_max_articles)
        self.google_news_fetcher = NewsFetcher(self.config.data.news_max_articles)
        
        # Portfolio
        self.portfolio = Portfolio(cash=self.config.initial_capital)
        
        # Results tracking
        self.decisions: List[TradeDecision] = []
        self.trades: List[Dict[str, Any]] = []
    
    def populate_memory(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        verbose: bool = True
    ):
        """Populate memory with historical data (train mode).
        
        Args:
            ticker: Stock ticker.
            start_date: Start date for data.
            end_date: End date for data.
            verbose: Print progress.
        """
        if verbose:
            print(f"Populating memory for {ticker} from {start_date.date()} to {end_date.date()}")
        
        # Get company fundamentals (deep memory)
        fundamentals = self.price_fetcher.get_fundamentals_summary(ticker)
        self.memory.add(
            content=fundamentals,
            source="fundamental",
            ticker=ticker,
            importance=0.8,
            timestamp=start_date,
            layer=MemoryLayer.DEEP
        )
        if verbose:
            print(f"  Added fundamentals to deep memory")
        
        # Get historical prices
        prices = self.price_fetcher.get_historical_prices(
            ticker, start_date, end_date
        )
        
        for price in prices:
            # Determine layer based on how old the data is
            # Normalize to avoid timezone comparison issues
            price_date = _normalize_datetime(price.date)
            age_days = (end_date - price_date).days
            
            if age_days <= 3:
                layer = MemoryLayer.SHALLOW
                importance = 0.7
            elif age_days <= 21:
                layer = MemoryLayer.INTERMEDIATE
                importance = 0.5
            else:
                layer = MemoryLayer.DEEP
                importance = 0.3
            
            self.memory.add(
                content=price.to_summary(),
                source="price",
                ticker=ticker,
                importance=importance,
                timestamp=price.date,
                layer=layer
            )
        
        if verbose:
            print(f"  Added {len(prices)} price records")
        
        # Get news (Finnhub primary, Google News RSS fallback)
        days_back = (end_date - start_date).days
        
        # Try Finnhub first
        news_items = self.finnhub_fetcher.fetch_and_format_for_memory(ticker, days_back)
        news_source = "Finnhub"
        
        # Fallback to Google News RSS if Finnhub returns nothing
        if not news_items:
            news_items = self.google_news_fetcher.fetch_and_format_for_memory(
                ticker, days_back=days_back
            )
            news_source = "Google News"
        
        for news in news_items:
            self.memory.add(
                content=news,
                source="news",
                ticker=ticker,
                importance=0.6,
                layer=MemoryLayer.SHALLOW
            )
        
        if verbose:
            print(f"  Added {len(news_items)} news articles (via {news_source})")
            stats = self.memory.stats()
            print(f"  Memory stats: {stats['by_layer']}")
    
    def make_decision(
        self,
        ticker: str,
        current_price: Optional[float] = None
    ) -> TradeDecision:
        """Make a trading decision for a ticker.
        
        Args:
            ticker: Stock ticker.
            current_price: Current price (fetched if not provided).
            
        Returns:
            TradeDecision object.
        """
        # Get current price if not provided
        if current_price is None:
            price_data = self.price_fetcher.get_current_price(ticker)
            current_price = price_data.close if price_data else None
        
        # Build portfolio context
        position_size = self.portfolio.get_position_size(ticker)
        portfolio_context = (
            f"Cash: ${self.portfolio.cash:,.2f}, "
            f"Current {ticker} position: {position_size:.1%} of portfolio"
        )
        
        # Make decision
        decision = self.decision_engine.decide(
            ticker=ticker,
            current_price=current_price,
            portfolio_context=portfolio_context
        )
        
        self.decisions.append(decision)
        return decision
    
    def execute_decision(
        self,
        decision: TradeDecision,
        price: float
    ) -> Optional[Dict[str, Any]]:
        """Execute a trading decision.
        
        Args:
            decision: The trading decision.
            price: Execution price.
            
        Returns:
            Trade record or None if no trade executed.
        """
        ticker = decision.ticker
        
        if decision.action == TradeAction.BUY:
            # Calculate position size
            max_investment = self.portfolio.cash * min(
                decision.suggested_size,
                self.config.max_position_size
            )
            
            if max_investment < 100:  # Minimum trade size
                return None
            
            shares = max_investment / price
            
            # Update portfolio
            self.portfolio.cash -= max_investment
            
            if ticker in self.portfolio.positions:
                # Average into existing position
                existing = self.portfolio.positions[ticker]
                total_shares = existing.shares + shares
                avg_price = (
                    (existing.shares * existing.entry_price + shares * price)
                    / total_shares
                )
                self.portfolio.positions[ticker] = Position(
                    ticker=ticker,
                    shares=total_shares,
                    entry_price=avg_price,
                    entry_date=existing.entry_date
                )
            else:
                self.portfolio.positions[ticker] = Position(
                    ticker=ticker,
                    shares=shares,
                    entry_price=price,
                    entry_date=decision.timestamp
                )
            
            trade = {
                "action": "BUY",
                "ticker": ticker,
                "shares": shares,
                "price": price,
                "value": max_investment,
                "timestamp": decision.timestamp.isoformat()
            }
            self.trades.append(trade)
            return trade
            
        elif decision.action == TradeAction.SELL:
            if ticker not in self.portfolio.positions:
                return None
            
            position = self.portfolio.positions[ticker]
            shares = position.shares
            value = shares * price
            
            # Update portfolio
            self.portfolio.cash += value
            del self.portfolio.positions[ticker]
            
            # Calculate P&L
            pnl = value - (shares * position.entry_price)
            
            trade = {
                "action": "SELL",
                "ticker": ticker,
                "shares": shares,
                "price": price,
                "value": value,
                "pnl": pnl,
                "timestamp": decision.timestamp.isoformat()
            }
            self.trades.append(trade)
            return trade
        
        return None  # HOLD
    
    def run(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        mode: str = "train",
        verbose: bool = True
    ) -> SimulationResult:
        """Run the simulation.
        
        Args:
            ticker: Stock ticker to trade.
            start_date: Start date.
            end_date: End date.
            mode: 'train' or 'test'.
            verbose: Print progress.
            
        Returns:
            SimulationResult object.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"FinMEM Trading Simulation")
            print(f"Ticker: {ticker}")
            print(f"Period: {start_date.date()} to {end_date.date()}")
            print(f"Mode: {mode.upper()}")
            print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
            print(f"{'='*60}\n")
        
        # Populate memory
        self.populate_memory(ticker, start_date, end_date, verbose)
        
        if mode == "train":
            if verbose:
                print("\nTraining complete. Memory populated.")
            
            return SimulationResult(
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.config.initial_capital,
                final_value=self.portfolio.total_value,
                decisions=self.decisions,
                trades=self.trades
            )
        
        # Test mode: make decisions
        if verbose:
            print("\nMaking trading decision...")
        
        price_data = self.price_fetcher.get_current_price(ticker)
        current_price = price_data.close if price_data else None
        
        decision = self.make_decision(ticker, current_price)
        
        if verbose:
            print(f"\nDecision: {decision}")
        
        # Execute if not HOLD
        if decision.action != TradeAction.HOLD and current_price:
            trade = self.execute_decision(decision, current_price)
            if trade and verbose:
                print(f"Executed: {trade['action']} {trade['shares']:.2f} shares @ ${trade['price']:.2f}")
        
        return SimulationResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.config.initial_capital,
            final_value=self.portfolio.total_value,
            decisions=self.decisions,
            trades=self.trades
        )
    
    def save_state(self, filepath: str):
        """Save simulator state to file.
        
        Args:
            filepath: Path to save state.
        """
        state = {
            "portfolio": {
                "cash": self.portfolio.cash,
                "positions": {
                    k: {
                        "ticker": v.ticker,
                        "shares": v.shares,
                        "entry_price": v.entry_price,
                        "entry_date": v.entry_date.isoformat()
                    }
                    for k, v in self.portfolio.positions.items()
                }
            },
            "decisions": [d.to_dict() for d in self.decisions],
            "trades": self.trades
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
