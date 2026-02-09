"""
Stock Price Data Fetcher

Uses yfinance (Yahoo Finance) for free stock price data.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import pandas as pd


@dataclass
class PriceData:
    """Stock price data container."""
    ticker: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # Computed fields
    change: float = 0.0
    change_percent: float = 0.0
    
    def to_summary(self) -> str:
        """Generate a text summary for memory storage."""
        direction = "up" if self.change >= 0 else "down"
        return (
            f"{self.ticker} on {self.date.strftime('%Y-%m-%d')}: "
            f"Closed at ${self.close:.2f} ({direction} {abs(self.change_percent):.2f}%). "
            f"High: ${self.high:.2f}, Low: ${self.low:.2f}, Volume: {self.volume:,}"
        )


class PriceFetcher:
    """Fetches stock price data from Yahoo Finance."""
    
    def __init__(self):
        """Initialize the price fetcher."""
        self._yf = None
    
    @property
    def yf(self):
        """Lazy load yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                raise ImportError(
                    "yfinance is required. Install with: pip install yfinance"
                )
        return self._yf
    
    def get_current_price(self, ticker: str) -> Optional[PriceData]:
        """Get the most recent price for a ticker.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            PriceData object or None if not found.
        """
        try:
            stock = self.yf.Ticker(ticker)
            hist = stock.history(period="5d")
            
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            prev_close = hist.iloc[-2]["Close"] if len(hist) > 1 else latest["Open"]
            
            change = latest["Close"] - prev_close
            change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
            
            return PriceData(
                ticker=ticker,
                date=hist.index[-1].to_pydatetime(),
                open=float(latest["Open"]),
                high=float(latest["High"]),
                low=float(latest["Low"]),
                close=float(latest["Close"]),
                volume=int(latest["Volume"]),
                change=change,
                change_percent=change_pct
            )
        except Exception as e:
            print(f"Error fetching price for {ticker}: {e}")
            return None
    
    def get_historical_prices(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = "3mo"
    ) -> List[PriceData]:
        """Get historical price data.
        
        Args:
            ticker: Stock ticker symbol.
            start_date: Start date for data.
            end_date: End date for data.
            period: Period string if dates not provided (1d, 5d, 1mo, 3mo, etc.).
            
        Returns:
            List of PriceData objects.
        """
        try:
            stock = self.yf.Ticker(ticker)
            
            if start_date and end_date:
                hist = stock.history(start=start_date, end=end_date)
            else:
                hist = stock.history(period=period)
            
            if hist.empty:
                return []
            
            prices = []
            prev_close = None
            
            for idx, row in hist.iterrows():
                close = float(row["Close"])
                change = close - prev_close if prev_close else 0
                change_pct = (change / prev_close * 100) if prev_close else 0
                
                prices.append(PriceData(
                    ticker=ticker,
                    date=idx.to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=close,
                    volume=int(row["Volume"]),
                    change=change,
                    change_percent=change_pct
                ))
                prev_close = close
            
            return prices
        except Exception as e:
            print(f"Error fetching historical prices for {ticker}: {e}")
            return []
    
    def get_info(self, ticker: str) -> Dict[str, Any]:
        """Get company information for fundamental analysis.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            Dictionary with company info.
        """
        try:
            stock = self.yf.Ticker(ticker)
            info = stock.info
            
            # Extract key fundamental data
            return {
                "name": info.get("longName", ticker),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "eps": info.get("trailingEps", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "avg_volume": info.get("averageVolume", 0),
                "description": info.get("longBusinessSummary", "")[:500]  # Truncate
            }
        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")
            return {"name": ticker, "error": str(e)}
    
    def get_fundamentals_summary(self, ticker: str) -> str:
        """Get a text summary of fundamentals for memory storage.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            Text summary of company fundamentals.
        """
        info = self.get_info(ticker)
        
        if "error" in info:
            return f"{ticker}: Unable to fetch fundamental data."
        
        pe_str = f"P/E: {info['pe_ratio']:.2f}" if info['pe_ratio'] else "P/E: N/A"
        eps_str = f"EPS: ${info['eps']:.2f}" if info['eps'] else "EPS: N/A"
        
        return (
            f"{info['name']} ({ticker}) - {info['sector']} / {info['industry']}. "
            f"Market Cap: ${info['market_cap']:,.0f}. {pe_str}, {eps_str}. "
            f"52-week range: ${info['52_week_low']:.2f} - ${info['52_week_high']:.2f}. "
            f"{info['description'][:200]}..."
        )
