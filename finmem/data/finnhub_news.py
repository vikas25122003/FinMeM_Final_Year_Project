"""
Finnhub News Fetcher

Uses Finnhub API for stock-specific news (free tier: 60 calls/min, 1 year history).
Get your free API key at: https://finnhub.io/
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Optional, List
from dataclasses import dataclass

from ..config import DEFAULT_CONFIG


@dataclass
class FinnhubNewsArticle:
    """Finnhub news article container."""
    headline: str
    source: str
    published: datetime
    url: str
    summary: str
    ticker: Optional[str] = None
    category: Optional[str] = None
    sentiment: Optional[float] = None  # Finnhub provides sentiment on paid plans
    
    def to_memory_content(self) -> str:
        """Generate text content for memory storage."""
        date_str = self.published.strftime("%Y-%m-%d")
        summary_text = f" {self.summary[:200]}..." if len(self.summary) > 200 else f" {self.summary}"
        return f"[{date_str}] {self.source}: {self.headline}.{summary_text}"


class FinnhubNewsFetcher:
    """Fetches stock news from Finnhub API.
    
    Free tier:
    - 60 API calls per minute
    - 30 API calls per second
    - 1 year of company news history
    - Real-time news updates
    """
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: Optional[str] = None, max_articles: int = 20):
        """Initialize Finnhub news fetcher.
        
        Args:
            api_key: Finnhub API key. Uses FINNHUB_API_KEY env var if not provided.
            max_articles: Maximum articles to fetch per query.
        """
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY", "")
        self.max_articles = max_articles
        
        if not self.api_key:
            print("Warning: FINNHUB_API_KEY not set. Add it to your .env file.")
    
    def _make_request(self, endpoint: str, params: dict) -> dict:
        """Make authenticated request to Finnhub API.
        
        Args:
            endpoint: API endpoint path.
            params: Query parameters.
            
        Returns:
            JSON response as dict.
        """
        params["token"] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Finnhub API error: {e}")
            return {}
    
    def fetch_company_news(
        self,
        ticker: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[FinnhubNewsArticle]:
        """Fetch news for a specific company/stock.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'NVDA').
            from_date: Start date. Defaults to 7 days ago.
            to_date: End date. Defaults to today.
            
        Returns:
            List of FinnhubNewsArticle objects.
        """
        if not self.api_key:
            return []
        
        # Default date range: last 7 days
        if to_date is None:
            to_date = datetime.now()
        if from_date is None:
            from_date = to_date - timedelta(days=7)
        
        params = {
            "symbol": ticker.upper(),
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d")
        }
        
        data = self._make_request("company-news", params)
        
        if not data or not isinstance(data, list):
            return []
        
        articles = []
        for item in data[:self.max_articles]:
            try:
                # Finnhub returns Unix timestamp
                pub_time = datetime.fromtimestamp(item.get("datetime", 0))
                
                articles.append(FinnhubNewsArticle(
                    headline=item.get("headline", ""),
                    source=item.get("source", "Unknown"),
                    published=pub_time,
                    url=item.get("url", ""),
                    summary=item.get("summary", ""),
                    ticker=ticker.upper(),
                    category=item.get("category", None)
                ))
            except Exception:
                continue
        
        return articles
    
    def fetch_market_news(
        self,
        category: str = "general"
    ) -> List[FinnhubNewsArticle]:
        """Fetch general market news.
        
        Args:
            category: News category ('general', 'forex', 'crypto', 'merger').
            
        Returns:
            List of FinnhubNewsArticle objects.
        """
        if not self.api_key:
            return []
        
        params = {"category": category}
        data = self._make_request("news", params)
        
        if not data or not isinstance(data, list):
            return []
        
        articles = []
        for item in data[:self.max_articles]:
            try:
                pub_time = datetime.fromtimestamp(item.get("datetime", 0))
                
                articles.append(FinnhubNewsArticle(
                    headline=item.get("headline", ""),
                    source=item.get("source", "Unknown"),
                    published=pub_time,
                    url=item.get("url", ""),
                    summary=item.get("summary", ""),
                    category=item.get("category", category)
                ))
            except Exception:
                continue
        
        return articles
    
    def fetch_and_format_for_memory(
        self,
        ticker: str,
        days_back: int = 7
    ) -> List[str]:
        """Fetch news and format for memory storage.
        
        Args:
            ticker: Stock ticker symbol.
            days_back: How many days back to search.
            
        Returns:
            List of formatted text strings for memory storage.
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        articles = self.fetch_company_news(ticker, from_date, to_date)
        return [article.to_memory_content() for article in articles]
    
    def test_connection(self) -> bool:
        """Test API connection.
        
        Returns:
            True if connection successful.
        """
        if not self.api_key:
            print("Error: FINNHUB_API_KEY not set")
            return False
        
        try:
            # Try to fetch market news as a simple test
            data = self._make_request("news", {"category": "general"})
            return isinstance(data, list) and len(data) > 0
        except Exception:
            return False
