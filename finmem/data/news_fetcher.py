"""
News Fetcher

Uses Google News RSS for free news data (no API key required).
"""

import re
from datetime import datetime, timedelta
from typing import Optional, List
from dataclasses import dataclass
from urllib.parse import quote_plus
import feedparser
from bs4 import BeautifulSoup


@dataclass
class NewsArticle:
    """News article container."""
    title: str
    source: str
    published: datetime
    link: str
    summary: Optional[str] = None
    ticker: Optional[str] = None
    
    def to_memory_content(self) -> str:
        """Generate text content for memory storage."""
        date_str = self.published.strftime("%Y-%m-%d")
        summary_text = f" {self.summary}" if self.summary else ""
        return f"[{date_str}] {self.source}: {self.title}.{summary_text}"


class NewsFetcher:
    """Fetches news from Google News RSS (free, no API key required)."""
    
    def __init__(self, max_articles: int = 20):
        """Initialize the news fetcher.
        
        Args:
            max_articles: Maximum articles to fetch per query.
        """
        self.max_articles = max_articles
        self.base_url = "https://news.google.com/rss/search"
    
    def _parse_date(self, date_string: str) -> datetime:
        """Parse RSS date string to datetime.
        
        Args:
            date_string: RSS date string.
            
        Returns:
            Parsed datetime object.
        """
        try:
            # Try common RSS date formats
            for fmt in [
                "%a, %d %b %Y %H:%M:%S %Z",
                "%a, %d %b %Y %H:%M:%S %z",
                "%Y-%m-%dT%H:%M:%SZ"
            ]:
                try:
                    return datetime.strptime(date_string, fmt)
                except ValueError:
                    continue
            
            # Fallback: use feedparser's date parsing
            import time
            parsed = feedparser._parse_date(date_string)
            if parsed:
                return datetime.fromtimestamp(time.mktime(parsed))
        except Exception:
            pass
        
        return datetime.now()
    
    def _clean_html(self, html_text: str) -> str:
        """Remove HTML tags from text.
        
        Args:
            html_text: Text potentially containing HTML.
            
        Returns:
            Clean text without HTML.
        """
        if not html_text:
            return ""
        soup = BeautifulSoup(html_text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    
    def fetch_stock_news(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        days_back: int = 7
    ) -> List[NewsArticle]:
        """Fetch news for a specific stock.
        
        Args:
            ticker: Stock ticker symbol.
            company_name: Company name for better search results.
            days_back: How many days back to search.
            
        Returns:
            List of NewsArticle objects.
        """
        # Build search query
        search_terms = [ticker]
        if company_name:
            search_terms.append(company_name)
        
        query = " OR ".join(search_terms) + " stock"
        return self.fetch_news(query, days_back)
    
    def fetch_news(
        self,
        query: str,
        days_back: int = 7
    ) -> List[NewsArticle]:
        """Fetch news for a general query.
        
        Args:
            query: Search query.
            days_back: How many days back to consider.
            
        Returns:
            List of NewsArticle objects.
        """
        encoded_query = quote_plus(query)
        url = f"{self.base_url}?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(url)
            
            if not feed.entries:
                return []
            
            articles = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for entry in feed.entries[:self.max_articles]:
                # Parse publication date
                pub_date = self._parse_date(
                    entry.get("published", entry.get("updated", ""))
                )
                
                # Skip old articles
                if pub_date < cutoff_date:
                    continue
                
                # Extract source from title (Google News format: "Title - Source")
                title = entry.get("title", "")
                source = "Unknown"
                
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    if len(parts) == 2:
                        title = parts[0]
                        source = parts[1]
                
                # Get summary
                summary = self._clean_html(entry.get("summary", ""))
                summary = summary[:300] if summary else None  # Truncate
                
                articles.append(NewsArticle(
                    title=title,
                    source=source,
                    published=pub_date,
                    link=entry.get("link", ""),
                    summary=summary
                ))
            
            return articles
            
        except Exception as e:
            print(f"Error fetching news for '{query}': {e}")
            return []
    
    def fetch_market_news(self, days_back: int = 3) -> List[NewsArticle]:
        """Fetch general market news.
        
        Args:
            days_back: How many days back to search.
            
        Returns:
            List of NewsArticle objects.
        """
        queries = [
            "stock market today",
            "S&P 500 news",
            "NASDAQ news"
        ]
        
        all_articles = []
        for query in queries:
            articles = self.fetch_news(query, days_back)
            all_articles.extend(articles)
        
        # Remove duplicates by title
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            if article.title not in seen_titles:
                seen_titles.add(article.title)
                unique_articles.append(article)
        
        # Sort by date, most recent first
        unique_articles.sort(key=lambda a: a.published, reverse=True)
        
        return unique_articles[:self.max_articles]
    
    def fetch_and_format_for_memory(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        days_back: int = 7
    ) -> List[str]:
        """Fetch news and format for memory storage.
        
        Args:
            ticker: Stock ticker symbol.
            company_name: Company name.
            days_back: How many days back to search.
            
        Returns:
            List of formatted text strings for memory storage.
        """
        articles = self.fetch_stock_news(ticker, company_name, days_back)
        return [article.to_memory_content() for article in articles]
