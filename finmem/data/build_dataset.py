"""
Dataset Builder

Builds pickle-format datasets from live APIs (Yahoo Finance, Google News RSS)
for use with the MarketEnvironment.

Output format:
{
    date(2024, 1, 1): {
        "price": {"AAPL": 150.0},
        "filing_k": {"AAPL": ""},     # Placeholder — filings not fetched live
        "filing_q": {"AAPL": ""},     # Placeholder
        "news": {"AAPL": ["headline 1", "headline 2"]},
    },
    ...
}
"""

import os
import pickle
import time
import logging
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any

import yfinance as yf

logger = logging.getLogger(__name__)


def build_dataset(
    ticker: str,
    start_date: date,
    end_date: date,
    output_path: Optional[str] = None,
    include_news: bool = True,
    news_source: str = "google",
    include_filings: bool = True,
) -> Dict[date, Dict[str, Any]]:
    """Build a date-indexed dataset for MarketEnvironment.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        start_date: Start date.
        end_date: End date.
        output_path: If set, save as pickle to this path.
        include_news: Whether to fetch news (requires API calls).
        news_source: "google" or "finnhub".
        include_filings: Whether to fetch SEC EDGAR filings.
        
    Returns:
        Date-indexed dictionary with price, news, and filing data.
    """
    logger.info(f"Building dataset for {ticker}: {start_date} to {end_date}")
    
    # Fetch price data from Yahoo Finance
    stock = yf.Ticker(ticker)
    hist = stock.history(
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
    )
    
    if hist.empty:
        logger.error(f"No price data for {ticker}")
        return {}
    
    # Fetch SEC filings (once, applied to all days)
    filing_k_text = ""
    filing_q_text = ""
    if include_filings:
        try:
            from .sec_filings import fetch_filings_for_dataset
            filings = fetch_filings_for_dataset(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            filing_k_text = filings.get("filing_k", "")
            filing_q_text = filings.get("filing_q", "")
            if filing_k_text:
                logger.info(f"Fetched 10-K: {len(filing_k_text)} chars")
            if filing_q_text:
                logger.info(f"Fetched 10-Q: {len(filing_q_text)} chars")
        except Exception as e:
            logger.warning(f"SEC filing fetch failed: {e}")
    
    # Build the dataset
    dataset: Dict[date, Dict[str, Any]] = {}
    
    for idx, row in hist.iterrows():
        day = idx.date() if hasattr(idx, 'date') else idx
        
        dataset[day] = {
            "price": {ticker: float(row["Close"])},
            "filing_k": {ticker: filing_k_text},
            "filing_q": {ticker: filing_q_text},
            "news": {ticker: []},
        }
    
    logger.info(f"Fetched {len(dataset)} days of price data")
    
    # Fetch news if requested
    if include_news:
        try:
            _add_news_to_dataset(dataset, ticker, news_source)
        except Exception as e:
            logger.warning(f"News fetching failed: {e}")
    
    # Add price summaries as simple news-like entries
    _add_price_summaries(dataset, ticker)
    
    # Save if path specified
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(dataset, f)
        logger.info(f"Dataset saved to {output_path}")
    
    return dataset


def _add_news_to_dataset(
    dataset: Dict[date, Dict[str, Any]],
    ticker: str,
    news_source: str,
) -> None:
    """Add news to the dataset using Google RSS or Finnhub."""
    
    if news_source == "google":
        try:
            from .news_fetcher import GoogleNewsFetcher
            fetcher = GoogleNewsFetcher()
            articles = fetcher.fetch_stock_news(ticker, max_results=50)
            
            for article in articles:
                article_date = article.published.date() if hasattr(article.published, 'date') else article.published
                if article_date in dataset:
                    content = article.to_memory_content()
                    dataset[article_date]["news"][ticker].append(content)
            
            logger.info(f"Added {len(articles)} news articles from Google RSS")
        except ImportError:
            logger.warning("Google News fetcher not available")
    
    elif news_source == "finnhub":
        try:
            from .finnhub_news import FinnhubNewsFetcher
            fetcher = FinnhubNewsFetcher()
            
            # Fetch in chunks to avoid rate limits
            dates = sorted(dataset.keys())
            start = dates[0]
            end = dates[-1]
            
            articles = fetcher.fetch_stock_news(
                ticker,
                from_date=start.isoformat(),
                to_date=end.isoformat(),
            )
            
            for article in articles:
                article_date = article.published.date() if hasattr(article.published, 'date') else article.published
                if article_date in dataset:
                    content = article.to_memory_content()
                    dataset[article_date]["news"][ticker].append(content)
            
            logger.info(f"Added {len(articles)} news articles from Finnhub")
        except ImportError:
            logger.warning("Finnhub news fetcher not available")


def _add_price_summaries(
    dataset: Dict[date, Dict[str, Any]],
    ticker: str,
) -> None:
    """Add price movement summaries as pseudo-news entries."""
    dates = sorted(dataset.keys())
    
    for i, day in enumerate(dates):
        price = dataset[day]["price"][ticker]
        
        if i > 0:
            prev_price = dataset[dates[i-1]]["price"][ticker]
            change = price - prev_price
            pct_change = (change / prev_price * 100) if prev_price else 0
            direction = "up" if change >= 0 else "down"
            
            summary = (
                f"[{day}] Price update: {ticker} closed at ${price:.2f} "
                f"({direction} {abs(pct_change):.2f}% from previous day)"
            )
            dataset[day]["news"][ticker].append(summary)


def load_dataset(path: str) -> Dict[date, Dict[str, Any]]:
    """Load a dataset from a pickle file.
    
    Args:
        path: Path to the pickle file.
        
    Returns:
        Date-indexed dataset dictionary.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FinMEM dataset")
    parser.add_argument("--ticker", required=True, help="Stock ticker")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default=None, help="Output path")
    parser.add_argument("--news", default="google", help="News source: google or finnhub")
    parser.add_argument("--no-news", action="store_true", help="Skip news fetching")
    
    args = parser.parse_args()
    
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    output = args.output or f"data/datasets/{args.ticker}_{args.start}_{args.end}.pkl"
    
    logging.basicConfig(level=logging.INFO)
    
    dataset = build_dataset(
        ticker=args.ticker,
        start_date=start,
        end_date=end,
        output_path=output,
        include_news=not args.no_news,
        news_source=args.news,
    )
    
    print(f"Dataset built: {len(dataset)} trading days")
    for day in sorted(list(dataset.keys()))[:3]:
        print(f"  {day}: price={dataset[day]['price']}, news_count={len(dataset[day]['news'][args.ticker])}")
