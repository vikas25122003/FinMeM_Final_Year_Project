"""
SEC EDGAR Filing Fetcher — Paper-Faithful Implementation

Paper: Uses real SEC Form 10-K (annual) and Form 10-Q (quarterly) filing text.

Fetches filing data from the SEC EDGAR API (free, no API key required).
EDGAR requires a User-Agent header with a name and email.

Rate limit: 10 requests/second per SEC EDGAR fair access policy.
"""

import logging
import time
import requests
from datetime import date, timedelta
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# SEC EDGAR API base URLs
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"
EDGAR_FULLTEXT_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={start}&enddt={end}&forms={form_type}"
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Common headers required by SEC
EDGAR_HEADERS = {
    "User-Agent": "FinMEM-Research research@finmem.edu",
    "Accept-Encoding": "gzip, deflate",
}

# Cache for CIK lookups
_cik_cache: Dict[str, str] = {}


def _get_cik(ticker: str) -> Optional[str]:
    """Get the CIK number for a stock ticker from SEC.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'TSLA').
        
    Returns:
        CIK number as zero-padded string, or None if not found.
    """
    if ticker in _cik_cache:
        return _cik_cache[ticker]
    
    try:
        resp = requests.get(
            EDGAR_COMPANY_TICKERS_URL,
            headers=EDGAR_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                _cik_cache[ticker] = cik
                return cik
        
        logger.warning(f"CIK not found for ticker: {ticker}")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch CIK for {ticker}: {e}")
        return None


def _fetch_filing_text(
    cik: str,
    form_type: str,
    before_date: date,
    max_length: int = 5000,
) -> Optional[str]:
    """Fetch the most recent filing text of a given type from EDGAR.
    
    Args:
        cik: Company CIK number.
        form_type: '10-K' or '10-Q'.
        before_date: Only consider filings before this date.
        max_length: Max characters to return (filings can be huge).
        
    Returns:
        Truncated filing text, or None if not found.
    """
    try:
        # Fetch company submissions
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        
        # Find the most recent filing of the given type before the date
        for i, form in enumerate(forms):
            if form != form_type:
                continue
            filing_date = date.fromisoformat(dates[i])
            if filing_date > before_date:
                continue
            
            # Found a matching filing — fetch its primary document
            accession = accessions[i].replace("-", "")
            filename = primary_docs[i]
            
            filing_url = EDGAR_FILING_URL.format(
                cik=cik.lstrip("0"),
                accession=accession,
                filename=filename,
            )
            
            time.sleep(0.15)  # SEC rate limit: 10 req/s
            
            doc_resp = requests.get(
                filing_url,
                headers=EDGAR_HEADERS,
                timeout=30,
            )
            doc_resp.raise_for_status()
            
            text = doc_resp.text
            
            # Strip HTML tags for a rough text extraction
            import re
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Truncate to max_length
            if len(text) > max_length:
                text = text[:max_length] + "... [truncated]"
            
            logger.info(
                f"Fetched {form_type} filing dated {filing_date} "
                f"({len(text)} chars)"
            )
            return f"[SEC {form_type} filed {filing_date}] {text}"
        
        logger.info(f"No {form_type} filing found before {before_date}")
        return None
        
    except Exception as e:
        logger.warning(f"Failed to fetch {form_type} for CIK {cik}: {e}")
        return None


def fetch_10k(ticker: str, before_date: date, max_length: int = 5000) -> Optional[str]:
    """Fetch the most recent 10-K (annual) filing for a ticker.
    
    Args:
        ticker: Stock ticker symbol.
        before_date: Only consider filings before this date.
        max_length: Max characters to return.
        
    Returns:
        Filing text or None.
    """
    cik = _get_cik(ticker)
    if not cik:
        return None
    return _fetch_filing_text(cik, "10-K", before_date, max_length)


def fetch_10q(ticker: str, before_date: date, max_length: int = 5000) -> Optional[str]:
    """Fetch the most recent 10-Q (quarterly) filing for a ticker.
    
    Args:
        ticker: Stock ticker symbol.
        before_date: Only consider filings before this date.
        max_length: Max characters to return.
        
    Returns:
        Filing text or None.
    """
    cik = _get_cik(ticker)
    if not cik:
        return None
    return _fetch_filing_text(cik, "10-Q", before_date, max_length)


def fetch_filings_for_dataset(
    ticker: str,
    start_date: date,
    end_date: date,
    max_length: int = 5000,
) -> Dict[str, Optional[str]]:
    """Fetch both 10-K and 10-Q filings for a ticker valid during a date range.
    
    Returns the most recent filings that were filed BEFORE the start_date 
    (i.e., information available to the agent at simulation start).
    
    Args:
        ticker: Stock ticker.
        start_date: Simulation start date.
        end_date: Simulation end date.
        max_length: Max chars per filing.
        
    Returns:
        Dict with 'filing_k' and 'filing_q' keys.
    """
    logger.info(f"Fetching SEC filings for {ticker} (before {start_date})")
    
    filing_k = fetch_10k(ticker, before_date=start_date, max_length=max_length)
    time.sleep(0.2)  # Rate limit
    filing_q = fetch_10q(ticker, before_date=start_date, max_length=max_length)
    
    return {
        "filing_k": filing_k or "",
        "filing_q": filing_q or "",
    }
