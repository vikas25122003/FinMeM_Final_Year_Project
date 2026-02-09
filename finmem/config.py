"""
FinMEM Configuration Settings

Centralized configuration for the FinMEM trading agent.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class LLMConfig:
    """LLM configuration for OpenRouter."""
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "deepseek/deepseek-chat"))
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 2048
    temperature: float = 0.7


@dataclass
class EmbeddingConfig:
    """Embedding configuration using sentence-transformers."""
    model_name: str = "all-MiniLM-L6-v2"  # Fast and good quality
    device: str = "cpu"  # Use 'cuda' if GPU available


@dataclass
class MemoryConfig:
    """Layered memory configuration based on the paper."""
    
    # Memory decay rates (per time unit)
    shallow_decay: float = 0.9      # Fast decay for daily events
    intermediate_decay: float = 0.7  # Medium decay for weekly trends
    deep_decay: float = 0.3          # Slow decay for fundamentals
    
    # Time horizons (in days)
    shallow_horizon: int = 3         # 1-3 days
    intermediate_horizon: int = 21   # ~3 weeks
    deep_horizon: int = 90           # ~3 months
    
    # Scoring weights (recency, relevancy, importance)
    alpha: float = 0.3  # Recency weight
    beta: float = 0.4   # Relevancy weight  
    gamma: float = 0.3  # Importance weight
    
    # Vector store settings
    collection_name: str = "finmem_memories"
    persist_directory: str = "./data/chromadb"


@dataclass
class ProfileConfig:
    """Agent profiling configuration."""
    
    # Risk profiles: conservative=0.3, moderate=0.5, aggressive=0.7
    risk_tolerance: float = 0.5
    
    # Trading styles
    trading_style: str = "balanced"  # value, growth, momentum, balanced
    
    # Agent personality prompt
    personality: str = """You are a professional financial analyst with expertise in stock trading.
You analyze market data, news, and trends to make informed trading decisions.
You are {risk_level} in your approach and focus on {trading_style} investing."""


@dataclass
class DataConfig:
    """Data source configuration."""
    
    # Yahoo Finance settings
    price_interval: str = "1d"  # Daily data
    price_period: str = "3mo"   # Last 3 months
    
    # Google News RSS
    news_base_url: str = "https://news.google.com/rss/search"
    news_max_articles: int = 20
    
    # Cache settings
    cache_dir: str = "./data/cache"
    cache_expiry_hours: int = 1


@dataclass
class FinMEMConfig:
    """Main configuration container."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Simulation settings
    initial_capital: float = 100000.0
    max_position_size: float = 0.2  # Max 20% of portfolio per stock
    

# Default configuration
DEFAULT_CONFIG = FinMEMConfig()
