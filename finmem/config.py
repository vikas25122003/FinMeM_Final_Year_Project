"""
FinMEM Configuration Settings

Centralized configuration for the FinMEM trading agent.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
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
class MemoryLayerConfig:
    """Configuration for a single memory layer."""
    jump_threshold_upper: float = 999999.0  # No promotion by default
    jump_threshold_lower: float = -999999.0  # No demotion by default
    decay_params: Dict[str, float] = field(default_factory=lambda: {
        "decay_rate": 14.0,       # Q_l: stability term in days (paper default)
        "importance_base": 0.9,   # α_l: importance decay base
    })
    clean_up_threshold_dict: Dict[str, float] = field(default_factory=lambda: {
        "recency_threshold": 0.01,
        "importance_threshold": 0.01,
    })
    compound_score_params: Dict[str, float] = field(default_factory=lambda: {
        "w_recency": 1.0,
        "w_importance": 1.0,
        "w_similarity": 1.0,
        "w_compound": 1.0,
    })


@dataclass
class MemoryConfig:
    """Layered memory configuration based on the paper."""
    
    # Per-layer configs
    # Paper: Q_shallow=14, Q_intermediate=90, Q_deep=365
    # Paper: α_shallow=0.9, α_intermediate=0.967, α_deep=0.988
    short: MemoryLayerConfig = field(default_factory=lambda: MemoryLayerConfig(
        jump_threshold_upper=0.8,
        jump_threshold_lower=-999999.0,  # No demotion from short
        decay_params={"decay_rate": 14.0, "importance_base": 0.9},
        clean_up_threshold_dict={"recency_threshold": 0.01, "importance_threshold": 0.01},
    ))
    mid: MemoryLayerConfig = field(default_factory=lambda: MemoryLayerConfig(
        jump_threshold_upper=0.85,
        jump_threshold_lower=0.1,
        decay_params={"decay_rate": 90.0, "importance_base": 0.967},
        clean_up_threshold_dict={"recency_threshold": 0.01, "importance_threshold": 0.01},
    ))
    long: MemoryLayerConfig = field(default_factory=lambda: MemoryLayerConfig(
        jump_threshold_upper=999999.0,  # No promotion from long
        jump_threshold_lower=0.15,
        decay_params={"decay_rate": 365.0, "importance_base": 0.988},
        clean_up_threshold_dict={"recency_threshold": 0.005, "importance_threshold": 0.005},
    ))
    reflection: MemoryLayerConfig = field(default_factory=lambda: MemoryLayerConfig(
        jump_threshold_upper=999999.0,
        jump_threshold_lower=-999999.0,
        decay_params={"decay_rate": 365.0, "importance_base": 0.988},
        clean_up_threshold_dict={"recency_threshold": 0.005, "importance_threshold": 0.005},
    ))
    
    # Cognitive span: how many memories to retrieve per layer
    top_k: int = 5
    
    # Vector store settings (used for checkpointing)
    persist_directory: str = "./data/checkpoints"


@dataclass
class ProfileConfig:
    """Agent profiling configuration."""
    
    # Risk profiles: conservative=0.3, moderate=0.5, aggressive=0.7
    risk_tolerance: float = 0.5
    
    # Trading styles
    trading_style: str = "balanced"  # value, growth, momentum, balanced
    
    # Character string for memory queries
    character_string: str = (
        "A professional financial analyst specializing in stock trading. "
        "Expert in analyzing market trends, financial statements, and news sentiment."
    )
    
    # Agent personality prompt template
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
    
    # Agent settings
    agent_name: str = "finmem_agent"
    look_back_window_size: int = 7
    
    def get_brain_config(self) -> Dict[str, Any]:
        """Generate config dict for BrainDB.from_config()."""
        return {
            "agent_name": self.agent_name,
            "short": {
                "jump_threshold_upper": self.memory.short.jump_threshold_upper,
                "jump_threshold_lower": self.memory.short.jump_threshold_lower,
                "decay_params": self.memory.short.decay_params,
                "clean_up_threshold_dict": self.memory.short.clean_up_threshold_dict,
                "compound_score_params": self.memory.short.compound_score_params,
            },
            "mid": {
                "jump_threshold_upper": self.memory.mid.jump_threshold_upper,
                "jump_threshold_lower": self.memory.mid.jump_threshold_lower,
                "decay_params": self.memory.mid.decay_params,
                "clean_up_threshold_dict": self.memory.mid.clean_up_threshold_dict,
                "compound_score_params": self.memory.mid.compound_score_params,
            },
            "long": {
                "jump_threshold_upper": self.memory.long.jump_threshold_upper,
                "jump_threshold_lower": self.memory.long.jump_threshold_lower,
                "decay_params": self.memory.long.decay_params,
                "clean_up_threshold_dict": self.memory.long.clean_up_threshold_dict,
                "compound_score_params": self.memory.long.compound_score_params,
            },
            "reflection": {
                "jump_threshold_upper": self.memory.reflection.jump_threshold_upper,
                "jump_threshold_lower": self.memory.reflection.jump_threshold_lower,
                "decay_params": self.memory.reflection.decay_params,
                "clean_up_threshold_dict": self.memory.reflection.clean_up_threshold_dict,
                "compound_score_params": self.memory.reflection.compound_score_params,
            },
        }


# Default configuration
DEFAULT_CONFIG = FinMEMConfig()
