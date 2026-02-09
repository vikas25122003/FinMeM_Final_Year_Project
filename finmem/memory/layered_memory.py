"""
Layered Memory System

The core innovation of FinMEM - a 3-layer memory system that processes
financial data with different decay rates and time horizons.

Based on the paper's Memory module design:
- Shallow Layer: Daily events (fast decay)
- Intermediate Layer: Weekly trends (medium decay)  
- Deep Layer: Fundamental knowledge (slow decay)
"""

import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import math

from ..config import MemoryConfig, DEFAULT_CONFIG
from .embeddings import get_embedding_model, EmbeddingModel


def _normalize_datetime(dt: datetime) -> datetime:
    """Convert timezone-aware datetime to timezone-naive (strip tzinfo)."""
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


class MemoryLayer(Enum):
    """Memory layer types with their characteristics."""
    SHALLOW = "shallow"         # Daily events, news
    INTERMEDIATE = "intermediate"  # Weekly trends, patterns
    DEEP = "deep"               # Fundamental knowledge


@dataclass
class MemoryItem:
    """A single memory item stored in the system."""
    
    id: str
    content: str
    layer: MemoryLayer
    timestamp: datetime
    
    # Metadata
    source: str = "unknown"      # news, price, fundamental, etc.
    ticker: Optional[str] = None
    
    # Scoring components (set during retrieval)
    importance: float = 0.5      # Base importance (0-1)
    
    # Computed embedding (set after creation)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "layer": self.layer.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "ticker": self.ticker,
            "importance": self.importance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            layer=MemoryLayer(data["layer"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data.get("source", "unknown"),
            ticker=data.get("ticker"),
            importance=data.get("importance", 0.5)
        )


class LayeredMemory:
    """
    Three-layer memory system based on the FinMEM paper.
    
    Implements:
    - Shallow memory: Short-term events with fast decay
    - Intermediate memory: Medium-term trends
    - Deep memory: Long-term fundamental knowledge
    
    Memory scoring: Score = α * Recency + β * Relevancy + γ * Importance
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the layered memory system.
        
        Args:
            config: Memory configuration. Uses default if not provided.
        """
        self.config = config or DEFAULT_CONFIG.memory
        
        # Initialize embedding model (lazy loaded)
        self._embedding_model: Optional[EmbeddingModel] = None
        
        # In-memory storage (can be replaced with ChromaDB for persistence)
        self._memories: Dict[str, MemoryItem] = {}
        self._embeddings: Dict[str, List[float]] = {}
        
        # Memory counter for ID generation
        self._counter = 0
    
    @property
    def embedding_model(self) -> EmbeddingModel:
        """Get the embedding model (lazy loaded)."""
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model
    
    def _generate_id(self) -> str:
        """Generate a unique memory ID."""
        self._counter += 1
        return f"mem_{int(time.time())}_{self._counter}"
    
    def _determine_layer(self, source: str, age_days: float) -> MemoryLayer:
        """Determine which layer a memory belongs to.
        
        Args:
            source: Source type of the information.
            age_days: Age of the information in days.
            
        Returns:
            Appropriate memory layer.
        """
        # News and recent price data go to shallow layer
        if source in ["news", "price"] and age_days <= self.config.shallow_horizon:
            return MemoryLayer.SHALLOW
        
        # Weekly trends go to intermediate
        if age_days <= self.config.intermediate_horizon:
            return MemoryLayer.INTERMEDIATE
        
        # Everything else (fundamentals, old info) goes to deep
        return MemoryLayer.DEEP
    
    def add(
        self,
        content: str,
        source: str = "unknown",
        ticker: Optional[str] = None,
        importance: float = 0.5,
        timestamp: Optional[datetime] = None,
        layer: Optional[MemoryLayer] = None
    ) -> str:
        """Add a memory item to the system.
        
        Args:
            content: The text content to store.
            source: Source type (news, price, fundamental, etc.).
            ticker: Stock ticker if applicable.
            importance: Base importance score (0-1).
            timestamp: When the information occurred. Defaults to now.
            layer: Specific layer to store in. Auto-determined if not provided.
            
        Returns:
            The ID of the stored memory.
        """
        timestamp = timestamp or datetime.now()
        
        # Determine layer based on source and age
        if layer is None:
            ts = _normalize_datetime(timestamp)
            age_days = (datetime.now() - ts).days
            layer = self._determine_layer(source, age_days)
        
        # Create memory item
        memory_id = self._generate_id()
        memory = MemoryItem(
            id=memory_id,
            content=content,
            layer=layer,
            timestamp=timestamp,
            source=source,
            ticker=ticker,
            importance=importance
        )
        
        # Generate and store embedding
        embedding = self.embedding_model.embed(content)
        memory.embedding = embedding
        
        # Store
        self._memories[memory_id] = memory
        self._embeddings[memory_id] = embedding
        
        return memory_id
    
    def _calculate_recency_score(self, memory: MemoryItem) -> float:
        """Calculate recency score with layer-specific decay.
        
        Args:
            memory: The memory item to score.
            
        Returns:
            Recency score (0-1).
        """
        ts = _normalize_datetime(memory.timestamp)
        age = datetime.now() - ts
        age_days = age.total_seconds() / 86400
        
        # Get decay rate for this layer
        decay_rates = {
            MemoryLayer.SHALLOW: self.config.shallow_decay,
            MemoryLayer.INTERMEDIATE: self.config.intermediate_decay,
            MemoryLayer.DEEP: self.config.deep_decay
        }
        decay = decay_rates[memory.layer]
        
        # Exponential decay: score = e^(-decay * age)
        score = math.exp(-decay * age_days)
        return max(0.0, min(1.0, score))
    
    def _calculate_relevancy_score(
        self,
        memory: MemoryItem,
        query_embedding: List[float]
    ) -> float:
        """Calculate relevancy score using cosine similarity.
        
        Args:
            memory: The memory item to score.
            query_embedding: Embedding of the query.
            
        Returns:
            Relevancy score (0-1).
        """
        import numpy as np
        
        if memory.embedding is None:
            return 0.0
        
        mem_emb = np.array(memory.embedding)
        query_emb = np.array(query_embedding)
        
        dot_product = np.dot(mem_emb, query_emb)
        norm1 = np.linalg.norm(mem_emb)
        norm2 = np.linalg.norm(query_emb)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity, normalized to 0-1
        similarity = (dot_product / (norm1 * norm2) + 1) / 2
        return float(similarity)
    
    def _calculate_total_score(
        self,
        memory: MemoryItem,
        query_embedding: List[float]
    ) -> float:
        """Calculate total memory score.
        
        Score = α * Recency + β * Relevancy + γ * Importance
        
        Args:
            memory: The memory item to score.
            query_embedding: Embedding of the query.
            
        Returns:
            Total score (0-1).
        """
        recency = self._calculate_recency_score(memory)
        relevancy = self._calculate_relevancy_score(memory, query_embedding)
        importance = memory.importance
        
        score = (
            self.config.alpha * recency +
            self.config.beta * relevancy +
            self.config.gamma * importance
        )
        
        return min(1.0, score)
    
    def retrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        layer: Optional[MemoryLayer] = None,
        top_k: int = 10
    ) -> List[tuple[MemoryItem, float]]:
        """Retrieve relevant memories for a query.
        
        Args:
            query: The query text.
            ticker: Filter by specific ticker.
            layer: Filter by specific layer.
            top_k: Maximum number of memories to return.
            
        Returns:
            List of (memory, score) tuples, sorted by score descending.
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed(query)
        
        # Score all memories
        scored_memories = []
        for memory in self._memories.values():
            # Apply filters
            if ticker and memory.ticker and memory.ticker != ticker:
                continue
            if layer and memory.layer != layer:
                continue
            
            score = self._calculate_total_score(memory, query_embedding)
            scored_memories.append((memory, score))
        
        # Sort by score and return top-k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return scored_memories[:top_k]
    
    def retrieve_by_layer(
        self,
        layer: MemoryLayer,
        ticker: Optional[str] = None,
        top_k: int = 10
    ) -> List[MemoryItem]:
        """Retrieve memories from a specific layer.
        
        Args:
            layer: The memory layer to retrieve from.
            ticker: Filter by specific ticker.
            top_k: Maximum number of memories to return.
            
        Returns:
            List of memory items, sorted by recency.
        """
        memories = []
        for memory in self._memories.values():
            if memory.layer != layer:
                continue
            if ticker and memory.ticker and memory.ticker != ticker:
                continue
            memories.append(memory)
        
        # Sort by timestamp (most recent first)
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        return memories[:top_k]
    
    def get_context_summary(
        self,
        ticker: str,
        query: Optional[str] = None,
        max_items_per_layer: int = 5
    ) -> Dict[str, List[str]]:
        """Get a summary of relevant context from all layers.
        
        Args:
            ticker: The stock ticker to get context for.
            query: Optional query for relevancy filtering.
            max_items_per_layer: Max items to include per layer.
            
        Returns:
            Dictionary with layer names as keys and content lists as values.
        """
        summary = {}
        
        for layer in MemoryLayer:
            if query:
                # Use relevancy-based retrieval
                results = self.retrieve(
                    query=query,
                    ticker=ticker,
                    layer=layer,
                    top_k=max_items_per_layer
                )
                contents = [m.content for m, _ in results]
            else:
                # Use recency-based retrieval
                memories = self.retrieve_by_layer(
                    layer=layer,
                    ticker=ticker,
                    top_k=max_items_per_layer
                )
                contents = [m.content for m in memories]
            
            summary[layer.value] = contents
        
        return summary
    
    def clear(self, older_than: Optional[datetime] = None):
        """Clear memories, optionally only those older than a date.
        
        Args:
            older_than: If provided, only clear memories older than this date.
        """
        if older_than is None:
            self._memories.clear()
            self._embeddings.clear()
        else:
            to_remove = [
                mid for mid, mem in self._memories.items()
                if mem.timestamp < older_than
            ]
            for mid in to_remove:
                del self._memories[mid]
                if mid in self._embeddings:
                    del self._embeddings[mid]
    
    def stats(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dictionary with memory statistics.
        """
        layer_counts = {layer.value: 0 for layer in MemoryLayer}
        for memory in self._memories.values():
            layer_counts[memory.layer.value] += 1
        
        return {
            "total_memories": len(self._memories),
            "by_layer": layer_counts,
            "config": {
                "shallow_decay": self.config.shallow_decay,
                "intermediate_decay": self.config.intermediate_decay,
                "deep_decay": self.config.deep_decay
            }
        }
