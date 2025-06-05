# Save as memory/retriever.py

import logging
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger('edgeformer')

class MemoryRetriever:
    """
    Memory retrieval component for EdgeFormer's associative memory system.
    Retrieves relevant memories based on query similarity.
    """
    
    def __init__(self, hidden_size, num_attention_heads=4, dropout=0.1):
        """
        Initialize the memory retriever.
        
        Args:
            hidden_size: Dimension of embeddings
            num_attention_heads: Number of attention heads for retrieval
            dropout: Dropout probability
        """
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        
        # Initialize attention params
        self.query_proj = None
        self.key_proj = None
        self.value_proj = None
        self.output_proj = None
        
        self._init_attention_layers()
        
        logger.info(f"Initialized MemoryRetriever with hidden_size={hidden_size}, heads={num_attention_heads}")
    
    def _init_attention_layers(self):
        """Initialize attention projection layers."""
        # Only initialize if not already initialized
        if self.query_proj is None:
            self.query_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.key_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.value_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.output_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
    
    def retrieve_memories(self, query_hidden, memory_module, top_k=3, capture_attention=False):
        """
        Retrieve relevant memories for a given query.
        
        Args:
            query_hidden: Query hidden states
            memory_module: The HTPSMemory module
            top_k: Number of top memories to retrieve
            capture_attention: Whether to capture attention maps for visualization
            
        Returns:
            memory_vectors: Retrieved memory vectors
            attention_weights: Attention map for visualization
            memory_texts: List of retrieved memory texts
        """
        # If no memories or empty query, return None
        if not hasattr(memory_module, 'get_all_entries') or not memory_module.get_all_entries():
            return None, None, []
        
        # Get query representation - use final token for generation, average for other cases
        if query_hidden.dim() == 3 and query_hidden.size(1) > 1:
            # For full sequence input, use the last token
            query_vector = query_hidden[:, -1:, :]
        else:
            # Already a single token or properly shaped
            query_vector = query_hidden
        
        # Retrieve relevant memories from the memory module
        retrieved_memories = memory_module.get_relevant_memories(query_vector, top_k=top_k)
        
        if not retrieved_memories:
            return None, None, []
        
        # Extract texts and vectors
        memory_texts = [item[0] for item in retrieved_memories]
        memory_vectors = torch.stack([item[1] for item in retrieved_memories], dim=1)
        
        # Create attention map
        attention_weights = self._compute_attention(query_vector, memory_vectors)
        
        # Return memory vectors, attention weights, and texts
        return memory_vectors, attention_weights, memory_texts
    
    def _compute_attention(self, query, memory_vectors):
        """
        Compute attention weights between query and memory vectors.
        
        Args:
            query: Query vectors [batch_size, seq_len, hidden_size]
            memory_vectors: Memory vectors [batch_size, num_memories, hidden_size]
            
        Returns:
            attention_weights: Attention weights [batch_size, seq_len, num_memories]
        """
        # If attention layers not initialized, initialize them
        if self.query_proj is None:
            self._init_attention_layers()
        
        # Project query and memory vectors
        q = self.query_proj(query)
        k = self.key_proj(memory_vectors)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        
        # Apply softmax to get attention weights
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        return attention_weights
    
    def retrieve_memories_legacy(self, query, memory_module, top_k=3):
        """
        Legacy method for compatibility with the demo script.
        
        Args:
            query: Text query
            memory_module: The memory module
            top_k: Number of top memories to retrieve
            
        Returns:
            list: Retrieved memory texts
        """
        if not hasattr(memory_module, 'list_memories') or not memory_module.list_memories():
            return []
        
        # For legacy compatibility, just return the most recent memories
        memories = memory_module.list_memories()
        return memories[-top_k:]
