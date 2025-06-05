# Save as memory/htps_memory.py

import logging
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger('edgeformer')

class HTPSMemory:
    """
    Hyper-Tree Parameter Selection (HTPS) associative memory implementation.
    Provides efficient memory storage and retrieval for enhanced reasoning.
    """
    
    def __init__(self, capacity=100, hidden_size=768, selection_strategy='htps'):
        """
        Initialize the HTPS memory module.
        
        Args:
            capacity: Maximum number of memories to store
            hidden_size: Dimension of memory embeddings
            selection_strategy: Strategy for memory selection ('htps', 'recency', 'random')
        """
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.selection_strategy = selection_strategy
        
        # Initialize memory storage as a list of (text, vector) tuples
        self.memories = []
        
        logger.info(f"Initialized HTPSMemory with capacity={capacity}, hidden_size={hidden_size}")
    
    def add_entry(self, text, vector):
        """
        Add a new memory entry.
        
        Args:
            text: The memory text
            vector: The embedding vector
            
        Returns:
            bool: Success status
        """
        # Check if at capacity
        if len(self.memories) >= self.capacity:
            # Remove the oldest memory if at capacity
            self.memories.pop(0)
        
        # Add the new memory
        self.memories.append((text, vector))
        
        return True
    
    def add_memory(self, text):
        """
        Legacy method for compatibility - creates a mock vector
        
        Args:
            text: The memory text
            
        Returns:
            bool: Success status
        """
        # Create a mock embedding for compatibility
        if torch.cuda.is_available():
            mock_vector = torch.randn(1, self.hidden_size).cuda()
        else:
            mock_vector = torch.randn(1, self.hidden_size)
        
        return self.add_entry(text, mock_vector)
    
    def get_all_entries(self):
        """
        Get all memory entries.
        
        Returns:
            list: List of (text, vector) tuples
        """
        return self.memories
    
    def get_relevant_memories(self, query_vector, top_k=3):
        """
        Get the most relevant memories for a query.
        
        Args:
            query_vector: The query embedding
            top_k: Number of top memories to retrieve
            
        Returns:
            list: List of (text, vector, score) tuples
        """
        if not self.memories:
            return []
        
        results = []
        
        if self.selection_strategy == 'htps':
            # HTPS strategy: Combine importance and recency
            # Here we use a simple implementation - in a real system this would be more sophisticated
            
            # Compute similarity scores
            scores = []
            for text, vector in self.memories:
                # Ensure vectors are the right shape for similarity calculation
                q_vec = query_vector.view(-1, self.hidden_size)
                m_vec = vector.view(-1, self.hidden_size)
                
                # Compute cosine similarity
                similarity = torch.nn.functional.cosine_similarity(q_vec, m_vec)
                scores.append(similarity.item())
            
            # Combine with recency (more recent = higher score)
            for i, (score, (text, vector)) in enumerate(zip(scores, self.memories)):
                # Add recency bias
                recency_boost = i / len(self.memories)
                combined_score = 0.7 * score + 0.3 * recency_boost
                results.append((text, vector, combined_score))
            
            # Sort by combined score (descending)
            results.sort(key=lambda x: x[2], reverse=True)
            
        elif self.selection_strategy == 'recency':
            # Recency strategy: Return the most recent memories
            for i, (text, vector) in enumerate(reversed(self.memories)):
                results.append((text, vector, 1.0 - i/len(self.memories)))
                
        else:
            # Default to similarity-based retrieval
            for text, vector in self.memories:
                # Ensure vectors are the right shape for similarity calculation
                q_vec = query_vector.view(-1, self.hidden_size)
                m_vec = vector.view(-1, self.hidden_size)
                
                # Compute cosine similarity
                similarity = torch.nn.functional.cosine_similarity(q_vec, m_vec)
                results.append((text, vector, similarity.item()))
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x[2], reverse=True)
        
        # Return top-k results
        return results[:top_k]
    
    def clear(self):
        """Clear all memories."""
        self.memories = []
        return True
    
    def list_memories(self):
        """Return a list of all memory texts (for compatibility)."""
        return [text for text, _ in self.memories]
    
    def clear_memories(self):
        """Clear all memories (for compatibility)."""
        return self.clear()
    
    def size(self):
        """Return the number of stored memories."""
        return len(self.memories)
