import torch
import numpy as np

class HTRSMemory:
    """
    HyperTree-inspired memory storage system for associative memory.
    Implements a key-value memory with retrieval based on embedding similarity
    and HTPS-inspired selection mechanisms.
    """
    def __init__(self, embedding_dim=256, capacity=1000, selection_temperature=0.1):
        self.embedding_dim = embedding_dim
        self.capacity = capacity
        self.selection_temperature = selection_temperature
        
        # Storage for key-value pairs
        self.memory_keys = []
        self.memory_values = []
        
        # Metadata for enhanced retrieval
        self.access_counts = []
        self.last_access_time = []
        self.creation_time = []
        self.importance_scores = []
        
        # Current timestep counter
        self.current_time = 0
        
    def store(self, key_embedding, value, importance=None):
        """
        Store a key-value pair in memory with metadata.
        
        Args:
            key_embedding: Tensor embedding that serves as the key
            value: Any object to store as the value
            importance: Optional importance score (0-1)
        """
        if isinstance(key_embedding, torch.Tensor):
            # Convert to numpy for storage efficiency if tensor
            key_embedding = key_embedding.detach().cpu().numpy()
            
        # If at capacity, either replace lowest importance item or oldest
        if len(self.memory_keys) >= self.capacity:
            if importance is not None:
                # Find least important item
                min_importance_idx = np.argmin(self.importance_scores)
                if importance > self.importance_scores[min_importance_idx]:
                    # Replace least important item
                    self._replace_item_at_index(min_importance_idx, key_embedding, value, importance)
                    return
            
            # If no importance provided or not more important than least important,
            # replace oldest item (FIFO)
            self.memory_keys.pop(0)
            self.memory_values.pop(0)
            self.access_counts.pop(0)
            self.last_access_time.pop(0)
            self.creation_time.pop(0)
            self.importance_scores.pop(0)
            
        # Append new item with metadata
        self.memory_keys.append(key_embedding)
        self.memory_values.append(value)
        self.access_counts.append(0)
        self.last_access_time.append(self.current_time)
        self.creation_time.append(self.current_time)
        self.importance_scores.append(importance if importance is not None else 0.5)
        
        # Update timestep
        self.current_time += 1
        
    def _replace_item_at_index(self, idx, key_embedding, value, importance):
        """Replace an item at a specific index."""
        self.memory_keys[idx] = key_embedding
        self.memory_values[idx] = value
        self.access_counts[idx] = 0
        self.last_access_time[idx] = self.current_time
        self.creation_time[idx] = self.current_time
        self.importance_scores[idx] = importance
        
    def retrieve(self, query_embedding, top_k=5, selection_strategy="similarity"):
        """
        Retrieve top-k memories based on query similarity and selection strategy.
        
        Args:
            query_embedding: Embedding to compare against memory keys
            top_k: Number of results to return
            selection_strategy: Strategy for selecting memories
                - "similarity": Pure similarity-based retrieval
                - "importance": Weight by importance scores
                - "recency": Weight by recency of access
                - "htps": HyperTree-inspired probabilistic selection
        
        Returns:
            List of (value, similarity_score) tuples
        """
        if not self.memory_keys:
            return []
            
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()
            
        # Compute similarities between query and all keys
        similarities = []
        for key in self.memory_keys:
            # Cosine similarity
            similarity = np.dot(query_embedding, key) / (np.linalg.norm(query_embedding) * np.linalg.norm(key))
            similarities.append(float(similarity))
            
        # Apply selection strategy
        if selection_strategy == "similarity":
            # Pure similarity-based retrieval
            scores = similarities
        elif selection_strategy == "importance":
            # Weight by importance
            scores = [sim * imp for sim, imp in zip(similarities, self.importance_scores)]
        elif selection_strategy == "recency":
            # Weight by recency (linear decay with time)
            recency_weights = [1.0 - (self.current_time - t) / max(1, self.current_time) for t in self.last_access_time]
            scores = [sim * rec for sim, rec in zip(similarities, recency_weights)]
        elif selection_strategy == "htps":
            # HyperTree-inspired probabilistic selection
            # Combination of similarity, importance, and recency with temperature
            recency_weights = [1.0 - (self.current_time - t) / max(1, self.current_time) for t in self.last_access_time]
            importance_weights = [i + 0.2 for i in self.importance_scores]  # Add offset to ensure non-zero
            
            # Combine factors with temperature scaling
            combined_scores = []
            for s, i, r in zip(similarities, importance_weights, recency_weights):
                # Apply temperature to make selection more deterministic/focused
                scaled_score = (s * i * r) ** (1.0 / self.selection_temperature)
                combined_scores.append(scaled_score)
            
            scores = combined_scores
        else:
            raise ValueError(f"Unknown selection strategy: {selection_strategy}")
            
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]  # Descending order
        
        # Update metadata for retrieved items
        for idx in top_indices:
            self.access_counts[idx] += 1
            self.last_access_time[idx] = self.current_time
        
        # Increment time
        self.current_time += 1
        
        # Return corresponding values with similarity scores
        return [(self.memory_values[i], similarities[i]) for i in top_indices]
        
    def clear(self):
        """Clear all memory."""
        self.memory_keys = []
        self.memory_values = []
        self.access_counts = []
        self.last_access_time = []
        self.creation_time = []
        self.importance_scores = []
        
    def get_stats(self):
        """Get memory usage statistics."""
        return {
            "size": len(self.memory_keys),
            "capacity": self.capacity,
            "avg_importance": np.mean(self.importance_scores) if self.importance_scores else 0,
            "avg_access_count": np.mean(self.access_counts) if self.access_counts else 0,
            "current_time": self.current_time
        }