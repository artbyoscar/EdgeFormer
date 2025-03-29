# src/model/associative_memory/htps_memory.py

import torch
import numpy as np
from collections import deque

class HTPSMemory:
    """
    HyperTree-inspired memory storage with multiple selection strategies.
    
    This class provides a simple but effective memory system that stores embeddings
    and their associated text, with multiple strategies for selecting which memories
    to retain and retrieve.
    """
    
    def __init__(self, capacity=100, embedding_dim=768, selection_strategy='htps'):
        """
        Initialize the memory storage.
        
        Args:
            capacity (int): Maximum number of entries in memory
            embedding_dim (int): Dimension of embedding vectors
            selection_strategy (str): Strategy for selecting memories to keep/retrieve
                Options: 'importance', 'recency', 'frequency', 'htps'
        """
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.selection_strategy = selection_strategy
        
        # Storage for memory entries
        self.embeddings = []  # List of embedding vectors
        self.texts = []       # List of text entries
        self.metadata = []    # List of metadata dicts
        
        # Initialize metadata keys based on selection strategy
        self.metadata_keys = ['importance', 'recency', 'access_count', 'last_access_time']
    
    def __len__(self):
        """Return the number of entries in memory."""
        return len(self.embeddings)
    
    def add_entry(self, embedding, text, importance=0.5):
        """
        Add a new entry to memory.
        
        Args:
            embedding (torch.Tensor or numpy.ndarray): Embedding vector for the entry
            text (str): Text associated with the embedding
            importance (float): Initial importance score (0-1)
        """
        # Convert embedding to numpy if it's a torch tensor
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        
        # Create metadata
        metadata = {
            'importance': importance,
            'recency': 1.0,  # New entries are most recent
            'access_count': 0,
            'last_access_time': 0,
            'creation_time': len(self.embeddings)  # Use entry index as time proxy
        }
        
        # Add entry
        self.embeddings.append(embedding)
        self.texts.append(text)
        self.metadata.append(metadata)
        
        # Update recency scores for all entries
        self._update_recency()
        
        # If over capacity, remove entries according to selection strategy
        if len(self.embeddings) > self.capacity:
            self._prune_memory()
    
    def clear(self):
        """Clear all entries from memory."""
        self.embeddings = []
        self.texts = []
        self.metadata = []
    
    def _update_recency(self):
        """Update recency scores for all entries."""
        # Newest entry already has recency=1, decrement older entries
        decay_factor = 0.99
        for i in range(len(self.metadata) - 1):
            self.metadata[i]['recency'] *= decay_factor
    
    def _prune_memory(self):
        """Remove entries according to the selection strategy."""
        if self.selection_strategy == 'importance':
            # Remove least important entry
            scores = [meta['importance'] for meta in self.metadata]
        elif self.selection_strategy == 'recency':
            # Remove oldest entry
            scores = [meta['recency'] for meta in self.metadata]
        elif self.selection_strategy == 'frequency':
            # Remove least accessed entry
            scores = [meta['access_count'] for meta in self.metadata]
        elif self.selection_strategy == 'htps':
            # HyperTree-inspired combined strategy
            scores = []
            for meta in self.metadata:
                # Combine importance, recency and frequency with weights
                score = (
                    0.4 * meta['importance'] + 
                    0.3 * meta['recency'] + 
                    0.3 * (meta['access_count'] / (max(1, max(m['access_count'] for m in self.metadata))))
                )
                scores.append(score)
        else:
            # Default to recency
            scores = [meta['recency'] for meta in self.metadata]
        
        # Find index of the entry with lowest score
        min_idx = scores.index(min(scores))
        
        # Remove the entry
        self.embeddings.pop(min_idx)
        self.texts.pop(min_idx)
        self.metadata.pop(min_idx)
    
    def retrieve(self, query_embedding, k=5):
        """
        Retrieve the k most relevant memories for a query.
        
        Args:
            query_embedding (torch.Tensor or numpy.ndarray): Query embedding vector
            k (int): Number of memories to retrieve
        
        Returns:
            tuple: (retrieved_embeddings, retrieval_scores, retrieved_texts)
        """
        if not self.embeddings:
            # Return empty results if no memories
            return torch.tensor([]), torch.tensor([]), []
        
        # Convert query to numpy if it's a torch tensor
        if isinstance(query_embedding, torch.Tensor):
            query_np = query_embedding.detach().cpu().numpy()
        else:
            query_np = query_embedding
        
        # Calculate cosine similarity
        similarities = []
        for idx, emb in enumerate(self.embeddings):
            # L2 normalize embeddings
            emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
            query_norm = query_np / (np.linalg.norm(query_np) + 1e-9)
            
            # Calculate cosine similarity
            similarity = np.dot(emb_norm, query_norm)
            similarities.append(similarity)
            
            # Update metadata for this entry
            self.metadata[idx]['access_count'] += 1
            self.metadata[idx]['last_access_time'] = max(m['last_access_time'] for m in self.metadata) + 1
        
        # Convert to torch tensor for easier manipulation
        similarity_scores = torch.tensor(similarities)
        
        # Get top k indices
        k = min(k, len(self.embeddings))
        topk_values, topk_indices = torch.topk(similarity_scores, k)
        
        # Gather results
        retrieved_embeddings = [self.embeddings[i] for i in topk_indices]
        retrieved_texts = [self.texts[i] for i in topk_indices]
        
        # Convert retrieved embeddings to torch tensor
        retrieved_embeddings_tensor = torch.tensor(np.stack(retrieved_embeddings))
        
        return retrieved_embeddings_tensor, topk_values, retrieved_texts
    
    def get_all_entries(self):
        """
        Return all memory entries with their importance scores.
        
        Returns:
            list: List of (importance, text) tuples
        """
        return [(meta['importance'], text) for meta, text in zip(self.metadata, self.texts)]