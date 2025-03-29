import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryRetriever(nn.Module):
    """
    Neural network module for retrieving and integrating information from associative memory.
    Implements attention-based memory retrieval with HTPS-inspired selection mechanisms.
    """
    def __init__(self, hidden_size, memory_size=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        
        # Query projection
        self.query_proj = nn.Linear(hidden_size, memory_size)
        
        # Multi-head attention for memory retrieval
        self.mem_attention = nn.MultiheadAttention(
            embed_dim=memory_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating mechanism to control memory integration
        self.gate = nn.Sequential(
            nn.Linear(hidden_size + memory_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Memory integration layers
        self.integration = nn.Sequential(
            nn.Linear(hidden_size + memory_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(memory_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Selection strategy parameters
        self.selection_params = nn.Parameter(torch.ones(3) / 3)  # Equal weights initially
        
    def create_memory_embeddings(self, memory, selection_strategy="htps"):
        """
        Create embeddings from memory items for attention-based retrieval.
        
        Args:
            memory: HTRSMemory object
            selection_strategy: Strategy for memory selection
            
        Returns:
            memory_embeddings: Tensor of shape [num_memories, memory_size]
            memory_values: List of corresponding values
            memory_scores: Scores for each memory item based on selection strategy
        """
        if not memory.memory_keys or len(memory.memory_keys) == 0:
            # Return empty tensors if memory is empty
            return (
                torch.zeros((0, self.memory_size), device=self.query_proj.weight.device),
                [],
                torch.zeros(0, device=self.query_proj.weight.device)
            )
            
        # Convert memory keys to tensor
        mem_keys = []
        for key in memory.memory_keys:
            if isinstance(key, torch.Tensor):
                mem_keys.append(key.to(self.query_proj.weight.device))
            else:
                mem_keys.append(torch.tensor(key, device=self.query_proj.weight.device))
                
        mem_keys_tensor = torch.stack(mem_keys)
        
        # Compute selection scores based on strategy
        if selection_strategy == "importance":
            importance_scores = torch.tensor(memory.importance_scores, device=mem_keys_tensor.device)
            selection_scores = importance_scores
        elif selection_strategy == "recency":
            max_time = float(memory.current_time) or 1.0
            recency_scores = torch.tensor([
                1.0 - (memory.current_time - t) / max_time for t in memory.last_access_time
            ], device=mem_keys_tensor.device)
            selection_scores = recency_scores
        elif selection_strategy == "frequency":
            max_count = max(memory.access_counts) if memory.access_counts else 1
            frequency_scores = torch.tensor([
                count / max_count for count in memory.access_counts
            ], device=mem_keys_tensor.device)
            selection_scores = frequency_scores
        elif selection_strategy == "htps":
            # Normalized strategy weights using softmax
            strategy_weights = F.softmax(self.selection_params, dim=0)
            
            # Compute individual scores
            max_time = float(memory.current_time) or 1.0
            importance_scores = torch.tensor(memory.importance_scores, device=mem_keys_tensor.device)
            recency_scores = torch.tensor([
                1.0 - (memory.current_time - t) / max_time for t in memory.last_access_time
            ], device=mem_keys_tensor.device)
            max_count = max(memory.access_counts) if memory.access_counts else 1
            frequency_scores = torch.tensor([
                count / max_count for count in memory.access_counts
            ], device=mem_keys_tensor.device)
            
            # Combine scores using learned weights
            selection_scores = (
                strategy_weights[0] * importance_scores +
                strategy_weights[1] * recency_scores +
                strategy_weights[2] * frequency_scores
            )
        else:
            # Default: uniform scores
            selection_scores = torch.ones(len(mem_keys_tensor), device=mem_keys_tensor.device)
            
        return mem_keys_tensor, memory.memory_values, selection_scores
        
    def forward(self, query_hidden, memory, top_k=None, selection_strategy="htps"):
        """
        Retrieve and integrate information from memory based on query.
        
        Args:
            query_hidden: Hidden state to use as query [batch_size, seq_len, hidden_size]
            memory: HTRSMemory object to retrieve from
            top_k: Optional limit to number of memory items to consider
            selection_strategy: Strategy for memory selection
            
        Returns:
            Updated hidden state with memory integration
        """
        batch_size, seq_len, _ = query_hidden.shape
        
        # Project query to memory space
        query_embedding = self.query_proj(query_hidden)  # [batch_size, seq_len, memory_size]
        
        # Get memory embeddings and values
        memory_embeddings, memory_values, selection_scores = self.create_memory_embeddings(
            memory, selection_strategy
        )
        
        # If memory is empty, return unchanged hidden state
        if memory_embeddings.size(0) == 0:
            return query_hidden
            
        # Apply top-k selection if specified
        if top_k is not None and memory_embeddings.size(0) > top_k:
            top_k_indices = torch.topk(selection_scores, top_k).indices
            memory_embeddings = memory_embeddings[top_k_indices]
            memory_values = [memory_values[i] for i in top_k_indices.cpu().tolist()]
            selection_scores = selection_scores[top_k_indices]
            
        # Use selection scores as attention bias
        selection_bias = selection_scores.unsqueeze(0).unsqueeze(0)  # [1, 1, num_memories]
        selection_bias = selection_bias.expand(batch_size, seq_len, -1)  # [batch_size, seq_len, num_memories]
        
        # Normalize memory embeddings
        memory_embeddings = self.layer_norm1(memory_embeddings)
        
        # Expand memory embeddings for all items in batch/sequence
        # From [num_memories, memory_size] to [batch_size, num_memories, memory_size]
        expanded_memory = memory_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # For each position in the sequence, retrieve relevant memory
        memory_integrated_hidden = []
        
        for i in range(seq_len):
            # Get query embedding for this position
            pos_query = query_embedding[:, i:i+1, :]  # [batch_size, 1, memory_size]
            
            # Retrieve from memory using attention with selection bias
            attn_output, attn_weights = self.mem_attention(
                query=pos_query,
                key=expanded_memory,
                value=expanded_memory,
                attn_mask=None,
                key_padding_mask=None,
            )
            
            # Get hidden state for this position
            pos_hidden = query_hidden[:, i:i+1, :]  # [batch_size, 1, hidden_size]
            
            # Compute gate value to control memory integration
            gate_value = self.gate(torch.cat([pos_hidden, attn_output], dim=-1))
            
            # Integrate memory with hidden state
            integrated = self.integration(torch.cat([pos_hidden, attn_output], dim=-1))
            
            # Apply gating
            updated_hidden = pos_hidden + gate_value * integrated
            
            # Normalize
            updated_hidden = self.layer_norm2(updated_hidden)
            
            memory_integrated_hidden.append(updated_hidden)
            
        # Combine all positions
        return torch.cat(memory_integrated_hidden, dim=1)
    
    def batch_retrieve(self, query_hidden_batch, memory, top_k=None, selection_strategy="htps"):
        """
        Process a batch of queries efficiently.
        
        Args:
            query_hidden_batch: Batch of hidden states [batch_size, seq_len, hidden_size]
            memory: HTRSMemory object
            top_k: Optional limit to number of memory items
            selection_strategy: Strategy for memory selection
            
        Returns:
            Batch of updated hidden states
        """
        return self.forward(query_hidden_batch, memory, top_k, selection_strategy)