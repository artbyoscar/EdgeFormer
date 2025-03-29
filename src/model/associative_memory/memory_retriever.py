"""
HTPS-Enhanced Associative Memory - Memory Retriever Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBasedRetriever(nn.Module):
    """
    Attention-based memory retriever that integrates retrieved memories 
    with the current model state through an attention mechanism.
    """
    
    def __init__(self, model_dim=768, num_heads=4, dropout=0.1, temperature=1.0, use_gating=True):
        """
        Initialize the attention-based memory retriever.
        
        Args:
            model_dim (int): Dimension of the model embeddings
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
            temperature (float): Temperature for attention softmax
            use_gating (bool): Whether to use a gating mechanism
        """
        super().__init__()
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.temperature = temperature
        self.use_gating = use_gating
        
        assert self.head_dim * num_heads == model_dim, "model_dim must be divisible by num_heads"
        
        # Query, key, value projections
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        
        # Output projection
        self.out_proj = nn.Linear(model_dim, model_dim)
        
        # Gating mechanism
        if use_gating:
            self.gate = nn.Linear(model_dim * 2, model_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # For visualization
        self.last_attention_weights = None
    
    def forward(self, query_hidden_states, memory_embeddings, memory_scores=None):
        """
        Retrieve and integrate memories with the current hidden states.
        
        Args:
            query_hidden_states (torch.Tensor): Hidden states from the model [batch, seq_len, hidden_dim]
            memory_embeddings (torch.Tensor): Retrieved memory embeddings [batch, num_memories, hidden_dim]
            memory_scores (torch.Tensor, optional): Scores for each memory [batch, num_memories]
            
        Returns:
            torch.Tensor: Updated hidden states with memory integration
        """
        # Handle empty memory case
        if memory_embeddings.shape[1] == 0:
            return query_hidden_states
        
        batch_size, seq_len, _ = query_hidden_states.shape
        num_memories = memory_embeddings.shape[1]
        
        # Project query, key, value
        q = self.q_proj(query_hidden_states)
        k = self.k_proj(memory_embeddings)
        v = self.v_proj(memory_embeddings)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_memories, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_memories, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale query
        q = q / (self.head_dim ** 0.5)
        
        # Calculate attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply temperature
        attention_scores = attention_scores / self.temperature
        
        # Incorporate memory scores if provided
        if memory_scores is not None:
            # Reshape scores to add to attention
            reshaped_scores = memory_scores.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, seq_len, -1)
            attention_scores = attention_scores + reshaped_scores
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Store for visualization
        self.last_attention_weights = attention_weights.detach().mean(dim=1)
        
        # Apply attention
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape back
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.model_dim
        )
        
        # Output projection
        attention_output = self.out_proj(attention_output)
        
        # Apply gating if enabled
        if self.use_gating:
            # Concatenate original and memory-enhanced states
            concat = torch.cat([query_hidden_states, attention_output], dim=-1)
            
            # Calculate gate values
            gate_values = torch.sigmoid(self.gate(concat))
            
            # Apply gate
            output = gate_values * attention_output + (1 - gate_values) * query_hidden_states
        else:
            # Simple residual connection
            output = query_hidden_states + attention_output
        
        return output

# Keep the original MemoryRetriever class for backward compatibility
class MemoryRetriever(nn.Module):
    """
    Neural network module for retrieving and integrating information from associative memory.
    Implements attention-based memory retrieval with HTPS-inspired selection mechanisms.
    """
    def __init__(self, hidden_size, num_attention_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = hidden_size  # Use hidden_size as memory_size
        self.num_heads = num_attention_heads  # Rename parameter
        
        # Query projection
        self.query_proj = nn.Linear(hidden_size, self.memory_size)  # Use self.memory_size
        
        # Multi-head attention for memory retrieval
        self.mem_attention = nn.MultiheadAttention(
            embed_dim=self.memory_size,  # Use self.memory_size
            num_heads=self.num_heads,  # Use self.num_heads
            dropout=dropout,
            batch_first=True
        )
        
        # Gating mechanism to control memory integration
        self.gate = nn.Sequential(
            nn.Linear(hidden_size + self.memory_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Memory integration layers
        self.integration = nn.Sequential(
            nn.Linear(hidden_size + self.memory_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(self.memory_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Selection strategy parameters
        self.selection_params = nn.Parameter(torch.ones(3) / 3)  # Equal weights initially
        
        # For visualization
        self.last_attention_weights = None
        
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
        # Check if memory is empty
        if not memory.embeddings or len(memory.embeddings) == 0:
            # Return empty tensors if memory is empty
            return (
                torch.zeros((0, self.memory_size), device=self.query_proj.weight.device),
                [],
                torch.zeros(0, device=self.query_proj.weight.device)
            )
            
        # Convert memory keys to tensor
        mem_keys = []
        for emb in memory.embeddings:
            if isinstance(emb, torch.Tensor):
                mem_keys.append(emb.to(self.query_proj.weight.device))
            else:
                mem_keys.append(torch.tensor(emb, device=self.query_proj.weight.device))
                
        mem_keys_tensor = torch.stack(mem_keys)

        # Extract metadata for scoring based on HTPSMemory structure
        importance_scores = [meta['importance'] for meta in memory.metadata]
        recency_scores = [meta['recency'] for meta in memory.metadata]
        access_counts = [meta['access_count'] for meta in memory.metadata]
        
        # Compute selection scores based on strategy
        if selection_strategy == "importance":
            selection_scores = torch.tensor(importance_scores, device=mem_keys_tensor.device)
        elif selection_strategy == "recency":
            selection_scores = torch.tensor(recency_scores, device=mem_keys_tensor.device)
        elif selection_strategy == "frequency":
            max_count = max(access_counts) if access_counts else 1
            frequency_scores = [count / max_count for count in access_counts]
            selection_scores = torch.tensor(frequency_scores, device=mem_keys_tensor.device)
        elif selection_strategy == "htps":
            # Normalized strategy weights using softmax
            strategy_weights = F.softmax(self.selection_params, dim=0)
            
            # Normalize scores
            max_count = max(access_counts) if access_counts else 1
            normalized_frequencies = [count / max_count for count in access_counts]
            
            # Combine scores using learned weights
            combined_scores = []
            for imp, rec, freq in zip(importance_scores, recency_scores, normalized_frequencies):
                score = (
                    strategy_weights[0] * imp +
                    strategy_weights[1] * rec +
                    strategy_weights[2] * freq
                )
                combined_scores.append(score)
        
            selection_scores = torch.tensor(combined_scores, device=mem_keys_tensor.device)
        else:
            # Default: uniform scores
            selection_scores = torch.ones(len(mem_keys_tensor), device=mem_keys_tensor.device)
            
        return mem_keys_tensor, memory.texts, selection_scores  # Return texts instead of memory_values
        
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
        all_attn_weights = []
        
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
            
            all_attn_weights.append(attn_weights)
            
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
        
        # Store attention weights for visualization
        if all_attn_weights:
            self.last_attention_weights = torch.cat(all_attn_weights, dim=1)
            
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