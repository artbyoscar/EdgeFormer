# src/model/transformer/gqa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) implementation.
    
    Reduces parameter count and computation by sharing key-value heads
    across groups of query heads.
    """
    
    def __init__(self, config):
        super().__init__()
        if not hasattr(config, "num_key_value_heads"):
            config.num_key_value_heads = max(1, config.num_attention_heads // 4)
            
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.kv_head_size = self.num_key_value_heads * self.attention_head_size
        
        # Number of queries per key-value head
        self.num_queries_per_kv = self.num_attention_heads // self.num_key_value_heads
        
        # Query projects to full dimension
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Key and value project to reduced dimension (fewer heads)
        self.key = nn.Linear(config.hidden_size, self.kv_head_size)
        self.value = nn.Linear(config.hidden_size, self.kv_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x, num_heads):
        """Reshape for multi-head attention."""
        new_shape = x.size()[:-1] + (num_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def repeat_kv_heads(self, hidden_states):
        """Repeat KV heads to match the number of query heads."""
        batch_size, seq_length, _ = hidden_states.size()
        
        # Reshape to extract the key/value heads
        hidden_states = self.transpose_for_scores(hidden_states, self.num_key_value_heads)
        
        # Create a new tensor with repeated KV heads
        expanded_shape = (
            batch_size,
            self.num_attention_heads,
            seq_length,
            self.attention_head_size
        )
        expanded = torch.zeros(expanded_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        
        # For each query head, copy the appropriate KV head
        for q_idx in range(self.num_attention_heads):
            kv_idx = q_idx // self.num_queries_per_kv
            expanded[:, q_idx, :, :] = hidden_states[:, kv_idx, :, :]
        
        return expanded
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """Forward pass with grouped attention."""
        mixed_query_layer = self.query(hidden_states)
        
        # If past key value is used, only process the last token
        if past_key_value is not None:
            if isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                # Standard format
                past_key, past_value = past_key_value
                mixed_query_layer = mixed_query_layer[:, -1:, :]
                key_layer = past_key
                value_layer = past_value
            else:
                # Handle empty past_key_value
                key_layer = self.transpose_for_scores(self.key(hidden_states), self.num_key_value_heads)
                value_layer = self.transpose_for_scores(self.value(hidden_states), self.num_key_value_heads)
        else:
            # Project keys and values (fewer heads)
            key_layer = self.transpose_for_scores(self.key(hidden_states), self.num_key_value_heads)
            value_layer = self.transpose_for_scores(self.value(hidden_states), self.num_key_value_heads)
        
        # Reshape queries for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer, self.num_attention_heads)
        
        # Repeat KV heads to match the number of query heads
        if past_key_value is None:
            # For first pass where we compute new key/value
            key_layer_expanded = self.repeat_kv_heads(self.key(hidden_states))
            value_layer_expanded = self.repeat_kv_heads(self.value(hidden_states))
        else:
            # Re-expand cached keys/values
            key_layer_expanded = self.repeat_kv_heads(past_key.permute(0, 2, 1, 3).view(past_key.size(0), past_key.size(2), -1))
            value_layer_expanded = self.repeat_kv_heads(past_value.permute(0, 2, 1, 3).view(past_value.size(0), past_value.size(2), -1))
        
        # Take the dot product to get attention scores
        attention_scores = torch.matmul(query_layer, key_layer_expanded.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # Apply the attention mask
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer_expanded)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Create outputs tuple
        outputs = (context_layer,)
        if output_attentions:
            outputs = outputs + (attention_probs,)
        
        # Always include past_key_value in the outputs
        past_key_value = (key_layer, value_layer)
        outputs = outputs + (past_key_value,)
        
        return outputs
