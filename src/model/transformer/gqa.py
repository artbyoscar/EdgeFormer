"""Grouped-Query Attention (GQA) for EdgeFormer."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention implementation.
    
    GQA groups query heads together to share KV heads, reducing
    memory requirements while maintaining performance.
    """
    
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_attention_heads // 4)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_head_size = config.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query projects to full number of heads
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Key and value project to reduced number of heads
        self.key = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size)
        self.value = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x, num_heads):
        """Reshape for multi-head attention."""
        new_shape = x.size()[:-1] + (num_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """Forward pass with grouped-query attention."""
        mixed_query_layer = self.query(hidden_states)
        
        # If past key value is used, only the last token needs to be processed
        if past_key_value is not None:
            mixed_query_layer = mixed_query_layer[:, -1:, :]
            past_key, past_value = past_key_value
            key_layer = past_key
            value_layer = past_value
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states), self.num_key_value_heads)
            value_layer = self.transpose_for_scores(self.value(hidden_states), self.num_key_value_heads)
        
        query_layer = self.transpose_for_scores(mixed_query_layer, self.num_attention_heads)
        
        # Repeat KV heads to match query heads for attention computation
        if self.num_key_value_groups > 1:
            key_layer = self._repeat_kv(key_layer, self.num_key_value_groups)
            value_layer = self._repeat_kv(value_layer, self.num_key_value_groups)
        
        # Take the dot product to get attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # Apply the attention mask
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        
        if past_key_value is None:
            # Only store the original KV heads to save memory
            past_key_value = (key_layer[:, :self.num_key_value_heads], value_layer[:, :self.num_key_value_heads])
        
        outputs = outputs + (past_key_value,)
        
        return outputs
    
    def _repeat_kv(self, hidden_states, n_rep):
        """Repeat the key and value tensors for each query head."""
        batch, kv_heads, seq_len, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        
        # [batch, kv_heads, seq_len, head_dim] -> [batch, kv_heads, seq_len, 1, head_dim]
        hidden_states = hidden_states.unsqueeze(3)
        # [batch, kv_heads, seq_len, 1, head_dim] -> [batch, kv_heads, seq_len, n_rep, head_dim]
        hidden_states = hidden_states.expand(-1, -1, -1, n_rep, -1)
        # [batch, kv_heads, seq_len, n_rep, head_dim] -> [batch, kv_heads * n_rep, seq_len, head_dim]
        hidden_states = hidden_states.reshape(batch, kv_heads * n_rep, seq_len, head_dim)
        return hidden_states