"""Base Transformer implementation for EdgeFormer."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .config import EdgeFormerConfig
from .embeddings import EdgeFormerEmbeddings

class EdgeFormerSelfAttention(nn.Module):
    """Self-attention layer with support for different attention types."""
    
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attention_type = getattr(config, "attention_type", "standard")
        self.sliding_window_size = getattr(config, "sliding_window_size", 512)
        
    def transpose_for_scores(self, x):
        """Reshape for multi-head attention."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
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
        """Forward pass with different attention types."""
        mixed_query_layer = self.query(hidden_states)
        
        # If past key value is used, only the last token needs to be processed
        if past_key_value is not None:
            mixed_query_layer = mixed_query_layer[:, -1:, :]
            past_key, past_value = past_key_value
            key_layer = past_key
            value_layer = past_value
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        # Take the dot product to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply sliding window attention if specified
        if self.attention_type == "sliding_window" and attention_mask is not None:
            # Create a sliding window mask
            seq_length = hidden_states.size(1)
            window_mask = torch.ones(
                (seq_length, seq_length), device=attention_scores.device
            )
            
            for i in range(seq_length):
                window_start = max(0, i - self.sliding_window_size // 2)
                window_end = min(seq_length, i + self.sliding_window_size // 2 + 1)
                window_mask[i, window_start:window_end] = 0
                
            # Add large negative values to positions outside window
            window_mask = window_mask.unsqueeze(0).unsqueeze(0)
            attention_scores = attention_scores - 10000.0 * window_mask
        
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
            past_key_value = (key_layer, value_layer)
            
        outputs = outputs + (past_key_value,)
        
        return outputs