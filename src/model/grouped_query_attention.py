import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupedQueryLatentAttention(nn.Module):
    """
    Combines Grouped-Query Attention (GQA) with Multi-Head Latent Attention (MLA)
    for even greater memory savings.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_groups = config.num_kv_groups
        self.head_dim = self.hidden_size // self.num_heads
        self.latent_size = config.latent_size
        
        # Project queries (for all heads)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Project to smaller latent space (shared across heads)
        self.kv_latent_proj = nn.Linear(self.hidden_size, self.latent_size)
        
        # Project from latent space to keys and values (for fewer groups)
        kv_size = self.hidden_size * self.num_kv_groups // self.num_heads
        self.latent_to_k = nn.Linear(self.latent_size, kv_size)
        self.latent_to_v = nn.Linear(self.latent_size, kv_size)
        
        # Output projection
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Sliding window attention
        self.use_sliding_window = config.use_sliding_window
        self.window_size = config.sliding_window_size if config.use_sliding_window else None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        batch_size, seq_len = hidden_states.size()[:2]
        
        # Calculate query vectors for all heads
        q = self.q_proj(hidden_states)
        
        # Calculate latent, key, value projections
        if past_key_value is None:
            # Project to latent space (shared across heads)
            latent = self.kv_latent_proj(hidden_states)
            
            # Project from latent to keys and values (for fewer groups)
            k = self.latent_to_k(latent)
            v = self.latent_to_v(latent)
            
            past_key_value = (latent, k, v) if use_cache else None
        else:
            # Use cached values for previous tokens
            past_latent, past_k, past_v = past_key_value
            
            # Compute latent for new tokens only
            latent = self.kv_latent_proj(hidden_states)
            
            # Project from latent to keys and values
            k = self.latent_to_k(latent)
            v = self.latent_to_v(latent)
            
            # For sliding window attention, only keep recent tokens
            if self.use_sliding_window:
                past_seq_len = past_k.size(1)
                if past_seq_len + seq_len > self.window_size:
                    # Keep only the most recent window_size tokens
                    start_idx = past_seq_len + seq_len - self.window_size
                    if start_idx < past_seq_len:
                        past_latent = past_latent[:, start_idx:]
                        past_k = past_k[:, start_idx:]
                        past_v = past_v[:, start_idx:]
                    else:
                        # All past tokens are outside window
                        past_latent = None
                        past_k = None
                        past_v = None
            
            # Concatenate with past
            if past_latent is not None:
                latent = torch.cat([past_latent, latent], dim=1)
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)
            
            past_key_value = (latent, k, v) if use_cache else None
        
        # Reshape queries for all heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Reshape keys and values for fewer groups
        kv_head_dim = self.head_dim
        k = k.view(batch_size, -1, self.num_kv_groups, kv_head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_kv_groups, kv_head_dim).transpose(1, 2)
        
        # Expand KV heads to match query heads through repetition
        heads_per_group = self.num_heads // self.num_kv_groups
        expanded_k = k.repeat_interleave(heads_per_group, dim=1)
        expanded_v = v.repeat_interleave(heads_per_group, dim=1)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, expanded_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply position bias if provided
        if position_bias is not None:
            attention_scores = attention_scores + position_bias
            
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        # Apply head mask if provided
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # Compute context vector
        context = torch.matmul(attention_probs, expanded_v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Apply output projection
        output = self.out_proj(context)
        output = self.resid_dropout(output)
        
        outputs = (output, past_key_value)
        if output_attentions:
            outputs = outputs + (attention_probs,)
            
        return outputs
