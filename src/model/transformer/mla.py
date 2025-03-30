"""Multi-Head Latent Attention (MLA) implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention implementation.
    
    Reduces KV cache size by projecting keys and values into a shared
    latent space while maintaining model quality.
    """
    
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Latent dimension for keys and values (smaller than original)
        self.latent_size = getattr(config, "latent_size", self.all_head_size // 4)
        
        # Query projects to full dimension
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Key and value project to latent space (reduced dimension)
        self.key = nn.Linear(config.hidden_size, self.latent_size)
        self.value = nn.Linear(config.hidden_size, self.latent_size)
        
        # Projections from latent space back to attention head dimensions
        self.key_up = nn.Linear(self.latent_size, self.all_head_size)
        self.value_up = nn.Linear(self.latent_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x, num_heads):
        """Reshape for multi-head attention."""
        new_shape = x.size()[:-1] + (num_heads, -1)
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
        """Forward pass with latent space projection."""
        mixed_query_layer = self.query(hidden_states)
        
        # If past key value is used, only process the last token
        if past_key_value is not None:
            mixed_query_layer = mixed_query_layer[:, -1:, :]
            
            # Use cached latent projections
            latent_key, latent_value = past_key_value
        else:
            # Project to latent space (reduced dimension)
            latent_key = self.key(hidden_states)
            latent_value = self.value(hidden_states)
        
        # Project back to attention dimensions
        key_layer = self.key_up(latent_key)
        value_layer = self.value_up(latent_value)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer, self.num_attention_heads)
        key_layer = self.transpose_for_scores(key_layer, self.num_attention_heads)
        value_layer = self.transpose_for_scores(value_layer, self.num_attention_heads)
        
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
        
        # Cache the latent projections instead of full KV
        past_key_value = (latent_key, latent_value)
        outputs = outputs + (past_key_value,)
        
        return outputs