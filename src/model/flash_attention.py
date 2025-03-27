import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FlashMultiHeadLatentAttention(nn.Module):
    """
    Combines Flash Attention technique with Multi-Head Latent Attention.
    Processes attention in blocks to reduce memory footprint.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.latent_size = config.latent_size
        
        # Project queries (not shared across heads)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Project to smaller latent space (shared across heads)
        self.kv_latent_proj = nn.Linear(self.hidden_size, self.latent_size)
        
        # Project from latent space to keys and values for each head
        self.latent_to_k = nn.Linear(self.latent_size, self.hidden_size)
        self.latent_to_v = nn.Linear(self.latent_size, self.hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Block size for flash attention
        self.block_size = 128  # Can be tuned based on hardware

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
        
        # Calculate query vectors
        q = self.q_proj(hidden_states)
        
        # Calculate latent, key, value projections
        if past_key_value is None:
            # Project to latent space (shared across heads)
            latent = self.kv_latent_proj(hidden_states)
            
            # Project from latent to keys and values
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
            
            # Concatenate with past
            latent = torch.cat([past_latent, latent], dim=1)
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
            
            past_key_value = (latent, k, v) if use_cache else None
        
        # Reshape queries, keys, values for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Initialize output tensor
        context = torch.zeros_like(q)
        
        # Flash Attention implementation - process in blocks to save memory
        kv_seq_len = k.size(2)
        for i in range(0, kv_seq_len, self.block_size):
            block_end = min(i + self.block_size, kv_seq_len)
            
            # Get key/value block
            k_block = k[:, :, i:block_end, :]
            v_block = v[:, :, i:block_end, :]
            
            # Compute scores for this block
            scores = torch.matmul(q, k_block.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply position bias if provided
            if position_bias is not None:
                if position_bias.size(-1) >= block_end:
                    block_bias = position_bias[:, :, :, i:block_end]
                    scores = scores + block_bias
            
            # Apply attention mask for this block
            if attention_mask is not None:
                if attention_mask.size(-1) >= block_end:
                    block_mask = attention_mask[:, :, :, i:block_end]
                    scores = scores + block_mask
            
            # Apply softmax within block (causal masking handled by attention_mask)
            block_probs = F.softmax(scores, dim=-1)
            block_probs = self.attn_dropout(block_probs)
            
            # Apply head mask if provided
            if head_mask is not None:
                block_probs = block_probs * head_mask
            
            # Compute weighted sum for this block
            context = context + torch.matmul(block_probs, v_block)
        
        # Reshape context
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Apply output projection
        output = self.out_proj(context)
        output = self.resid_dropout(output)
        
        outputs = (output, past_key_value)
        if output_attentions:
            outputs = outputs + (None,)  # We don't return attention probabilities for flash attention
            
        return outputs