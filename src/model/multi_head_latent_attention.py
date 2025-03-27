import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging import setup_logging, log_tensor_shape

# Set up logging
logger = setup_logging(debug_mode=True)
logger.info("Starting EdgeFormer test...")

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention implementation based on DeepSeek's approach.
    Reduces KV cache size by projecting into a shared latent space.
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
        
        # Save sliding window configuration
        self.use_sliding_window = getattr(config, "use_sliding_window", False)
        self.sliding_window_size = getattr(config, "sliding_window_size", 512)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        sliding_window_size=None,
    ):
        batch_size, seq_len = hidden_states.size()[:2]

        # Add logging for input tensor shapes
        logger = logging.getLogger("edgeformer")
        logger.debug(f"Forward pass - Input hidden_states shape: {hidden_states.shape}")
        if attention_mask is not None:
            logger.debug(f"Forward pass - Input attention_mask shape: {attention_mask.shape}")
        
        # Determine sliding window size (parameter override, then config, then default)
        use_sliding_window = False
        window_size = None
        
        if sliding_window_size is not None and sliding_window_size > 0:
            use_sliding_window = True
            window_size = sliding_window_size
        elif self.use_sliding_window and self.sliding_window_size > 0:
            use_sliding_window = True
            window_size = self.sliding_window_size
        
        # Calculate query vectors
        q = self.q_proj(hidden_states)
        logger.debug(f"Query projection shape: {q.shape}")
        
        # Calculate latent, key, value projections
        if past_key_value is None:
            # Project to latent space (shared across heads)
            latent = self.kv_latent_proj(hidden_states)
            logger.debug(f"Latent projection shape: {latent.shape}")
            
            # Project from latent to keys and values
            k = self.latent_to_k(latent)
            v = self.latent_to_v(latent)
            logger.debug(f"Key projection shape: {k.shape}")
            logger.debug(f"Value projection shape: {v.shape}")
            
            # CHANGE THIS PART: Store complete key and value states instead of just latent
            past_key_value = (k, v) if use_cache else None
        else:
            # CHANGE THIS PART: Unpack key and value from past_key_value
            past_key, past_value = past_key_value
            
            # Compute latent for new tokens only
            latent = self.kv_latent_proj(hidden_states)

            # Project to keys and values for new tokens
            new_k = self.latent_to_k(latent)
            new_v = self.latent_to_v(latent)
            
            # Concatenate with past keys and values
            k = torch.cat([past_key, new_k], dim=1)
            v = torch.cat([past_value, new_v], dim=1)

            # Update cache with full keys and values
            past_key_value = (k, v) if use_cache else None
        
        # Reshape queries, keys, values for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        logger.debug(f"Reshaped query shape: {q.shape}")
        logger.debug(f"Reshaped key shape: {k.shape}")
        logger.debug(f"Reshaped value shape: {v.shape}")   

        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logger.debug(f"Attention scores shape: {attention_scores.shape}")

        # Apply position bias if provided
        if position_bias is not None:
            logger.debug(f"Position bias shape: {position_bias.shape}")
            attention_scores = attention_scores + position_bias

        # Get the dimensions for creating masks
        batch_size, num_heads, q_len, k_len = attention_scores.shape
        
        # Apply sliding window if specified
        if use_sliding_window:
            # Create sliding window mask - initialize with ones (visible)
            window_mask = torch.ones(q_len, k_len, device=attention_scores.device)
            
            # Create upper and lower triangular masks with the sliding window size
            window_mask = torch.tril(window_mask, diagonal=window_size)
            window_mask = torch.triu(window_mask, diagonal=-window_size)
            
            # Convert mask: 1 = keep, 0 = mask out
            # Then convert to attention mask format where 0 = keep, large negative = mask out
            window_mask = (1.0 - window_mask) * -10000.0
            
            # Add batch and head dimensions
            window_mask = window_mask.view(1, 1, q_len, k_len).expand(batch_size, num_heads, -1, -1)
            
            # Apply window mask
            attention_scores = attention_scores + window_mask
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Debug logging
            print(f"attention_scores shape: {attention_scores.shape}")
            print(f"attention_mask shape: {attention_mask.shape}")
            logger.debug(f"attention_scores shape: {attention_scores.shape}")
            logger.debug(f"attention_mask shape: {attention_mask.shape}")

            # Ensure proper broadcasting by expanding attention mask
            # Keep the batch dimension, expand the head dimension if needed
            if attention_mask.dim() == 4:  # [batch_size, 1/num_heads, q_len, k_len]
                if attention_mask.size(1) == 1:
                    attention_mask = attention_mask.expand(-1, num_heads, -1, -1)
            elif attention_mask.dim() == 3:  # [batch_size, q_len, k_len]
                attention_mask = attention_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
            else:  # [batch_size, seq_len] or unexpected
                # Create a proper 4D mask
                attention_mask = attention_mask.view(batch_size, 1, 1, -1).expand(-1, 1, q_len, -1)
                attention_mask = attention_mask.expand(-1, num_heads, -1, -1)
            
            # Ensure mask dimensions match attention_scores
            # Slice or pad as needed (prioritize slicing to avoid errors)
            mask_q_len, mask_k_len = attention_mask.size(-2), attention_mask.size(-1)
            
            if mask_q_len > q_len or mask_k_len > k_len:
                # Slice the mask if it's too large
                attention_mask = attention_mask[:, :, :q_len, :k_len]
            
            logger.debug(f"Expanded attention_mask shape: {attention_mask.shape}")
            
            # Add the mask to attention scores
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        logger.debug(f"Attention probabilities shape: {attention_probs.shape}")
        attention_probs = self.attn_dropout(attention_probs)
        
        # Apply head mask if provided
        if head_mask is not None:
            logger.debug(f"Head mask shape: {head_mask.shape}")
            attention_probs = attention_probs * head_mask
        
        # Compute context vector
        context = torch.matmul(attention_probs, v)
        logger.debug(f"Raw context vector shape: {context.shape}")

        # Fix: Make sure the shapes match when reshaping
        context_length = context.size(2)
        context_reshape = context.transpose(1, 2).contiguous()
        logger.debug(f"Transposed context shape: {context_reshape.shape}")
        context = context_reshape.view(batch_size, context_length, self.hidden_size)
        logger.debug(f"Final context shape: {context.shape}")
        
        # Apply output projection
        output = self.out_proj(context)
        logger.debug(f"Output projection shape: {output.shape}")
        output = self.resid_dropout(output)
        
        outputs = (output, past_key_value)
        if output_attentions:
            outputs = outputs + (attention_probs,)
            
        return outputs