import torch
import torch.nn as nn
from .multi_head_latent_attention import MultiHeadLatentAttention
from .grouped_query_attention import GroupedQueryLatentAttention
from .flash_attention import FlashMultiHeadLatentAttention
from .sparse_mlp import SparseMLP


# ---- Transformer Block Implementation ----
class EdgeFormerBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Layer normalization
        self.ln_1 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.ln_2 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        
        # Attention
        self.attention = MultiHeadLatentAttention(config)
        
        # MLP
        if config.use_sparse_mlp:
            self.mlp = SparseMLP(config)
        else:
            # Standard MLP
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.hidden_dropout_prob)
            )

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
        # Self-attention block
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            sliding_window_size=sliding_window_size,
        )
        
        attn_output = attn_outputs[0]
        
        # Create temporary outputs for attention layer
        temp_outputs = ()
        
        # Add past key-value if needed
        if use_cache:
            past_key_value = attn_outputs[1]
            temp_outputs = temp_outputs + (past_key_value,)
        
        # Add attention weights if needed
        if output_attentions:
            attention_index = 2 if use_cache else 1
            attention_outputs = attn_outputs[attention_index]
            temp_outputs = temp_outputs + (attention_outputs,)
        
        # Residual connection
        hidden_states = residual + attn_output
        
        # MLP block
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_output = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + feed_forward_output
        
        # Return hidden states as first element followed by the rest
        return (hidden_states,) + temp_outputs
