#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom EdgeFormer Loader

This script defines a custom EdgeFormer model class that matches the structure
of the saved model file with 'layers' prefix instead of 'encoder.layer'.
"""

import os
import sys
import torch
import logging
from torch import nn

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.config import EdgeFormerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("custom-edgeformer")

class CustomMultiHeadLatentAttention(nn.Module):
    """
    Custom Multi-Head Latent Attention layer that matches the structure of the saved model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_size = hidden_size // num_heads
        latent_size = hidden_size // config.latent_size_factor
        
        # Query projection
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        
        # Key-Value latent projection
        self.kv_latent_proj = nn.Linear(hidden_size, latent_size)
        
        # Latent to K/V projections
        self.latent_to_k = nn.Linear(latent_size, hidden_size)
        self.latent_to_v = nn.Linear(latent_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # This is a placeholder implementation
        # We only need the structure for loading weights
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Query projection
        q = self.q_proj(hidden_states)
        
        # Key-Value latent projection
        kv_latent = self.kv_latent_proj(hidden_states)
        
        # Latent to K/V projections
        k = self.latent_to_k(kv_latent)
        v = self.latent_to_v(kv_latent)
        
        # Output projection
        output = self.out_proj(hidden_states)
        
        return {"output": output, "attentions": None}

class CustomMLP(nn.Module):
    """
    Custom MLP layer that matches the structure of the saved model.
    """
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Add masks for sparsity if used in the original model
        self.up_mask = nn.Parameter(torch.ones(config.intermediate_size))
        self.down_mask = nn.Parameter(torch.ones(config.hidden_size))
    
    def forward(self, hidden_states):
        # This is a placeholder implementation
        # We only need the structure for loading weights
        h = self.dense_h_to_4h(hidden_states)
        h = self.dense_4h_to_h(h)
        return h

class CustomTransformerLayer(nn.Module):
    """
    Custom transformer layer that matches the structure of the saved model.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.attention = CustomMultiHeadLatentAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.mlp = CustomMLP(config)
    
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # This is a placeholder implementation
        # We only need the structure for loading weights
        
        # Layer norm 1 and attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attention(hidden_states, attention_mask, output_attentions)
        hidden_states = residual + attn_outputs["output"]
        
        # Layer norm 2 and MLP
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return {"hidden_states": hidden_states, "attentions": attn_outputs["attentions"]}

class CustomEdgeFormer(nn.Module):
    """
    Custom EdgeFormer model that matches the structure of the saved model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = nn.ModuleDict({
            "word_embeddings": nn.Embedding(config.vocab_size, config.hidden_size),
            "position_embeddings": nn.Embedding(config.max_position_embeddings, config.hidden_size),
            "LayerNorm": nn.LayerNorm(config.hidden_size, eps=1e-12),
        })
        self.embeddings.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            CustomTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids, attention_mask=None, output_attentions=False, use_cache=False):
        # This is a placeholder implementation
        # We only need the structure for loading weights
        
        # Get embeddings
        position_ids = self.embeddings["position_ids"][:, :input_ids.size(1)]
        word_embeds = self.embeddings["word_embeddings"](input_ids)
        position_embeds = self.embeddings["position_embeddings"](position_ids)
        
        # Add embeddings
        hidden_states = word_embeds + position_embeds
        
        # Apply embedding layer norm
        hidden_states = self.embeddings["LayerNorm"](hidden_states)
        
        # Apply transformer layers
        all_attentions = () if output_attentions else None
        past_key_values = []
        
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(hidden_states, attention_mask, output_attentions)
            hidden_states = layer_outputs["hidden_states"]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs["attentions"],)
            
            if use_cache:
                # This is just a placeholder for the cache structure
                past_key_values.append({"k": None, "v": None})
        
        # Apply LM head
        logits = self.lm_head(hidden_states)
        
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "attentions": all_attentions,
            "past_key_values": past_key_values if use_cache else None
        }
    
    def continue_generation(self, next_token, past_key_values):
        """Helper method for continuing generation with KV cache."""
        # This is a simplified implementation just for structure
        outputs = self.forward(next_token, use_cache=True)
        return outputs

def load_custom_model(model_path, config=None):
    """
    Load a model with the custom structure.
    
    Args:
        model_path: Path to the model file
        config: Model configuration (optional)
        
    Returns:
        Loaded model
    """
    # Create configuration if not provided
    if config is None:
        config = EdgeFormerConfig(
            vocab_size=30522,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            latent_size_factor=8,
            intermediate_size=1024,
            max_position_embeddings=128
        )
    
    # Create custom model
    model = CustomEdgeFormer(config)
    
    # Load state dict
    logger.info(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    
    return model, config

def test_custom_model(model_path):
    """
    Test loading and running the custom model.
    
    Args:
        model_path: Path to the model file
    """
    # Load model
    logger.info("Loading custom model...")
    model, config = load_custom_model(model_path)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded with {total_params:,} parameters")
    
    # Create dummy input
    logger.info("Testing forward pass...")
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    attention_mask = torch.ones(1, 10)
    
    # Run model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    logger.info(f"Forward pass successful, output shape: {outputs['logits'].shape}")
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Custom EdgeFormer Loader")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model file")
    
    args = parser.parse_args()
    test_custom_model(args.model_path)