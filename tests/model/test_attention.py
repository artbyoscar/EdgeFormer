# tests/model/test_attention.py
import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.transformer.config import EdgeFormerConfig
from src.model.transformer.mla import MultiHeadLatentAttention

class TestMultiHeadLatentAttention(unittest.TestCase):
    def setUp(self):
        self.config = EdgeFormerConfig(
            hidden_size=768,
            num_attention_heads=12,
            attention_type="mla",
            latent_size=192,  # 1/4 of hidden size
        )
        self.mla = MultiHeadLatentAttention(self.config)
        self.batch_size = 2
        self.seq_length = 10
        self.hidden_states = torch.rand(self.batch_size, self.seq_length, self.config.hidden_size)
    
    def test_shapes(self):
        """Test if the output shapes are correct."""
        outputs = self.mla(self.hidden_states)
        context_layer, past_key_value = outputs[0], outputs[-1]
        
        # Check context layer shape
        self.assertEqual(
            context_layer.shape, 
            (self.batch_size, self.seq_length, self.config.hidden_size)
        )
        
        # Check latent key/value shapes
        latent_key, latent_value = past_key_value
        self.assertEqual(
            latent_key.shape, 
            (self.batch_size, self.seq_length, self.config.latent_size)
        )
        self.assertEqual(
            latent_value.shape, 
            (self.batch_size, self.seq_length, self.config.latent_size)
        )
    
    def test_attention_mask(self):
        """Test if attention mask is applied correctly."""
        # Create a causal mask (can't attend to future tokens)
        attention_mask = torch.tril(torch.ones(self.batch_size, 1, self.seq_length, self.seq_length))
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        outputs_with_mask = self.mla(self.hidden_states, attention_mask)
        outputs_without_mask = self.mla(self.hidden_states)
        
        # Outputs should be different with mask
        self.assertFalse(
            torch.allclose(outputs_with_mask[0], outputs_without_mask[0])
        )
    
    def test_past_key_value(self):
        """Test if past key value caching works correctly."""
        # First forward pass to get past_key_value
        outputs = self.mla(self.hidden_states)
        past_key_value = outputs[-1]
        
        # Second forward pass with last token only
        last_token = self.hidden_states[:, -1:, :]
        outputs_with_past = self.mla(last_token, past_key_value=past_key_value)
        
        # Shape should match full sequence for context
        self.assertEqual(
            outputs_with_past[0].shape, 
            (self.batch_size, 1, self.config.hidden_size)
        )

if __name__ == "__main__":
    unittest.main()