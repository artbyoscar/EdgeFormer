import unittest
import torch
from src.model.config import EdgeFormerConfig
from src.model.multi_head_latent_attention import MultiHeadLatentAttention

class TestMultiHeadLatentAttention(unittest.TestCase):
    def setUp(self):
        # Create a small config for testing
        self.config = EdgeFormerConfig(
            hidden_size=256,
            num_attention_heads=8,
            latent_size_factor=8
        )
        self.attention = MultiHeadLatentAttention(self.config)
        
    def test_shapes(self):
        # Test with small batch and sequence length
        batch_size = 2
        seq_len = 4
        hidden_size = self.config.hidden_size
        
        # Create input tensor
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Create attention mask
        attention_mask = torch.ones(batch_size, 1, 1, seq_len)
        
        # Forward pass
        outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask
        )
        
        # Check output shapes
        self.assertEqual(outputs[0].shape, (batch_size, seq_len, hidden_size))
        
    def test_masking(self):
        # Test if masking works correctly
        batch_size = 2
        seq_len = 4
        hidden_size = self.config.hidden_size
        
        # Create input tensor
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Create attention mask that masks the last token
        attention_mask = torch.ones(batch_size, 1, 1, seq_len)
        attention_mask[:, :, :, -1] = -10000.0
        
        # Forward pass
        outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask
        )
        
        # Get attention probabilities
        attn_probs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=True
        )[-1]
        
        # Check that last token has near-zero attention probability
        self.assertTrue(torch.all(attn_probs[:, :, :, -1] < 0.001))

if __name__ == "__main__":
    unittest.main()