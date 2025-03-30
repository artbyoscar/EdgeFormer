# tests/model/test_gqa_simplified.py
import unittest
import torch
from src.model.transformer.config import EdgeFormerConfig
from src.model.transformer.gqa import GroupedQueryAttention

class TestGQASimplified(unittest.TestCase):
    def setUp(self):
        # Create a standard test configuration
        self.config = EdgeFormerConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=2,
            attention_probs_dropout_prob=0.1
        )
        
        # Create test inputs
        self.batch_size = 2
        self.seq_length = 4
        self.hidden_states = torch.randn(
            self.batch_size, self.seq_length, self.config.hidden_size
        )
        
        # Create an attention mask
        self.attention_mask = torch.ones(
            self.batch_size, 1, 1, self.seq_length
        )
        
        # Initialize the GQA module
        self.gqa = GroupedQueryAttention(self.config)

    def test_basic_forward(self):
        """Test a basic forward pass."""
        outputs = self.gqa(self.hidden_states)
        
        # Check that outputs[0] is the correct shape
        self.assertEqual(
            outputs[0].shape,
            (self.batch_size, self.seq_length, self.config.hidden_size)
        )

    def test_with_attention(self):
        """Test with attention output."""
        outputs = self.gqa(
            self.hidden_states,
            output_attentions=True
        )
        
        # Check that there are at least 2 outputs
        self.assertTrue(len(outputs) >= 2)
        
        # Check first output shape (context layer)
        self.assertEqual(
            outputs[0].shape,
            (self.batch_size, self.seq_length, self.config.hidden_size)
        )
        
        if len(outputs) >= 2:
            # If attention probs are returned
            attention_probs = outputs[1]
            # Check attention shape
            self.assertEqual(
                attention_probs.shape[0], 
                self.batch_size
            )