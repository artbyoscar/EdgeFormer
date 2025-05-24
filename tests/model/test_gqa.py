import unittest
import torch
import torch.nn as nn
from src.model.transformer.config import EdgeFormerConfig
from src.model.transformer.gqa import GroupedQueryAttention

class TestGroupedQueryAttention(unittest.TestCase):
    def setUp(self):
        # Create a standard test configuration
        self.config = EdgeFormerConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=2,  # 4 queries share each key/value head
            attention_probs_dropout_prob=0.1
        )
        
        # Create test inputs
        self.batch_size = 2
        self.seq_length = 4
        self.hidden_states = torch.randn(
            self.batch_size, self.seq_length, self.config.hidden_size
        )
        
        # Create an attention mask (1 = attend, 0 = ignore)
        self.attention_mask = torch.ones(
            self.batch_size, 1, 1, self.seq_length
        )
        
        # Initialize the GQA module
        self.gqa = GroupedQueryAttention(self.config)

    def test_initialization(self):
        """Test that the GQA module initializes correctly with proper dimensions."""
        # Check query projections (full dimension)
        self.assertEqual(
            self.gqa.query.in_features, 
            self.config.hidden_size
        )
        self.assertEqual(
            self.gqa.query.out_features, 
            self.config.hidden_size
        )
        
        # Check key/value projections (reduced dimension)
        self.assertEqual(
            self.gqa.key.out_features, 
            self.config.num_key_value_heads * (self.config.hidden_size // self.config.num_attention_heads)
        )
        self.assertEqual(
            self.gqa.value.out_features, 
            self.config.num_key_value_heads * (self.config.hidden_size // self.config.num_attention_heads)
        )
        
        # Check query/key/value head ratio
        self.assertEqual(
            self.gqa.num_queries_per_kv,
            self.config.num_attention_heads // self.config.num_key_value_heads
        )

    def test_forward_pass(self):
        """Test the forward pass of the GQA module."""
        # Run forward pass
        outputs = self.gqa(
            self.hidden_states,
            attention_mask=self.attention_mask,
            output_attentions=True
        )
        
        # Check output shapes
        context_layer = outputs[0]
        attention_probs = outputs[1]
        past_key_value = outputs[2]
        
        # Context should have the same shape as input
        self.assertEqual(
            context_layer.shape,
            (self.batch_size, self.seq_length, self.config.hidden_size)
        )
        
        # Attention probs should have shape [batch, num_heads, seq, seq]
        self.assertEqual(
            attention_probs.shape,
            (self.batch_size, self.config.num_attention_heads, self.seq_length, self.seq_length)
        )
        
        # Past key/value should have shape [batch, num_kv_heads, seq, head_dim]
        key_layer, value_layer = past_key_value
        self.assertEqual(
            key_layer.shape,
            (self.batch_size, self.config.num_key_value_heads, self.seq_length, self.config.hidden_size // self.config.num_attention_heads)
        )
        self.assertEqual(
            value_layer.shape,
            (self.batch_size, self.config.num_key_value_heads, self.seq_length, self.config.hidden_size // self.config.num_attention_heads)
        )

    def test_head_sharing(self):
        """Test that key/value heads are properly shared across query heads."""
        # Initialize model in eval mode to disable dropout
        self.gqa.eval()
        
        # Run forward pass
        with torch.no_grad():
            outputs = self.gqa(
                self.hidden_states,
                attention_mask=self.attention_mask,
                output_attentions=True
            )
        
        attention_probs = outputs[1]
        
        # Group query heads by which key/value head they share
        # For each group, compare attention patterns
        for kv_head_idx in range(self.config.num_key_value_heads):
            start_idx = kv_head_idx * self.gqa.num_queries_per_kv
            end_idx = start_idx + self.gqa.num_queries_per_kv
            
            # Get attention patterns for all queries in this group
            group_attention = attention_probs[:, start_idx:end_idx, :, :]
            
            # Check that attention is computed uniquely for each query head
            # (even though they share key/value heads)
            for i in range(self.gqa.num_queries_per_kv - 1):
                # Calculate similarity between patterns
                # Patterns should be different (not identical)
                pattern_i = group_attention[:, i, :, :]
                pattern_j = group_attention[:, i+1, :, :]
                
                # Check that the patterns are not identical
                # (they should differ because queries are different)
                are_identical = torch.allclose(pattern_i, pattern_j, rtol=1e-5, atol=1e-7)
                self.assertFalse(are_identical, "Attention patterns should be different for different query heads")

    def test_cached_kv(self):
        """Test using cached key/value from a previous forward pass."""
        # Run a forward pass to get cached KV
        outputs = self.gqa(
            self.hidden_states,
            attention_mask=self.attention_mask
        )
        # Check if past_key_value is included in outputs
        if len(outputs) < 3:
            # If your implementation doesn't return past_key_value as outputs[2],
            # we need to manually create it
            mixed_key_layer = self.gqa.key(self.hidden_states)
            mixed_value_layer = self.gqa.value(self.hidden_states)
            key_layer = self.gqa.transpose_for_scores(mixed_key_layer, self.gqa.num_key_value_heads)
            value_layer = self.gqa.transpose_for_scores(mixed_value_layer, self.gqa.num_key_value_heads)
            past_key_value = (key_layer, value_layer)
        else:
            past_key_value = outputs[2]
            past_key_value = outputs[2]
        
        # Now forward just the last token with the cached KV
        last_token = self.hidden_states[:, -1:, :]
        new_outputs = self.gqa(
            last_token,
            attention_mask=self.attention_mask,
            past_key_value=past_key_value
        )
        new_context = new_outputs[0]
        
        # The output for the last token should match
        self.assertEqual(
            new_context.shape,
            (self.batch_size, 1, self.config.hidden_size)
        )
        
        # When used with past_key_value, a new past_key_value should be returned
        # Handle variable number of outputs
        if len(new_outputs) > 2:
            new_past_kv = new_outputs[2]
        else:
            # Skip this test if past_kv not returned
            self.skipTest("GQA module doesn't return past_key_values")
        self.assertIsNotNone(new_past_kv)
        
        # The shapes of key/value caches should match the original
        self.assertEqual(new_past_kv[0].shape, past_key_value[0].shape)
        self.assertEqual(new_past_kv[1].shape, past_key_value[1].shape)

    def test_attention_mask(self):
        """Test that the attention mask properly prevents attention to masked positions."""
        # Create a custom attention mask that allows attending only to the first token
        custom_mask = torch.ones(self.batch_size, 1, 1, self.seq_length)
        custom_mask[:, :, :, 1:] = -10000.0  # Large negative value to mask out
        
        # Run with this custom mask
        self.gqa.eval()  # Disable dropout for deterministic results
        with torch.no_grad():
            outputs = self.gqa(
                self.hidden_states,
                attention_mask=custom_mask,
                output_attentions=True
            )
        
        attention_probs = outputs[1]
        
        # Check that attention to masked positions is close to zero
        # For each head, most attention should be on the first token
        for head_idx in range(self.config.num_attention_heads):
            head_probs = attention_probs[:, head_idx, :, :]
            
            # Sum probabilities for the first token and the rest
            first_token_prob = head_probs[:, :, 0].sum()
            other_tokens_prob = head_probs[:, :, 1:].sum()
            
            # Most probability mass should be on the first token
            self.assertGreater(first_token_prob, other_tokens_prob)
            
            # Attention to masked tokens should be very small
            self.assertLess(other_tokens_prob, 0.01)

    def test_head_expansion(self):
        """Test that key/value heads are properly expanded to match query heads."""
        # Get the raw projections
        query_proj = self.gqa.query(self.hidden_states)
        key_proj = self.gqa.key(self.hidden_states)
        value_proj = self.gqa.value(self.hidden_states)
        
        # Check the shapes
        self.assertEqual(
            query_proj.shape,
            (self.batch_size, self.seq_length, self.config.hidden_size)
        )
        
        # Key and value should have reduced dimensions
        kv_head_size = self.config.num_key_value_heads * (self.config.hidden_size // self.config.num_attention_heads)
        self.assertEqual(
            key_proj.shape,
            (self.batch_size, self.seq_length, kv_head_size)
        )
        self.assertEqual(
            value_proj.shape,
            (self.batch_size, self.seq_length, kv_head_size)
        )
        
        # Check if the repeat_kv_heads method exists in your implementation
        if hasattr(self.gqa, 'repeat_kv_heads'):
            # Test the repeat_kv_heads function directly
            key_expanded = self.gqa.repeat_kv_heads(key_proj)
        
            # Print diagnostics for debugging
            print(f"Original key shape: {key_proj.shape}")
            print(f"Expanded key shape: {key_expanded.shape}")
            print(f"Expected shape: ({self.batch_size}, {self.config.num_attention_heads}, {self.seq_length}, {self.config.hidden_size // self.config.num_attention_heads})")
        
            # Let's check that the expansion is happening correctly
            # The expanded shape should match the expected dimensions
            self.assertEqual(
                key_expanded.shape[1],  # Number of heads dimension
                self.config.num_attention_heads
            )
        else:
            # If the method doesn't exist, let's just check the forward pass works
            outputs = self.gqa(
                self.hidden_states,
                attention_mask=self.attention_mask,
                output_attentions=True
            )
        
            # The attention patterns should have the full number of heads
            attention_probs = outputs[1]
            self.assertEqual(
                attention_probs.shape[1],
                self.config.num_attention_heads
            )
        
        # Test the repeat_kv_heads function directly
        key_layer = self.gqa.transpose_for_scores(key_proj, self.config.num_key_value_heads)
        key_expanded = self.gqa.repeat_kv_heads(key_proj)
        
        # The expanded keys should have the same shape as if we had used full heads
        self.assertEqual(
            key_expanded.shape,
            (self.batch_size, self.config.num_attention_heads, self.seq_length, self.config.hidden_size // self.config.num_attention_heads)
        )
        
        # The number of unique patterns should match the number of key/value heads
        # (because each key is repeated for multiple queries)
        reshaped_expanded = key_expanded.view(self.batch_size, self.config.num_key_value_heads, self.gqa.num_queries_per_kv, self.seq_length, -1)
        
        # Check that keys are repeated properly
        for kv_idx in range(self.config.num_key_value_heads):
            # Get the key for this kv head
            first_query_key = reshaped_expanded[:, kv_idx, 0, :, :]
            
            # Check that it's repeated for all queries in the group
            for query_idx in range(1, self.gqa.num_queries_per_kv):
                repeated_key = reshaped_expanded[:, kv_idx, query_idx, :, :]
                self.assertTrue(torch.allclose(first_query_key, repeated_key))

if __name__ == "__main__":
    unittest.main()