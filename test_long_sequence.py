import torch
from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer

# Create a configuration with smaller dimensions for testing
config = EdgeFormerConfig(
    vocab_size=30522,
    hidden_size=256, 
    num_hidden_layers=2,
    num_attention_heads=8,
    latent_size_factor=8,
    max_position_embeddings=2048
)

# Initialize the model
model = EdgeFormer(config)
model.eval()

def test_incremental_generation(seq_length=64, continuation_length=8):
    """Test incremental generation with KV cache."""
    print(f"Testing incremental generation with {seq_length} tokens + {continuation_length} new tokens")
    
    # Initial sequence
    input_ids = torch.randint(0, config.vocab_size, (1, seq_length))
    attention_mask = torch.ones(1, seq_length)
    
    # First pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True
    )
    
    # New tokens
    next_tokens = torch.randint(0, config.vocab_size, (1, continuation_length))
    
    # Continue generation with new helper method
    continuation_outputs = model.continue_generation(next_tokens, outputs["past_key_values"])
    
    print(f"✅ Successfully processed {seq_length + continuation_length} total tokens")
    return continuation_outputs

# Test with different sequence lengths
for length in [32, 64, 128, 256]:
    test_incremental_generation(length)

print("All tests completed successfully!")