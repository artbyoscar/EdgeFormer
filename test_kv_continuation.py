# test_kv_continuation.py
import torch
import logging
import time
import os
from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from src.utils.weight_quantization import kv_cache_offload

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("kv_test")

def test_kv_continuation():
    """Test KV cache continuation with proper attention mask handling"""
    logger.info("Testing KV cache continuation...")
    
    # Create a small model for testing
    config = EdgeFormerConfig(
        vocab_size=30522,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        latent_size_factor=8,
        max_position_embeddings=2048,
    )
    
    model = EdgeFormer(config)
    model.eval()
    
    # Enable KV cache offloading
    offloaded_model = kv_cache_offload(model)
    
    # Create initial sequence
    seq_length = 32
    logger.info(f"Testing with initial sequence length {seq_length}")
    
    input_ids = torch.randint(0, config.vocab_size, (1, seq_length))
    attention_mask = torch.ones(1, seq_length)
    
    # First pass
    logger.info("Running first pass...")
    outputs = offloaded_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True
    )
    
    # Get KV cache ID
    kv_cache_id = outputs["past_key_values"]
    logger.info(f"KV cache ID: {kv_cache_id}")
    
    # Second pass with just one token
    next_token = torch.randint(0, config.vocab_size, (1, 1))
    
    # IMPORTANT: Next attention mask needs to include all tokens (previous + new)
    next_attention_mask = torch.ones(1, seq_length + 1)
    
    logger.info("Running second pass with one token...")
    try:
        next_outputs = offloaded_model(
            input_ids=next_token,  # Just one new token
            attention_mask=next_attention_mask,  # Full mask (previous + new)
            past_key_values=kv_cache_id,
            use_cache=True
        )
        logger.info("Second pass successful!")
        
        # Get new KV cache ID
        new_kv_cache_id = next_outputs["past_key_values"]
        logger.info(f"New KV cache ID: {new_kv_cache_id}")
        
        # Third pass with one more token
        next_next_token = torch.randint(0, config.vocab_size, (1, 1))
        next_next_attention_mask = torch.ones(1, seq_length + 2)  # Original + 2 tokens
        
        logger.info("Running third pass with one more token...")
        next_next_outputs = offloaded_model(
            input_ids=next_next_token,
            attention_mask=next_next_attention_mask,
            past_key_values=new_kv_cache_id,
            use_cache=True
        )
        logger.info("Third pass successful!")
        
        # Clean up
        offloaded_model.cleanup_kv_cache()
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    test_kv_continuation()