# debug_kv_cache.py
import torch
import logging
import time
import os
import psutil
from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from src.utils.weight_quantization import kv_cache_offload

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("kv_debug")

def test_kv_cache_basics():
    """Test basic KV cache functionality with detailed logging"""
    logger.info("Initializing model for detailed debugging...")
    
    # Create a small model for debugging
    config = EdgeFormerConfig(
        vocab_size=30522,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        latent_size_factor=8,
        max_position_embeddings=2048,  # Ensure this is large enough
    )
    
    model = EdgeFormer(config)
    model.eval()
    
    # Enable KV cache offloading
    offloaded_model = kv_cache_offload(model)
    
    # Create a simple input sequence
    seq_length = 128  # Start with a short sequence
    input_ids = torch.randint(0, config.vocab_size, (1, seq_length))
    attention_mask = torch.ones(1, seq_length)
    
    logger.info(f"Running initial pass with sequence length {seq_length}...")
    
    # First pass
    outputs = offloaded_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True
    )
    
    # Log output structure
    logger.info(f"Output keys: {list(outputs.keys())}")
    if "past_key_values" in outputs:
        past_kv = outputs["past_key_values"]
        logger.info(f"past_key_values type: {type(past_kv)}")
        
        if isinstance(past_kv, str):
            logger.info(f"past_key_values is a string: {past_kv}")
        elif isinstance(past_kv, tuple):
            logger.info(f"past_key_values is a tuple with {len(past_kv)} elements")
            if len(past_kv) > 0:
                logger.info(f"First element type: {type(past_kv[0])}")
                if isinstance(past_kv[0], tuple):
                    logger.info(f"First element is a tuple with {len(past_kv[0])} sub-elements")
                    logger.info(f"Key shape: {past_kv[0][0].shape}, Value shape: {past_kv[0][1].shape}")
    else:
        logger.error("No past_key_values in output!")
        return
    
    # Test second pass with one new token
    logger.info("Running second pass with one new token...")
    
    next_token = torch.randint(0, config.vocab_size, (1, 1))
    next_mask = torch.ones(1, seq_length + 1)  # Include all previous tokens + new one
    
    # Detailed logging of second pass inputs
    logger.info(f"Next token shape: {next_token.shape}")
    logger.info(f"Next mask shape: {next_mask.shape}")
    logger.info(f"past_key_values being passed: {past_kv}")
    
    try:
        next_outputs = offloaded_model(
            input_ids=next_token,
            attention_mask=next_mask,
            past_key_values=past_kv,
            use_cache=True
        )
        
        logger.info("Second pass completed successfully!")
        
        # Log output structure
        logger.info(f"Second pass output keys: {list(next_outputs.keys())}")
        if "past_key_values" in next_outputs:
            next_past_kv = next_outputs["past_key_values"]
            logger.info(f"New past_key_values type: {type(next_past_kv)}")
            
            if isinstance(next_past_kv, str):
                logger.info(f"New past_key_values is a string: {next_past_kv}")
        else:
            logger.error("No past_key_values in second pass output!")
        
        # Clean up
        if hasattr(offloaded_model, "cleanup_kv_cache"):
            offloaded_model.cleanup_kv_cache()
            logger.info("Cleaned up KV cache")
        
        logger.info("Test completed successfully!")
    
    except Exception as e:
        logger.error(f"Error during second pass: {str(e)}")
        logger.exception("Detailed error stack:")

if __name__ == "__main__":
    test_kv_cache_basics()