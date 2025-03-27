# test_extreme_length_kv.py
import torch
import logging
import argparse
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
logger = logging.getLogger("extreme_sequence_test")

def test_extreme_length():
    """Test KV cache offloading with extremely long sequences"""
    parser = argparse.ArgumentParser(description="Test extreme sequence lengths with KV cache offloading")
    parser.add_argument("--length", type=int, default=32768, help="Sequence length to test")
    parser.add_argument("--hidden-size", type=int, default=128, help="Model hidden size (smaller for extreme test)")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers (smaller for extreme test)")
    args = parser.parse_args()
    
    # Initialize model (smaller than usual for extreme length test)
    logger.info(f"Initializing model for extreme sequence length: {args.length}")
    config = EdgeFormerConfig(
        vocab_size=30522,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=4,
        latent_size_factor=8,
    )
    
    model = EdgeFormer(config)
    model.eval()
    
    # Enable KV cache offloading with chunking
    offloaded_model = kv_cache_offload(
        model, 
        chunk_size=1024,  # Use chunking for extreme sequences
        compression_level=1  # Light compression
    )
    
    # Create input tensors
    input_ids = torch.randint(0, config.vocab_size, (1, args.length))
    attention_mask = torch.ones(1, args.length)
    
    # Run forward pass with timing and error handling
    try:
        logger.info("Starting forward pass...")
        start_time = time.time()
        
        outputs = offloaded_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True
        )
        
        elapsed = time.time() - start_time
        tokens_per_second = args.length / elapsed
        
        logger.info(f"Successfully processed sequence of length {args.length}")
        logger.info(f"Processing time: {elapsed:.2f} seconds")
        logger.info(f"Speed: {tokens_per_second:.2f} tokens/second")
        
        if "past_key_values" in outputs:
            logger.info(f"KV cache ID: {outputs['past_key_values']}")
            
            # Try a follow-up token
            next_token = torch.randint(0, config.vocab_size, (1, 1))
            next_attention_mask = torch.ones(1, args.length + 1)
            
            logger.info("Testing continuation with one additional token...")
            start_time = time.time()
            
            next_outputs = offloaded_model(
                input_ids=next_token,
                attention_mask=next_attention_mask,
                past_key_values=outputs["past_key_values"],
                use_cache=True
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Continuation processing time: {elapsed:.2f} seconds")
            
            if "past_key_values" in next_outputs:
                logger.info(f"New KV cache ID: {next_outputs['past_key_values']}")
                
                # Clean up
                if hasattr(offloaded_model, "cleanup_kv_cache"):
                    offloaded_model.cleanup_kv_cache(outputs["past_key_values"])
                    offloaded_model.cleanup_kv_cache(next_outputs["past_key_values"])
    
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.exception("Detailed error information:")

if __name__ == "__main__":
    test_extreme_length()