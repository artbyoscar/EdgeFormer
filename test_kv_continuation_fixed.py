# test_kv_continuation_fixed.py
import torch
import logging
from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from src.utils.weight_quantization import kv_cache_offload

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kv_test")

def main():
    """Test KV cache continuation with the improved approach"""
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
    
    # Add the continue_generation helper method to the model
    def continue_generation(self, new_tokens, past_key_values):
        """Helper method to continue generation with KV cache"""
        # Get past length if it's a tensor tuple
        past_length = 0
        if not isinstance(past_key_values, str) and past_key_values is not None:
            if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
                if isinstance(past_key_values[0], tuple) and len(past_key_values[0]) > 0:
                    if past_key_values[0][0] is not None:
                        past_length = past_key_values[0][0].size(1)
        
        # Get batch size and new sequence length
        batch_size = new_tokens.shape[0]
        new_seq_length = new_tokens.shape[1]
        
        # Create attention mask that spans both past tokens and new tokens
        total_seq_length = past_length + new_seq_length
        attention_mask = torch.ones((batch_size, total_seq_length), device=new_tokens.device)
        
        # Run forward pass with prepared inputs
        outputs = self(
            input_ids=new_tokens,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )
        
        return outputs
    
    # Add the method to the model
    model.continue_generation = continue_generation.__get__(model, type(model))
    
    # Enable KV cache offloading
    offloaded_model = kv_cache_offload(model)
    
    # First pass
    seq_length = 32  # Start with a short sequence
    logger.info(f"Running first pass with sequence length {seq_length}")
    
    input_ids = torch.randint(0, config.vocab_size, (1, seq_length))
    attention_mask = torch.ones(1, seq_length)
    
    outputs = offloaded_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True
    )
    
    # Log the KV cache
    past_key_values = outputs["past_key_values"]
    logger.info(f"KV cache ID: {past_key_values}")
    
    # Second pass using the helper method
    logger.info("Running second pass with one new token...")
    next_token = torch.randint(0, config.vocab_size, (1, 1))
    
    next_outputs = offloaded_model.continue_generation(
        next_token,
        past_key_values
    )
    
    logger.info("Second pass completed successfully!")
    
    # Third pass using the helper method again
    logger.info("Running third pass with one more token...")
    next_next_token = torch.randint(0, config.vocab_size, (1, 1))
    
    next_next_outputs = offloaded_model.continue_generation(
        next_next_token,
        next_outputs["past_key_values"]
    )
    
    logger.info("Third pass completed successfully!")
    
    # Clean up
    if hasattr(offloaded_model, "cleanup_kv_cache"):
        offloaded_model.cleanup_kv_cache()
        logger.info("Cleaned up KV cache")
    
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    main()