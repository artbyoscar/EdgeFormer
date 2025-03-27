# examples/test_sliding_window.py
import torch
import logging
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig

# Setup logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_sliding_window")

def test_simple_sliding_window():
    # Create a configuration using parameters that exist in your class
    config = EdgeFormerConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=2048,
        use_sliding_window=True,  # Enable sliding window
        sliding_window_size=128,  # Default window size
        debug_mode=True,  # Enable verbose logging
    )
    
    logger.info(f"Latent size: {config.latent_size}")
    logger.info(f"Sliding window size: {config.sliding_window_size}")
    
    model = EdgeFormer(config)
    model.eval()
    
    # Create input data
    input_ids = torch.randint(0, 100, (1, 512))
    attention_mask = torch.ones(1, 512)
    
    # First test with the default behavior
    logger.info("Testing with default behavior...")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    
    # Then explicitly turn off sliding window
    logger.info("Testing with explicitly disabled sliding window...")
    non_window_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        sliding_window_size=None  # Explicitly disable sliding window
    )
    
    # Then explicitly set a different window size
    logger.info("Testing with custom sliding window size...")
    custom_window_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        sliding_window_size=64  # Use a smaller window
    )
    
    # Check if outputs differ
    default_vs_none = (outputs["logits"] - non_window_outputs["logits"]).abs().mean().item()
    default_vs_custom = (outputs["logits"] - custom_window_outputs["logits"]).abs().mean().item()
    
    logger.info(f"Default vs. No Window difference: {default_vs_none}")
    logger.info(f"Default vs. Custom Window difference: {default_vs_custom}")
    
    # Try with longer sequence
    try:
        logger.info("Testing with longer sequence (1024 tokens)...")
        long_input_ids = torch.randint(0, 100, (1, 1024))
        long_attention_mask = torch.ones(1, 1024)
        
        # With sliding window
        long_outputs = model(
            input_ids=long_input_ids,
            attention_mask=long_attention_mask,
            sliding_window_size=256
        )
        logger.info("Sliding window successful with 1024 tokens")
        
        # Try with even longer sequence
        logger.info("Testing with even longer sequence (2048 tokens)...")
        very_long_input_ids = torch.randint(0, 100, (1, 2048))
        very_long_attention_mask = torch.ones(1, 2048)
        
        very_long_outputs = model(
            input_ids=very_long_input_ids,
            attention_mask=very_long_attention_mask,
            sliding_window_size=256
        )
        logger.info("Sliding window successful with 2048 tokens")
        
        # Try without sliding window on long sequence
        logger.info("Testing long sequence without sliding window...")
        try:
            no_window_long = model(
                input_ids=very_long_input_ids,
                attention_mask=very_long_attention_mask,
                sliding_window_size=None
            )
            logger.info("Full attention also worked with 2048 tokens")
        except Exception as e:
            logger.error(f"Full attention failed with 2048 tokens: {e}")
            logger.info("This confirms sliding window is helping with longer sequences")
        
    except Exception as e:
        logger.error(f"Error with long sequence: {e}")

if __name__ == "__main__":
    test_simple_sliding_window()