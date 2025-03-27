# examples/debug_sliding_window.py
import torch
import logging
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.multi_head_latent_attention import MultiHeadLatentAttention
from src.model.config import EdgeFormerConfig

# Setup logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("debug_sliding_window")

def test_sliding_window_directly():
    """Test the sliding window implementation directly on the attention module"""
    # Create a simple config
    config = EdgeFormerConfig(
        vocab_size=1000,
        hidden_size=128,
        num_attention_heads=4,
        # Make sure latent_size is here if added to config
    )
    
    # Create just the attention module
    attn = MultiHeadLatentAttention(config)
    
    # Create sample input
    batch_size = 1
    seq_len = 64
    hidden_size = config.hidden_size
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Create attention mask (all ones for simplicity)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
    
    # Run without sliding window
    logger.info("Running without sliding window...")
    outputs_standard = attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask
    )
    
    # Run with sliding window
    logger.info("Running with sliding window...")
    sliding_window_size = 16
    outputs_sliding = attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        sliding_window_size=sliding_window_size
    )
    
    # Compare outputs
    output_diff = (outputs_standard[0] - outputs_sliding[0]).abs().mean().item()
    logger.info(f"Average absolute difference between outputs: {output_diff}")
    
    if output_diff < 0.01:
        logger.warning("Outputs almost identical - sliding window might not be working")
    else:
        logger.info("Sliding window appears to be working (outputs are different)")

if __name__ == "__main__":
    test_sliding_window_directly()