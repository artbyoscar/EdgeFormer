# examples/test_kv_offload.py
import torch
import logging
import argparse
from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from src.utils.weight_quantization import kv_cache_offload

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edgeformer")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seq-length", type=int, default=32, help="Sequence length for testing")
parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
args = parser.parse_args()

# Set logging level
logger.setLevel(args.log_level)

# Print startup message
logger.info("Starting EdgeFormer test...")

# Create model configuration
config = EdgeFormerConfig(
    vocab_size=30522,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=8,
    latent_size_factor=8,  # Lower latent size factor for efficiency
)

# Initialize model
model = EdgeFormer(config)
model.eval()

# Enable KV cache offloading
logger.info("Starting EdgeFormer KV cache offload test...")
model = kv_cache_offload(model)

# Create input tensors
input_ids = torch.randint(0, config.vocab_size, (1, args.seq_length))
attention_mask = torch.ones(1, args.seq_length)

# First forward pass
logger.info("Running first forward pass...")
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    use_cache=True
)

# Check if past_key_values exists
past_key_values_id = outputs.get("past_key_values")
logger.info(f"Generated KV cache with ID: {past_key_values_id}")

# Create next tokens
next_tokens = torch.randint(0, config.vocab_size, (1, 4))  # Just 4 new tokens
next_attention_mask = torch.ones(1, args.seq_length + 4)  # seq_length + 4

# Second forward pass
logger.info("Running second forward pass with offloaded KV cache...")
next_outputs = model(
    input_ids=next_tokens,
    attention_mask=next_attention_mask,
    past_key_values=past_key_values_id,
    use_cache=True
)

logger.info("KV cache offloading test completed successfully!")