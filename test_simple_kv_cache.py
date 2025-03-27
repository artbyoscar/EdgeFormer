# test_simple_kv_cache.py
import torch
from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from src.utils.weight_quantization import kv_cache_offload
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edgeformer")

# Create model configuration
config = EdgeFormerConfig(
    vocab_size=30522,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=8,
    latent_size_factor=8
)

# Initialize model
model = EdgeFormer(config)
model.eval()

# Enable KV cache offloading
model = kv_cache_offload(model)

# Create input tensors
input_ids = torch.randint(0, config.vocab_size, (1, 32))
attention_mask = torch.ones(1, 32)

# First forward pass
logger.info("Running first forward pass...")
outputs1 = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    use_cache=True
)

# Check if past_key_values exists
if "past_key_values" in outputs1 and outputs1["past_key_values"] is not None:
    logger.info(f"past_key_values found: {outputs1['past_key_values']}")
else:
    logger.error("past_key_values not found in outputs!")
    
# Add a printout of outputs keys to debug
logger.info(f"Output keys: {list(outputs1.keys())}")

# Create next tokens
next_input_ids = torch.randint(0, config.vocab_size, (1, 4))
next_attention_mask = torch.ones(1, 36)  # 32 + 4

# Second forward pass
logger.info("Running second forward pass...")
outputs2 = model(
    input_ids=next_input_ids,
    attention_mask=next_attention_mask,
    past_key_values=outputs1["past_key_values"],
    use_cache=True
)

logger.info("Test completed successfully!")