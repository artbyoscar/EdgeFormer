# Create a new test script first
# test_kv_simple.py
import torch
import logging
from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from src.utils.weight_quantization import kv_cache_offload

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kv_test")

# Create a small model
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

# --- Modification to prepare_inputs_for_generation ---
# Add this method to your EdgeFormer class

def prepare_inputs_for_continuation(self, new_input_ids, past_key_values, prev_attention_mask=None):
    """Prepare inputs specifically for continuation with KV cache"""
    # Get batch size and sequence length
    batch_size = new_input_ids.shape[0]
    new_seq_length = new_input_ids.shape[1]
    
    # Create attention mask for all tokens (past + new)
    past_length = 0
    if past_key_values is not None:
        # Get past length from the past_key_values if it's a tuple
        if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
            if isinstance(past_key_values[0], tuple) and len(past_key_values[0]) > 0:
                # Get length from the first layer's key tensor
                past_length = past_key_values[0][0].size(1) if past_key_values[0][0] is not None else 0
    
    # Create new attention mask for all tokens
    total_length = past_length + new_seq_length
    attention_mask = torch.ones((batch_size, total_length), device=new_input_ids.device)
    
    return {
        "input_ids": new_input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": True
    }

# Add the method to the model
offloaded_model.prepare_inputs_for_continuation = prepare_inputs_for_continuation.__get__(offloaded_model, type(offloaded_model))

# Test continuation
initial_ids = torch.randint(0, config.vocab_size, (1, 32))
initial_mask = torch.ones(1, 32)

logger.info("Running first forward pass...")
outputs = offloaded_model(
    input_ids=initial_ids,
    attention_mask=initial_mask,
    use_cache=True
)

kv_cache_id = outputs["past_key_values"]
logger.info(f"KV cache ID: {kv_cache_id}")

# Next token
next_token = torch.randint(0, config.vocab_size, (1, 1))

# Prepare inputs for continuation
continuation_inputs = offloaded_model.prepare_inputs_for_continuation(
    next_token, 
    kv_cache_id,
    initial_mask
)

logger.info("Running continuation pass...")
next_outputs = offloaded_model(**continuation_inputs)

logger.info("Test completed successfully!")