import sys
import os
import torch
from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.logging import setup_logging

# Set up logging
logger = setup_logging()

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a test configuration with extended position embeddings
test_config = EdgeFormerConfig(
    hidden_size=256,
    num_hidden_layers=6,
    num_attention_heads=8,
    latent_size_factor=8,
    intermediate_size=1024,
    max_position_embeddings=8192,  # Increased from 2048
)

# Initialize the model
model = EdgeFormer(test_config)
model.eval()

# Test specific sequence lengths
test_lengths = [2048, 4096]

for seq_len in test_lengths:
    try:
        print(f"\nTesting sequence length: {seq_len}")
        # Create input tensors
        input_ids = torch.randint(0, test_config.vocab_size, (1, seq_len))
        attention_mask = torch.ones(1, seq_len)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        print(f"✅ Successfully processed sequence length: {seq_len}")
    except Exception as e:
        print(f"❌ Failed at sequence length: {seq_len}")
        print(f"Error: {str(e)}")