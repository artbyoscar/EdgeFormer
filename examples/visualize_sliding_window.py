import torch
import matplotlib.pyplot as plt
import numpy as np
from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig

# Test script to visualize attention patterns
def visualize_attention(model, input_ids, with_sliding_window=True, window_size=128):
    # Get attention scores with and without sliding window
    # You'll need to modify your model to return attention scores
    outputs_with_window = model(
        input_ids=input_ids,
        sliding_window_size=window_size if with_sliding_window else None,
        output_attentions=True
    )
    
    # Extract and visualize attention scores
    attention_scores = outputs_with_window["attentions"][0][0]  # First layer, first head
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_scores.detach().cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title(f"Attention Pattern {'With' if with_sliding_window else 'Without'} Sliding Window")
    plt.savefig(f"attention_pattern_{'with' if with_sliding_window else 'without'}_window.png")
    
# Run visualization
config = EdgeFormerConfig(hidden_size=128, num_attention_heads=4, latent_size=16)
model = EdgeFormer(config)
model.eval()

seq_len = 256
input_ids = torch.randint(0, 100, (1, seq_len))

visualize_attention(model, input_ids, with_sliding_window=False)
visualize_attention(model, input_ids, with_sliding_window=True, window_size=64)