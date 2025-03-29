import torch
import torch.nn as nn
from src.model.config import EdgeFormerConfig
import os

def create_edgeformer_adapter():
    """Create a simplified adapter for EdgeFormer"""
    
    # Define the adapter class
    adapter_code = """import torch
import torch.nn as nn
from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer

class EdgeFormerAdapter(nn.Module):
    \"\"\"
    Adapter to make EdgeFormer work with the same interface as the simple model
    \"\"\"
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.edgeformer = EdgeFormer(config)
        
    def forward(self, input_ids):
        \"\"\"
        Handles input shape compatibility with EdgeFormer
        
        Args:
            input_ids: Input tensor of shape [batch_size, seq_len]
            
        Returns:
            logits: Output tensor of shape [batch_size, seq_len, vocab_size]
        \"\"\"
        # Check input dimensions
        if input_ids.dim() == 2:
            # EdgeFormer expects inputs of shape [batch_size, seq_len]
            outputs = self.edgeformer(input_ids)
            return outputs
        else:
            raise ValueError(f"Expected input_ids to have 2 dimensions, got {input_ids.dim()}")
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=None):
        \"\"\"
        Generate text using the EdgeFormer model
        
        Args:
            input_ids: Starting token IDs of shape [batch_size, seq_len]
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated token IDs of shape [batch_size, max_length]
        \"\"\"
        self.eval()
        current_input = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Get prediction for next token
                outputs = self(current_input)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply top-k sampling
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")
                
                # Sample from the distribution
                probs = nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to current input
                current_input = torch.cat([current_input, next_token], dim=1)
                
        return current_input
"""
    
    # Create directory if it doesn't exist
    os.makedirs("src/model", exist_ok=True)
    
    # Write the adapter file
    with open("src/model/edgeformer_adapter.py", "w", encoding="utf-8") as f:
        f.write(adapter_code)
    
    print("Created EdgeFormerAdapter class")

if __name__ == "__main__":
    create_edgeformer_adapter()