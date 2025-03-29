"""
Test script for Value Estimator integration with EdgeFormer.
"""
import torch
import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.value_estimator import ValueEstimator

def parse_args():
    parser = argparse.ArgumentParser(description="Test Value Estimator integration")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size")
    parser.add_argument("--iterations", type=int, default=5, help="Number of recurrent iterations")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Testing Value Estimator integration with {args.iterations} iterations")
    
    # Create a small config for testing
    config = EdgeFormerConfig(
        vocab_size=1000,
        hidden_size=args.hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=args.hidden_size * 4
    )
    
    # Create model
    model = EdgeFormer(config)
    
    # Create value estimator
    value_estimator = ValueEstimator(config.hidden_size, config)
    
    # Set value estimator in model
    model.set_value_estimator(value_estimator)
    
    # Generate random input
    input_ids = torch.randint(0, 1000, (1, 10))
    
    # Initial forward pass
    with torch.no_grad():
        outputs = model.forward_with_hidden_states(
            input_ids=input_ids,
            return_hidden_states=True
        )
        logits, hidden_states = outputs
    
    print(f"Initial hidden states shape: {hidden_states.shape}")
    print(f"Initial logits shape: {logits.shape}")
    
    # Test recurrent processing
    values = []
    for i in range(args.iterations):
        with torch.no_grad():
            # Forward pass with existing hidden states
            outputs = model.forward_with_hidden_states(
                hidden_states=hidden_states,
                return_hidden_states=True
            )
            logits, hidden_states = outputs
            
            # Get value estimate
            value = value_estimator(hidden_states).item()
            values.append(value)
            
            print(f"Iteration {i+1}: Value = {value:.4f}")
            
            # Check for convergence
            if value_estimator.check_convergence(hidden_states):
                print(f"Converged at iteration {i+1}")
                break
    
    print(f"Final hidden states shape: {hidden_states.shape}")
    print(f"Final logits shape: {logits.shape}")
    print(f"Value history: {values}")

if __name__ == "__main__":
    main()