import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add current directory to path
sys.path.append(".")

from src.model.config import EdgeFormerConfig
from src.model.recurrent_block import RecurrentTransformerBlock
from src.utils.value_estimator import ValueEstimator

def test_recurrent_depth_processing():
    """Test the value-based recurrent depth processing"""
    print("Testing value-based recurrent depth processing...")
    
    # Create a simple configuration
    class SimpleConfig:
        def __init__(self):
            self.hidden_size = 64
            self.intermediate_size = 256
            self.num_attention_heads = 4
            self.attention_probs_dropout_prob = 0.1
            self.hidden_dropout_prob = 0.1
    
    config = SimpleConfig()
    
    # Create recurrent block and value estimator
    recurrent_block = RecurrentTransformerBlock(config)
    value_estimator = ValueEstimator(config.hidden_size)
    
    # Create some dummy input
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Test different max_iterations
    iterations_to_test = [1, 2, 4, 8, 16, 32]
    value_histories = []
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    for max_iterations in iterations_to_test:
        print(f"\nTesting with max_iterations={max_iterations}")
        
        # Reset
        value_estimator.reset()
        recurrent_block.reset_iteration_count()
        
        # Clone the input for this test
        test_hidden = hidden_states.clone()
        
        # Iterate the recurrent block
        value_history = []
        
        for i in range(max_iterations):
            # Process hidden states
            test_hidden = recurrent_block(test_hidden)
            
            # Get value estimate
            value = value_estimator(test_hidden).mean().item()
            value_history.append(value)
            
            print(f"Iteration {i+1}, Value: {value:.4f}")
            
            # Optional: check for convergence
            if i >= 3:  # Wait a few iterations before checking
                if value_estimator.check_convergence(test_hidden):
                    print(f"Converged after {i+1} iterations!")
                    break
        
        # Store value history
        value_histories.append(value_history)
    
    # Plot value histories
    plt.figure(figsize=(10, 6))
    for i, history in enumerate(value_histories):
        iterations = iterations_to_test[i]
        plt.plot(range(1, len(history) + 1), history, label=f"Max {iterations}")
    
    plt.xlabel("Iteration")
    plt.ylabel("Value Estimate")
    plt.title("Value Estimates During Recurrent Processing")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/value_recurrent_depth.png")
    
    # Test convergence properties
    print("\nTesting convergence properties...")
    
    # Create more challenging input with pattern
    pattern_input = torch.zeros(1, 16, config.hidden_size)
    for i in range(16):
        # Create a pattern that changes more at the beginning
        pattern_input[0, i, :] = torch.sin(torch.tensor([j / 5 + i / 3 for j in range(config.hidden_size)]))
    
    # Reset
    value_estimator.reset()
    recurrent_block.reset_iteration_count()
    
    # Iterate with convergence checking
    test_pattern = pattern_input.clone()
    convergence_values = []
    
    max_test_iterations = 50
    for i in range(max_test_iterations):
        # Process hidden states
        test_pattern = recurrent_block(test_pattern)
        
        # Get value estimate
        value = value_estimator(test_pattern).mean().item()
        convergence_values.append(value)
        
        print(f"Pattern Iteration {i+1}, Value: {value:.4f}")
        
        # Check for convergence
        if i >= 3 and value_estimator.check_convergence(test_pattern):
            print(f"Pattern converged after {i+1} iterations!")
            break
    
    # Plot convergence values
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(convergence_values) + 1), convergence_values, 'b-o')
    plt.xlabel("Iteration")
    plt.ylabel("Value Estimate")
    plt.title("Convergence Pattern for Structured Input")
    plt.grid(True)
    plt.savefig("plots/convergence_pattern.png")
    
    print("\nValue-based recurrent depth processing test completed.")
    print("Results saved to plots/value_recurrent_depth.png and plots/convergence_pattern.png")

if __name__ == "__main__":
    test_recurrent_depth_processing()