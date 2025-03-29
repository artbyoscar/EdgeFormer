import torch
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path
sys.path.append('.')

from src.utils.value_estimator import ValueEstimator

def test_value_estimator():
    """Test the value estimator functionality"""
    print("Testing value estimator...")
    
    # Create a value estimator
    hidden_size = 64
    value_estimator = ValueEstimator(hidden_size)
    
    # Create some dummy hidden states
    batch_size = 2
    seq_len = 16
    
    # Create different types of hidden states to test
    # 1. Random noise (low quality)
    random_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 2. Increasingly structured states (medium quality)
    structured_states = []
    for i in range(10):
        # Create more structured hidden states with each iteration
        coherence = i / 10.0  # 0.0 to 0.9
        state = torch.randn(batch_size, seq_len, hidden_size) * (1.0 - coherence) + torch.sin(torch.arange(0, hidden_size) * 0.1) * coherence
        structured_states.append(state)
    
    # 3. Highly structured states (high quality)
    pattern_states = []
    for i in range(10):
        # Create increasingly refined patterns
        refinement = 0.5 + i / 20.0  # 0.5 to 0.95
        base_pattern = torch.sin(torch.arange(0, hidden_size) * 0.1)
        state = torch.zeros(batch_size, seq_len, hidden_size)
        for j in range(seq_len):
            state[:, j, :] = base_pattern + torch.randn(hidden_size) * (1.0 - refinement)
        pattern_states.append(state)
    
    # Test value estimates
    print("\nTesting random states:")
    random_value = value_estimator(random_states).mean().item()
    print(f"Random states value: {random_value:.4f}")
    
    print("\nTesting increasingly structured states:")
    structured_values = []
    for i, state in enumerate(structured_states):
        value = value_estimator(state).mean().item()
        structured_values.append(value)
        print(f"Structured state {i+1}, coherence {(i+1)/10:.1f}, value: {value:.4f}")
    
    print("\nTesting highly structured states:")
    pattern_values = []
    for i, state in enumerate(pattern_states):
        value = value_estimator(state).mean().item()
        pattern_values.append(value)
        print(f"Pattern state {i+1}, refinement {0.5+(i+1)/20:.2f}, value: {value:.4f}")
    
    # Test convergence detection
    print("\nTesting convergence detection:")
    value_estimator.reset()
    for i, state in enumerate(pattern_states):
        converged = value_estimator.check_convergence(state)
        value = value_estimator.value_history[-1]
        print(f"Iteration {i+1}, value: {value:.4f}, converged: {converged}")
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Plot value estimates
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), structured_values, 'b-o', label="Structured (medium)")
    plt.plot(range(1, 11), pattern_values, 'r-o', label="Pattern (high)")
    plt.axhline(y=random_value, color='g', linestyle='--', label="Random (low)")
    plt.xlabel("Iteration")
    plt.ylabel("Value Estimate")
    plt.title("Value Estimates for Different Hidden State Types")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/value_estimates.png")
    
    print("\nValue estimator test completed. Results saved to plots/value_estimates.png")

if __name__ == "__main__":
    test_value_estimator()