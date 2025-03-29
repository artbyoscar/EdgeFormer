"""
Test script for Value-Based Recurrent Processing
"""
import torch
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.value_estimator import ValueEstimator

def parse_args():
    parser = argparse.ArgumentParser(description="Test Value-Based Recurrent Processing")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--seq_length", type=int, default=32, help="Sequence length")
    parser.add_argument("--iterations", type=int, default=20, help="Maximum iterations")
    parser.add_argument("--convergence_threshold", type=float, default=0.005, help="Convergence threshold")
    return parser.parse_args()

def generate_test_data(hidden_size, seq_length, device="cpu"):
    """Generate test data with improving structure over iterations"""
    base_tensor = torch.randn(1, seq_length, hidden_size, device=device)
    
    # Generate a target pattern (structured)
    target = torch.zeros(1, seq_length, hidden_size, device=device)
    # Create a wave pattern
    for i in range(hidden_size):
        target[0, :, i] = torch.sin(torch.linspace(0, 3*3.14159, seq_length) + i * 0.5)
    
    # List of increasingly structured tensors
    tensors = []
    
    # Start with random
    tensors.append(base_tensor.clone())
    
    # Generate increasingly structured tensors
    mix_ratios = torch.linspace(0.1, 0.9, 19)
    for ratio in mix_ratios:
        # Mix random and structured with increasing ratio of structure
        mixed = (1 - ratio) * base_tensor + ratio * target
        tensors.append(mixed)
    
    return tensors

def test_value_convergence(iterations, hidden_size, seq_length):
    """Test value estimator convergence on increasingly structured data"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize value estimator
    value_estimator = ValueEstimator(hidden_size)
    value_estimator.to(device)
    
    # Generate test data
    test_tensors = generate_test_data(hidden_size, seq_length, device)
    
    # Evaluate each tensor
    values = []
    convergence_points = []
    
    for i, tensor in enumerate(test_tensors):
        # Reset estimator
        value_estimator.reset()
        
        # Get value
        with torch.no_grad():
            value = value_estimator(tensor).item()
        
        values.append(value)
        
        # Only test convergence on first tensor
        if i == 0:
            # Test convergence detection
            value_estimator.reset()
            iterations_to_converge = iterations  # Default if no convergence
            
            for j in range(iterations):
                # Small random noise to simulate successive iterations
                noise_level = 0.1 * (1.0 - j/iterations)  # Decreasing noise
                noisy_tensor = tensor + noise_level * torch.randn_like(tensor)
                
                with torch.no_grad():
                    value_estimator(noisy_tensor)
                
                if value_estimator.check_convergence(noisy_tensor):
                    iterations_to_converge = j + 1
                    break
            
            convergence_points.append(iterations_to_converge)
    
    return values, convergence_points, value_estimator.get_value_history()

def visualize_results(values, convergence_points, value_history):
    """Visualize test results"""
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Values for increasingly structured tensors
    plt.subplot(2, 1, 1)
    plt.plot(values, 'b-o')
    plt.title('Value Estimates for Increasingly Structured Data')
    plt.xlabel('Structure Level (0 = Random, 19 = Highly Structured)')
    plt.ylabel('Value Estimate')
    plt.grid(True)
    
    # Plot 2: Value history during convergence test
    plt.subplot(2, 1, 2)
    plt.plot(value_history, 'r-o')
    plt.axvline(x=convergence_points[0]-1, color='g', linestyle='--', 
                label=f'Convergence at iteration {convergence_points[0]}')
    plt.title('Value History During Convergence Test')
    plt.xlabel('Iteration')
    plt.ylabel('Value Estimate')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('value_recurrent_test.png')
    print("Visualization saved to 'value_recurrent_test.png'")
    plt.show()

def main():
    args = parse_args()
    print(f"Testing Value-Based Recurrent Processing with {args.iterations} max iterations")
    
    # Run test
    values, convergence_points, value_history = test_value_convergence(
        args.iterations, args.hidden_size, args.seq_length
    )
    
    # Print results
    print("\nResults:")
    print(f"- Value for random tensor: {values[0]:.4f}")
    print(f"- Value for most structured tensor: {values[-1]:.4f}")
    print(f"- Value increase: {values[-1] - values[0]:.4f} ({(values[-1] - values[0])/values[0]*100:.1f}%)")
    print(f"- Detected convergence at iteration: {convergence_points[0]}")
    
    # Visualize
    visualize_results(values, convergence_points, value_history)

if __name__ == "__main__":
    main()