"""
Value Estimator Test Script
===========================

Tests the Value Estimator's ability to differentiate between random and structured states.

Usage:
    python examples/test_value_estimator.py --iterations 10 --visualize
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.value_estimator import ValueEstimator
# For testing the improved version (when implemented)
# from src.utils.improved_value_estimator import ImprovedValueEstimator

def parse_arguments():
    parser = argparse.ArgumentParser(description="Test Value Estimator")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden state size")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=16, help="Sequence length")
    parser.add_argument("--iterations", type=int, default=10, help="Number of test iterations")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--improved", action="store_true", help="Use improved value estimator")
    return parser.parse_args()

def generate_random_state(hidden_size, batch_size, seq_length):
    """Generate a completely random hidden state."""
    return torch.randn(batch_size, seq_length, hidden_size)

def generate_structured_state(hidden_size, batch_size, seq_length, pattern="repeating"):
    """Generate a structured hidden state with recognizable patterns."""
    if pattern == "repeating":
        # Create a repeating pattern
        base_pattern = torch.randn(1, 1, hidden_size)
        repeats = torch.ones(batch_size, seq_length, 1)
        return base_pattern * repeats + torch.randn(batch_size, seq_length, hidden_size) * 0.1
    
    elif pattern == "sequential":
        # Create a sequential pattern
        base = torch.arange(0, seq_length).float().unsqueeze(0).unsqueeze(-1) / seq_length
        base = base.repeat(batch_size, 1, hidden_size)
        return base + torch.randn(batch_size, seq_length, hidden_size) * 0.1
    
    elif pattern == "clustered":
        # Create clustered data (similar within batch)
        states = []
        for i in range(batch_size):
            center = torch.randn(1, hidden_size)
            state = center.repeat(seq_length, 1) + torch.randn(seq_length, hidden_size) * 0.2
            states.append(state.unsqueeze(0))
        return torch.cat(states, dim=0)
    
    else:
        raise ValueError(f"Unknown pattern type: {pattern}")

def test_value_estimator(args):
    print(f"Testing {'improved ' if args.improved else ''}Value Estimator...")
    
    # Create value estimator
    if args.improved and 'ImprovedValueEstimator' in globals():
        value_estimator = ImprovedValueEstimator(args.hidden_size)
    else:
        value_estimator = ValueEstimator(args.hidden_size)
    
    # Test data structures
    pattern_types = ["repeating", "sequential", "clustered"]
    
    # Results storage
    results = {
        "random": [],
        **{pattern: [] for pattern in pattern_types}
    }
    
    # Run multiple iterations for stability
    for i in range(args.iterations):
        print(f"Iteration {i+1}/{args.iterations}")
        
        # Test with random data
        random_state = generate_random_state(args.hidden_size, args.batch_size, args.seq_length)
        with torch.no_grad():
            random_value = value_estimator(random_state).mean().item()
        results["random"].append(random_value)
        
        # Test with structured data
        for pattern in pattern_types:
            structured_state = generate_structured_state(
                args.hidden_size, args.batch_size, args.seq_length, pattern
            )
            with torch.no_grad():
                structured_value = value_estimator(structured_state).mean().item()
            results[pattern].append(structured_value)
    
    # Calculate statistics
    stats = {}
    for key, values in results.items():
        stats[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }
    
    # Print results
    print("\nResults Summary:")
    print("="*50)
    for key, stat in stats.items():
        print(f"{key.capitalize()} states:")
        print(f"  Mean value: {stat['mean']:.4f}")
        print(f"  Std dev: {stat['std']:.4f}")
        print(f"  Range: {stat['min']:.4f} - {stat['max']:.4f}")
    
    # Calculate discrimination ability
    for pattern in pattern_types:
        diff = stats[pattern]["mean"] - stats["random"]["mean"]
        print(f"\nDiscrimination ({pattern} vs random): {diff:.4f}")
        if diff > 0:
            effectiveness = diff / (stats[pattern]["std"] + stats["random"]["std"])
            print(f"Effectiveness: {effectiveness:.4f}")
        else:
            print("Warning: Structured states not valued higher than random states!")
    
    # Visualize if requested
    if args.visualize:
        visualize_results(results)
    
    return results, stats

def visualize_results(results):
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    data = []
    labels = []
    
    for key, values in results.items():
        data.append(values)
        labels.append(key.capitalize())
    
    # Create box plot
    plt.boxplot(data, labels=labels)
    plt.title("Value Estimator Results: Random vs. Structured States")
    plt.ylabel("Value Estimate")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add scatter points for individual data points
    for i, values in enumerate(data):
        x = np.random.normal(i+1, 0.04, size=len(values))
        plt.scatter(x, values, alpha=0.4)
    
    # Save figure
    plt.tight_layout()
    plt.savefig("value_estimator_test.png")
    print("Visualization saved to 'value_estimator_test.png'")

def main():
    args = parse_arguments()
    test_value_estimator(args)

if __name__ == "__main__":
    main()