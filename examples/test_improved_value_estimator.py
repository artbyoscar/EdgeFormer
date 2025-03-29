import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from tqdm import tqdm

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.improved_value_estimator import ImprovedValueEstimator

def generate_random_hidden_states(batch_size, seq_len, hidden_size, device):
    """Generate random hidden states tensor"""
    return torch.randn(batch_size, seq_len, hidden_size, device=device)

def generate_structured_hidden_states(batch_size, seq_len, hidden_size, device, pattern_type="linear"):
    """
    Generate structured hidden states with recognizable patterns.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden dimension size
        device: Device to put tensor on
        pattern_type: Type of pattern to generate
        
    Returns:
        Tensor of structured hidden states
    """
    if pattern_type == "linear":
        # Linear pattern across sequence length
        base = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1).to(device)
        base = base.repeat(batch_size, 1, hidden_size)
        # Add some randomness but maintain pattern
        noise = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.2
        return base + noise
        
    elif pattern_type == "sinusoidal":
        # Sinusoidal pattern
        x = torch.linspace(0, 4*np.pi, seq_len).to(device)
        sin_base = torch.sin(x).unsqueeze(0).unsqueeze(-1)
        sin_base = sin_base.repeat(batch_size, 1, hidden_size)
        noise = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.2
        return sin_base + noise
        
    elif pattern_type == "step":
        # Step function pattern
        steps = torch.zeros(seq_len, device=device)
        steps[seq_len//3:2*seq_len//3] = 0.5
        steps[2*seq_len//3:] = 1.0
        base = steps.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, hidden_size)
        noise = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.2
        return base + noise
        
    else:  # random pattern with some structure
        # Random but with correlation between dimensions
        base = torch.randn(batch_size, seq_len, 1, device=device)
        base = base.repeat(1, 1, hidden_size)
        # Add dimension-specific patterns
        for d in range(hidden_size):
            if d % 4 == 0:  # Every 4th dimension gets a unique pattern
                pattern = torch.linspace(0, 1, seq_len).to(device)
                base[:, :, d] += pattern
        noise = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.1
        return base + noise

def test_structured_vs_random(value_estimator, args):
    """Test ability to distinguish structured vs random patterns"""
    print("=== Testing Structured vs Random Pattern Detection ===")
    
    device = args.device
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    seq_len = args.seq_len
    num_tests = args.test_samples
    
    random_values = []
    structured_values = []
    pattern_types = ["linear", "sinusoidal", "step", "random"]
    
    for i in tqdm(range(num_tests), desc="Testing patterns"):
        # For each test, try different patterns
        pattern = pattern_types[i % len(pattern_types)]
        
        # Generate hidden states
        random_states = generate_random_hidden_states(batch_size, seq_len, hidden_size, device)
        structured_states = generate_structured_hidden_states(batch_size, seq_len, hidden_size, device, pattern)
        
        # Get value estimates
        with torch.no_grad():
            random_value = value_estimator(random_states).mean().item()
            structured_value = value_estimator(structured_states).mean().item()
        
        random_values.append(random_value)
        structured_values.append(structured_value)
        
        print(f"Test {i+1} ({pattern} pattern)")
        print(f"  Random value: {random_value:.4f}")
        print(f"  Structured value: {structured_value:.4f}")
        print(f"  Difference: {structured_value - random_value:.4f}")
    
    avg_random = sum(random_values) / len(random_values)
    avg_structured = sum(structured_values) / len(structured_values)
    avg_diff = avg_structured - avg_random
    
    print("\nResults:")
    print(f"Average random value: {avg_random:.4f}")
    print(f"Average structured value: {avg_structured:.4f}")
    print(f"Average difference: {avg_diff:.4f}")
    
    if args.visualization:
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(range(num_tests), random_values, alpha=0.7, label='Random', color='red')
        plt.scatter(range(num_tests), structured_values, alpha=0.7, label='Structured', color='blue')
        plt.axhline(y=avg_random, color='red', linestyle='--', alpha=0.5, label=f'Avg Random: {avg_random:.3f}')
        plt.axhline(y=avg_structured, color='blue', linestyle='--', alpha=0.5, label=f'Avg Structured: {avg_structured:.3f}')
        plt.xlabel('Test Number')
        plt.ylabel('Value Estimate')
        plt.title('ImprovedValueEstimator: Structured vs Random Pattern Detection')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('value_estimator_patterns.png')
        print("Saved pattern comparison plot to value_estimator_patterns.png")

def test_convergence(value_estimator, args):
    """Test convergence detection capability"""
    print("\n=== Testing Convergence Detection ===")
    
    device = args.device
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    seq_len = args.seq_len
    
    # Reset estimator
    value_estimator.reset()
    
    # Create a sequence of improving hidden states
    steps = 20
    convergence_points = []
    values = []
    
    print("Testing with gradually improving hidden states...")
    
    for i in tqdm(range(steps), desc="Testing convergence"):
        # Generate hidden states that gradually improve
        progress = i / (steps - 1)  # 0 to 1
        
        # More structure as we progress
        random_component = 1.0 - min(1.0, progress * 1.5)
        structured_component = min(1.0, progress * 1.5)
        
        random_states = generate_random_hidden_states(batch_size, seq_len, hidden_size, device)
        structured_states = generate_structured_hidden_states(batch_size, seq_len, hidden_size, device, "linear")
        
        # Blend random and structured based on progress
        hidden_states = random_states * random_component + structured_states * structured_component
        
        # Check convergence
        with torch.no_grad():
            value = value_estimator(hidden_states).mean().item()
            values.append(value)
            converged = value_estimator.check_convergence(hidden_states)
        
        print(f"Step {i+1}: Value = {value:.4f}, Converged = {converged}")
        
        if converged:
            convergence_points.append(i)
    
    print(f"Convergence detected at steps: {convergence_points}")
    
    if args.visualization and values:
        plt.figure(figsize=(10, 6))
        plt.plot(range(steps), values, marker='o', label='Value')
        
        for point in convergence_points:
            plt.axvline(x=point, color='green', linestyle='--', alpha=0.7)
        
        plt.xlabel('Step')
        plt.ylabel('Value Estimate')
        plt.title('Convergence Detection Test')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('convergence_test.png')
        print("Saved convergence test plot to convergence_test.png")

def test_iteration_decision(value_estimator, args):
    """Test decisions about whether to continue iterations"""
    print("\n=== Testing Iteration Continuation Decisions ===")
    
    device = args.device
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    seq_len = args.seq_len
    min_iterations = args.min_iterations
    max_iterations = args.max_iterations
    
    # Create a sequence of improving hidden states
    steps = max_iterations * 2
    continue_decisions = []
    values = []
    
    # Reset estimator
    value_estimator.reset()
    
    print(f"Testing with parameters: min_iterations={min_iterations}, max_iterations={max_iterations}")
    
    for i in tqdm(range(steps), desc="Testing iteration decisions"):
        # Simulate gradual improvement that plateaus
        if i < steps // 2:
            # First half - improving
            progress = i / (steps // 2)
            improvement_factor = progress
        else:
            # Second half - plateauing with slight fluctuations
            progress = 1.0
            improvement_factor = 1.0 + 0.05 * np.sin(i)
        
        # Generate hidden states with structure proportional to improvement
        random_states = generate_random_hidden_states(batch_size, seq_len, hidden_size, device)
        structured_states = generate_structured_hidden_states(batch_size, seq_len, hidden_size, device, "sinusoidal")
        
        # Blend random and structured based on improvement factor
        blend_factor = min(1.0, improvement_factor)
        hidden_states = random_states * (1 - blend_factor) + structured_states * blend_factor
        
        # Check if iteration should continue
        with torch.no_grad():
            value = value_estimator(hidden_states).mean().item()
            values.append(value)
            should_continue = value_estimator.should_continue_iteration(
                hidden_states, i, min_iterations, max_iterations
            )
            continue_decisions.append(should_continue)
        
        print(f"Iteration {i+1}: Value = {value:.4f}, Continue = {should_continue}")
        
        if not should_continue:
            print(f"Stopping iteration at step {i+1}")
            break
    
    # Get value history
    value_history = value_estimator.get_value_history()
    change_history = value_estimator.get_change_history()
    
    print(f"Final value history: {value_history}")
    print(f"Value change history: {change_history}")
    
    if args.visualization:
        plt.figure(figsize=(12, 8))
        
        # Plot values
        plt.subplot(2, 1, 1)
        plt.plot(range(len(values)), values, marker='o', label='Value', color='blue')
        
        # Mark continue/stop decisions
        for i, decision in enumerate(continue_decisions):
            color = 'green' if decision else 'red'
            marker = 'o' if decision else 'x'
            plt.plot(i, values[i], marker=marker, color=color, markersize=10)
        
        # Mark min and max iteration boundaries
        plt.axvline(x=min_iterations-1, color='orange', linestyle='--', 
                   label=f'Min Iterations ({min_iterations})')
        plt.axvline(x=max_iterations-1, color='red', linestyle='--', 
                   label=f'Max Iterations ({max_iterations})')
        
        plt.xlabel('Iteration')
        plt.ylabel('Value Estimate')
        plt.title('Iteration Decision Test - Values')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot changes if we have them
        if change_history:
            plt.subplot(2, 1, 2)
            plt.plot(range(len(change_history)), change_history, marker='o', label='Value Change', color='purple')
            
            # Plot convergence threshold if available
            if hasattr(value_estimator, 'convergence_threshold'):
                plt.axhline(y=value_estimator.convergence_threshold, color='red', linestyle='--', 
                           label=f'Threshold: {value_estimator.convergence_threshold:.4f}')
            
            plt.xlabel('Iteration')
            plt.ylabel('Value Change')
            plt.title('Value Changes Between Iterations')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('iteration_decisions.png')
        print("Saved iteration decision plot to iteration_decisions.png")

def main():
    parser = argparse.ArgumentParser(description='Test Improved Value Estimator')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden state size')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--seq_len', type=int, default=16,
                        help='Sequence length')
    parser.add_argument('--test_samples', type=int, default=10,
                        help='Number of test samples')
    parser.add_argument('--min_iterations', type=int, default=2,
                        help='Minimum iterations for continuation test')
    parser.add_argument('--max_iterations', type=int, default=10,
                        help='Maximum iterations for continuation test')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu/cuda)')
    parser.add_argument('--structured_vs_random', action='store_true',
                        help='Run structured vs random test')
    parser.add_argument('--test_convergence', action='store_true',
                        help='Run convergence test')
    parser.add_argument('--test_iterations', action='store_true',
                        help='Run iteration decision test')
    parser.add_argument('--visualization', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device)
    
    # Create an instance of the improved value estimator
    print(f"Creating ImprovedValueEstimator with hidden_size={args.hidden_size}")
    value_estimator = ImprovedValueEstimator(args.hidden_size)
    value_estimator.to(device)
    
    # Run specified tests or all if none specified
    run_all = not (args.structured_vs_random or args.test_convergence or args.test_iterations)
    
    if args.structured_vs_random or run_all:
        test_structured_vs_random(value_estimator, args)
    
    if args.test_convergence or run_all:
        test_convergence(value_estimator, args)
    
    if args.test_iterations or run_all:
        test_iteration_decision(value_estimator, args)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()