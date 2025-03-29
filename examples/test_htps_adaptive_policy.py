import argparse
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.htps_adaptive_policy import HTPSAdaptivePolicy

def test_task_detection(policy):
    """Test task type detection with various prompts"""
    test_prompts = [
        "What is the capital of France?",
        "Solve this math problem: 5 + 7 * 3 =",
        "Explain the causes of World War II",
        "Write a short story about a robot that dreams",
        "List the top 5 programming languages"
    ]
    
    print("=== Task Detection Test ===")
    for prompt in test_prompts:
        task_type = policy.detect_task_type(prompt)
        print(f"Prompt: '{prompt}'")
        print(f"Detected task: {task_type}")
        min_iter, max_iter, threshold = policy.get_iteration_params(prompt)
        print(f"Parameters: min_iter={min_iter}, max_iter={max_iter}, threshold={threshold:.4f}\n")

def test_path_selection(policy):
    """Test computation path selection logic"""
    print("=== Path Selection Test ===")
    
    # Test cases with different value histories and token info
    test_cases = [
        {
            "name": "Improving values",
            "value_history": [0.5, 0.6, 0.7],
            "token_info": {"id": 1000, "position": 5},
            "paths": ["standard", "deep", "focused"]
        },
        {
            "name": "Declining values",
            "value_history": [0.8, 0.7, 0.6],
            "token_info": {"id": 2000, "position": 15},
            "paths": ["standard", "deep", "focused"]
        },
        {
            "name": "Stable values",
            "value_history": [0.7, 0.7, 0.7],
            "token_info": {"id": 3000, "position": 25},
            "paths": ["standard", "deep", "focused"]
        }
    ]
    
    for case in test_cases:
        print(f"\nCase: {case['name']}")
        print(f"Value history: {case['value_history']}")
        
        # Select path multiple times to see if exploration affects choices
        for i in range(3):
            path = policy.select_computation_path(
                case["value_history"], 
                case["token_info"], 
                case["paths"]
            )
            print(f"  Selection {i+1}: {path}")
            
            # Update with simulated performance
            if "Improving" in case["name"]:
                # Good performance
                policy.update_path_performance(path, 0.8)
            elif "Declining" in case["name"]:
                # Poor performance
                policy.update_path_performance(path, 0.4)
            else:
                # Average performance
                policy.update_path_performance(path, 0.6)

def test_dynamic_iterations(policy):
    """Test dynamic iteration count calculations"""
    print("=== Dynamic Iterations Test ===")
    
    # Test cases with different token positions and context values
    test_cases = [
        {"token_id": 1000, "position": 5, "task": "math", "context_value": 0.2},
        {"token_id": 1000, "position": 50, "task": "math", "context_value": 0.7},
        {"token_id": 2000, "position": 3, "task": "reasoning", "context_value": 0.5},
        {"token_id": 2000, "position": 120, "task": "reasoning", "context_value": 0.3},
        {"token_id": 3000, "position": 10, "task": "simple", "context_value": 0.9},
        {"token_id": 3000, "position": 80, "task": "simple", "context_value": 0.4}
    ]
    
    for case in test_cases:
        iterations = policy.get_dynamic_iterations(
            case["token_id"], 
            case["position"], 
            case["task"], 
            case["context_value"]
        )
        print(f"Token ID: {case['token_id']}, Position: {case['position']}, "
              f"Task: {case['task']}, Context value: {case['context_value']:.2f}")
        print(f"  Dynamic max iterations: {iterations}")
        
        # Simulate recording iterations
        policy.record_token_iterations(
            case["token_id"], 
            case["position"], 
            iterations // 2,  # Simulated actual iterations used
            case["task"]
        )

def test_task_strategies(policy):
    """Test task-specific strategies"""
    print("=== Task Strategy Test ===")
    
    for task_type in ["simple", "retrieval", "reasoning", "math", "creative"]:
        strategy = policy.get_task_strategy(task_type)
        print(f"\nStrategy for {task_type} tasks:")
        for key, value in strategy.items():
            print(f"  {key}: {value}")

def plot_iteration_stats(policy):
    """Plot iteration statistics"""
    stats = policy.get_stats()
    
    # Check if we have enough data to plot
    if stats["total_tokens_processed"] < 5:
        print("Not enough data collected for meaningful plots")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Task statistics
    task_stats = stats["task_stats"]
    tasks = list(task_stats.keys())
    avg_iters = [task_stats[task]["avg_iterations"] for task in tasks]
    
    ax1.bar(tasks, avg_iters)
    ax1.set_title('Average Iterations by Task Type')
    ax1.set_ylabel('Average Iterations')
    ax1.set_ylim(0, max(avg_iters) * 1.2)
    
    # Plot 2: Position statistics
    position_stats = stats["position_stats"]
    positions = sorted([int(pos) for pos in position_stats.keys()])
    avg_pos_iters = [position_stats[pos]["avg_iterations"] for pos in positions]
    
    ax2.plot(positions, avg_pos_iters, marker='o')
    ax2.set_title('Average Iterations by Token Position')
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Average Iterations')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('htps_policy_stats.png')
    print("Saved statistics plot to htps_policy_stats.png")

def simulate_sequence_generation(policy, args):
    """Simulate sequence generation and adaptive iteration"""
    print("=== Sequence Generation Simulation ===")
    
    # Simulate sequence of tokens with different positions
    sequence_length = args.sequence_length
    task_type = args.task_type
    
    print(f"Simulating {sequence_length} token sequence for task type: {task_type}")
    
    # Generate random token IDs and positions
    np.random.seed(42)  # For reproducibility
    token_ids = np.random.randint(1000, 10000, size=sequence_length)
    
    # Simulate context values (quality of generation so far)
    # Start lower and improve as sequence progresses
    base_context_values = np.linspace(0.3, 0.9, sequence_length)
    # Add some noise
    noise = np.random.normal(0, 0.1, sequence_length)
    context_values = np.clip(base_context_values + noise, 0.1, 0.95)
    
    # Track iterations used
    iterations_used = []
    
    # Process each token
    for i in range(sequence_length):
        token_id = int(token_ids[i])
        position = i
        context_value = float(context_values[i])
        
        # Get dynamic max iterations
        max_iter = policy.get_dynamic_iterations(token_id, position, task_type, context_value)
        
        # Simulate actual iterations used (random fraction of max)
        actual_iterations = int(max_iter * np.random.uniform(0.5, 1.0))
        iterations_used.append(actual_iterations)
        
        # Record iterations for this token
        policy.record_token_iterations(token_id, position, actual_iterations, task_type)
    
    # Plot iterations used
    plt.figure(figsize=(10, 6))
    plt.plot(range(sequence_length), iterations_used, marker='o', alpha=0.5)
    plt.title(f'Iterations Used During Sequence Generation ({task_type} task)')
    plt.xlabel('Token Position')
    plt.ylabel('Iterations Used')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'sequence_iterations_{task_type}.png')
    print(f"Saved sequence simulation plot to sequence_iterations_{task_type}.png")
    
    # Get and display statistics
    stats = policy.get_stats()
    print("\nSimulation Statistics:")
    print(f"Total tokens processed: {stats['total_tokens_processed']}")
    print(f"Total iterations: {stats['total_iterations']}")
    print(f"Average iterations per token: {stats['avg_iterations_per_token']:.2f}")
    
    # Plot overall statistics
    plot_iteration_stats(policy)

def main():
    parser = argparse.ArgumentParser(description='Test HTPS Adaptive Policy')
    parser.add_argument('--sequence_length', type=int, default=128,
                        help='Sequence length for simulation')
    parser.add_argument('--min_iterations', type=int, default=2,
                        help='Minimum number of iterations')
    parser.add_argument('--max_iterations', type=int, default=20,
                        help='Maximum number of iterations')
    parser.add_argument('--task_type', type=str, default='reasoning',
                        choices=['simple', 'retrieval', 'reasoning', 'math', 'creative'],
                        help='Task type for simulation')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu/cuda)')
    parser.add_argument('--visualization', action='store_true',
                        help='Generate additional visualizations')
    
    args = parser.parse_args()
    
    # Create policy
    policy = HTPSAdaptivePolicy()
    
    # Run tests
    test_task_detection(policy)
    print("\n" + "="*50 + "\n")
    
    test_path_selection(policy)
    print("\n" + "="*50 + "\n")
    
    test_dynamic_iterations(policy)
    print("\n" + "="*50 + "\n")
    
    test_task_strategies(policy)
    print("\n" + "="*50 + "\n")
    
    simulate_sequence_generation(policy, args)

if __name__ == "__main__":
    main()