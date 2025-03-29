import argparse
import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.value_estimator import ImprovedValueEstimator
from src.utils.htps_budget_manager import HTPSBudgetManager
from src.utils.kv_cache_manager import KVCacheManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('benchmark')

# Define test tasks
TASKS = {
    "simple": "EdgeFormer is a high-performance Transformer implementation",
    "math": "Solve this math problem step by step: If a rectangle has a length of 8 meters and a width of 5 meters, what is its area and perimeter?",
    "reasoning": "Explain the concept of quantum entanglement in simple terms that a high school student could understand.",
    "creative": "Write a short story about a robot that discovers it has emotions."
}

# Define feature combinations to test
FEATURE_COMBINATIONS = [
    {"name": "Base", "kv_cache": False, "recurrent": False, "budget": False},
    {"name": "KV Cache", "kv_cache": True, "recurrent": False, "budget": False},
    {"name": "Recurrent", "kv_cache": False, "recurrent": True, "budget": False},
    {"name": "Budget", "kv_cache": False, "recurrent": False, "budget": True},
    {"name": "KV + Recurrent", "kv_cache": True, "recurrent": True, "budget": False},
    {"name": "KV + Budget", "kv_cache": True, "recurrent": False, "budget": True},
    {"name": "Recurrent + Budget", "kv_cache": False, "recurrent": True, "budget": True},
    {"name": "All Features", "kv_cache": True, "recurrent": True, "budget": True}
]

# Define sequence lengths to test
SEQUENCE_LENGTHS = [128, 256, 512, 1024]

def track_memory_usage(device):
    """Track memory usage for the given device."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 * 1024)  # MB
    else:
        # For CPU, use psutil if available
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # MB
        except ImportError:
            return 0  # Fallback if psutil is not available

def run_benchmark(args):
    """Run comprehensive benchmarks for EdgeFormer features."""
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Initialize results dictionary
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": args.device,
        "tasks": {},
        "feature_combinations": FEATURE_COMBINATIONS,
        "sequence_lengths": SEQUENCE_LENGTHS
    }
    
    # Test each task
    for task_name, prompt in TASKS.items():
        logger.info(f"Benchmarking task: {task_name}")
        results["tasks"][task_name] = {}
        
        # Test each feature combination
        for feature_idx, features in enumerate(FEATURE_COMBINATIONS):
            feature_name = features["name"]
            logger.info(f"  Testing feature combination: {feature_name}")
            results["tasks"][task_name][feature_name] = {}
            
            # Test each sequence length
            for seq_length in SEQUENCE_LENGTHS:
                if seq_length < len(prompt):
                    # Skip if prompt is longer than sequence length
                    continue
                
                logger.info(f"    Sequence length: {seq_length}")
                
                # Initialize model with the appropriate configuration
                config = EdgeFormerConfig(
                    vocab_size=256,  # Character-level tokenization
                    hidden_size=256,
                    num_hidden_layers=6,
                    num_attention_heads=8,
                    intermediate_size=1024,
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    max_position_embeddings=2048,
                    type_vocab_size=2,
                    initializer_range=0.02,
                    layer_norm_eps=1e-12,
                    attention_type="mla",  # Using MLA attention
                    # Enable features based on combination
                    enable_budget_forcing=features["budget"],
                    max_budget_tokens=args.max_budget_tokens,
                    max_thinking_extensions=args.extensions,
                    extension_token="Wait",
                    enable_recurrent_depth=features["recurrent"],
                    max_iterations=args.max_iterations,
                    convergence_threshold=args.convergence_threshold
                )
                
                model = EdgeFormer(config)
                model.to(args.device)
                
                # Initialize KV cache manager if needed
                kv_cache_manager = None
                if features["kv_cache"]:
                    kv_cache_manager = KVCacheManager(
                        max_batch_size=1,
                        max_seq_length=seq_length,
                        num_layers=config.num_hidden_layers,
                        num_heads=config.num_attention_heads,
                        head_dim=config.hidden_size // config.num_attention_heads,
                        device=args.device,
                        enable_offload=True,
                        max_gpu_cache_size=args.offload_threshold
                    )
                    model.kv_cache_manager = kv_cache_manager
                
                # Initialize value estimator if needed
                value_estimator = None
                if features["recurrent"]:
                    value_estimator = ImprovedValueEstimator(config.hidden_size, config)
                    value_estimator.to(args.device)
                
                # Initialize budget manager if needed
                budget_manager = None
                if features["budget"]:
                    budget_manager = HTPSBudgetManager(
                        budget_tokens=args.max_budget_tokens,
                        max_thinking_extensions=args.extensions,
                        extension_token="Wait",
                        confidence_threshold=0.9,
                        complexity_threshold=0.6
                    )
                
                # Tokenize input
                tokens = [ord(c) % config.vocab_size for c in prompt]
                input_ids = torch.tensor([tokens], dtype=torch.long, device=args.device)
                
                # Track memory before generation
                start_memory = track_memory_usage(args.device)
                
                # Measure generation time
                start_time = time.time()
                
                # Generate text based on feature combination
                if features["recurrent"] and args.unified_demo:
                    # Use unified demo for recurrent processing
                    # This would be a wrapper around the process implemented in unified_features_demo.py
                    generated_text, stats = generate_with_unified_demo(
                        model=model,
                        input_ids=input_ids,
                        max_length=seq_length,
                        use_kv_cache=features["kv_cache"],
                        use_recurrent=features["recurrent"],
                        use_budget=features["budget"],
                        min_iterations=args.min_iterations,
                        max_iterations=args.max_iterations,
                        convergence_threshold=args.convergence_threshold,
                        value_estimator=value_estimator,
                        budget_manager=budget_manager,
                        device=args.device
                    )
                    iteration_stats = stats.get("iterations", [])
                    avg_iterations = sum(iteration_stats) / len(iteration_stats) if iteration_stats else 0
                else:
                    # Standard generation
                    generated_ids = model.generate(
                        input_ids,
                        max_length=seq_length,
                        do_sample=True,
                        temperature=0.8,
                        top_k=40,
                        top_p=0.9,
                        budget_manager=budget_manager,
                        task_complexity=0.7  # Medium complexity
                    )
                    
                    # Convert to text
                    generated_text = ""
                    for token_id in generated_ids[0]:
                        generated_text += chr(token_id.item() % 128)
                    
                    avg_iterations = 0  # Not applicable for non-recurrent generation
                
                # Calculate generation time
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Calculate tokens per second
                tokens_generated = len(generated_text) - len(prompt)
                tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
                
                # Track memory after generation
                end_memory = track_memory_usage(args.device)
                memory_used = end_memory - start_memory
                
                # Store results
                results["tasks"][task_name][feature_name][seq_length] = {
                    "generation_time_seconds": generation_time,
                    "tokens_generated": tokens_generated,
                    "tokens_per_second": tokens_per_second,
                    "memory_used_mb": memory_used,
                    "avg_iterations": avg_iterations
                }
                
                logger.info(f"      Generation time: {generation_time:.2f}s")
                logger.info(f"      Tokens generated: {tokens_generated}")
                logger.info(f"      Tokens/second: {tokens_per_second:.2f}")
                logger.info(f"      Memory used: {memory_used:.2f} MB")
                if features["recurrent"]:
                    logger.info(f"      Avg iterations: {avg_iterations:.2f}")
                
                # Clean up to free memory
                del model
                if features["kv_cache"]:
                    del kv_cache_manager
                if features["recurrent"]:
                    del value_estimator
                if features["budget"]:
                    del budget_manager
                torch.cuda.empty_cache()
    
    # Save results to JSON
    results_file = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")
    
    # Generate visualizations
    generate_visualizations(results, args.output_dir)
    
    return results

def generate_with_unified_demo(model, input_ids, max_length, use_kv_cache, use_recurrent, use_budget, min_iterations, max_iterations, convergence_threshold, value_estimator, budget_manager, device):
    """
    Implementation of recurrent generation similar to unified_features_demo.py.
    This is a simplified version for benchmarking purposes.
    """
    # Initialize tracking variables
    generated_ids = input_ids.clone()
    token_iterations = []
    value_histories = []
    
    # Generate tokens one at a time
    for i in range(max_length - input_ids.size(1)):
        # Forward pass to get logits and hidden states
        with torch.no_grad():
            # Get the last token's hidden states
            outputs = model.forward(generated_ids)
            logits = outputs["logits"]
            
            # Get initial hidden state value
            hidden_states = outputs["hidden_states"]
            if hidden_states is None:
                # If hidden_states isn't returned, use forward_with_hidden_states
                logits, all_hidden_states = model.forward_with_hidden_states(generated_ids)
                last_hidden = all_hidden_states[-1][:, -1:, :].clone()
            else:
                last_hidden = hidden_states[-1][:, -1:, :].clone()
            
            current_hidden = last_hidden.clone()
            
            # Only perform recurrent processing if enabled
            if use_recurrent and value_estimator is not None:
                # Get initial value
                initial_value = value_estimator(current_hidden).item()
                value_history = [initial_value]
                
                # Recurrent processing loop
                iterations = 0
                prev_value = initial_value
                continue_iterating = True
                change = float('inf')
                
                # Run at least min_iterations iterations
                while continue_iterating and iterations < max_iterations:
                    # Pass through the last transformer layer again
                    current_hidden = model.layers[-1].forward(current_hidden)[0]
                    
                    # Calculate value
                    value = value_estimator(current_hidden).item()
                    value_history.append(value)
                    
                    # Check convergence after min_iterations
                    iterations += 1
                    if iterations >= min_iterations:
                        change = abs(value - prev_value)
                        if change < convergence_threshold:
                            continue_iterating = False
                    
                    prev_value = value
                
                # Record data for statistics
                token_iterations.append(iterations)
                value_histories.append(value_history)
                
                # Sample next token from improved hidden state
                improved_logits = model.lm_head(current_hidden)
                next_token_logits = improved_logits
            else:
                # Just get the last token's logits
                next_token_logits = logits[:, -1, :]
                
            # Apply budget forcing if enabled
            if use_budget and budget_manager is not None:
                input_ids, continue_gen = budget_manager.enforce_budget(
                    model, generated_ids, next_token_logits, 0.7
                )
                
                if not continue_gen:
                    break
            
            # Apply temperature and sampling
            next_token_logits = next_token_logits / 0.8  # Temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs[0], 1)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token.view(1, 1)], dim=1)
    
    # Convert to text
    generated_text = ""
    for token_id in generated_ids[0]:
        generated_text += chr(token_id.item() % 128)
    
    # Gather statistics
    stats = {
        "iterations": token_iterations,
        "avg_iterations": sum(token_iterations) / len(token_iterations) if token_iterations else 0,
        "max_iterations": max(token_iterations) if token_iterations else 0,
        "value_histories": value_histories
    }
    
    return generated_text, stats

def generate_visualizations(results, output_dir):
    """Generate visualizations for benchmark results."""
    
    # Plot tokens per second for each feature combination across sequence lengths
    plt.figure(figsize=(12, 8))
    
    for task_name in results["tasks"]:
        plt.figure(figsize=(12, 8))
        
        # Create a bar for each feature combination
        feature_names = [fc["name"] for fc in results["feature_combinations"]]
        sequence_lengths = results["sequence_lengths"]
        
        # Set up the colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
        
        # Plot tokens per second for each sequence length
        for seq_idx, seq_length in enumerate(sequence_lengths):
            values = []
            labels = []
            
            for feat_idx, feat_name in enumerate(feature_names):
                if str(seq_length) in results["tasks"][task_name].get(feat_name, {}):
                    values.append(results["tasks"][task_name][feat_name][str(seq_length)]["tokens_per_second"])
                    labels.append(feat_name)
            
            x = np.arange(len(labels))
            width = 0.8 / len(sequence_lengths)
            offset = width * seq_idx - 0.4 + width/2
            
            plt.bar(x + offset, values, width, label=f'{seq_length} tokens', color=colors[seq_idx])
        
        plt.xlabel('Feature Combination')
        plt.ylabel('Tokens per Second')
        plt.title(f'Generation Speed by Feature Combination - {task_name.capitalize()} Task')
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'tokens_per_second_{task_name}.png'))
        plt.close()
    
    # Plot memory usage
    for task_name in results["tasks"]:
        plt.figure(figsize=(12, 8))
        
        feature_names = [fc["name"] for fc in results["feature_combinations"]]
        sequence_lengths = results["sequence_lengths"]
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
        
        for feat_idx, feat_name in enumerate(feature_names):
            memory_values = []
            seq_labels = []
            
            for seq_length in sequence_lengths:
                if str(seq_length) in results["tasks"][task_name].get(feat_name, {}):
                    memory_values.append(results["tasks"][task_name][feat_name][str(seq_length)]["memory_used_mb"])
                    seq_labels.append(seq_length)
            
            plt.plot(seq_labels, memory_values, 'o-', label=feat_name, color=colors[feat_idx])
        
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory Usage (MB)')
        plt.title(f'Memory Usage by Feature Combination - {task_name.capitalize()} Task')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'memory_usage_{task_name}.png'))
        plt.close()
    
    # If there are recurrent processing results, plot average iterations
    for task_name in results["tasks"]:
        has_recurrent = False
        for feat_name in feature_names:
            if "Recurrent" in feat_name:
                for seq_length in sequence_lengths:
                    if str(seq_length) in results["tasks"][task_name].get(feat_name, {}) and \
                       "avg_iterations" in results["tasks"][task_name][feat_name][str(seq_length)]:
                        has_recurrent = True
                        break
        
        if has_recurrent:
            plt.figure(figsize=(12, 8))
            
            for feat_idx, feat_name in enumerate(feature_names):
                if "Recurrent" in feat_name:
                    iter_values = []
                    seq_labels = []
                    
                    for seq_length in sequence_lengths:
                        if str(seq_length) in results["tasks"][task_name].get(feat_name, {}) and \
                           "avg_iterations" in results["tasks"][task_name][feat_name][str(seq_length)]:
                            iter_values.append(results["tasks"][task_name][feat_name][str(seq_length)]["avg_iterations"])
                            seq_labels.append(seq_length)
                    
                    if iter_values:
                        plt.plot(seq_labels, iter_values, 'o-', label=feat_name)
            
            plt.xlabel('Sequence Length')
            plt.ylabel('Average Iterations per Token')
            plt.title(f'Recurrent Processing Iterations - {task_name.capitalize()} Task')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'avg_iterations_{task_name}.png'))
            plt.close()
    
    # Generate a summary heatmap
    for task_name in results["tasks"]:
        feature_names = [fc["name"] for fc in results["feature_combinations"]]
        valid_features = []
        valid_seq_lengths = []
        
        # Find valid feature combinations and sequence lengths that have data
        for feat_name in feature_names:
            if feat_name in results["tasks"][task_name]:
                valid_features.append(feat_name)
                for seq_length in sequence_lengths:
                    if str(seq_length) in results["tasks"][task_name][feat_name] and \
                       seq_length not in valid_seq_lengths:
                        valid_seq_lengths.append(seq_length)
        
        valid_seq_lengths.sort()
        
        if valid_features and valid_seq_lengths:
            # Create heatmap data for tokens per second
            heatmap_data = np.zeros((len(valid_features), len(valid_seq_lengths)))
            
            for i, feat_name in enumerate(valid_features):
                for j, seq_length in enumerate(valid_seq_lengths):
                    if str(seq_length) in results["tasks"][task_name][feat_name]:
                        heatmap_data[i, j] = results["tasks"][task_name][feat_name][str(seq_length)]["tokens_per_second"]
            
            plt.figure(figsize=(10, 8))
            plt.imshow(heatmap_data, cmap='viridis')
            plt.colorbar(label='Tokens per Second')
            plt.xlabel('Sequence Length')
            plt.ylabel('Feature Combination')
            plt.title(f'Performance Heatmap - {task_name.capitalize()} Task')
            plt.xticks(np.arange(len(valid_seq_lengths)), valid_seq_lengths)
            plt.yticks(np.arange(len(valid_features)), valid_features)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'performance_heatmap_{task_name}.png'))
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='EdgeFormer Features Benchmark')
    
    # Basic settings
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cpu|cuda)')
    parser.add_argument('--output_dir', type=str, default=f'benchmark_results_{datetime.now().strftime("%Y%m%d-%H%M%S")}', 
                        help='Directory to save results')
    
    # Feature settings
    parser.add_argument('--offload_threshold', type=int, default=1024, help='Token threshold for offloading to RAM')
    parser.add_argument('--min_iterations', type=int, default=2, help='Minimum recurrent iterations')
    parser.add_argument('--max_iterations', type=int, default=10, help='Maximum recurrent iterations')
    parser.add_argument('--convergence_threshold', type=float, default=0.005, help='Convergence threshold')
    parser.add_argument('--max_budget_tokens', type=int, default=2048, help='Maximum budget tokens')
    parser.add_argument('--extensions', type=int, default=2, help='Maximum thinking extensions')
    
    # Special flags
    parser.add_argument('--unified_demo', action='store_true', help='Use unified demo implementation for recurrent processing')
    parser.add_argument('--quick_test', action='store_true', help='Run a quick test with fewer combinations')
    
    args = parser.parse_args()
    
    # Adjust parameters for quick test
    if args.quick_test:
        global FEATURE_COMBINATIONS, SEQUENCE_LENGTHS, TASKS
        FEATURE_COMBINATIONS = [
            {"name": "Base", "kv_cache": False, "recurrent": False, "budget": False},
            {"name": "All Features", "kv_cache": True, "recurrent": True, "budget": True}
        ]
        SEQUENCE_LENGTHS = [128, 256]
        TASKS = {"simple": TASKS["simple"]}
    
    # Run benchmarks
    run_benchmark(args)

if __name__ == "__main__":
    main()