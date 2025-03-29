"""
EdgeFormer Features Demo
========================

This script demonstrates the key features of EdgeFormer:
1. Language modeling capabilities
2. KV Cache management with offloading
3. Value-based recurrent depth processing

Usage:
    python examples/demo_edgeformer_features.py --prompt "EdgeFormer is" --max_length 100 --iterations 10
"""

import argparse
import os
import sys
import time
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.text_dataset import SimpleTokenizer, get_tokenizer
from src.utils.fixed_kv_cache_manager import KVCacheManager
from src.utils.value_estimator import ValueEstimator

def parse_arguments():
    parser = argparse.ArgumentParser(description="EdgeFormer Features Demo")
    parser.add_argument("--prompt", type=str, default="EdgeFormer is", help="Initial prompt for text generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--model_path", type=str, default="checkpoints/final_model.pt", help="Path to the model checkpoint")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.pt", help="Path to the vocabulary")
    parser.add_argument("--attention_type", type=str, default="mla", choices=["standard", "mla", "mla_window"], help="Attention mechanism to use")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for inference")
    parser.add_argument("--offload", action="store_true", help="Enable KV cache offloading")
    parser.add_argument("--recurrent_depth", type=int, default=5, help="Recurrent depth for value-based processing")
    parser.add_argument("--min_iterations", type=int, default=2, help="Minimum recurrent iterations")
    parser.add_argument("--max_iterations", type=int, default=10, help="Maximum recurrent iterations")
    parser.add_argument("--visualize", action="store_true", help="Visualize value estimation and memory usage")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    print(f"Loading model from {args.model_path}...")
    
    # Get device
    device = torch.device(args.device)
    
    # Load tokenizer
    try:
        tokenizer = torch.load(args.vocab_path)
        print(f"Loaded tokenizer with vocabulary size: {len(tokenizer)}")
    except FileNotFoundError:
        print(f"Tokenizer not found at {args.vocab_path}, creating a new one...")
        tokenizer = SimpleTokenizer()
        tokenizer.fit([args.prompt])
        print(f"Created basic tokenizer with vocabulary size: {len(tokenizer)}")
    
    # Create or load model
    try:
        model_state = torch.load(args.model_path, map_location=device)
        config = model_state.get("config", None)
        
        if config is None:
            print("Config not found in checkpoint, creating default config...")
            config = EdgeFormerConfig(
                vocab_size=len(tokenizer),
                hidden_size=128,
                attention_type=args.attention_type,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=512,
                max_position_embeddings=512
            )
        else:
            print("Using config from checkpoint")
            # Update attention type if specified
            if hasattr(config, 'attention_type'):
                config.attention_type = args.attention_type
        
        model = EdgeFormer(config)
        
        # Load model weights
        if "model_state_dict" in model_state:
            model.load_state_dict(model_state["model_state_dict"])
        else:
            model.load_state_dict(model_state)
            
        print("Model loaded successfully")
    except FileNotFoundError:
        print(f"Model not found at {args.model_path}, creating a new one...")
        config = EdgeFormerConfig(
            vocab_size=len(tokenizer),
            hidden_size=128,
            attention_type=args.attention_type,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=512
        )
        model = EdgeFormer(config)
        print("Created new model with default parameters")
    
    # Move model to device
    model.to(device)
    model.eval()
    
    return model, tokenizer, device, config

def initialize_kv_cache_manager(model, config, device, enable_offload=False):
    print("Initializing KV Cache Manager...")
    
    # Create KV Cache Manager
    kv_cache_manager = KVCacheManager(
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        max_seq_length=1024,  # This will grow automatically as needed
        device=device,
        enable_offload=enable_offload
    )
    
    # Set model to use the KV cache manager
    model.set_kv_cache_manager(kv_cache_manager)
    
    print(f"KV Cache Manager initialized (offloading: {enable_offload})")
    return kv_cache_manager

def initialize_value_estimator(model, config, device):
    print("Initializing Value Estimator...")
    
    # Create Value Estimator
    value_estimator = ValueEstimator(config.hidden_size, config)
    value_estimator.to(device)
    
    # Set model to use the value estimator
    model.set_value_estimator(value_estimator)
    
    print("Value Estimator initialized")
    return value_estimator

def encode_prompt(tokenizer, prompt, device):
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    return input_ids

def generate_with_value_based_recurrence(model, tokenizer, input_ids, device, args):
    print(f"Generating text with value-based recurrent processing (max depth: {args.max_iterations})...")
    
    # Track memory usage and value history for visualization
    memory_usage = []
    value_histories = []
    avg_iterations = []
    tokens_generated = []
    generation_times = []
    
    # Initialize output with input
    output_ids = input_ids.clone()
    
    # Get value estimator
    value_estimator = model.get_value_estimator()
    
    # Reset value estimator history
    value_estimator.reset()
    
    # Store the starting time
    start_time = time.time()
    
    # Generate tokens one by one
    for i in range(args.max_length):
        # Measure memory before generation
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # Store token generation start time
        token_start_time = time.time()
        
        # Forward pass with recurrent processing
        with torch.no_grad():
            # Get last token as input for the next prediction
            next_token_input = output_ids[:, -1].unsqueeze(-1)
            
            # Initialize hidden states
            hidden_states = None
            
            # Recurrent processing with value estimation
            iteration = 0
            while iteration < args.max_iterations:
                # Forward pass to get hidden states (not logits yet)
                logits, hidden_states = model.forward_with_hidden_states(
                    next_token_input if iteration == 0 else None,
                    use_kv_cache=True,
                    hidden_states=hidden_states,
                    return_hidden_states=True
                )
                
                # Check if we should continue iterating
                iteration += 1
                if iteration >= args.min_iterations:
                    if value_estimator.check_convergence(hidden_states) or iteration >= args.max_iterations:
                        break
            
            # Record the number of iterations used
            avg_iterations.append(iteration)
            
            # Get output token
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            output_ids = torch.cat([output_ids, next_token], dim=1)
        
        # Calculate token generation time
        token_time = time.time() - token_start_time
        generation_times.append(token_time)
        
        # Record memory usage
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_usage.append(mem_after)
        
        # Store value history for this token
        value_histories.append(value_estimator.get_value_history().copy())
        
        # Reset value estimator for next token
        value_estimator.reset()
        
        # Track generated token
        tokens_generated.append(tokenizer.decode([next_token.item()]))
        
        # Print progress
        if (i+1) % 10 == 0:
            print(f"Generated {i+1} tokens...")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Decode the generated text
    output_text = tokenizer.decode(output_ids[0].tolist())
    
    # Prepare statistics
    stats = {
        "output_text": output_text,
        "total_time": total_time,
        "tokens_per_second": args.max_length / total_time,
        "avg_iterations": sum(avg_iterations) / len(avg_iterations),
        "memory_usage": memory_usage,
        "value_histories": value_histories,
        "tokens_generated": tokens_generated,
        "generation_times": generation_times
    }
    
    return output_text, stats

def visualize_results(stats, args):
    print("Visualizing results...")
    
    # Create a figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Memory usage
    if stats["memory_usage"]:
        axs[0].plot(stats["memory_usage"], 'b-')
        axs[0].set_title('Memory Usage During Generation')
        axs[0].set_xlabel('Token Position')
        axs[0].set_ylabel('Memory (MB)')
        axs[0].grid(True)
    else:
        axs[0].text(0.5, 0.5, 'Memory tracking not available (CPU mode)', 
                 horizontalalignment='center', verticalalignment='center')
    
    # Plot 2: Iteration counts
    iteration_counts = [len(vh) for vh in stats["value_histories"]]
    axs[1].bar(range(len(iteration_counts)), iteration_counts)
    axs[1].set_title('Iterations Per Token')
    axs[1].set_xlabel('Token Position')
    axs[1].set_ylabel('Iteration Count')
    axs[1].grid(True)
    
    # Plot 3: Value convergence for selected tokens
    # Choose a few tokens to visualize (evenly spaced)
    num_tokens = len(stats["value_histories"])
    tokens_to_show = min(5, num_tokens)
    token_indices = [int(i * (num_tokens-1) / (tokens_to_show-1)) for i in range(tokens_to_show)] if tokens_to_show > 1 else [0]
    
    for idx in token_indices:
        if idx < len(stats["value_histories"]):
            values = stats["value_histories"][idx]
            token = stats["tokens_generated"][idx]
            axs[2].plot(values, marker='o', label=f'Token "{token}" (pos {idx})')
    
    axs[2].set_title('Value Convergence During Recurrent Processing')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Value Estimate')
    axs[2].grid(True)
    axs[2].legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('value_recurrent_analysis.png')
    print("Visualization saved to 'value_recurrent_analysis.png'")
    
    # Display additional statistics
    print(f"\nStatistics:")
    print(f"- Total generation time: {stats['total_time']:.2f} seconds")
    print(f"- Tokens per second: {stats['tokens_per_second']:.2f}")
    print(f"- Average iterations per token: {stats['avg_iterations']:.2f}")
    
    # Display token-specific information
    print("\nToken generation details:")
    for i, (token, time) in enumerate(zip(stats["tokens_generated"][:10], stats["generation_times"][:10])):
        iterations = len(stats["value_histories"][i])
        print(f"Token {i}: '{token}' - {iterations} iterations, {time*1000:.2f}ms")
    print("...")

def main():
    args = parse_arguments()
    
    # Load model and tokenizer
    model, tokenizer, device, config = load_model_and_tokenizer(args)
    
    # Initialize KV Cache Manager
    kv_cache_manager = initialize_kv_cache_manager(model, config, device, args.offload)
    
    # Initialize Value Estimator
    value_estimator = initialize_value_estimator(model, config, device)
    
    # Encode the prompt
    input_ids = encode_prompt(tokenizer, args.prompt, device)
    
    # Generate text with value-based recurrence
    output_text, stats = generate_with_value_based_recurrence(
        model, tokenizer, input_ids, device, args
    )
    
    # Print the generated text
    print("\n" + "="*50)
    print("Generated Text:")
    print("="*50)
    print(output_text)
    print("="*50)
    
    # Visualize results if requested
    if args.visualize:
        visualize_results(stats, args)
    
    # Report KV Cache statistics
    if kv_cache_manager.enable_offload:
        print("\nKV Cache Statistics:")
        print(f"- Current sequence length: {kv_cache_manager.current_seq_length}")
        print(f"- Offload operations: {kv_cache_manager.offload_count}")
        print(f"- CPU RAM usage: {kv_cache_manager.get_cpu_memory_usage() / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()