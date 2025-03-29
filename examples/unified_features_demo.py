import argparse
import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

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
logger = logging.getLogger('unified_demo')

def main():
    parser = argparse.ArgumentParser(description='EdgeFormer Unified Features Demo')
    parser.add_argument('--prompt', type=str, default="EdgeFormer is", help='Input prompt for generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cpu|cuda)')
    
    # Feature flags
    parser.add_argument('--use_kv_cache', action='store_true', help='Enable KV cache management')
    parser.add_argument('--use_recurrent', action='store_true', help='Enable value-based recurrent processing')
    parser.add_argument('--use_budget', action='store_true', help='Enable HyperTree budget forcing')
    
    # KV cache parameters
    parser.add_argument('--offload_threshold', type=int, default=1024, help='Token threshold for offloading to RAM')
    
    # Recurrent processing parameters
    parser.add_argument('--min_iterations', type=int, default=2, help='Minimum recurrent iterations')
    parser.add_argument('--max_iterations', type=int, default=10, help='Maximum recurrent iterations')
    parser.add_argument('--convergence_threshold', type=float, default=0.005, help='Convergence threshold')
    
    # Budget forcing parameters
    parser.add_argument('--max_budget_tokens', type=int, default=2048, help='Maximum budget tokens')
    parser.add_argument('--extension_token', type=str, default="Wait", help='Token for extending thinking')
    parser.add_argument('--extensions', type=int, default=2, help='Maximum thinking extensions')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Initialize model
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
        # Enable features based on arguments
        enable_budget_forcing=args.use_budget,
        max_budget_tokens=args.max_budget_tokens,
        max_thinking_extensions=args.extensions,
        extension_token=args.extension_token,
        enable_recurrent_depth=args.use_recurrent,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold
    )
    
    logger.info(f"Initializing EdgeFormer with features: KV Cache: {args.use_kv_cache}, "
                f"Recurrent Processing: {args.use_recurrent}, Budget Forcing: {args.use_budget}")
    
    model = EdgeFormer(config)
    model.to(args.device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize KV cache manager if requested
    if args.use_kv_cache:
        kv_cache_manager = KVCacheManager(
            max_batch_size=1,
            max_seq_length=1024,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            device=args.device,
            enable_offload=True,
            offload_threshold=args.offload_threshold
        )
        model.kv_cache_manager = kv_cache_manager
        logger.info(f"KV Cache Manager initialized with offload threshold: {args.offload_threshold}")
    
    # Initialize value estimator if using recurrent processing
    value_estimator = None
    if args.use_recurrent:
        value_estimator = ImprovedValueEstimator(config.hidden_size, config)
        value_estimator.to(args.device)
        logger.info(f"Value Estimator initialized with {sum(p.numel() for p in value_estimator.parameters())} parameters")
    
    # Initialize budget manager if using budget forcing
    budget_manager = None
    if args.use_budget:
        budget_manager = HTPSBudgetManager(
            max_budget_tokens=args.max_budget_tokens,
            max_thinking_extensions=args.extensions,
            extension_token=args.extension_token,
            confidence_threshold=0.9,
            complexity_threshold=0.6
        )
        logger.info(f"Budget Manager initialized with max budget: {args.max_budget_tokens}, "
                    f"extensions: {args.extensions}")
    
    # Tokenize input
    logger.info(f"Prompt: \"{args.prompt}\"")
    tokens = [ord(c) % config.vocab_size for c in args.prompt]
    input_ids = torch.tensor([tokens], dtype=torch.long, device=args.device)
    
    # Run generation with all requested features
    if args.use_recurrent:
        # Implement recurrent generation
        logger.info(f"Generating with recurrent processing: min_iter={args.min_iterations}, max_iter={args.max_iterations}")
        
        # Initialize tracking variables
        generated_ids = input_ids.clone()
        token_iterations = []
        value_histories = []
        
        # Generate tokens one at a time
        for i in range(args.max_length):
            # Forward pass to get logits and hidden states
            with torch.no_grad():
                # Get hidden states for the entire sequence so far
                outputs = model.forward(generated_ids)
                logits = outputs["logits"]
                
                # Get the last token's logits (what we'll use to generate the next token)
                next_token_logits = logits[:, -1, :]
                
                # Get initial hidden state value
                hidden_states = outputs["hidden_states"]
                if hidden_states is None:
                    # If hidden_states isn't returned, use forward_with_hidden_states
                    logits, all_hidden_states = model.forward_with_hidden_states(generated_ids)
                    last_hidden = all_hidden_states[-1][:, -1:, :].clone()
                else:
                    last_hidden = hidden_states[-1][:, -1:, :].clone()
                
                current_hidden = last_hidden.clone()
                
                # Get initial value
                initial_value = value_estimator(current_hidden).item()
                value_history = [initial_value]
                
                # Recurrent processing loop
                iterations = 0
                prev_value = initial_value
                continue_iterating = True
                change = float('inf')
                
                # Run at least min_iterations iterations
                while continue_iterating and iterations < args.max_iterations:
                    # Pass through the last transformer layer again
                    current_hidden = model.layers[-1].forward(current_hidden)[0]
                    
                    # Calculate value
                    value = value_estimator(current_hidden).item()
                    value_history.append(value)
                    
                    # Check convergence after min_iterations
                    iterations += 1
                    if iterations >= args.min_iterations:
                        change = abs(value - prev_value)
                        if change < args.convergence_threshold:
                            continue_iterating = False
                    
                    prev_value = value
                    
                    # Logging
                    if iterations % 5 == 0 or iterations == 1 or iterations == args.max_iterations or not continue_iterating:
                        logger.info(f"  Iteration {iterations}: value = {value:.4f}, change = {change:.6f}")
                
                # Record data for visualization
                token_iterations.append(iterations)
                value_histories.append(value_history)
                
                # Sample next token from improved hidden state
                # Get logits from the improved hidden state
                improved_logits = model.lm_head(current_hidden)
                
                # Apply temperature and sampling
                improved_logits = improved_logits / 0.8  # Temperature
                probs = torch.nn.functional.softmax(improved_logits, dim=-1)
                next_token = torch.multinomial(probs[0], 1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token.view(1, 1)], dim=1)
        
        # Convert to text
        generated_text = ""
        for token_id in generated_ids[0]:
            generated_text += chr(token_id.item() % 128)
        
        logger.info("==================================================")
        logger.info("GENERATED TEXT:")
        logger.info("--------------------------------------------------")
        logger.info(generated_text)
        logger.info("==================================================")
        logger.info("ITERATION STATISTICS:")
        logger.info(f"Total tokens generated: {len(token_iterations)}")
        logger.info(f"Total iterations: {sum(token_iterations)}")
        logger.info(f"Average iterations per token: {sum(token_iterations)/len(token_iterations):.2f}")
        logger.info(f"Max iterations for a token: {max(token_iterations)}")
        logger.info(f"Min iterations for a token: {min(token_iterations)}")
        
        # Value improvement statistics
        avg_initial = sum([histories[0] for histories in value_histories]) / len(value_histories)
        avg_final = sum([histories[-1] for histories in value_histories]) / len(value_histories)
        avg_improvement = avg_final - avg_initial
        
        logger.info("VALUE STATISTICS:")
        logger.info(f"Average starting value: {avg_initial:.4f}")
        logger.info(f"Average final value: {avg_final:.4f}")
        logger.info(f"Average improvement: {avg_improvement:.4f}")
        
        # Create visualization if requested
        if args.visualize:
            # Plot iterations per token
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(token_iterations)), token_iterations)
            plt.xlabel("Token Position")
            plt.ylabel("Iterations")
            plt.title("Iterations per Token")
            plt.savefig("iterations_per_token.png")
            logger.info("Saved iterations visualization to iterations_per_token.png")
            
            # Plot value convergence for a sample of tokens
            plt.figure(figsize=(12, 8))
            
            # Take a sample of tokens to visualize (to avoid overcrowding)
            sample_indices = list(range(0, len(value_histories), max(1, len(value_histories) // 10)))
            
            for idx in sample_indices:
                plt.plot(value_histories[idx], label=f"Token {idx}")
            
            plt.xlabel("Iteration")
            plt.ylabel("Value")
            plt.title("Value Convergence During Recurrent Processing")
            plt.legend()
            plt.savefig("value_convergence.png")
            logger.info("Saved value convergence plot to value_convergence.png")
            
    elif args.use_budget:
        # Implement budget-forced generation
        logger.info(f"Generating with budget forcing: max_tokens={args.max_budget_tokens}, extensions={args.extensions}")
        
        # Generate with budget forcing
        generated_ids = model.generate(
            input_ids,
            max_length=args.max_length,
            do_sample=True,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            budget_manager=budget_manager,
            task_complexity=0.7  # Medium complexity - can be adjusted
        )
        
        # Convert to text
        generated_text = ""
        for token_id in generated_ids[0]:
            generated_text += chr(token_id.item() % 128)
        
        logger.info("==================================================")
        logger.info("GENERATED TEXT:")
        logger.info("--------------------------------------------------")
        logger.info(generated_text)
        logger.info("==================================================")
        logger.info("BUDGET FORCING STATISTICS:")
        logger.info(f"  Total tokens generated: {budget_manager.token_count}")
        logger.info(f"  Extensions used: {budget_manager.extensions_used}")
        
        # Create visualization if requested
        if args.visualize:
            # Plot extension points
            extension_points = budget_manager.extension_points
            if extension_points:
                plt.figure(figsize=(10, 3))
                for point in extension_points:
                    plt.axvline(x=point, color='r', linestyle='--')
                plt.xlim(0, budget_manager.token_count)
                plt.xlabel("Token Position")
                plt.yticks([])
                plt.title("Budget Extension Points")
                plt.savefig("budget_extensions.png")
                logger.info("Saved budget extension visualization to budget_extensions.png")
    else:
        # Regular generation
        generated_ids = model.generate(
            input_ids,
            max_length=args.max_length,
            do_sample=True,
            temperature=0.8,
            top_k=40,
            top_p=0.9
        )
        
        # Convert IDs to text
        generated_text = ""
        for token_id in generated_ids[0]:
            generated_text += chr(token_id.item() % 128)
        
        logger.info("==================================================")
        logger.info("GENERATED TEXT:")
        logger.info("--------------------------------------------------")
        logger.info(generated_text)
        logger.info("==================================================")
        
        # Implement visualization if requested
        if args.visualize:
            # Create simple visualization for regular generation
            plt.figure(figsize=(10, 2))
            plt.text(0.5, 0.5, f"Generated {len(generated_text) - len(args.prompt)} tokens", 
                    horizontalalignment='center', fontsize=14)
            plt.axis('off')
            plt.savefig("regular_generation.png")
            logger.info("Saved regular generation info to regular_generation.png")
    
    # If multiple features are enabled, show how they work together
    if args.use_recurrent and args.use_kv_cache:
        logger.info("FEATURE INTEGRATION NOTES:")
        logger.info("- KV Cache management was active during recurrent processing")
        logger.info(f"- RAM offloading threshold: {args.offload_threshold} tokens")
        if args.use_budget:
            logger.info("- Budget forcing was integrated with recurrent processing")
            logger.info(f"- Total budget tokens: {args.max_budget_tokens}")
            logger.info(f"- Extensions used: {budget_manager.extensions_used}")

if __name__ == "__main__":
    main()