import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from tqdm import tqdm
import time
import logging

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer, EdgeFormerConfig
from src.utils.improved_value_estimator import ImprovedValueEstimator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger('value_recurrent_demo')

class SimpleTokenizer:
    """Very simple character-level tokenizer for demo purposes"""
    def __init__(self):
        self.vocab = {ch: i+1 for i, ch in enumerate(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?:;\'"-+=/\\(){}[]<>*&^%$#@_')}
        self.vocab_size = len(self.vocab) + 1  # +1 for unknown tokens
        
    def encode(self, text):
        return torch.tensor([self.vocab.get(c, 0) for c in text], dtype=torch.long)
    
    def decode(self, tokens):
        # Create reverse mapping
        id_to_token = {v: k for k, v in self.vocab.items()}
        return ''.join([id_to_token.get(t.item(), '') for t in tokens if t.item() in id_to_token])

def create_model(vocab_size, device, config=None):
    """Create a small EdgeFormer model for demo purposes"""
    if config is None:
        # Default configuration for demo
        config = EdgeFormerConfig(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=512,
            max_position_embeddings=512,
            attention_type="mla",  # Use MLA by default
        )
    
    model = EdgeFormer(config)
    model.to(device)
    
    # Initialize improved value estimator
    value_estimator = ImprovedValueEstimator(config.hidden_size, config)
    value_estimator.to(device)
    
    return model, value_estimator

def generate_with_recurrent_processing(
    model, 
    tokenizer, 
    prompt, 
    value_estimator,
    device,
    max_length=100, 
    min_iterations=2,
    max_iterations=20,
    convergence_threshold=0.005,
    temperature=0.7,
    visualize=False
):
    """
    Generate text with value-based recurrent depth processing.
    """
    logger.info(f"Generating with recurrent processing: min_iter={min_iterations}, max_iter={max_iterations}")
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt).to(device)
    
    # Initialize for tracking
    generated_ids = input_ids.clone()
    all_token_iterations = []
    all_value_histories = []
    total_iterations = 0
    
    start_time = time.time()

    # Generation loop
    with torch.no_grad():
        for i in range(max_length):
            # Forward pass
            logits, hidden_states = model.forward_with_hidden_states(
                generated_ids.unsqueeze(0)
            )
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Initialize value estimator for this token
            value_estimator.reset()
            token_value_history = []
            
            # Recurrent processing loop for current token
            last_hidden = hidden_states[-1][:, -1:, :]  # Get last token's hidden state
            current_hidden = last_hidden.clone()
            
            # Track iterations for this token
            iteration_count = 0
            
            # Recurrent processing loop
            pbar = tqdm(total=max_iterations, desc=f"Token {i+1} iterations", disable=not visualize)
            
            while iteration_count < max_iterations:
                # Estimate value
                value = value_estimator(current_hidden).mean().item()
                token_value_history.append(value)
                
                # Print verbose iteration info
                if visualize and iteration_count % 2 == 0:
                    logger.info(f"  Iteration {iteration_count}: value = {value:.4f}")
                
                # Check if we should continue iterating
                should_continue = value_estimator.should_continue_iteration(
                    current_hidden, iteration_count, min_iterations, max_iterations
                )
                
                # Exit if convergence reached
                if not should_continue:
                    logger.info(f"  Converged after {iteration_count} iterations with value {value:.4f}")
                    break
                
                # Apply iterative recurrent processing using appropriate layers from model
                # This is a simplified version - in a full implementation, you would use
                # dedicated recurrent modules
                current_hidden = model.layers[-1].forward(current_hidden)[0]
                
                iteration_count += 1
                total_iterations += 1
                pbar.update(1)
            
            pbar.close()
            
            # Store iteration information
            all_token_iterations.append(iteration_count)
            all_value_histories.append(token_value_history)
            
            # Sample from the vocabulary distribution
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add the next token to the sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Print progress
            if i % 5 == 0 or i == max_length - 1:
                logger.info(f"Generated {i+1}/{max_length} tokens")
                
    end_time = time.time()
    time_taken = end_time - start_time
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(generated_ids)
    
    logger.info(f"Generation complete: {len(generated_ids)} tokens, {total_iterations} total iterations")
    logger.info(f"Average iterations per token: {total_iterations / (len(generated_ids) - len(input_ids)):.2f}")
    logger.info(f"Time taken: {time_taken:.2f} seconds")
    
    # Create visualizations if requested
    if visualize:
        # Plot iterations per token
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(all_token_iterations)), all_token_iterations, marker='o')
        plt.title(f'Iterations per Token (min={min_iterations}, max={max_iterations})')
        plt.xlabel('Token Position')
        plt.ylabel('Iterations')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('token_iterations.png')
        logger.info("Saved iterations plot to token_iterations.png")
        
        # Plot value history for a sample of tokens
        plt.figure(figsize=(12, 6))
        # Take up to 5 tokens to visualize
        tokens_to_plot = min(5, len(all_value_histories))
        for i in range(tokens_to_plot):
            plt.plot(range(len(all_value_histories[i])), all_value_histories[i], 
                     marker='o', label=f'Token {len(input_ids) + i}')
        plt.title('Value Convergence During Recurrent Processing')
        plt.xlabel('Iteration')
        plt.ylabel('Value Estimate')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('value_convergence.png')
        logger.info("Saved value convergence plot to value_convergence.png")
    
    return generated_text, all_token_iterations, all_value_histories

def main():
    parser = argparse.ArgumentParser(description='Value-Based Recurrent Reasoning Demo')
    parser.add_argument('--prompt', type=str, default="Solve this math problem:",
                        help='Text prompt for generation')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum length of generated sequence')
    parser.add_argument('--min_iterations', type=int, default=2,
                        help='Minimum iterations per token')
    parser.add_argument('--max_iterations', type=int, default=20,
                        help='Maximum iterations per token')
    parser.add_argument('--convergence_threshold', type=float, default=0.005,
                        help='Convergence threshold for early stopping')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu/cuda)')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device)
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    
    # Create model and value estimator
    model, value_estimator = create_model(tokenizer.vocab_size, device)
    
    logger.info(f"Initialized EdgeFormer model with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Value Estimator has {sum(p.numel() for p in value_estimator.parameters())} parameters")
    
    # Print prompt
    logger.info(f"Prompt: \"{args.prompt}\"")
    
    # Generate text with recurrent processing
    generated_text, token_iterations, value_histories = generate_with_recurrent_processing(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        value_estimator=value_estimator,
        device=device,
        max_length=args.max_length,
        min_iterations=args.min_iterations,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        temperature=args.temperature,
        visualize=args.visualize
    )
    
    # Print generated text
    print("\n" + "="*50)
    print("GENERATED TEXT:")
    print("-"*50)
    print(generated_text)
    print("="*50)
    
    # Print iteration statistics
    total_iterations = sum(token_iterations)
    avg_iterations = total_iterations / len(token_iterations)
    
    print("\nITERATION STATISTICS:")
    print(f"Total tokens generated: {len(token_iterations)}")
    print(f"Total iterations: {total_iterations}")
    print(f"Average iterations per token: {avg_iterations:.2f}")
    print(f"Max iterations for a token: {max(token_iterations)}")
    print(f"Min iterations for a token: {min(token_iterations)}")
    
    # Print value estimator statistics
    if value_histories:
        avg_start_value = sum(history[0] for history in value_histories) / len(value_histories)
        avg_end_value = sum(history[-1] for history in value_histories) / len(value_histories)
        
        print("\nVALUE STATISTICS:")
        print(f"Average starting value: {avg_start_value:.4f}")
        print(f"Average final value: {avg_end_value:.4f}")
        print(f"Average improvement: {avg_end_value - avg_start_value:.4f}")

if __name__ == "__main__":
    main()