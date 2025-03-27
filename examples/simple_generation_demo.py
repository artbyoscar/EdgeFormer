#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Text Generation Demo for EdgeFormer

This script demonstrates text generation with EdgeFormer using a random model
(no pre-trained weights) for demonstration purposes.
"""

import os
import sys
import time
import torch
import logging
import argparse

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
try:
    from better_tokenization import BetterTokenizer
except ImportError:
    # Fallback to simple tokenizer
    from text_generation_demo import SimpleTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("simple-demo")

def generate_random_text(
    prompt="EdgeFormer is a custom transformer that",
    max_length=50,
    temperature=0.7,
    hidden_size=128,
    num_layers=2
):
    """
    Generate text using a randomly initialized EdgeFormer model.
    
    Args:
        prompt: Text prompt to continue from
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        hidden_size: Hidden size for the model
        num_layers: Number of transformer layers
    
    Returns:
        Generated text
    """
    logger.info(f"Initializing model with hidden_size={hidden_size}, num_layers={num_layers}")
    
    # Create model configuration
    config = EdgeFormerConfig(
        vocab_size=30522,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=hidden_size // 32,  # Common ratio
        latent_size_factor=8,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=128
    )
    
    # Create model
    model = EdgeFormer(config)
    model.eval()
    
    # Create tokenizer
    try:
        tokenizer = BetterTokenizer()
        logger.info("Using BetterTokenizer")
    except NameError:
        tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
        logger.info("Using SimpleTokenizer")
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    
    # Create attention mask
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    # Record the time
    start_time = time.time()
    
    # Initial forward pass
    logger.info("Running initial forward pass")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True
    )
    
    # Get past key values and logits
    past_key_values = outputs["past_key_values"]
    next_token_logits = outputs["logits"][:, -1, :]
    
    # Generate text token by token
    generated_text = prompt
    generated_ids = input_ids.clone()
    
    logger.info(f"Generating {max_length} tokens...")
    for i in range(max_length):
        # Apply temperature
        scaled_logits = next_token_logits / temperature
        
        # Sample from the distribution
        probs = torch.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Add the new token to the sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # Continue with the model
        continuation_outputs = model.continue_generation(next_token, past_key_values)
        past_key_values = continuation_outputs["past_key_values"]
        next_token_logits = continuation_outputs["logits"][:, -1, :]
        
        # Decode the token and update the generated text
        next_token_text = tokenizer.decode([next_token.item()])
        generated_text += next_token_text
        
        # Print progress
        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i + 1} tokens")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    tokens_per_second = max_length / elapsed_time
    
    logger.info(f"Generation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Throughput: {tokens_per_second:.2f} tokens/second")
    
    return generated_text

def main(args):
    """Main function."""
    # Generate text
    generated_text = generate_random_text(
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )
    
    # Print the result
    print("\n" + "=" * 40 + " GENERATED TEXT " + "=" * 40)
    print(generated_text)
    print("=" * 90 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeFormer Simple Text Generation Demo")
    
    parser.add_argument("--prompt", type=str, default="EdgeFormer is a custom transformer that",
                        help="Text prompt to continue from")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size for the model")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of transformer layers")
    
    args = parser.parse_args()
    main(args)