#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo for Custom EdgeFormer Model

This script demonstrates text generation using the custom EdgeFormer model
that matches the structure of the saved model file.
"""

import os
import sys
import torch
import time
import logging
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom model loader
from custom_edgeformer_loader import load_custom_model
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
logger = logging.getLogger("custom-demo")

def generate_text(
    model,
    tokenizer,
    prompt="EdgeFormer is a custom transformer that",
    max_length=50,
    temperature=0.7
):
    """
    Generate text using the custom EdgeFormer model.
    
    Args:
        model: The custom EdgeFormer model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Text prompt to continue
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text
    """
    logger.info("Starting text generation...")
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    
    # Create attention mask
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    # Record the time
    start_time = time.time()
    
    # Initial forward pass
    logger.info("Running initial forward pass")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True
        )
    
    # Get logits and past key values
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
        with torch.no_grad():
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
    # Load the model
    logger.info(f"Loading model from {args.model_path}")
    
    try:
        model, config = load_custom_model(args.model_path)
        
        # Create tokenizer
        if args.use_better_tokenizer:
            logger.info("Using BetterTokenizer")
            tokenizer = BetterTokenizer()
        else:
            logger.info("Using SimpleTokenizer")
            from text_generation_demo import SimpleTokenizer
            tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
        
        # Generate text
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        # Print the result
        print("\n" + "=" * 40 + " GENERATED TEXT " + "=" * 40)
        print(generated_text)
        print("=" * 90 + "\n")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error loading or running model: {str(e)}")
        
        # Fall back to random model
        logger.info("Falling back to random model...")
        
        # Run the simple generation demo as a fallback
        from simple_generation_demo import generate_random_text
        
        generated_text = generate_random_text(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        # Print the result
        print("\n" + "=" * 40 + " GENERATED TEXT (RANDOM MODEL) " + "=" * 40)
        print(generated_text)
        print("=" * 90 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom EdgeFormer Demo")
    
    parser.add_argument("--model_path", type=str, default="./mobile_model/edgeformer.pt",
                        help="Path to model file")
    parser.add_argument("--prompt", type=str, default="EdgeFormer is a custom transformer that",
                        help="Text prompt to continue from")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--use_better_tokenizer", action="store_true",
                        help="Use the better tokenizer if available")
    
    args = parser.parse_args()
    main(args)