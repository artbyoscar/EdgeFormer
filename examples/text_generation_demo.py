#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text Generation Demo for EdgeFormer

This script demonstrates text generation capabilities of the EdgeFormer model,
including the ability to use KV cache offloading for longer sequences.
"""

import argparse
import time
import torch
import logging
import os
import sys
from typing import List, Optional

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from src.utils.kv_cache_offload import kv_cache_offload

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("text-generation-demo")

# Simple tokenizer implementation for demo purposes
class SimpleTokenizer:
    """Simple tokenizer for demonstration purposes."""
    
    def __init__(self, vocab_size: int = 30522):
        """Initialize a simple tokenizer."""
        self.vocab_size = vocab_size
        # In a real implementation, you would have a proper tokenizer
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        # This is a placeholder. In a real implementation, you would use a proper tokenizer
        # For demo, we'll just use character-level encoding
        encoded = [ord(c) % (self.vocab_size-1024) + 1000 for c in text]
        return encoded
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids to text."""
        # This is a placeholder. In a real implementation, you would use a proper tokenizer
        # For demo, we'll convert back from our simple character encoding
        decoded = ''.join([chr((token_id - 1000) + ord('a')) if 1000 <= token_id < self.vocab_size else ' ' 
                         for token_id in token_ids])
        return decoded


def generate_text(
    model: EdgeFormer,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    use_kv_offload: bool = False,
) -> str:
    """
    Generate text using the EdgeFormer model.
    
    Args:
        model: The EdgeFormer model
        tokenizer: Tokenizer to convert between text and token ids
        prompt: The text prompt to continue from
        max_length: Maximum length of generated text (in tokens)
        temperature: Temperature for sampling (higher = more random)
        top_k: Number of highest probability tokens to consider for sampling
        top_p: Cumulative probability cutoff for sampling
        use_kv_offload: Whether to use KV cache offloading
        
    Returns:
        The generated text including the prompt
    """
    model.eval()
    
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    
    # Create attention mask (all 1s for the prompt)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    # Store the original prompt length to return the generated part later
    prompt_length = input_ids.shape[1]
    
    # Set up offload directory if using KV offloading
    offload_directory = None
    if use_kv_offload:
        offload_directory = "./kv_cache_offload"
        os.makedirs(offload_directory, exist_ok=True)
    
    # Initial forward pass to get the first set of logits and past key values
    logger.info(f"Initial pass with sequence length: {input_ids.shape[1]}")
    start_time = time.time()
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True
    )
    
    past_key_values = outputs["past_key_values"]
    
    # Get logits for the next token prediction
    next_token_logits = outputs["logits"][:, -1, :]
    
    # Track generation time
    time_elapsed = time.time() - start_time
    logger.info(f"Initial pass completed in {time_elapsed:.4f} seconds")
    
    # Generate tokens one by one
    generated_ids = input_ids.clone()
    
    for i in range(max_length):
        # Apply temperature
        scaled_logits = next_token_logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
            scaled_logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            scaled_logits[0, indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = torch.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append the new token to the sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # If using KV cache offloading, handle the offloading process here
        if use_kv_offload and i > 0 and i % 64 == 0:
            logger.info(f"Offloading KV cache at step {i}")
            # Create a temporary directory for offloading if needed
            if not hasattr(model, 'kv_cache_offload_path'):
                model = kv_cache_offload(model, offload_directory)

        
        # Use the helper method for continuation with the KV cache
        start_time = time.time()
        continuation_outputs = model.continue_generation(next_token, past_key_values)
        past_key_values = continuation_outputs["past_key_values"]
        next_token_logits = continuation_outputs["logits"][:, -1, :]
        
        # Track token generation time
        time_elapsed = time.time() - start_time
        
        # Print progress
        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i + 1} tokens, last token time: {time_elapsed:.4f}s")
            
        # Early stopping if we generate an EOS token (use 102 as sample EOS token)
        if next_token.item() == 102:
            break
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    # Clean up offload directory if it was created
    if use_kv_offload and offload_directory and os.path.exists(offload_directory):
        for file in os.listdir(offload_directory):
            os.remove(os.path.join(offload_directory, file))
        os.rmdir(offload_directory)
    
    return generated_text


def main(args):
    # Create the tokenizer
    if args.use_tokenizer == "basic":
        # Use the existing SimpleTokenizer
        tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
        logger.info("Using SimpleTokenizer")
    elif args.use_tokenizer == "better":
        # Import and use the BetterTokenizer
        try:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from better_tokenization import BetterTokenizer
            tokenizer = BetterTokenizer()  # No vocab_size parameter here
            logger.info(f"Using BetterTokenizer with vocabulary size {tokenizer.vocab_size}")
        except ImportError:
            logger.warning("BetterTokenizer not found, falling back to SimpleTokenizer")
            tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    
    def load_model(args):
        """Load the EdgeFormer model."""
        config = EdgeFormerConfig(
            vocab_size=30522,
            hidden_size=128,  # Use the detected value from model_load_fix.py
            num_hidden_layers=2,  # Use the detected value
            num_attention_heads=4,  # Use the detected value
            latent_size_factor=8,
            intermediate_size=1024,
            max_position_embeddings=128  # Use the detected value
        )
    
        model = EdgeFormer(config)
    
        if args.model_path:
            logger.info(f"Loading model from {args.model_path}")
            model.load_state_dict(torch.load(args.model_path))
    
        model.eval()
        return model

    # In main() function:
    # Replace the model configuration and loading code with:
    logger.info("Loading EdgeFormer model...")
    model = load_model(args)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    
    # Generate text
    logger.info(f"Generating text from prompt: '{args.prompt}'")
    logger.info(f"Using KV cache offloading: {args.use_kv_offload}")
    
    # Perform text generation
    start_time = time.time()
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_kv_offload=args.use_kv_offload,
    )
    
    time_elapsed = time.time() - start_time
    logger.info(f"Text generation completed in {time_elapsed:.2f} seconds")
    
    # Print the generated text
    print("\n" + "=" * 40 + " GENERATED TEXT " + "=" * 40)
    print(generated_text)
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeFormer Text Generation Demo")
    
    # Model parameters
    parser.add_argument("--vocab_size", type=int, default=30522, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of hidden layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--latent_size_factor", type=int, default=8, help="Latent size factor for MLA")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="EdgeFormer is a custom transformer that", 
                        help="Text prompt to continue from")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    
    # Additional parameters
    parser.add_argument("--model_path", type=str, default="", help="Path to pretrained model")
    parser.add_argument("--use_kv_offload", action="store_true", help="Use KV cache offloading")
    parser.add_argument("--use_tokenizer", type=str, default="basic", choices=["basic", "better"],help="Which tokenizer to use (basic or better)")
    
    args = parser.parse_args()
    main(args)