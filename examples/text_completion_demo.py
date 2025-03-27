#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text Completion Demo for EdgeFormer

This script demonstrates the code completion capabilities of the EdgeFormer model
with a focus on completing Python code snippets.
"""

import argparse
import time
import torch
import logging
import os
import sys
from typing import List, Dict, Any

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("text-completion-demo")

# Simple Python-aware tokenizer (for demonstration)
class SimplePythonTokenizer:
    """Simple tokenizer with added awareness of Python syntax."""
    
    def __init__(self, vocab_size: int = 30522):
        """Initialize a simple tokenizer."""
        self.vocab_size = vocab_size
        self.python_keywords = [
            "def", "class", "import", "from", "return", "if", "else", "elif", "for", 
            "while", "try", "except", "finally", "with", "as", "True", "False", "None",
            "and", "or", "not", "is", "in", "break", "continue", "pass", "lambda"
        ]
        # In a real implementation, you would have a proper tokenizer with vocab
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids with some Python-specific handling."""
        # This is a simplified implementation
        tokens = []
        # Split text into words and handle indentation
        lines = text.split('\n')
        for line in lines:
            # Count leading spaces for indentation
            indent_count = len(line) - len(line.lstrip(' '))
            if indent_count > 0:
                # Add special indentation tokens (space groups of 4)
                for _ in range(indent_count // 4):
                    tokens.append(5000)  # Special indent token
                line = line.lstrip(' ')
            
            # Split the line into words and special characters
            import re
            parts = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|\S)', line)
            
            for part in parts:
                if part in self.python_keywords:
                    # Use special token IDs for Python keywords
                    token_id = 10000 + self.python_keywords.index(part)
                elif part.isdigit():
                    # Use special token IDs for numbers
                    token_id = 20000 + (int(part) % 1000)
                else:
                    # Use character-level encoding for other tokens
                    token_id = ord(part[0]) % (self.vocab_size-1024) + 1000
                tokens.append(token_id)
            
            # Add newline token
            tokens.append(4000)
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids to text."""
        decoded = []
        indent_level = 0
    
        for token_id in token_ids:
            if token_id == 5000:
                # Indentation token
                indent_level += 1
            elif token_id == 4000:
                # Newline token
                decoded.append('\n' + '    ' * indent_level)
            elif 10000 <= token_id < 11000:
                # Python keyword
                keyword_idx = token_id - 10000
                if keyword_idx < len(self.python_keywords):
                    decoded.append(self.python_keywords[keyword_idx])
                else:
                    decoded.append("keyword")
            elif 20000 <= token_id < 21000:
                # Number
                decoded.append(str(token_id - 20000))
            else:
                # Regular character - with safety check
                try:
                    char_value = (token_id - 1000) + ord('a')
                    if 0 <= char_value <= 0x10FFFF:  # Valid Unicode range
                        char = chr(char_value)
                    else:
                        char = '?'  # Replacement for invalid characters
                    decoded.append(char)
                except (ValueError, OverflowError):
                    # Handle any other errors
                    decoded.append('?')
        
            # Add space between tokens except after indentation and before newlines
            if token_id != 5000 and token_id != 4000 and token_ids.index(token_id) < len(token_ids) - 1 and token_ids[token_ids.index(token_id) + 1] != 4000:
                decoded.append(' ')
            
        return ''.join(decoded)


def complete_code(
    model: EdgeFormer,
    tokenizer: SimplePythonTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.2,  # Lower temperature for more focused code completion
) -> str:
    """
    Complete Python code using the EdgeFormer model.
    
    Args:
        model: The EdgeFormer model
        tokenizer: Tokenizer for code text
        prompt: The code prompt to continue
        max_length: Maximum length of generated code (in tokens)
        temperature: Temperature for sampling (lower = more deterministic)
        
    Returns:
        The completed code including the prompt
    """
    model.eval()
    
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    
    # Create attention mask (all 1s for the prompt)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    # Initial forward pass to get the first set of logits and past key values
    logger.info(f"Initial pass with sequence length: {input_ids.shape[1]}")
    start_time = time.time()
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True
    )
    
    past_key_values = outputs["past_key_values"]
    next_token_logits = outputs["logits"][:, -1, :]
    
    # Track generation time
    time_elapsed = time.time() - start_time
    logger.info(f"Initial pass completed in {time_elapsed:.4f} seconds")
    
    # Generate tokens one by one
    generated_ids = input_ids.clone()
    
    # Track indentation for more realistic code generation
    indent_stack = []
    for token in input_ids[0]:
        if token.item() == 10001:  # 'def' token
            indent_stack.append('def')
        elif token.item() == 10000:  # 'class' token
            indent_stack.append('class')
        elif token.item() == 10007:  # 'if' token
            indent_stack.append('if')
        elif token.item() == 10008:  # 'else' token
            # Replace the last 'if' with 'else'
            if indent_stack and indent_stack[-1] == 'if':
                indent_stack[-1] = 'else'
        elif token.item() == 10009:  # 'elif' token
            # Replace the last 'if' with 'elif'
            if indent_stack and indent_stack[-1] == 'if':
                indent_stack[-1] = 'elif'
    
    # Maintain a tracking system for bracket/parenthesis balance
    open_brackets = {"(": 0, "[": 0, "{": 0}
    matching_close = {"(": ")", "[": "]", "{": "}"}
    
    for i in range(max_length):
        # Apply temperature
        scaled_logits = next_token_logits / temperature
        
        # Special handling for code structures (indentation, brackets, etc.)
        if generated_ids[0, -1].item() == 10000:  # 'class' token
            # Bias towards adding a name after 'class'
            scaled_logits[0, 1000:1026] += 2.0  # Bias toward letter characters
        elif generated_ids[0, -1].item() == 10001:  # 'def' token
            # Bias towards adding a name after 'def'
            scaled_logits[0, 1000:1026] += 2.0  # Bias toward letter characters
        
        # Handle bracket balancing
        if i > 0:
            last_token = generated_ids[0, -1].item()
            if last_token == ord('(') % (tokenizer.vocab_size-1024) + 1000:
                open_brackets["("] += 1
            elif last_token == ord('[') % (tokenizer.vocab_size-1024) + 1000:
                open_brackets["["] += 1
            elif last_token == ord('{') % (tokenizer.vocab_size-1024) + 1000:
                open_brackets["{"] += 1
            elif last_token == ord(')') % (tokenizer.vocab_size-1024) + 1000:
                open_brackets["("] = max(0, open_brackets["("] - 1)
            elif last_token == ord(']') % (tokenizer.vocab_size-1024) + 1000:
                open_brackets["["] = max(0, open_brackets["["] - 1)
            elif last_token == ord('}') % (tokenizer.vocab_size-1024) + 1000:
                open_brackets["{"] = max(0, open_brackets["{"] - 1)
        
        # Apply bracket balancing as we approach the end of generation
        if i > max_length * 0.75:
            # Increase probability of closing brackets if we have open ones
            for bracket, count in open_brackets.items():
                if count > 0:
                    close_bracket_token = ord(matching_close[bracket]) % (tokenizer.vocab_size-1024) + 1000
                    scaled_logits[0, close_bracket_token] += 1.0
        
        # Sample from the filtered distribution
        probs = torch.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append the new token to the sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # Use the helper method for continuation with the KV cache
        start_time = time.time()
        continuation_outputs = model.continue_generation(next_token, past_key_values)
        past_key_values = continuation_outputs["past_key_values"]
        next_token_logits = continuation_outputs["logits"][:, -1, :]
        
        # Update indentation tracking
        if next_token.item() == 10001:  # 'def' token
            indent_stack.append('def')
        elif next_token.item() == 10000:  # 'class' token
            indent_stack.append('class')
        elif next_token.item() == 10007:  # 'if' token
            indent_stack.append('if')
        elif next_token.item() == 10008:  # 'else' token
            if indent_stack and indent_stack[-1] == 'if':
                indent_stack[-1] = 'else'
        
        # Track token generation time
        time_elapsed = time.time() - start_time
        
        # Print progress
        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i + 1} tokens, last token time: {time_elapsed:.4f}s")
        
        # Check for early stopping if we've completed a function or class definition
        if (len(indent_stack) == 0 and i > 20) or i == max_length - 1:
            # If we've returned to the original indentation level and generated a fair bit
            # or if we've reached the maximum length, stop generation
            break
    
    # Decode the generated tokens
    completed_code = tokenizer.decode(generated_ids[0].tolist())
    
    return completed_code


def main(args):
    # Create the tokenizer
    tokenizer = SimplePythonTokenizer(vocab_size=args.vocab_size)
    
    # Create model configuration
    config = EdgeFormerConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        latent_size_factor=args.latent_size_factor,
    )
    
    # Initialize model
    logger.info("Initializing EdgeFormer model...")
    model = EdgeFormer(config)
    model.eval()
    
    # Load pretrained weights if specified
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
        try:
            model.load_state_dict(torch.load(args.model_path))
        except Exception as e:
            logger.warning(f"Failed to load model: {str(e)}")
            logger.info("Continuing with randomly initialized model")
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    
    # Complete code
    logger.info(f"Completing code from prompt:\n{args.prompt}")
    
    # Perform code completion
    start_time = time.time()
    completed_code = complete_code(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
    )
    
    time_elapsed = time.time() - start_time
    logger.info(f"Code completion finished in {time_elapsed:.2f} seconds")
    
    # Print the completed code
    print("\n" + "=" * 40 + " COMPLETED CODE " + "=" * 40)
    print(completed_code)
    print("=" * 91 + "\n")
    
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(completed_code)
        logger.info(f"Saved completed code to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeFormer Code Completion Demo")
    
    # Model parameters
    parser.add_argument("--vocab_size", type=int, default=30522, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of hidden layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--latent_size_factor", type=int, default=8, help="Latent size factor for MLA")
    
    # Completion parameters
    parser.add_argument("--prompt", type=str, default="def calculate_fibonacci(n):", 
                        help="Code prompt to complete")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    
    # Additional parameters
    parser.add_argument("--model_path", type=str, default="", help="Path to pretrained model")
    parser.add_argument("--output_file", type=str, default="", help="Optional file to save completed code")
    
    args = parser.parse_args()
    main(args)