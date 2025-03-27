# examples/optimized_demo.py
import torch
import time
import sys
import os
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig

def generate_text(model, prompt, max_length=100, temperature=0.8):
    """Generate text using the model."""
    # Tokenize prompt (simplified for demo)
    input_ids = torch.tensor([[ord(c) % model.config.vocab_size for c in prompt]])
    
    # Generate text
    print(f"Generating text from prompt: '{prompt}'")
    print(f"Using {model.__class__.__name__} with {args.attention_type} attention")
    
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=len(prompt) + max_length,
            temperature=temperature,
            do_sample=True
        )
    end_time = time.time()
    
    # Convert output ids to text (simplified for demo)
    output_text = "".join([chr((id % 26) + 97) for id in output_ids[0][len(prompt):]])
    
    print(f"\nGeneration completed in {end_time - start_time:.2f}s")
    print(f"Generated text:\n{prompt}{output_text}")
    
    return output_text

def main():
    parser = argparse.ArgumentParser(description="EdgeFormer Text Generation Demo")
    parser.add_argument("--attention_type", type=str, default="standard",
                        choices=["standard", "mla", "mla_window"],
                        help="Type of attention to use")
    parser.add_argument("--prompt", type=str, default="EdgeFormer is a transformer that",
                        help="Prompt for text generation")
    parser.add_argument("--length", type=int, default=100,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    args = parser.parse_args()
    
    # Create config based on attention type
    if args.attention_type == "standard":
        config = EdgeFormerConfig(
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            latent_size_factor=8,
            intermediate_size=1024,
            max_position_embeddings=2048,
        )
    elif args.attention_type == "mla":
        config = EdgeFormerConfig(
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            latent_size_factor=8,
            max_position_embeddings=2048,
        )
    else:  # mla_window
        config = EdgeFormerConfig(
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            latent_size_factor=8,
            use_sliding_window=True,
            sliding_window_size=512,
            max_position_embeddings=2048,
        )
    
    # Initialize model
    model = EdgeFormer(config)
    model.eval()
    
    # Generate text
    generate_text(model, args.prompt, args.length, args.temperature)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeFormer Text Generation Demo")
    parser.add_argument("--attention_type", type=str, default="standard",
                        choices=["standard", "mla", "mla_window"],
                        help="Type of attention to use")
    parser.add_argument("--prompt", type=str, default="EdgeFormer is a transformer that",
                        help="Prompt for text generation")
    parser.add_argument("--length", type=int, default=100,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    args = parser.parse_args()
    main()